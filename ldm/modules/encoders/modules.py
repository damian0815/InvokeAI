import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import kornia
from ldm.dream.devices import choose_torch_device

from ldm.modules.x_transformer import (
    Encoder,
    TransformerWrapper,
)  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


def _expand_mask(mask, dtype, tgt_len=None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = (
        mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    )

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def _build_causal_attention_mask(bsz, seq_len, dtype):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""

    def __init__(
        self,
        n_embed,
        n_layer,
        vocab_size,
        max_seq_len=77,
        device=choose_torch_device(),
    ):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
        )

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(
        self, device=choose_torch_device(), vq_interface=True, max_length=77
    ):
        super().__init__()
        from transformers import (
            BertTokenizerFast,
        )  # TODO: add to reuquirements

        # Modified to allow to run on non-internet connected compute nodes.
        # Model needs to be loaded into cache from an internet-connected machine
        # by running:
        #   from transformers import BertTokenizerFast
        #   BertTokenizerFast.from_pretrained("bert-base-uncased")
        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(
                'bert-base-uncased', local_files_only=True
            )
        except OSError:
            raise SystemExit(
                "* Couldn't load Bert tokenizer files. Try running scripts/preload_models.py from an internet-conected machine."
            )
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding='max_length',
            return_tensors='pt',
        )
        tokens = batch_encoding['input_ids'].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""

    def __init__(
        self,
        n_embed,
        n_layer,
        vocab_size=30522,
        max_seq_len=77,
        device=choose_torch_device(),
        use_tokenizer=True,
        embedding_dropout=0.0,
    ):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(
                vq_interface=False, max_length=max_seq_len
            )
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
            emb_dropout=embedding_dropout,
        )

    def forward(self, text, embedding_manager=None):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)  # .to(self.device)
        else:
            tokens = text
        z = self.transformer(
            tokens, return_embeddings=True, embedding_manager=embedding_manager
        )
        return z

    def encode(self, text, **kwargs):
        # output of length 77
        return self(text, **kwargs)


class SpatialRescaler(nn.Module):
    def __init__(
        self,
        n_stages=1,
        method='bilinear',
        multiplier=0.5,
        in_channels=3,
        out_channels=None,
        bias=False,
    ):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in [
            'nearest',
            'linear',
            'bilinear',
            'trilinear',
            'bicubic',
            'area',
        ]
        self.multiplier = multiplier
        self.interpolator = partial(
            torch.nn.functional.interpolate, mode=method
        )
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(
                f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.'
            )
            self.channel_mapper = nn.Conv2d(
                in_channels, out_channels, 1, bias=bias
            )

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self,
        version='openai/clip-vit-large-patch14',
        device=choose_torch_device(),
        max_length=77,
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(
            version, local_files_only=True
        )
        self.transformer = CLIPTextModel.from_pretrained(
            version, local_files_only=True
        )
        self.device = device
        self.max_length = max_length
        self.freeze()

        def embedding_forward(
            self,
            input_ids=None,
            position_ids=None,
            inputs_embeds=None,
            embedding_manager=None,
        ) -> torch.Tensor:

            seq_length = (
                input_ids.shape[-1]
                if input_ids is not None
                else inputs_embeds.shape[-2]
            )

            if position_ids is None:
                position_ids = self.position_ids[:, :seq_length]

            if inputs_embeds is None:
                inputs_embeds = self.token_embedding(input_ids)

            if embedding_manager is not None:
                inputs_embeds = embedding_manager(input_ids, inputs_embeds)

            position_embeddings = self.position_embedding(position_ids)
            embeddings = inputs_embeds + position_embeddings

            return embeddings

        self.transformer.text_model.embeddings.forward = (
            embedding_forward.__get__(self.transformer.text_model.embeddings)
        )

        def encoder_forward(
            self,
            inputs_embeds,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )
            return_dict = (
                return_dict
                if return_dict is not None
                else self.config.use_return_dict
            )

            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            hidden_states = inputs_embeds
            for idx, encoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)

                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            return hidden_states

        self.transformer.text_model.encoder.forward = encoder_forward.__get__(
            self.transformer.text_model.encoder
        )

        def text_encoder_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            embedding_manager=None,
        ):
            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )
            return_dict = (
                return_dict
                if return_dict is not None
                else self.config.use_return_dict
            )

            if input_ids is None:
                raise ValueError('You have to specify either input_ids')

            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            hidden_states = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                embedding_manager=embedding_manager,
            )

            bsz, seq_len = input_shape
            # CLIP's text model uses causal mask, prepare it here.
            # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
            causal_attention_mask = _build_causal_attention_mask(
                bsz, seq_len, hidden_states.dtype
            ).to(hidden_states.device)

            # expand attention_mask
            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _expand_mask(
                    attention_mask, hidden_states.dtype
                )

            last_hidden_state = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_state = self.final_layer_norm(last_hidden_state)

            return last_hidden_state

        self.transformer.text_model.forward = text_encoder_forward.__get__(
            self.transformer.text_model
        )

        def transformer_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            embedding_manager=None,
        ):
            return self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                embedding_manager=embedding_manager,
            )

        self.transformer.forward = transformer_forward.__get__(
            self.transformer
        )

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, **kwargs):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding='max_length',
            return_tensors='pt',
        )
        tokens = batch_encoding['input_ids'].to(self.device)
        z = self.transformer(input_ids=tokens, **kwargs)

        return z

    def encode(self, text, **kwargs):
        return self(text, **kwargs)

class WeightedFrozenCLIPEmbedder(FrozenCLIPEmbedder):

    attention_weights_key = "attention_weights"

    def build_token_list_fragment(self, fragment: str, weight: float) -> (torch.Tensor, torch.Tensor):
        batch_encoding = self.tokenizer(
            fragment,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding='none',
            return_tensors='pt',
        )
        return batch_encoding, torch.ones_like(batch_encoding) * weight

    def forward(self, text, **kwargs):

        if self.attention_weights_key not in kwargs:
            # fallback to base class implementation
            return super().forward(text, **kwargs)

        attention_weights = kwargs[self.attention_weights_key]

        batch_tokens = None
        batch_weights = None
        for item_fragments, item_weights in zip(text, attention_weights):
            item_encodings = self.tokenizer(
                item_fragments,
                truncation=True,
                max_length=self.max_length,
                return_overflowing_tokens=False,
                padding='do_not_pad',
                return_tensors=None,  # just give me a list of ints
            )['input_ids']
            all_tokens = []
            per_token_weights = []
            print("all fragments:", item_fragments, item_weights)
            for index, fragment in enumerate(item_encodings):
                weight = item_weights[index]
                print("processing fragment", fragment, weight)
                fragment_tokens = item_encodings[index]
                print("fragment", fragment, "processed to", fragment_tokens)
                # trim bos and eos markers before appending
                all_tokens.extend(fragment_tokens[1:-1])
                per_token_weights.extend([weight] * (len(fragment_tokens) - 2))

            if len(all_tokens) > self.max_length-2:
                print("prompt is too long and has been truncated")
                all_tokens = all_tokens[:self.max_length-2]

            # build a 77-entry array: [eos_token, <prompt tokens>, eos_token, ..., eos_token]
            # (77 = self.max_length)
            pad_length = self.max_length - 1 - len(all_tokens)
            all_tokens.insert(0, self.tokenizer.bos_token_id)
            all_tokens.extend([self.tokenizer.eos_token_id] * pad_length)
            per_token_weights.insert(0, 1)
            per_token_weights.extend([1] * pad_length)

            item_tokens = torch.tensor(all_tokens, dtype=torch.long).to(self.device)
            item_weights = torch.tensor(per_token_weights, dtype=torch.float32).to(self.device)
            print(f"assembled tokens for '{fragment}' into tensor of shape {item_tokens.shape}")

            # append to batch
            batch_weights = item_weights.unsqueeze(0) if batch_weights is None else torch.cat((batch_weights, item_weights), 1)
            batch_tokens = item_tokens.unsqueeze(0) if batch_tokens is None else torch.cat((batch_tokens, item_tokens), 1)

        # apply attention weighting to all items in batch

        # self.transformer doesn't like receiving "attention_weights" as an argument
        kwargs.pop(self.attention_weights_key)
        batch_z = self.transformer(input_ids=batch_tokens, **kwargs)
        original_mean = batch_z.mean()
        batch_weights_expanded = batch_weights.reshape(batch_weights.shape + (1,)).expand(batch_z.shape)

        weight_delta_from_empty = False
        if weight_delta_from_empty:
            empty_tokens = self.tokenizer([''] * batch_z.shape[0],
                                     truncation=True,
                                     max_length=self.max_length,
                                     padding='max_length',
                                     return_tensors='pt'
                                     )['input_ids'].to(self.device)
            empty_z = self.transformer(input_ids=empty_tokens, **kwargs)
            z_delta_from_empty = batch_z - empty_z
            weighted_z = empty_z + (z_delta_from_empty * batch_weights_expanded)

            weighted_z_delta_from_empty = (weighted_z-empty_z)
            print("weighted z has delta from empty with sum", weighted_z_delta_from_empty.sum(), "mean", weighted_z_delta_from_empty.mean() )

            #print("using empty-delta method, first 5 rows:")
            #print(weighted_z[:5])

            return weighted_z

        else:

            batch_z *= batch_weights_expanded
            after_weighting_mean = batch_z.mean()
            # correct the mean. not sure if this is right but it's what the automatic1111 fork of SD does
            mean_correction_factor = original_mean/after_weighting_mean
            batch_z *= mean_correction_factor
            return batch_z

class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """

    def __init__(
        self,
        version='ViT-L/14',
        device=choose_torch_device(),
        max_length=77,
        n_repeat=1,
        normalize=True,
    ):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device=device)
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim == 2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
    Uses the CLIP image encoder.
    """

    def __init__(
        self,
        model,
        jit=False,
        device=choose_torch_device(),
        antialias=False,
    ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer(
            'mean',
            torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
            persistent=False,
        )
        self.register_buffer(
            'std',
            torch.Tensor([0.26862954, 0.26130258, 0.27577711]),
            persistent=False,
        )

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation='bicubic',
            align_corners=True,
            antialias=self.antialias,
        )
        x = (x + 1.0) / 2.0
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))


if __name__ == '__main__':
    from ldm.util import count_params

    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)
