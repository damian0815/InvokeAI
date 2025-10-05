import math
from typing import Tuple, Self
import torch
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
import PIL.Image
from torchvision.transforms.functional import resize as torchvision_resize, InterpolationMode

@contextmanager
def collect_attention_maps(unet, text_encoder_hidden_size: int):
    collector = CrossAttentionMapCollector(unet, text_encoder_hidden_size)
    original_sdp_func = torch.nn.functional.scaled_dot_product_attention
    try:
        yield collector
    finally:
        collector.remove_hooks()
        # in case something went wrong - forward post-hooks might not have been called, so make sure we un-monkeypatch SDP
        torch.nn.functional.scaled_dot_product_attention = original_sdp_func


class CrossAttentionMapCollector:
    def __init__(self, unet, text_encoder_hidden_size: int, verbose: bool=True) -> None:
        self.unet = unet
        self.text_encoder_hidden_size = text_encoder_hidden_size
        self.attention_maps = defaultdict(list)
        self.hooks = []
        self._original_sdp_func = None
        self.verbose = verbose
        self.register_hooks()

    def register_hooks(self):
        """Register forward hooks on all cross-attention modules in the UNet."""

        def _sdp_with_map_saving(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, target: CrossAttentionMapCollector=None, map_name: str=""):
            attn_output, attn_weights = _scaled_dot_product_attention_with_weight_return(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
            if self.verbose and map_name not in target.attention_maps:
                # log on first addition
                print(f"storing maps of size {attn_weights.shape} for '{map_name}'")
            target.attention_maps[map_name].append(attn_weights.detach().cpu())
            return attn_output

        def pre_hook_fn(module, input, name):
            # oerride sdp
            self._original_sdp_func = torch.nn.functional.scaled_dot_product_attention
            torch.nn.functional.scaled_dot_product_attention = partial(_sdp_with_map_saving, target=self, map_name=name)

        def post_hook_fn(module, input, output):
            # restore sdp
            torch.nn.functional.scaled_dot_product_attention = self._original_sdp_func

        # Find all cross-attention modules and register hooks
        # attn_modules = {n: m for n, m in pipeline.unet.named_modules() if 'attn' in n.lower() and hasattr(m, 'to_q') and hasattr(m, 'to_k') and hasattr(m, 'to_v') and m.cross_attention_dim == pipeline.text_encoder.config.hidden_size}
        for name, module in self.unet.named_modules():
            # Look for cross-attention modules
            if 'attn' in name.lower() and (
                hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "to_v")
            ) and (
                module.cross_attention_dim == self.text_encoder_hidden_size
            ):
                module: torch.nn.Module
                pre_hook = module.register_forward_pre_hook(
                    lambda mod, inp, n=name: pre_hook_fn(mod, inp, n)
                )
                self.hooks.append(pre_hook)
                post_hook = module.register_forward_hook(post_hook_fn)
                self.hooks.append(post_hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_maps(self):
        """Clear collected attention maps."""
        self.attention_maps = defaultdict(list)

    def get_attention_maps(self):
        """Return the collected cross-attention maps."""
        return self.attention_maps

    def get_stacked_maps(self, latents_width: int, latents_height: int,
                         prompt_index: int, eos_token_index: int = None, drop_bos_eos=True,
                         merge_timesteps=False, timestep_weighting_alpha=0, timestep_weighting_beta=1,
                         contrast_boost_pow=2) -> torch.Tensor:
        """
        Scale all collected attention maps to the same size, blend them together and return as an image.
        latents_width and latents_height are the width and height of the latent space, e.g. 64x64 for 512x512 images with 8x downsampling.
        :param latents_width: The width of the latent space.
        :param latents_height: The height of the latent space.
        :param prompt_index: The index of the prompt to visualize.
        :param eos_token_index: The index of the end-of-sequence token - map stacking will be truncated to this index if provided.
        :param drop_bos_eos: If True, drop the the 0th and last token maps (BOS and EOS). Only applied if eos_token_index is provided. It's up to you to determine if your tokenizer actually outputs BOS/EOS.
        :param merge_timesteps: If True, blend all timesteps together, producing a single map per token. If False, stack timesteps horizontally, producing a grid.
        :param timestep_weighting_alpha: If merge_timesteps is True, this controls the weighting of timesteps (0 means equal weighting, 1 means linear ramp such that later timesteps carry more weight that earlier).
        :param timestep_weighting_beta: If merge_timesteps is True, this controls the sharpness of the weighting curve (1 means linear, >1 means later timesteps are weighted more strongly).
        :param contrast_boost_pow: If >1, increase the contrast of the attention maps by raising them to this power.
        :param _
        :return: An image containing a vertical stack of blended attention maps, one for each requested token.
        """
        maps_dict = self.get_attention_maps()
        merged = None
        for key, maps in maps_dict.items():
            maps = torch.stack(maps, dim=0)  # [steps, B, heads, (H*W), N]
            maps = torch.swapdims(maps, 0, 1)  # [B, steps, heads, (H*W), N]
            assert len(maps.shape) == 5  # [B, steps, heads, (H*W), N]

            maps = maps[prompt_index:prompt_index + 1, ...] # only one batch slot

            # drop padding tokens, bos, eos
            if eos_token_index is not None:
                maps = maps[..., :eos_token_index + 1]
                if drop_bos_eos:
                    if maps.shape[-1] > 2:
                        # drop EOS and BOS
                        maps = maps[..., 1:-1]
                    elif maps.shape[-1] == 2:
                        # drop BOS, keep EOS (must output at least 1 map)
                        maps = maps[..., 1:]
                    else:
                        raise RuntimeError("maps are missing for bos and/or eos tokens")

            # merge all heads together by averaging
            maps = torch.mean(maps, dim=2)  # now [B, steps, (H*W), N]
            # maps = torch.mean(maps, dim=1, keepdim=True)

            # maps has shape [B, steps, (H*W), N] for N tokens
            # but we want [B, steps, N, H, W] for torchvision_resize
            this_scale_factor = math.sqrt(maps.shape[2] / (latents_width * latents_height))
            this_maps_height = int(float(latents_height) * this_scale_factor)
            this_maps_width = int(float(latents_width) * this_scale_factor)
            # and we need to do some dimension juggling
            bsz = maps.shape[0]
            num_steps = maps.shape[1]
            num_tokens = maps.shape[-1]
            maps = torch.reshape(torch.swapdims(maps, -2, -1),
                                 [bsz, num_steps, num_tokens, this_maps_height, this_maps_width])

            # scale to output size if necessary
            if this_scale_factor != 1:
                # torchvision resize expects [..., H, W]
                maps = maps.reshape(bsz * num_steps, num_tokens, this_maps_height, this_maps_width)
                maps = torchvision_resize(maps, [latents_height, latents_width], InterpolationMode.BICUBIC)
                maps = maps.reshape(bsz, num_steps, num_tokens, latents_height, latents_width)

            # normalize in [N, H, W] where N=tokens
            maps_min = torch.amin(maps, dim=(-3, -2, -1), keepdim=True)
            maps_range = torch.amax(maps, dim=(-3, -2, -1), keepdim=True) - maps_min
            # print(f"map {key} size {[this_maps_width, this_maps_height]} range {[maps_min, maps_min + maps_range]}")
            maps_normalized = (maps - maps_min) / maps_range

            # increase contrast
            maps_normalized = torch.pow(maps_normalized, contrast_boost_pow)

            # stack tokens vertically
            maps_stacked = torch.reshape(maps_normalized,
                                         [bsz, num_steps, num_tokens * latents_height, latents_width])
            # map_stacked is [B, steps, (H*W), N]
            if merge_timesteps:
                # blend steps together, producing a single map per token
                num_steps = maps_stacked.shape[1]
                ramp = torch.linspace(0, 1, steps=num_steps,
                                      dtype=maps_stacked.dtype, device=maps_stacked.device)
                ramp_curve = ramp.pow(timestep_weighting_beta)

                weights = (1 - timestep_weighting_alpha) * torch.ones(num_steps,
                                                                      dtype=maps_stacked.dtype,
                                                                      device=maps_stacked.device) / num_steps \
                          + timestep_weighting_alpha * ramp_curve / ramp_curve.sum()
                weights = weights / weights.sum()  # Ensure weights sum to 1
                maps_stacked = torch.tensordot(maps_stacked, weights, dims=([1], [0]))
            else:
                # stack steps horizontally, producing a grid
                maps_stacked = maps_stacked.permute(0, 2, 1, 3).reshape(maps_stacked.shape[0], maps_stacked.shape[2],
                                                                        -1)
            # maps_stacked is now [B, (N*H), (W*steps)] where steps==1 if merge_timesteps is True

            if merged is None:
                merged = maps_stacked
            else:
                # screen blend
                merged = 1 - (1 - maps_stacked) * (1 - merged)

        return merged.squeeze(0)

    def get_stacked_maps_images(self, latents_width: int, latents_height: int,
                                prompt_index: int, eos_token_index: int = None, merge_timesteps=False) -> PIL.Image:
        merged = self.get_stacked_maps(latents_width=latents_width, latents_height=latents_height,
                                  prompt_index=prompt_index, eos_token_index=eos_token_index,
                                  merge_timesteps=merge_timesteps)
        # [(N*H), (W*steps)]
        assert len(merged.shape) == 2
        merged_bytes = merged.mul(0xff).byte()
        return PIL.Image.fromarray(merged_bytes.numpy(), mode='L')


def _scaled_dot_product_attention_with_weight_return(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False) -> Tuple[torch.Tensor, torch.Tensor]:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight_with_dropout = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight_with_dropout @ value, attn_weight
