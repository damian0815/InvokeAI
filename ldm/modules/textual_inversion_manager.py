import os
import traceback
from typing import Union

import torch
from attr import dataclass
from picklescan.scanner import scan_file_path

from ldm.invoke.concepts_lib import HuggingFaceConceptsLibrary
from ldm.modules.embedding_manager import get_clip_token_id_for_string
from ldm.modules.encoders.modules import FrozenCLIPEmbedder


@dataclass
class TextualInversion:
    trigger_string: str
    trigger_token_id: int
    pad_token_ids: list[int]
    embedding: torch.Tensor

    @property
    def embedding_vector_length(self) -> int:
        return self.embedding.shape[0]

class TextualInversionManager():
    def __init__(self, clip_embedder: FrozenCLIPEmbedder, full_precision: bool=True):
        self.clip_embedder = clip_embedder
        self.full_precision = full_precision
        self.hf_concepts_library = HuggingFaceConceptsLibrary()
        default_textual_inversions: list[TextualInversion] = []
        self.textual_inversions = default_textual_inversions

    def load_huggingface_concepts(self, concepts: list[str]):
        for concept_name in concepts:
            if concept_name in self.hf_concepts_library.concepts_loaded:
                continue
            bin_file = self.hf_concepts_library.get_concept_model_path(concept_name)
            if not bin_file:
                continue
            self.load_textual_inversion(bin_file)
            self.hf_concepts_library.concepts_loaded[concept_name]=True

    def get_all_trigger_strings(self) -> list[str]:
        return [ti.trigger_string for ti in self.textual_inversions]

    def load_textual_inversion(self, ckpt_path):
        try:
            scan_result = scan_file_path(ckpt_path)
            if scan_result.infected_files == 1:
                print(f'\n### Security Issues Found in Model: {scan_result.issues_count}')
                print('### For your safety, InvokeAI will not load this embed.')
                return
        except Exception:
            print(f"### WARNING::: Invalid or corrupt embeddings found. Ignoring: {ckpt_path}")
            return

        embedding_info = self._parse_embedding(ckpt_path)
        if embedding_info:
            self._add_textual_inversion(embedding_info['name'], embedding_info['embedding'])
        else:
            print(f'>> Failed to load embedding located at {ckpt_path}. Unsupported file.')

    def _add_textual_inversion(self, trigger_str, embedding) -> int:
        """
        Add a textual inversion to be recognised.
        :param trigger_str: The trigger text in the prompt that activates this textual inversion. If unknown to the embedder's tokenizer, will be added.
        :param embedding: The actual embedding data that will be inserted into the conditioning at the point where the token_str appears.
        :return: The token id for the added embedding, either existing or newly-added.
        """
        if trigger_str in [ti.trigger_string for ti in self.textual_inversions]:
            print(f">> TextualInversionManager refusing to overwrite already-loaded token '{trigger_str}'")
            return
        if not self.full_precision:
            embedding = embedding.half()
        if len(embedding.shape) == 1:
            embedding = embedding.unsqueeze(0)
        elif len(embedding.shape) > 2:
            raise ValueError(f"TextualInversionManager cannot add {trigger_str} because the embedding shape {embedding.shape} is incorrect. The embedding must have shape [token_dim] or [V, token_dim] where V is vector length and token_dim is 768 for SD1 or 1280 for SD2.")

        # for embeddings with vector length > 1
        pad_token_strings = [trigger_str + "-!pad-" + str(pad_index) for pad_index in range(1, embedding.shape[0])]

        try:
            trigger_token_id = self._get_or_create_token_id_and_assign_embedding(trigger_str, embedding[0])
            # todo: batched UI for faster loading when vector length >2
            pad_token_ids = [self._get_or_create_token_id_and_assign_embedding(pad_token_str, embedding[1 + i]) \
                             for (i, pad_token_str) in enumerate(pad_token_strings)]
            self.textual_inversions.append(TextualInversion(
                trigger_string=trigger_str,
                trigger_token_id=trigger_token_id,
                pad_token_ids=pad_token_ids,
                embedding=embedding
            ))
            return trigger_token_id

        except ValueError:
            traceback.print_exc()
            print(f">> TextualInversionManager was unable to add a textual inversion with trigger string {trigger_str}.")
            raise


    def has_textual_inversion_for_trigger_string(self, trigger_string: str) -> bool:
        try:
            ti = self.get_textual_inversion_for_trigger_string(trigger_string)
            return ti is not None
        except StopIteration:
            return False

    def get_textual_inversion_for_trigger_string(self, trigger_string: str) -> TextualInversion:
        return next(ti for ti in self.textual_inversions if ti.trigger_string == trigger_string)


    def get_textual_inversion_for_token_id(self, token_id: int) -> TextualInversion:
        return next(ti for ti in self.textual_inversions if ti.trigger_token_id == token_id)

    def expand_textual_inversion_token_ids(self, prompt_token_ids: list[int]) -> list[int]:
        """
        Insert padding tokens as necessary into the passed-in list of token ids to match any textual inversions it includes.

        :param prompt_token_ids: The prompt as a list of token ids (`int`s). Should not include bos and eos markers.
        :return: The prompt token ids with any necessary padding to account for textual inversions inserted. May be too
                long - caller is responsible for prepending/appending eos and bos token ids, and truncating if necessary.
        """
        if len(prompt_token_ids) == 0:
            return prompt_token_ids

        if prompt_token_ids[0] == self.clip_embedder.tokenizer.bos_token_id:
            raise ValueError("prompt_token_ids must not start with bos_token_id")
        if prompt_token_ids[-1] == self.clip_embedder.tokenizer.eos_token_id:
            raise ValueError("prompt_token_ids must not end with eos_token_id")
        textual_inversion_trigger_token_ids = [ti.trigger_token_id for ti in self.textual_inversions]
        prompt_token_ids = prompt_token_ids.copy()
        for i, token_id in reversed(list(enumerate(prompt_token_ids))):
            if token_id in textual_inversion_trigger_token_ids:
                textual_inversion = next(ti for ti in self.textual_inversions if ti.trigger_token_id == token_id)
                for pad_idx in range(0, textual_inversion.embedding_vector_length-1):
                    prompt_token_ids.insert(i+pad_idx+1, textual_inversion.pad_token_ids[pad_idx])

        return prompt_token_ids

    def _get_or_create_token_id_and_assign_embedding(self, token_str: str, embedding: torch.Tensor):
        if len(embedding.shape) != 1:
            raise ValueError("Embedding has incorrect shape - must be [token_dim] where token_dim is 768 for SD1 or 1280 for SD2")
        existing_token_id = get_clip_token_id_for_string(self.clip_embedder.tokenizer, token_str)
        if existing_token_id == self.clip_embedder.tokenizer.unk_token_id:
            num_tokens_added = self.clip_embedder.tokenizer.add_tokens(token_str)
            current_embeddings = self.clip_embedder.transformer.resize_token_embeddings(None)
            current_token_count = current_embeddings.num_embeddings
            new_token_count = current_token_count + num_tokens_added
            # the following call is slow - todo make batched for better performance with vector length >1
            self.clip_embedder.transformer.resize_token_embeddings(new_token_count)

        token_id = get_clip_token_id_for_string(self.clip_embedder.tokenizer, token_str)
        self.clip_embedder.transformer.get_input_embeddings().weight.data[token_id] = embedding

        return token_id
    def _parse_embedding(self, embedding_file: str):
        file_type = embedding_file.split('.')[-1]
        if file_type == 'pt':
            return self._parse_embedding_pt(embedding_file)
        elif file_type == 'bin':
            return self._parse_embedding_bin(embedding_file)
        else:
            print(f'>> Not a recognized embedding file: {embedding_file}')

    def _parse_embedding_pt(self, embedding_file):
        embedding_ckpt = torch.load(embedding_file, map_location='cpu')
        embedding_info = {}

        # Check if valid embedding file
        if 'string_to_token' and 'string_to_param' in embedding_ckpt:

            # Catch variants that do not have the expected keys or values.
            try:
                embedding_info['name'] = embedding_ckpt['name'] or os.path.basename(os.path.splitext(embedding_file)[0])

                # Check num of embeddings and warn user only the first will be used
                embedding_info['num_of_embeddings'] = len(embedding_ckpt["string_to_token"])
                if embedding_info['num_of_embeddings'] > 1:
                    print('>> More than 1 embedding found. Will use the first one')

                embedding = list(embedding_ckpt['string_to_param'].values())[0]
            except (AttributeError,KeyError):
                return self._handle_broken_pt_variants(embedding_ckpt, embedding_file)

            embedding_info['embedding'] = embedding
            embedding_info['num_vectors_per_token'] = embedding.size()[0]
            embedding_info['token_dim'] = embedding.size()[1]

            try:
                embedding_info['trained_steps'] = embedding_ckpt['step']
                embedding_info['trained_model_name'] = embedding_ckpt['sd_checkpoint_name']
                embedding_info['trained_model_checksum'] = embedding_ckpt['sd_checkpoint']
            except AttributeError:
                print(">> No Training Details Found. Passing ...")

        # .pt files found at https://cyberes.github.io/stable-diffusion-textual-inversion-models/
        # They are actually .bin files
        elif len(embedding_ckpt.keys())==1:
            print('>> Detected .bin file masquerading as .pt file')
            embedding_info = self._parse_embedding_bin(embedding_file)

        else:
            print('>> Invalid embedding format')
            embedding_info = None

        return embedding_info

    def _parse_embedding_bin(self, embedding_file):
        embedding_ckpt = torch.load(embedding_file, map_location='cpu')
        embedding_info = {}

        if list(embedding_ckpt.keys()) == 0:
            print(">> Invalid concepts file")
            embedding_info = None
        else:
            for token in list(embedding_ckpt.keys()):
                embedding_info['name'] = token or os.path.basename(os.path.splitext(embedding_file)[0])
                embedding_info['embedding'] = embedding_ckpt[token]
                embedding_info['num_vectors_per_token'] = 1 # All Concepts seem to default to 1
                embedding_info['token_dim'] = embedding_info['embedding'].size()[0]

        return embedding_info

    def _handle_broken_pt_variants(self, embedding_ckpt:dict, embedding_file:str)->dict:
        '''
        This handles the broken .pt file variants. We only know of one at present.
        '''
        embedding_info = {}
        if isinstance(list(embedding_ckpt['string_to_token'].values())[0],torch.Tensor):
            print('>> Detected .pt file variant 1') # example at https://github.com/invoke-ai/InvokeAI/issues/1829
            for token in list(embedding_ckpt['string_to_token'].keys()):
                embedding_info['name'] = token if token != '*' else os.path.basename(os.path.splitext(embedding_file)[0])
                embedding_info['embedding'] = embedding_ckpt['string_to_param'].state_dict()[token]
                embedding_info['num_vectors_per_token'] = embedding_info['embedding'].shape[0]
                embedding_info['token_dim'] = embedding_info['embedding'].size()[0]
        else:
            print('>> Invalid embedding format')
            embedding_info = None

        return embedding_info
