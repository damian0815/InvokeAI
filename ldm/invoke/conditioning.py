'''
This module handles the generation of the conditioning tensors.

Useful function exports:

get_uc_and_c_and_ec()           get the conditioned and unconditioned latent, and edited conditioning if we're doing cross-attention control

'''
import re
from dataclasses import dataclass
from typing import Union, Optional

import torch
from transformers import CLIPTokenizer, CLIPTextModel

from .prompt_parser import Blend, FlattenedPrompt, \
    CrossAttentionControlSubstitute, Fragment, parse_prompt_string
from ..models.diffusion import cross_attention_control
from ..models.diffusion.shared_invokeai_diffusion import ExtraConditioningInfo
from ..modules.prompt_to_embeddings_converter import PromptToEmbeddingsConverter
from ..modules.textual_inversion_manager import TextualInversionManager


def get_uc_and_c_and_ec(prompt_string, model, log_tokens=False):

    # lazy-load any deferred textual inversions.
    # this might take a couple of seconds the first time a textual inversion is used.
    model.textual_inversion_manager.create_deferred_token_ids_for_any_trigger_terms(prompt_string)

    prompt, negative_prompt = parse_prompt_string(prompt_string)

    conditioning = get_conditioning_for_prompt_structure(prompt, negative_prompt, model, log_tokens)

    return conditioning


def get_tokens_for_prompt(prompt_to_embeddings_convertor: PromptToEmbeddingsConverter,
                          parsed_prompt: FlattenedPrompt,
                          truncate_if_too_long=True) -> [str]:
    text_fragments = [x.text if type(x) is Fragment else
                      (" ".join([f.text for f in x.original]) if type(x) is CrossAttentionControlSubstitute else
                       str(x))
                      for x in parsed_prompt.children]
    tokens = prompt_to_embeddings_convertor.get_token_ids(text_fragments)
    if truncate_if_too_long:
        max_token_count = prompt_to_embeddings_convertor.max_token_count - 2 # typically 75
        tokens = tokens[0:max_token_count]
    return tokens


def get_conditioning_for_prompt_structure(parsed_prompt: Union[Blend, FlattenedPrompt], parsed_negative_prompt: FlattenedPrompt,
                                          model, log_tokens=False) \
    -> tuple[torch.Tensor, torch.Tensor, ExtraConditioningInfo]:
    """
    Process prompt structure and tokens, and return (conditioning, unconditioning, extra_conditioning_info)
    """

    if log_tokens:
        print(f">> Parsed prompt to {parsed_prompt}")
        print(f">> Parsed negative prompt to {parsed_negative_prompt}")

    conditioning = None
    cac_args: cross_attention_control.Arguments = None

    if type(parsed_prompt) is Blend:
        conditioning = _get_conditioning_for_blend(model, parsed_prompt, log_tokens)
    elif type(parsed_prompt) is FlattenedPrompt:
        if parsed_prompt.wants_cross_attention_control:
            conditioning, cac_args = _get_conditioning_for_cross_attention_control(model, parsed_prompt, log_tokens)

        else:
            conditioning, _ = _get_embeddings_and_tokens_for_flattened_prompt(model,
                                                                              parsed_prompt,
                                                                              log_tokens=log_tokens,
                                                                              log_display_label="(prompt)")
    else:
        raise ValueError(f"parsed_prompt is '{type(parsed_prompt)}' which is not a supported prompt type")

    unconditioning, _ = _get_embeddings_and_tokens_for_flattened_prompt(model,
                                                                        parsed_negative_prompt,
                                                                        log_tokens=log_tokens,
                                                                        log_display_label="(unconditioning)")
    if isinstance(conditioning, dict):
        # hybrid conditioning is in play
        unconditioning, conditioning = _flatten_hybrid_conditioning(unconditioning, conditioning)
        if cac_args is not None:
            print(
                ">> Hybrid conditioning cannot currently be combined with cross attention control. Cross attention control will be ignored.")
            cac_args = None

    if type(parsed_prompt) is Blend:
        blend: Blend = parsed_prompt
        all_token_sequences = [get_tokens_for_prompt(model, p) for p in blend.prompts]
        longest_token_sequence = max(all_token_sequences, key=lambda t: len(t))
        eos_token_index = len(longest_token_sequence)+1
    else:
        tokens = get_tokens_for_prompt(model, parsed_prompt)
        eos_token_index = len(tokens)+1
    return (
        unconditioning, conditioning, ExtraConditioningInfo(
            tokens_count_including_eos_bos=eos_token_index + 1,
            cross_attention_control_args=cac_args
        )
    )


def _get_conditioning_for_cross_attention_control(model, prompt: FlattenedPrompt, log_tokens: bool = True):
    original_prompt = FlattenedPrompt()
    edited_prompt = FlattenedPrompt()
    # for name, a0, a1, b0, b1 in edit_opcodes: only name == 'equal' is currently parsed
    original_token_count = 0
    edited_token_count = 0
    edit_options = []
    edit_opcodes = []
    # beginning of sequence
    edit_opcodes.append(
        ('equal', original_token_count, original_token_count + 1, edited_token_count, edited_token_count + 1))
    edit_options.append(None)
    original_token_count += 1
    edited_token_count += 1
    for fragment in prompt.children:
        if type(fragment) is CrossAttentionControlSubstitute:
            original_prompt.append(fragment.original)
            edited_prompt.append(fragment.edited)

            to_replace_token_count = _get_tokens_length(model, fragment.original)
            replacement_token_count = _get_tokens_length(model, fragment.edited)
            edit_opcodes.append(('replace',
                                 original_token_count, original_token_count + to_replace_token_count,
                                 edited_token_count, edited_token_count + replacement_token_count
                                 ))
            original_token_count += to_replace_token_count
            edited_token_count += replacement_token_count
            edit_options.append(fragment.options)
        # elif type(fragment) is CrossAttentionControlAppend:
        #    edited_prompt.append(fragment.fragment)
        else:
            # regular fragment
            original_prompt.append(fragment)
            edited_prompt.append(fragment)

            count = _get_tokens_length(model, [fragment])
            edit_opcodes.append(('equal', original_token_count, original_token_count + count, edited_token_count,
                                 edited_token_count + count))
            edit_options.append(None)
            original_token_count += count
            edited_token_count += count
    # end of sequence
    edit_opcodes.append(
        ('equal', original_token_count, original_token_count + 1, edited_token_count, edited_token_count + 1))
    edit_options.append(None)
    original_token_count += 1
    edited_token_count += 1
    original_embeddings, original_tokens = _get_embeddings_and_tokens_for_flattened_prompt(model,
                                                                                           original_prompt,
                                                                                           log_tokens=log_tokens,
                                                                                           log_display_label="(.swap originals)")
    # naïvely building a single edited_embeddings like this disregards the effects of changing the absolute location of
    # subsequent tokens when there is >1 edit and earlier edits change the total token count.
    # eg "a cat.swap(smiling dog, s_start=0.5) eating a hotdog.swap(pizza)" - when the 'pizza' edit is active but the
    # 'cat' edit is not, the 'pizza' feature vector will nevertheless be affected by the introduction of the extra
    # token 'smiling' in the inactive 'cat' edit.
    # todo: build multiple edited_embeddings, one for each edit, and pass just the edited fragments through to the CrossAttentionControl functions
    edited_embeddings, edited_tokens = _get_embeddings_and_tokens_for_flattened_prompt(model,
                                                                                       edited_prompt,
                                                                                       log_tokens=log_tokens,
                                                                                       log_display_label="(.swap replacements)")
    conditioning = original_embeddings
    edited_conditioning = edited_embeddings
    # print('>> got edit_opcodes', edit_opcodes, 'options', edit_options)
    cac_args = cross_attention_control.Arguments(
        edited_conditioning=edited_conditioning,
        edit_opcodes=edit_opcodes,
        edit_options=edit_options
    )
    return conditioning, cac_args

def _get_conditioning_for_flattened_prompt(prompt_to_embeddings_converter: PromptToEmbeddingsConverter,
                                           prompt: FlattenedPrompt,
                                           log_tokens: bool = False):
    conditioning, _ = _get_embeddings_and_tokens_for_flattened_prompt(prompt_to_embeddings_converter,
                                                                      prompt,
                                                                      log_tokens)
    return conditioning

def _get_conditioning_for_blend(prompt_to_embeddings_converter: PromptToEmbeddingsConverter,
                                blend: Blend,
                                log_tokens: bool = False):
    embeddings_to_blend = None
    for i, flattened_prompt in enumerate(blend.prompts):
        this_embedding, _ = _get_embeddings_and_tokens_for_flattened_prompt(prompt_to_embeddings_converter,
                                                                            flattened_prompt,
                                                                            log_tokens=log_tokens,
                                                                            log_display_label=f"(blend part {i + 1}, weight={blend.weights[i]})")
        embeddings_to_blend = this_embedding if embeddings_to_blend is None else torch.cat(
            (embeddings_to_blend, this_embedding))
    conditioning = PromptToEmbeddingsConverter.apply_embedding_weights(embeddings_to_blend.unsqueeze(0),
                                                                       blend.weights,
                                                                       normalize=blend.normalize_weights)
    return conditioning


def _get_embeddings_and_tokens_for_flattened_prompt(prompt_to_embeddings_converter: PromptToEmbeddingsConverter,
                                                    flattened_prompt: FlattenedPrompt,
                                                    log_tokens: bool = False,
                                                    log_display_label: str = None):
    if type(flattened_prompt) is not FlattenedPrompt:
        raise Exception(f"embeddings can only be made from FlattenedPrompts, got {type(flattened_prompt)} instead")
    fragments = [x.text for x in flattened_prompt.children]
    weights = [x.weight for x in flattened_prompt.children]
    embeddings, tokens = prompt_to_embeddings_converter.get_embeddings_for_weighted_prompt_fragments(
            text=[fragments],
            fragment_weights=[weights],
            should_return_tokens=True)
    if log_tokens:
        text = " ".join(fragments)
        log_tokenization(text, prompt_to_embeddings_converter, display_label=log_display_label)

    return embeddings, tokens


def _get_tokens_length(model, fragments: list[Fragment]):
    fragment_texts = [x.text for x in fragments]
    tokens = model.cond_stage_model.get_token_ids(fragment_texts, include_start_and_end_markers=False)
    return sum([len(x) for x in tokens])


def _flatten_hybrid_conditioning(uncond, cond):
    '''
    This handles the choice between a conditional conditioning
    that is a tensor (used by cross attention) vs one that has additional
    dimensions as well, as used by 'hybrid'
    '''
    assert isinstance(uncond, dict)
    assert isinstance(cond, dict)
    cond_flattened = dict()
    for k in cond:
        if isinstance(cond[k], list):
            cond_flattened[k] = [
                torch.cat([uncond[k][i], cond[k][i]])
                for i in range(len(cond[k]))
            ]
        else:
            cond_flattened[k] = torch.cat([uncond[k], cond[k]])
    return uncond, cond_flattened


def log_tokenization(text,
                     prompt_to_embeddings_converter: PromptToEmbeddingsConverter,
                     display_label=None):
    """ shows how the prompt is tokenized
    # usually tokens have '</w>' to indicate end-of-word,
    # but for readability it has been replaced with ' '
    """

    tokens = prompt_to_embeddings_converter.get_tokenization_description(text)
    tokenized = ""
    discarded = ""
    usedTokens = 0
    totalTokens = len(tokens)
    for i in range(0, totalTokens):
        token = tokens[i].replace('</w>', ' ')
        # alternate color
        s = (usedTokens % 6) + 1
        if i < prompt_to_embeddings_converter.max_token_count:
            tokenized = tokenized + f"\x1b[0;3{s};40m{token}"
            usedTokens += 1
        else:  # over max token length
            discarded = discarded + f"\x1b[0;3{s};40m{token}"
    print(f"\n>> Tokens {display_label or ''} ({usedTokens}):\n{tokenized}\x1b[0m")
    if discarded != "":
        print(
            f">> Tokens Discarded ({totalTokens - usedTokens}):\n{discarded}\x1b[0m"
        )



@dataclass
class Conditioning():
    """
    Conditioning. In all examples `B` is batch size, `77` is the text encoder's max token length,
    and `token_dim` is 768 for SD1 and 1280 for SD2.
    """
    negative_conditioning: torch.Tensor # shape [B x 77 x token_dim]
    positive_conditioning: torch.Tensor # shape [B x 77 x token_dim]
    cfg_scale: float # conditioner-free guidance scale

    cross_attention_control_arguments: Optional[cross_attention_control.Arguments]

class ConditioningScheduler():
    """
    Provides a mechanism to control which processes to apply for any given step of a Stable Diffusion generation.
    """

    def get_conditioning_for_step_pct(self, step_pct: float) -> Conditioning:
        """
        Return the conditioning to apply at the given step.
        :param step_pct: The step as a float `0..1`, where `0.0` is immediately before the start of image generation
        process (when the latent vector is 100% noise), and `1.0` is immediately after the end of the final step
        (when the latent vector represents the final noise-free generated image).
        :return: The Conditioning to apply for the requested step.
        """
        raise NotImplementedError("Subclasses must override")


class StaticConditioningScheduler(ConditioningScheduler):
    def __init__(self, positive_conditioning: torch.Tensor,
                 negative_conditioning: torch.Tensor,
                 cfg_scale: float,
                 cross_attention_control_args: Optional[cross_attention_control.Arguments]):
        self.positive_conditioning = positive_conditioning
        self.negative_conditioning = negative_conditioning
        self.cfg_scale = cfg_scale
        self.cross_attention_control_args = cross_attention_control_args

    def get_conditioning_for_step_pct(self, step_pct: float) -> Conditioning:
        """ See base class for docs. """
        return Conditioning(negative_conditioning=self.negative_conditioning,
                            positive_conditioning=self.standard_positive_conditioning,
                            cfg_scale=self.cfg,
                            cross_attention_control_arguments=self.cross_attention_control_args)


class ConditioningSchedulerFactory():

    def __init__(self, prompt_to_embeddings_converter: PromptToEmbeddingsConverter):
        self.prompt_to_embeddings_converter = prompt_to_embeddings_converter

    @property
    def max_token_count(self):
        return self.prompt_to_embeddings_converter.max_token_count



    def make_conditioning_scheduler(self, prompt_string: str, cfg_scale: float, log_tokens=False) -> ConditioningScheduler:

        prompt, negative_prompt = parse_prompt_string(prompt_string)
        cross_attention_control_args = None
        if type(prompt) is FlattenedPrompt:
            if prompt.wants_cross_attention_control:
                conditioning, cross_attention_control_args = _get_conditioning_for_cross_attention_control(
                    self.prompt_to_embeddings_converter, prompt, log_tokens)
            else:
                conditioning = _get_conditioning_for_flattened_prompt(
                    self.prompt_to_embeddings_converter, prompt, log_tokens)
        elif type(prompt) is Blend:
            conditioning = _get_conditioning_for_blend(self.prompt_to_embeddings_converter, prompt, log_tokens)

        if type(negative_prompt) is not FlattenedPrompt:
            raise RuntimeError("Only basic prompts are supported as negative prompts.")
        negative_conditioning = _get_conditioning_for_flattened_prompt(self.prompt_to_embeddings_converter, negative_prompt)

        return StaticConditioningScheduler(positive_conditioning=conditioning,
                                           negative_conditioning=negative_conditioning,
                                           cfg_scale=cfg_scale,
                                           cross_attention_control_args=cross_attention_control_args)
