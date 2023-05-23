'''
This module handles the generation of the conditioning tensors.

Useful function exports:

get_uc_and_c_and_ec()           get the conditioned and unconditioned latent, and edited conditioning if we're doing cross-attention control

'''
import re
from typing import Union, Optional, Any

from transformers import CLIPTokenizer

from compel import Compel
from compel.compel import BlendMode
from compel.prompt_parser import FlattenedPrompt, Blend, Fragment, CrossAttentionControlSubstitute, PromptParser, \
    Conjunction
import torch
from .devices import torch_dtype
from .generator.diffusers_pipeline import StableDiffusionGeneratorPipeline
from ..models.diffusion.shared_invokeai_diffusion import InvokeAIDiffuserComponent
from ldm.invoke.globals import Globals

class UnsqueezingLDMTransformer:
    def __init__(self, ldm_transformer):
        self.ldm_transformer = ldm_transformer

    @property
    def device(self):
        return self.ldm_transformer.device

    def __call__(self, *args, **kwargs):
        insufficiently_unsqueezed_tensor = self.ldm_transformer(*args, **kwargs)
        return insufficiently_unsqueezed_tensor.unsqueeze(0)


def get_uc_and_c_and_ec(prompt_string,
                        model: StableDiffusionGeneratorPipeline,
                        log_tokens=False, skip_normalize_legacy_blend=False):
    # lazy-load any deferred textual inversions.
    # this might take a couple of seconds the first time a textual inversion is used.
    model.textual_inversion_manager.create_deferred_token_ids_for_any_trigger_terms(prompt_string)

    blend_mode = BlendMode.CONCAT
    compel = Compel(tokenizer=model.tokenizer,
                    text_encoder=model.text_encoder,
                    textual_inversion_manager=model.textual_inversion_manager,
                    blend_mode=blend_mode,
                    #truncate_long_prompts=False,
                    dtype_for_device_getter=torch_dtype)

    # get rid of any newline characters
    prompt_string = prompt_string.replace("\n", " ")
    positive_prompt_string, negative_prompt_string = split_prompt_to_positive_and_negative(prompt_string)

    legacy_blend = try_parse_legacy_blend(positive_prompt_string,
                                          skip_normalize=(True
                                                          if blend_mode is BlendMode.CONCAT
                                                          else skip_normalize_legacy_blend))
    positive_conjunction: Conjunction
    if legacy_blend is not None:
        positive_conjunction = legacy_blend
    else:
        positive_conjunction = Compel.parse_prompt_string(positive_prompt_string)
    positive_prompt = positive_conjunction.prompts[0]

    should_use_lora_manager = True
    lora_weights = positive_conjunction.lora_weights
    lora_conditions = None
    if model.peft_manager:
        should_use_lora_manager = model.peft_manager.should_use(lora_weights)
        if not should_use_lora_manager:
            model.peft_manager.set_loras(lora_weights)
    if model.lora_manager and should_use_lora_manager:
        lora_conditions = model.lora_manager.set_loras_conditions(lora_weights)

    negative_conjunction = Compel.parse_prompt_string(negative_prompt_string)
    negative_prompt: FlattenedPrompt | Blend = negative_conjunction.prompts[0]

    tokens_count = get_max_token_count(model.tokenizer, positive_prompt)
    if log_tokens or getattr(Globals, "log_tokenization", False):
        log_tokenization(positive_prompt, negative_prompt, tokenizer=model.tokenizer)

    # some LoRA models also mess with the text encoder, so they must be active while compel builds conditioning tensors
    lora_conditioning_ec = InvokeAIDiffuserComponent.ExtraConditioningInfo(tokens_count_including_eos_bos=tokens_count,
                                                                                            lora_conditions=lora_conditions)
    with InvokeAIDiffuserComponent.custom_attention_context(model.unet,
                                                            extra_conditioning_info=lora_conditioning_ec,
                                                            step_count=-1):
        c, options = compel.build_conditioning_tensor_for_prompt_object(positive_prompt)
        uc, _ = compel.build_conditioning_tensor_for_prompt_object(negative_prompt)

    [c, uc] = compel.pad_conditioning_tensors_to_same_length([c, uc])

    # now build the "real" ec
    ec = InvokeAIDiffuserComponent.ExtraConditioningInfo(tokens_count_including_eos_bos=tokens_count,
                                                         cross_attention_control_args=options.get(
                                                             'cross_attention_control', None),
                                                         lora_conditions=lora_conditions)
    return uc, c, ec


def get_prompt_structure(prompt_string, skip_normalize_legacy_blend: bool = False) -> (
    Union[FlattenedPrompt, Blend], FlattenedPrompt):
    positive_prompt_string, negative_prompt_string = split_prompt_to_positive_and_negative(prompt_string)
    legacy_blend = try_parse_legacy_blend(positive_prompt_string, skip_normalize_legacy_blend)
    positive_conjunction: Conjunction
    if legacy_blend is not None:
        positive_conjunction = legacy_blend
    else:
        positive_conjunction = Compel.parse_prompt_string(positive_prompt_string)
    positive_prompt = positive_conjunction.prompts[0]
    negative_conjunction = Compel.parse_prompt_string(negative_prompt_string)
    negative_prompt: FlattenedPrompt|Blend = negative_conjunction.prompts[0]

    return positive_prompt, negative_prompt

def get_max_token_count(tokenizer, prompt: Union[FlattenedPrompt, Blend], truncate_if_too_long=True) -> int:
    if type(prompt) is Blend:
        blend: Blend = prompt
        return max([get_max_token_count(tokenizer, c, truncate_if_too_long) for c in blend.prompts])
    else:
        return len(get_tokens_for_prompt_object(tokenizer, prompt, truncate_if_too_long))


def get_tokens_for_prompt_object(tokenizer, parsed_prompt: FlattenedPrompt, truncate_if_too_long=True) -> [str]:

    if type(parsed_prompt) is Blend:
        raise ValueError("Blend is not supported here - you need to get tokens for each of its .children")

    text_fragments = [x.text if type(x) is Fragment else
                      (" ".join([f.text for f in x.original]) if type(x) is CrossAttentionControlSubstitute else
                       str(x))
                      for x in parsed_prompt.children]
    text = " ".join(text_fragments)
    tokens = tokenizer.tokenize(text)
    if truncate_if_too_long:
        max_tokens_length = tokenizer.model_max_length - 2  # typically 75
        tokens = tokens[0:max_tokens_length]
    return tokens


def split_prompt_to_positive_and_negative(prompt_string_uncleaned: str):
    unconditioned_words = ''
    unconditional_regex = r'\[(.*?)\]'
    unconditionals = re.findall(unconditional_regex, prompt_string_uncleaned)
    if len(unconditionals) > 0:
        unconditioned_words = ' '.join(unconditionals)

        # Remove Unconditioned Words From Prompt
        unconditional_regex_compile = re.compile(unconditional_regex)
        clean_prompt = unconditional_regex_compile.sub(' ', prompt_string_uncleaned)
        prompt_string_cleaned = re.sub(' +', ' ', clean_prompt)
    else:
        prompt_string_cleaned = prompt_string_uncleaned
    return prompt_string_cleaned, unconditioned_words


def log_tokenization(positive_prompt: Union[Blend, FlattenedPrompt],
                     negative_prompt: Union[Blend, FlattenedPrompt],
                     tokenizer):
    print(f"\n>> [TOKENLOG] Parsed Prompt: {positive_prompt}")
    print(f"\n>> [TOKENLOG] Parsed Negative Prompt: {negative_prompt}")

    log_tokenization_for_prompt_object(positive_prompt, tokenizer)
    log_tokenization_for_prompt_object(negative_prompt, tokenizer, display_label_prefix="(negative prompt)")


def log_tokenization_for_prompt_object(p: Union[Blend, FlattenedPrompt], tokenizer, display_label_prefix=None):
    display_label_prefix = display_label_prefix or ""
    if type(p) is Blend:
        blend: Blend = p
        for i, c in enumerate(blend.prompts):
            log_tokenization_for_prompt_object(
                c, tokenizer,
                display_label_prefix=f"{display_label_prefix}(blend part {i + 1}, weight={blend.weights[i]})")
    elif type(p) is FlattenedPrompt:
        flattened_prompt: FlattenedPrompt = p
        if flattened_prompt.wants_cross_attention_control:
            original_fragments = []
            edited_fragments = []
            for f in flattened_prompt.children:
                if type(f) is CrossAttentionControlSubstitute:
                    original_fragments += f.original
                    edited_fragments += f.edited
                else:
                    original_fragments.append(f)
                    edited_fragments.append(f)

            original_text = " ".join([x.text for x in original_fragments])
            log_tokenization_for_text(original_text, tokenizer,
                                      display_label=f"{display_label_prefix}(.swap originals)")
            edited_text = " ".join([x.text for x in edited_fragments])
            log_tokenization_for_text(edited_text, tokenizer,
                                      display_label=f"{display_label_prefix}(.swap replacements)")
        else:
            text = " ".join([x.text for x in flattened_prompt.children])
            log_tokenization_for_text(text, tokenizer, display_label=display_label_prefix)


def log_tokenization_for_text(text, tokenizer, display_label=None):
    """ shows how the prompt is tokenized
    # usually tokens have '</w>' to indicate end-of-word,
    # but for readability it has been replaced with ' '
    """
    tokens = tokenizer.tokenize(text)
    tokenized = ""
    discarded = ""
    usedTokens = 0
    totalTokens = len(tokens)

    for i in range(0, totalTokens):
        token = tokens[i].replace('</w>', ' ')
        # alternate color
        s = (usedTokens % 6) + 1
        if i < tokenizer.model_max_length:
            tokenized = tokenized + f"\x1b[0;3{s};40m{token}"
            usedTokens += 1
        else:  # over max token length
            discarded = discarded + f"\x1b[0;3{s};40m{token}"

    if usedTokens > 0:
        print(f'\n>> [TOKENLOG] Tokens {display_label or ""} ({usedTokens}):')
        print(f'{tokenized}\x1b[0m')

    if discarded != "":
        print(f'\n>> [TOKENLOG] Tokens Discarded ({totalTokens - usedTokens}):')
        print(f'{discarded}\x1b[0m')


def try_parse_legacy_blend(text: str, skip_normalize: bool=False) -> Optional[Conjunction]:
    weighted_subprompts = split_weighted_subprompts(text, skip_normalize=skip_normalize)
    if len(weighted_subprompts) <= 1:
        return None
    strings = [x[0] for x in weighted_subprompts]

    pp = PromptParser()
    parsed_conjunctions = [pp.parse_conjunction(x) for x in strings]
    flattened_prompts = []
    weights = []
    loras = []
    for i, x in enumerate(parsed_conjunctions):
        if len(x.prompts)>0:
            flattened_prompts.append(x.prompts[0])
            weights.append(weighted_subprompts[i][1])
        if len(x.lora_weights)>0:
            loras.extend(x.lora_weights)

    return Conjunction([Blend(prompts=flattened_prompts, weights=weights, normalize_weights=not skip_normalize)],
                       lora_weights = loras)


def split_weighted_subprompts(text, skip_normalize=False)->list:
    """
    Legacy blend parsing.

    grabs all text up to the first occurrence of ':'
    uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
    if ':' has no value defined, defaults to 1.0
    repeats until no text remaining
    """
    prompt_parser = re.compile("""
            (?P<prompt>     # capture group for 'prompt'
            (?:\\\:|[^:])+  # match one or more non ':' characters or escaped colons '\:'
            )               # end 'prompt'
            (?:             # non-capture group
            :+              # match one or more ':' characters
            (?P<weight>     # capture group for 'weight'
            -?\d+(?:\.\d+)? # match positive or negative integer or decimal number
            )?              # end weight capture group, make optional
            \s*             # strip spaces after weight
            |               # OR
            $               # else, if no ':' then match end of line
            )               # end non-capture group
            """, re.VERBOSE)
    parsed_prompts = [(match.group("prompt").replace("\\:", ":"), float(
        match.group("weight") or 1)) for match in re.finditer(prompt_parser, text)]
    if skip_normalize:
        return parsed_prompts
    weight_sum = sum(map(lambda x: x[1], parsed_prompts))
    if weight_sum == 0:
        print(
            "* Warning: Subprompt weights add up to zero. Discarding and using even weights instead.")
        equal_weight = 1 / max(len(parsed_prompts), 1)
        return [(x[0], equal_weight) for x in parsed_prompts]
    return [(x[0], x[1] / weight_sum) for x in parsed_prompts]

