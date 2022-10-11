'''
This module handles the generation of the conditioning tensors, including management of
weighted subprompts.

Useful function exports:

get_uc_and_c()                  get the conditioned and unconditioned latent
split_weighted_subpromopts()    split subprompts, normalize and weight them
log_tokenization()              print out colour-coded tokens and warn if truncated

'''
import re
import torch

from .prompt_parser import PromptParser, Fragment, Attention, Blend, Conjunction, FlattenedPrompt


def get_uc_and_c(prompt_string_uncleaned, model, log_tokens=False, skip_normalize=False):

    # Extract Unconditioned Words From Prompt
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


    subprompts_to_blend = split_weighted_subprompts(
        prompt_string_cleaned, True  # skip_normalize
    )
    if len(subprompts_to_blend)>1:
        quoted_prompts = ['"'+x[0]+'"' for x in subprompts_to_blend]
        weights = [str(x[1]) for x in subprompts_to_blend]
        quoted_prompts_string = ", ".join(quoted_prompts)
        weights_string = ", ".join(weights)
        prompt_string_cleaned = f"({quoted_prompts_string}).blend({weights_string})"

    #rebuilt_blends_prompt_string = prompt_string_cleaned if len(subprompts_to_blend)<2 else

    pp = PromptParser()
    parsed_positive_conjunction: Conjunction = pp.parse(prompt_string_cleaned)
    assert(type(parsed_positive_conjunction) is Conjunction)
    print(parsed_positive_conjunction)

    if len(parsed_positive_conjunction.parts)>1:
        print("only handling 1 conjunction part for now")

    positive_conditioning_list = []

    def make_embeddings_for_flattened_prompt(flattened_prompt: FlattenedPrompt):
        if type(flattened_prompt) is not FlattenedPrompt:
            raise f"embeddings can only be made from FlattenedPrompts, got {type(flattened_prompt)} instead"
        fragments = [x[0] for x in flattened_prompt.children]
        attention_weights = [x[1] for x in flattened_prompt.children]
        print(fragments, attention_weights)
        return model.get_learned_conditioning([fragments], attention_weights=[attention_weights])

    for part in parsed_positive_conjunction.parts:
        if type(part) is Blend:
            blend:Blend = part
            embeddings_to_blend = []
            for flattened_prompt in blend.children:
                embeddings_to_blend.append(make_embeddings_for_flattened_prompt(flattened_prompt))
            blended_embeddings = model.blend_embeddings(embeddings_to_blend, blend.weights)
            positive_conditioning_list.append((blended_embeddings, 1.0))
        else:
            flattened_prompt: FlattenedPrompt = part
            embeddings = make_embeddings_for_flattened_prompt(flattened_prompt)
            positive_conditioning_list.append((embeddings, 1.0))

    # get weighted sub-prompts
    def get_blend_prompts_and_weights(prompt):
        subprompts_to_blend = split_weighted_subprompts(
            prompt, True #skip_normalize
        )
        for subprompt, weight in subprompts_to_blend:
            log_tokenization(subprompt, model, log_tokens, weight)
        if len(subprompts_to_blend)==0:
            subprompts = ['']
            weights = [1]
        else:
            subprompts = [subprompt for subprompt, _ in subprompts_to_blend]
            weights = [weight for _, weight in subprompts_to_blend]
        return model.get_learned_conditioning([subprompts], attention_weights=[weights])

    negative_conditioning = get_blend_prompts_and_weights(unconditioned_words)
    #positive_conditioning_list.append((get_blend_prompts_and_weights(prompt), this_weight))
    #print("got empty_conditionining with shape", empty_conditioning.shape, "c[0][0] with shape", positive_conditioning[0][0].shape)

    # "unconditioned" means "the conditioning tensor is empty"
    uc = negative_conditioning
    c = positive_conditioning_list

    return (uc, c)

def split_weighted_subprompts(text, skip_normalize=False)->list:
    """
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
            "Warning: Subprompt weights add up to zero. Discarding and using even weights instead.")
        equal_weight = 1 / max(len(parsed_prompts), 1)
        return [(x[0], equal_weight) for x in parsed_prompts]
    return [(x[0], x[1] / weight_sum) for x in parsed_prompts]

# shows how the prompt is tokenized
# usually tokens have '</w>' to indicate end-of-word,
# but for readability it has been replaced with ' '
def log_tokenization(text, model, log=False, weight=1):
    if not log:
        return
    tokens    = model.cond_stage_model.tokenizer._tokenize(text)
    tokenized = ""
    discarded = ""
    usedTokens = 0
    totalTokens = len(tokens)
    for i in range(0, totalTokens):
        token = tokens[i].replace('</w>', ' ')
        # alternate color
        s = (usedTokens % 6) + 1
        if i < model.cond_stage_model.max_length:
            tokenized = tokenized + f"\x1b[0;3{s};40m{token}"
            usedTokens += 1
        else:  # over max token length
            discarded = discarded + f"\x1b[0;3{s};40m{token}"
    print(f"\n>> Tokens ({usedTokens}), Weight ({weight:.2f}):\n{tokenized}\x1b[0m")
    if discarded != "":
        print(
            f">> Tokens Discarded ({totalTokens-usedTokens}):\n{discarded}\x1b[0m"
        )
