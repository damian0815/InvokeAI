import string
from typing import Union, Optional
import re
import pyparsing as pp
'''
This module parses prompt strings and produces tree-like structures that can be used generate and control the conditioning tensors. 
weighted subprompts.

Useful class exports:

PromptParser - parses prompts

Useful function exports:

split_weighted_subpromopts()    split subprompts, normalize and weight them
log_tokenization()              print out colour-coded tokens and warn if truncated
'''

class Prompt():
    """
    Mid-level structure for storing the tree-like result of parsing a prompt. A Prompt may not represent the whole of
    the singular user-defined "prompt string" (although it can) - for example, if the user specifies a Blend, the objects
    that are to be blended together are stored individuall as Prompt objects.

    Nesting makes this object not suitable for directly tokenizing; instead call flatten() on the containing Conjunction
    to produce a FlattenedPrompt.
    """
    def __init__(self, parts: list):
        for c in parts:
            if type(c) is not Attention and not issubclass(type(c), BaseFragment) and type(c) is not pp.ParseResults:
                raise PromptParser.ParsingException(f"Prompt cannot contain {type(c).__name__} ({c}), only {[c.__name__ for c in BaseFragment.__subclasses__()]} are allowed")
        self.children = parts
    def __repr__(self):
        return f"Prompt:{self.children}"
    def __eq__(self, other):
        return type(other) is Prompt and other.children == self.children

class BaseFragment:
    pass

class FlattenedPrompt():
    """
    A Prompt that has been passed through flatten(). Its children can be readily tokenized.
    """
    def __init__(self, parts: list=[]):
        self.children = []
        for part in parts:
            self.append(part)

    def append(self, fragment: Union[list, BaseFragment, tuple]):
        # verify type correctness
        if type(fragment) is list:
            for x in fragment:
                self.append(x)
        elif issubclass(type(fragment), BaseFragment):
            self.children.append(fragment)
        elif type(fragment) is tuple:
            # upgrade tuples to Fragments
            if type(fragment[0]) is not str or (type(fragment[1]) is not float and type(fragment[1]) is not int):
                raise PromptParser.ParsingException(
                    f"FlattenedPrompt cannot contain {fragment}, only Fragments or (str, float) tuples are allowed")
            self.children.append(Fragment(fragment[0], fragment[1]))
        else:
            raise PromptParser.ParsingException(
                f"FlattenedPrompt cannot contain {fragment}, only Fragments or (str, float) tuples are allowed")

    @property
    def is_empty(self):
        return len(self.children) == 0 or \
               (len(self.children) == 1 and len(self.children[0].text) == 0)

    def __repr__(self):
        return f"FlattenedPrompt:{self.children}"
    def __eq__(self, other):
        return type(other) is FlattenedPrompt and other.children == self.children


class Fragment(BaseFragment):
    """
    A Fragment is a chunk of plain text and an optional weight. The text should be passed as-is to the CLIP tokenizer.
    """
    def __init__(self, text: str, weight: float=1):
        assert(type(text) is str)
        if '\\"' in text or '\\(' in text or '\\)' in text:
            #print("Fragment converting escaped \( \) \\\" into ( ) \"")
            text = text.replace('\\(', '(').replace('\\)', ')').replace('\\"', '"')
        self.text = text
        self.weight = float(weight)

    def __repr__(self):
        return "Fragment:'"+self.text+"'@"+str(self.weight)
    def __eq__(self, other):
        return type(other) is Fragment \
            and other.text == self.text \
            and other.weight == self.weight

class Attention():
    """
    Nestable weight control for fragments. Each object in the children array may in turn be an Attention object;
    weights should be considered to accumulate as the tree is traversed to deeper levels of nesting.

    Do not traverse directly; instead obtain a FlattenedPrompt by calling Flatten() on a top-level Conjunction object.
    """
    def __init__(self, weight: float, children: list):
        if type(weight) is not float:
            raise PromptParser.ParsingException(
                f"Attention weight must be float (got {type(weight).__name__} {weight})")
        self.weight = weight
        if type(children) is not list:
            raise PromptParser.ParsingException(f"cannot make Attention with non-list of children (got {type(children)})")
        assert(type(children) is list)
        self.children = children
        #print(f"A: requested attention '{children}' to {weight}")

    def __repr__(self):
        return f"Attention:{self.children} * {self.weight}"
    def __eq__(self, other):
        return type(other) is Attention and other.weight == self.weight and other.fragment == self.fragment

class CrossAttentionControlledFragment(BaseFragment):
    pass

class CrossAttentionControlSubstitute(CrossAttentionControlledFragment):
    """
    A Cross-Attention Controlled ('prompt2prompt') fragment, for use inside a Prompt, Attention, or FlattenedPrompt.
    Representing an "original" word sequence that supplies feature vectors for an initial diffusion operation, and an
    "edited" word sequence, to which the attention maps produced by the "original" word sequence are applied. Intuitively,
    the result should be an "edited" image that looks like the "original" image with concepts swapped.

    eg "a cat sitting on a car" (original) -> "a smiling dog sitting on a car" (edited): the edited image should look
    almost exactly the same as the original, but with a smiling dog rendered in place of the cat. The
    CrossAttentionControlSubstitute object representing this swap may be confined to the tokens being swapped:
        CrossAttentionControlSubstitute(original=[Fragment('cat')], edited=[Fragment('dog')])
    or it may represent a larger portion of the token sequence:
        CrossAttentionControlSubstitute(original=[Fragment('a cat sitting on a car')],
                                        edited=[Fragment('a smiling dog sitting on a car')])

    In either case expect it to be embedded in a Prompt or FlattenedPrompt:
    FlattenedPrompt([
            Fragment('a'),
            CrossAttentionControlSubstitute(original=[Fragment('cat')], edited=[Fragment('dog')]),
            Fragment('sitting on a car')
        ])
    """
    def __init__(self, original: list, edited: list, options: dict=None):
        self.original = original
        self.edited = edited if len(edited)>0 else [Fragment('')]

        default_options = {
            's_start': 0.0,
            's_end': 0.2062994740159002, # ~= shape_freedom=0.5
            't_start': 0.0,
            't_end': 1.0
        }
        merged_options = default_options
        if options is not None:
            shape_freedom = options.pop('shape_freedom', None)
            if shape_freedom is not None:
                # high shape freedom = SD can do what it wants with the shape of the object
                # high shape freedom => s_end = 0
                # low shape freedom => s_end = 1
                # shape freedom is in a "linear" space, while noticeable changes to s_end are typically closer around 0,
                # and there is very little perceptible difference as s_end increases above 0.5
                # so for shape_freedom = 0.5 we probably want s_end to be 0.2
                #  -> cube root and subtract from 1.0
                merged_options['s_end'] = 1.0 - shape_freedom ** (1. / 3.)
                #print('converted shape_freedom argument to', merged_options)
            merged_options.update(options)

        self.options = merged_options

    def __repr__(self):
        return f"CrossAttentionControlSubstitute:({self.original}->{self.edited} ({self.options})"
    def __eq__(self, other):
        return type(other) is CrossAttentionControlSubstitute \
               and other.original == self.original \
               and other.edited == self.edited \
               and other.options == self.options


class CrossAttentionControlAppend(CrossAttentionControlledFragment):
    def __init__(self, fragment: Fragment):
        self.fragment = fragment
    def __repr__(self):
        return "CrossAttentionControlAppend:",self.fragment
    def __eq__(self, other):
        return type(other) is CrossAttentionControlAppend \
               and other.fragment == self.fragment



class Conjunction():
    """
    Storage for one or more Prompts or Blends, each of which is to be separately diffused and then the results merged
    by weighted sum in latent space.
    """
    def __init__(self, prompts: list, weights: list = None):
        # force everything to be a Prompt
        print("making conjunction with", prompts, "types", [type(p).__name__ for p in prompts])
        self.prompts = [x if (type(x) is Prompt
                          or type(x) is Blend
                          or type(x) is FlattenedPrompt)
                      else Prompt(x) for x in prompts]
        self.weights = [1.0]*len(self.prompts) if weights is None else list(weights)
        if len(self.weights) != len(self.prompts):
            raise PromptParser.ParsingException(f"while parsing Conjunction: mismatched parts/weights counts {prompts}, {weights}")
        self.type = 'AND'

    def __repr__(self):
        return f"Conjunction:{self.prompts} | weights {self.weights}"
    def __eq__(self, other):
        return type(other) is Conjunction \
               and other.prompts == self.prompts \
               and other.weights == self.weights


class Blend():
    """
    Stores a Blend of multiple Prompts. To apply, build feature vectors for each of the child Prompts and then perform a
    weighted blend of the feature vectors to produce a single feature vector that is effectively a lerp between the
    Prompts.
    """
    def __init__(self, prompts: list, weights: list[float], normalize_weights: bool=True):
        #print("making Blend with prompts", prompts, "and weights", weights)
        if len(prompts) != len(weights):
            raise PromptParser.ParsingException(f"while parsing Blend: mismatched prompts/weights counts {prompts}, {weights}")
        for p in prompts:
            if type(p) is not Prompt and type(p) is not FlattenedPrompt:
                raise(PromptParser.ParsingException(f"{type(p)} cannot be added to a Blend, only Prompts or FlattenedPrompts"))
            for f in p.children:
                if isinstance(f, CrossAttentionControlSubstitute):
                    raise(PromptParser.ParsingException(f"while parsing Blend: sorry, you cannot do .swap() as part of a Blend"))

        # upcast all lists to Prompt objects
        self.prompts = [x if (type(x) is Prompt or type(x) is FlattenedPrompt)
                         else Prompt(x)
                        for x in prompts]
        self.prompts = prompts
        self.weights = weights
        self.normalize_weights = normalize_weights

    def __repr__(self):
        return f"Blend:{self.prompts} | weights {' ' if self.normalize_weights else '(non-normalized) '}{self.weights}"
    def __eq__(self, other):
        return other.__repr__() == self.__repr__()


class PromptParser():

    class ParsingException(Exception):
        pass

    class UnrecognizedOperatorException(Exception):
        def __init__(self, operator:str):
            super().__init__("Unrecognized operator: " + operator)

    def __init__(self, attention_plus_base=1.1, attention_minus_base=0.9):

        self.conjunction, self.prompt = build_parser_syntax(attention_plus_base, attention_minus_base)


    def parse_conjunction(self, prompt: str) -> Conjunction:
        '''
        :param prompt: The prompt string to parse
        :return: a Conjunction representing the parsed results.
        '''
        #print(f"!!parsing '{prompt}'")

        if len(prompt.strip()) == 0:
            return Conjunction(prompts=[FlattenedPrompt([('', 1.0)])], weights=[1.0])

        root = self.conjunction.parse_string(prompt)
        #print(f"'{prompt}' parsed to root", root)
        #fused = fuse_fragments(parts)
        #print("fused to", fused)

        return self.flatten(root[0])

    def parse_legacy_blend(self, text: str) -> Optional[Blend]:
        weighted_subprompts = split_weighted_subprompts(text, skip_normalize=False)
        if len(weighted_subprompts) <= 1:
            return None
        strings = [x[0] for x in weighted_subprompts]
        weights = [x[1] for x in weighted_subprompts]

        parsed_conjunctions = [self.parse_conjunction(x) for x in strings]
        flattened_prompts = [x.prompts[0] for x in parsed_conjunctions]

        return Blend(prompts=flattened_prompts, weights=weights, normalize_weights=True)


    def flatten(self, root: Conjunction) -> Conjunction:
        """
        Flattening a Conjunction traverses all of the nested tree-like structures in each of its Prompts or Blends,
        producing from each of these walks a linear sequence of Fragment or CrossAttentionControlSubstitute objects
        that can be readily tokenized without the need to walk a complex tree structure.

        :param root: The Conjunction to flatten.
        :return: A Conjunction containing the result of flattening each of the prompts in the passed-in root.
        """


        def fuse_fragments(items):
            # print("fusing fragments in ", items)
            result = []
            for x in items:
                if type(x) is CrossAttentionControlSubstitute:
                    original_fused = fuse_fragments(x.original)
                    edited_fused = fuse_fragments(x.edited)
                    result.append(CrossAttentionControlSubstitute(original_fused, edited_fused, options=x.options))
                else:
                    last_weight = result[-1].weight \
                        if (len(result) > 0 and not issubclass(type(result[-1]), CrossAttentionControlledFragment)) \
                        else None
                    this_text = x.text
                    this_weight = x.weight
                    if last_weight is not None and last_weight == this_weight:
                        last_text = result[-1].text
                        result[-1] = Fragment(last_text + ' ' + this_text, last_weight)
                    else:
                        result.append(x)
            return result

        def flatten_internal(node, weight_scale, results, prefix):
            print(prefix + "flattening", node, "...")
            if type(node) is pp.ParseResults or type(node) is list:
                for x in node:
                    results = flatten_internal(x, weight_scale, results, prefix+' pr ')
                #print(prefix, " ParseResults expanded, results is now", results)
            elif type(node) is Attention:
                # if node.weight < 1:
                # todo: inject a blend when flattening attention with weight <1"
                for index,c in enumerate(node.children):
                    results = flatten_internal(c, weight_scale * node.weight, results, prefix + f" att{index} ")
            elif type(node) is Fragment:
                results += [Fragment(node.text, node.weight*weight_scale)]
            elif type(node) is CrossAttentionControlSubstitute:
                original = flatten_internal(node.original, weight_scale, [], prefix + ' CAo ')
                edited = flatten_internal(node.edited, weight_scale, [], prefix + ' CAe ')
                results += [CrossAttentionControlSubstitute(original, edited, options=node.options)]
            elif type(node) is Blend:
                flattened_subprompts = []
                #print(" flattening blend with prompts", node.prompts, "weights", node.weights)
                for prompt in node.prompts:
                    # prompt is a list
                    flattened_subprompts = flatten_internal(prompt, weight_scale, flattened_subprompts, prefix+'B ')
                results += [Blend(prompts=flattened_subprompts, weights=node.weights, normalize_weights=node.normalize_weights)]
            elif type(node) is Prompt:
                #print(prefix + "about to flatten Prompt with children", node.children)
                flattened_prompt = []
                for child in node.children:
                    flattened_prompt = flatten_internal(child, weight_scale, flattened_prompt, prefix+'P ')
                results += [FlattenedPrompt(parts=fuse_fragments(flattened_prompt))]
                #print(prefix + "after flattening Prompt, results is", results)
            else:
                raise PromptParser.ParsingException(f"unhandled node type {type(node)} when flattening {node}")
            print(prefix + "-> after flattening", type(node).__name__, "results is", results)
            return results

        print("flattening", root)

        flattened_parts = []
        for part in root.prompts:
            flattened_parts += flatten_internal(part, 1.0, [], ' C| ')

        print("flattened to", flattened_parts)

        weights = root.weights
        return Conjunction(flattened_parts, weights)




def build_parser_syntax(attention_plus_base: float, attention_minus_base: float):
    def make_operator(x):
        print('making operator for', x)
        target = x[0]
        operator = x[1]
        arguments = x[2]
        if operator == '.swap':
            return CrossAttentionControlSubstitute(target, arguments, x.as_dict())
        elif operator == '.blend':
            prompts = [Prompt(p) for p in x[0]]
            weights = [float(w[0]) for w in x[2]]
            return Blend(prompts=prompts, weights=weights)

        raise PromptParser.UnrecognizedOperatorException(operator)

    def make_attention(x):
        print('making attention for', x, "types are", [type(p).__name__ for p in x])
        weight_raw = x[1]
        weight = 1.0
        if type(weight_raw) is float or type(weight_raw) is int:
            weight = weight_raw
        elif type(weight_raw) is str:
            base = attention_plus_base if weight_raw[0] == '+' else attention_minus_base
            weight = pow(base, len(weight_raw))

        assert(type(x[0]) is list or type(x[0]) is pp.ParseResults)
        children = [x for x in x[0]]
        return Attention(weight=weight, children=children)


    def parse_fragment_str(x, expression: pp.ParseExpression, in_quotes: bool = False, in_parens: bool = False):
        print(f"parsing fragment string for {x}")
        fragment_string = x[0]
        # print(f"ppparsing fragment string \"{fragment_string}\"")

        if len(fragment_string.strip()) == 0:
            return Fragment('')

        if in_quotes:
            # escape unescaped quotes
            fragment_string = fragment_string.replace('"', '\\"')

        # fragment_parser = pp.Group(pp.OneOrMore(attention | cross_attention_substitute | (greedy_word.set_parse_action(make_text_fragment))))
        try:
            result = (expression + pp.StringEnd()).parse_string(fragment_string)
            print("parsed to", result)
            return result
        except pp.ParseException as e:
            #print("parse_fragment_str couldn't parse prompt string:", e)
            raise

    # meaningful symbols
    lparen = pp.Literal("(").suppress()
    rparen = pp.Literal(")").suppress()
    quote = pp.Literal('"').suppress()
    comma = pp.Literal(",").suppress()
    dot = pp.Literal(".").suppress()
    equals = pp.Literal("=").suppress()

    escaped_lparen = pp.Literal('\\(')#.set_parse_action(lambda x: '(')
    escaped_rparen = pp.Literal('\\)')#.set_parse_action(lambda x: ')')
    escaped_quote = pp.Literal('\\"')#.set_parse_action(lambda x: '"')
    escaped_comma = pp.Literal('\\,')#.set_parse_action(lambda x: '"')
    escaped_dot = pp.Literal('\\.')#.set_parse_action(lambda x: '"')
    escaped_plus = pp.Literal('\\+')#.set_parse_action(lambda x: '"')
    escaped_minus = pp.Literal('\\-')#.set_parse_action(lambda x: '"')
    escaped_equals = pp.Literal('\\=')

    syntactic_symbols = {
        '(': escaped_lparen,
        ')': escaped_rparen,
        '"': escaped_quote,
        ',': escaped_comma,
        '.': escaped_dot,
        '+': escaped_plus,
        '-': escaped_minus,
        '=': escaped_equals,
    }
    syntactic_chars = "".join(syntactic_symbols.keys())

    #escaped_syntactic_symbol = escaped_lparen | escaped_rparen | escaped_quote | escaped_comma | escaped_dot

    # accepts int or float notation, always maps to float
    number = pp.pyparsing_common.real | \
             pp.Combine(pp.Optional("-")+pp.Word(pp.nums)).set_parse_action(pp.token_map(float))




    restricted_word = pp.Combine(pp.OneOrMore(pp.MatchFirst([
            pp.Or(syntactic_symbols.values()),
            pp.one_of(['-', '+']) + pp.NotAny(pp.White() | pp.Char(syntactic_chars) | pp.StringEnd()),
            pp.CharsNotIn(string.whitespace + syntactic_chars, exact=1)
        ])))
    restricted_word.set_parse_action(lambda x: [Fragment(t) for t in x])
    restricted_word.set_name('restricted_word')
    restricted_word.set_debug(False)

    restricted_fragment = pp.OneOrMore(restricted_word)

    attention = pp.Forward()
    quoted_fragment = pp.Forward()
    parenthesized_fragment = pp.Forward()
    unrestricted_word = pp.Forward()
    fragment = pp.ZeroOrMore(pp.MatchFirst([
        attention,
        parenthesized_fragment,
        quoted_fragment,
        restricted_word,
        pp.Literal(',').set_parse_action(lambda x: Fragment(x[0]))
    ]))

    quoted_fragment << pp.QuotedString(quote_char='"', esc_char=None, esc_quote='\\"')
    quoted_fragment.set_parse_action(lambda x: parse_fragment_str(x, fragment, in_quotes=True))
    prompt = pp.Forward()
    quoted_prompt = quoted_fragment.copy().set_parse_action(lambda x: parse_fragment_str(x, prompt, in_quotes=True))

    parenthesized_fragment << (lparen + fragment + rparen)
    parenthesized_fragment.set_name('parenthesized_fragment')
    parenthesized_fragment.set_debug(True)


    option = pp.Group(pp.MatchFirst([
        pp.Word(pp.alphanums + '_') + equals + (number | pp.Word(pp.alphanums + '_')),  # option=value
        number.copy().set_parse_action(pp.token_map(str)), # weight
        pp.Word(pp.alphanums + '_')  # flag
    ]))
    options = pp.Dict(pp.Optional(option + pp.ZeroOrMore(comma + option)))
    options.set_name('options')
    options.set_debug(True)

    potential_target_fragment = (quoted_fragment | parenthesized_fragment | restricted_word)

    cross_attention_substitute = (
        pp.Group(potential_target_fragment).set_name('ca-target').set_debug(True)
        + pp.Literal(".swap").set_name('ca-operator').set_debug(True)
        + lparen
        + pp.Group(quoted_fragment | parenthesized_fragment | attention | restricted_fragment | pp.Empty()).set_name('ca-replacement').set_debug(True)
        + pp.Optional(comma + options).set_name('ca-options').set_debug(True)
        + rparen
    )
    cross_attention_substitute.set_name('cross_attention_substitute')
    cross_attention_substitute.set_debug(True)
    cross_attention_substitute.set_parse_action(make_operator)

    blend = (
        lparen
        + pp.Group(pp.Group(potential_target_fragment | quoted_prompt) + pp.ZeroOrMore(comma + pp.Group(potential_target_fragment | quoted_prompt))).set_name('bl-target').set_debug(True)
        + rparen
        + pp.Literal(".blend").set_name('bl-operator').set_debug(True)
        + lparen
        + pp.Group(options).set_name('bl-options').set_debug(True)
        + rparen
    )
    blend.set_name('blend')
    blend.set_debug(True)
    blend.set_parse_action(make_operator)


    attention << pp.Group(quoted_fragment | parenthesized_fragment | restricted_word) + pp.NotAny(pp.White()) + (pp.Word('+') | pp.Word('-') | number)
    attention.set_parse_action(make_attention)
    attention.set_name('attention').set_debug(True)

    unrestricted_word << pp.CharsNotIn(string.whitespace).set_parse_action(lambda x: Fragment(x[0]))
    unrestricted_word.set_name('unrestricted_word')
    unrestricted_word.set_debug(False)

    """
    word_level_keywords = ['.swap']
    word_level_op = build_fragment_word(excluded_chars="".join(syntactic_symbols.keys())) + pp.one_of(word_level_keywords) + lparen + fragment_argument + pp.Optional(comma + options) + rparen
    word_level_op.set_name('word_level_op').set_parse_action(make_word_level_op)


    def build_fragment_word(excluded_chars):
        return pp.Combine(pp.OneOrMore(pp.MatchFirst([
            pp.Or([syntactic_symbols[x] for x in excluded_chars]), 
            pp.CharsNotIn(string.whitespace + excluded_chars, exact=1)
        ])))

    cross_attention_option_keyword = pp.Or([pp.Keyword("s_start"), pp.Keyword("s_end"), pp.Keyword("t_start"), pp.Keyword("t_end"), pp.Keyword("shape_freedom")])
    cross_attention_option = pp.Group(cross_attention_option_keyword + pp.Literal("=").suppress() + number)
    pp.Dict(pp.ZeroOrMore(comma + cross_attention_option)) +
    """

    prompt << pp.ZeroOrMore(pp.MatchFirst([
        cross_attention_substitute,
        attention,
        quoted_fragment,
        parenthesized_fragment,
        unrestricted_word,
        pp.White().suppress()
        #pp.Literal(',').set_parse_action(lambda x: Fragment(x[0])),
    ]))

    #def make_conjunction(x):
    #    return Conjunction([x])

    conjunction = (blend | pp.Group(prompt)) + pp.StringEnd()
    conjunction.set_parse_action(lambda x: Conjunction(x))

    return conjunction, prompt


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
            "Warning: Subprompt weights add up to zero. Discarding and using even weights instead.")
        equal_weight = 1 / max(len(parsed_prompts), 1)
        return [(x[0], equal_weight) for x in parsed_prompts]
    return [(x[0], x[1] / weight_sum) for x in parsed_prompts]


# shows how the prompt is tokenized
# usually tokens have '</w>' to indicate end-of-word,
# but for readability it has been replaced with ' '
def log_tokenization(text, model, display_label=None):
    tokens    = model.cond_stage_model.tokenizer._tokenize(text)
    tokenized = ""
    discarded = ""
    usedTokens = 0
    totalTokens = len(tokens)
    for i in range(0, totalTokens):
        token = tokens[i].replace('</w>', 'x` ')
        # alternate color
        s = (usedTokens % 6) + 1
        if i < model.cond_stage_model.max_length:
            tokenized = tokenized + f"\x1b[0;3{s};40m{token}"
            usedTokens += 1
        else:  # over max token length
            discarded = discarded + f"\x1b[0;3{s};40m{token}"
    print(f"\n>> Tokens {display_label or ''} ({usedTokens}):\n{tokenized}\x1b[0m")
    if discarded != "":
        print(
            f">> Tokens Discarded ({totalTokens-usedTokens}):\n{discarded}\x1b[0m"
        )
