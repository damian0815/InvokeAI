import pyparsing
import pyparsing as pp
from pyparsing import original_text_for


class CFGScale():
    def __init__(self, scale_factor: float, fragment: str):
        self.fragment = fragment
        self.scale_factor = scale_factor
        #print(f"S: requested CFGScale '{fragment}' x {scale_factor}")

    def __repr__(self):
        return f"CFGScale('{self.fragment}' x {self.scale_factor})"
    def __eq__(self, other):
        return type(other) is CFGScale and other.scale_factor == self.scale_factor and other.fragment == self.fragment


class Attention():
    def __init__(self, weight: float, fragment: str):
        self.weight = weight
        self.fragment = fragment
        #print(f"A: requested attention '{fragment}' to {weight}")

    def __repr__(self):
        return f"Attention('{self.fragment}' @ {self.weight})"
    def __eq__(self, other):
        return type(other) is Attention and other.weight == self.weight and other.fragment == self.fragment

class Fragment():
    def __init__(self, text: str):
        self.text = text

    def __repr__(self):
        return "Fragment('"+self.text+"')"
    def __eq__(self, other):
        return type(other) is Fragment and other.text == self.text

class Blend():
    def __init__(self, prompts: list, weights: list[float]):
        #print("making Blend with prompts", prompts, "and weights", weights)
        if len(prompts) != len(weights):
            raise PromptParser.ParsingException("().blend(): mismatched prompt/weight counts")
        self.prompts = [fuse_fragments(x) for x in prompts]
        self.weights = weights

    def __repr__(self):
        return f"Blend({[self.prompts]} | weights {self.weights}"
    def __eq__(self, other):
        return other.__repr__() == self.__repr__()


class PromptParser():

    class ParsingException(Exception):
        pass

    def __init__(self):

        lparen = pp.Suppress("(")
        rparen = pp.Suppress(")")

        number = pyparsing.pyparsing_common.real | pp.Word(pp.nums).set_parse_action(pp.token_map(float))

        attention_explicit = pp.Group(number + lparen + pp.CharsNotIn(')') + rparen)
        SPACE_CHARS = ' \t\n'
        attention_word = pp.CharsNotIn(SPACE_CHARS+'()')
        space = pp.Word(SPACE_CHARS)
        attention_plus = pp.Group(pp.Word('+') + ((lparen + attention_word + pp.ZeroOrMore(space + attention_word) + rparen) |
                                                  attention_word)).set_debug(False)
        attention_minus = pp.Group(pp.Word('-') + ((lparen + attention_word + pp.ZeroOrMore(space + attention_word) + rparen) |
                                                  attention_word)).set_debug(False)
        attention_explicit.set_parse_action(lambda x: Attention(weight=float(x[0][0]), fragment=x[0][1]))
        attention_plus.set_parse_action(lambda x: Attention(weight=pow(1.1, len(x[0][0])), fragment=x[0][1]))
        attention_minus.set_parse_action(lambda x: Attention(weight=pow(0.9, len(x[0][0])), fragment=x[0][1]))

        # attention: number(fragment)
        attention = attention_plus | attention_minus | attention_explicit

        word = pp.Word(pp.printables).set_parse_action(lambda x: Fragment(' '.join([s for s in x])))
        prompt_part = attention | word


        prompt = pp.Optional(pp.OneOrMore(prompt_part))

        # cfg scale: (fragment).scale(number)
        blend_terms = pp.QuotedString('"') + pp.ZeroOrMore(pp.Suppress(",") + pp.QuotedString('"'))
        blend_weights = number + pp.ZeroOrMore(pp.Suppress(",") + number)
        blend = pp.Group(lparen + pp.Group(blend_terms) + rparen + pp.Literal(".blend").suppress() + lparen + pp.Group(blend_weights) + rparen)

        self.prompt = blend | prompt

        blend.set_parse_action(lambda x: Blend(prompts=[list(prompt.parseString(p)) for p in x[0][0]], weights=x[0][1]))


    def parse(self, prompt):
        '''
        :param prompt: The prompt string to parse
        :return: a tuple
        '''
        #print("parsing", prompt)

        parts = self.prompt.parseString(prompt)

        return parts

def fuse_fragments(items):
    result = []
    for x in items:
        if type(x) is not Fragment:
            result.append(x)
            continue
        if len(result)==0 or type(result[-1]) is not Fragment:
            result.append(x)
            continue
        # append to existing
        result[-1] = Fragment(result[-1].text + " " + x.text)
    return result

def parse_prompt(prompt_string):
    pp = PromptParser()
    print(f"parsing '{prompt_string}'")
    parsed = pp.parse(prompt_string)
    #print(f"-> parsed '{prompt_string}' to ", parsed)
    return list(parsed)