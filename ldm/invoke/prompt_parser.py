import pyparsing
import pyparsing as pp
from pyparsing import original_text_for

class Prompt():

    def __init__(self, parts: list):
        self.children = parts
    def __repr__(self):
        return f"Prompt({self.children})"
    def __eq__(self, other):
        return type(other) is Prompt and other.children == self.children


class Attention():

    @classmethod
    def from_parsing(cls, x0, x1):
        print("making Attention from parsing with args", x0, x1)
        weight = 1
        if type(x0) is float or type(x0) is int:
            weight = float(x0)
        elif type(x0) is str:
            base = 1.1 if x0[0] == '+' else 0.9
            weight = pow(base, len(x0))
        return Attention(weight=weight, children=x1)

    def __init__(self, weight: float, children: list):
        self.weight = weight
        self.children = children
        print(f"A: requested attention '{children}' to {weight}")

    def __repr__(self):
        return f"Attention('{self.children}' @ {self.weight})"
    def __eq__(self, other):
        return type(other) is Attention and other.weight == self.weight and other.fragment == self.fragment


class CFGScale():
    def __init__(self, scale_factor: float, fragment: str):
        self.fragment = fragment
        self.scale_factor = scale_factor
        #print(f"S: requested CFGScale '{fragment}' x {scale_factor}")

    def __repr__(self):
        return f"CFGScale('{self.fragment}' x {self.scale_factor})"
    def __eq__(self, other):
        return type(other) is CFGScale and other.scale_factor == self.scale_factor and other.fragment == self.fragment



class Fragment():
    def __init__(self, text: str):
        assert(type(text) is str)
        self.text = text

    def __repr__(self):
        return "Fragment('"+self.text+"')"
    def __eq__(self, other):
        return type(other) is Fragment and other.text == self.text

class Conjunction():
    def __init__(self, parts: list):
        self.parts = parts
    def __repr__(self):
        return f"Conjunction({self.parts})"
    def __eq__(self, other):
        return type(other) is Conjunction and other.parts == self.parts


class Blend():
    def __init__(self, children: list, weights: list[float]):
        print("making Blend with prompts", children, "and weights", weights)
        if len(children) != len(weights):
            raise PromptParser.ParsingException("().blend(): mismatched child/weight counts")
        self.children = children
        self.weights = weights

    def __repr__(self):
        return f"Blend({self.children} | weights {self.weights})"
    def __eq__(self, other):
        return other.__repr__() == self.__repr__()


class PromptParser():

    class ParsingException(Exception):
        pass

    def __init__(self):

        lparen = pp.Literal("(").suppress()
        rparen = pp.Literal(")").suppress()
        number = pyparsing.pyparsing_common.real | pp.Word(pp.nums).set_parse_action(pp.token_map(float))
        SPACE_CHARS = ' \t\n'

        prompt_part = pp.Forward()
        word = pp.Forward()

        def make_fragment(x):
            if type(x) is str:
                return Fragment(x)
            elif type(x) is pp.ParseResults or type(x) is list:
                return Fragment(' '.join([s for s in x]))
            else:
                raise PromptParser.ParsingException("Cannot make fragment from " + str(x))

        fragment_inside_attention = pp.OneOrMore(pp.CharsNotIn('()'))\
            .set_parse_action(make_fragment)\
            .set_name("fragment_inside_attention")\
            .set_debug(False)
        # .set_parse_action(lambda x: Fragment(' '.join([s for s in x])))\
        # .set_parse_action\

        attention = pp.Forward()
        attention_head = (number | pp.Word('+') | pp.Word('-'))\
            .set_debug(False)\
            .set_name("attention_head")
        attention_with_parens = (attention_head + pp.nested_expr(content=(attention | pp.OneOrMore(fragment_inside_attention))))\
            .set_name("attention_with_parens")\
            .set_debug(False)

        attention_without_parens = ((pp.Word('+') | pp.Word('-')) + pp.CharsNotIn('()').set_parse_action(lambda x: [[make_fragment(x)]]))\
            .set_name("attention_without_parens")

        attention << (attention_with_parens | attention_without_parens)\
            .set_name("attention")
        attention.set_debug(False)

        def parse_attention_internal(x):
            print("attention guts is", x[1])
            #if type(x[1]) is str:
            #    return Attention.from_parsing(prompt_part.parse_string(x[0]), x[1])
            #else:
            return Attention.from_parsing(x[0], x[1])
        attention.set_parse_action(parse_attention_internal)

        #cfg_scale_tail = (pp.Literal(".scale(") + number + ")")
        #cfg_guts = pp.CharsNotIn(')')
        #cfg_scale = (lparen + cfg_guts + rparen + cfg_scale_tail)

        word << pp.Word(pp.printables).set_parse_action(lambda x: Fragment(' '.join([s for s in x])))
        word.set_name("word")
        word.set_debug(False)
        prompt_part << (attention | word)
        prompt_part.set_debug(False)
        prompt_part.set_name("prompt_part")

        prompt = pp.Group(pp.OneOrMore(prompt_part)).set_parse_action(lambda x: Prompt(x))

        # cfg scale: (fragment).scale(number)
        blend_terms = pp.delimited_list(pp.dbl_quoted_string)
        blend_weights = number + pp.ZeroOrMore(pp.Suppress(",") + number)
        blend = pp.Group(lparen + pp.Group(blend_terms) + rparen + pp.Literal(".blend").suppress() + lparen + pp.Group(blend_weights) + rparen)

        self.root = pp.OneOrMore(blend | prompt)

        def unquote_and_parse(x):
            print(f"unquoting x:'{x}' to '{x[1:-1]}'")
            x_unquoted = x[1:-1]
            if len(x_unquoted.strip()) == 0:
                return [Prompt([Fragment('')])]
            return prompt.parse_string(x_unquoted)

        blend.set_parse_action(lambda x: Blend(children=[unquote_and_parse(p) for p in x[0][0]], weights=x[0][1]))



    def parse(self, prompt: str) -> [list]:
        '''
        :param prompt: The prompt string to parse
        :return: a tuple
        '''
        #print("parsing", prompt)

        if len(prompt.strip()) == 0:
            return Conjunction(parts=[('', 1.0)])

        roots = self.root.parse_string(prompt)
        #fused = fuse_fragments(parts)
        #print("fused to", fused)

        result = []
        for x in roots:
            print("- Root:", x)
            if type(x) is Blend:
                blend_targets = []
                for c in x.children:
                    blend_targets += [self.flatten(c[0])]
                result.append(Blend(blend_targets, x.weights))
            else:
                result.append(self.flatten(x))
        return Conjunction(parts=result)

    def flatten(self, root: Prompt):

        def flatten_internal(node, weight_scale, results, prefix):
            print(prefix, "flattening", node, "...")
            if type(node) is pp.ParseResults:
                for x in node:
                    results = flatten_internal(x, weight_scale, results, prefix+'pr')
                print(prefix, " ParseResults expanded, results is now", results)
            elif type(node) is Fragment:
                results.append((node.text, float(weight_scale)))
            elif type(node) is Attention:
                if node.weight < 1:
                    print("todo: add a blend when flattening attention with weight <1")
                for c in node.children:
                    results = flatten_internal(c, weight_scale*node.weight, results, prefix+'  ')
            elif type(node) is Blend:
                flattened_subprompts = []
                print(" flattening blend with prompts", node.prompts)
                for prompt in node.prompts:
                    # prompt is a list
                    flattened_subprompt = [flatten_internal(p, weight_scale, [], prefix+'  ') for p in prompt]
                    print(" blend flattened", prompt, "to", flattened_subprompt)
                    flattened_subprompts.append(flattened_subprompt)
                results += [Blend(prompts=fuse_fragments(flattened_subprompts), weights=node.weights)]
            else:
                raise PromptParser.ParsingException(f"unhandled node type {type(node)} when flattening {node}")
            print(prefix, "-> node", node, "appended, results now", results)
            return results

        print("flattening", root)
        all_results = []
        for c in root.children:
            flattened = flatten_internal(c, 1, [], '|')
            print("| child", c, "flattened to", flattened)
            fused = fuse_fragments([x for x in flattened])
            print("| child", c, "flattened and fused to", flattened)
            all_results += fused
        return all_results






def fuse_fragments(items):
    print("fusing fragments in ", items)
    result = []
    for x in items:
        last_weight = result[-1][1] if len(result)>0 else None
        this_text = x[0]
        this_weight = x[1]
        if last_weight is not None and last_weight == this_weight:
            last_text = result[-1][0]
            result[-1] = (last_text + ' ' + this_text, last_weight)
        else:
            result.append(x)
    return result

def parse_prompt(prompt_string):
    pp = PromptParser()
    print(f"parsing '{prompt_string}'")
    parse_result = pp.parse(prompt_string)
    print(f"-> parsed '{prompt_string}' to {parse_result}")
    return parse_result