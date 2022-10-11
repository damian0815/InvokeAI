import pyparsing
import pyparsing as pp
from pyparsing import original_text_for

class Prompt():

    def __init__(self, parts: list):
        for c in parts:
            allowed_types = [Fragment, Attention, CFGScale]
            if type(c) not in allowed_types:
                raise PromptParser.ParsingException(f"Prompt cannot contain {type(c)}, only {allowed_types} are allowed")
        self.children = parts
    def __repr__(self):
        return f"Prompt:{self.children}"
    def __eq__(self, other):
        return type(other) is Prompt and other.children == self.children

class FlattenedPrompt():
    def __init__(self, parts: list):
        for c in parts:
            if type(c) is not tuple or type(c[0]) is not str or (type(c[1]) is not float and type(c[1]) is not int):
                raise PromptParser.ParsingException(f"FlattenedPrompt cannot contain {type(c)}, only (str, float) tuples are allowed")
        self.children = parts

    def __repr__(self):
        return f"FlattenedPrompt:{self.children}"
    def __eq__(self, other):
        return type(other) is FlattenedPrompt and other.children == self.children


class Attention():

    @classmethod
    def from_parsing(cls, x0, x1):
        #print("making Attention from parsing with args", x0, x1)
        weight = 1
        if type(x0) is float or type(x0) is int:
            weight = float(x0)
        elif type(x0) is str:
            base = 1.1 if x0[0] == '+' else 0.9
            weight = pow(base, len(x0))
        #print("Making attention with children of type", [str(type(x)) for x in x1])
        return Attention(weight=weight, children=x1)

    def __init__(self, weight: float, children: list):
        self.weight = weight
        self.children = children
        #print(f"A: requested attention '{children}' to {weight}")

    def __repr__(self):
        return f"Attention:'{self.children}' @ {self.weight}"
    def __eq__(self, other):
        return type(other) is Attention and other.weight == self.weight and other.fragment == self.fragment


class CFGScale():
    def __init__(self, scale_factor: float, fragment: str):
        self.fragment = fragment
        self.scale_factor = scale_factor
        #print(f"S: requested CFGScale '{fragment}' x {scale_factor}")

    def __repr__(self):
        return f"CFGScale:'{self.fragment}' x {self.scale_factor}"
    def __eq__(self, other):
        return type(other) is CFGScale and other.scale_factor == self.scale_factor and other.fragment == self.fragment



class Fragment():
    def __init__(self, text: str):
        assert(type(text) is str)
        self.text = text

    def __repr__(self):
        return "Fragment:'"+self.text+"'"
    def __eq__(self, other):
        return type(other) is Fragment and other.text == self.text

class Conjunction():
    def __init__(self, parts: list):
        # force everything to be a Prompt
        self.parts = [x if (type(x) is Prompt or type(x) is Blend or type(x) is FlattenedPrompt)
                      else Prompt(x) for x in parts]

    def __repr__(self):
        return f"Conjunction:{self.parts}"
    def __eq__(self, other):
        return type(other) is Conjunction and other.parts == self.parts


class Blend():
    def __init__(self, children: list, weights: list[float]):
        print("making Blend with prompts", children, "and weights", weights)
        if len(children) != len(weights):
            raise PromptParser.ParsingException("().blend(): mismatched child/weight counts")
        for c in children:
            if type(c) is not Prompt and type(c) is not FlattenedPrompt:
                raise(PromptParser.ParsingException(f"{type(c)} cannot be added to a Blend, only Prompts or FlattenedPrompts"))
        # upcast all lists to Prompt objects
        self.children = [x if (type(x) is Prompt or type(x) is FlattenedPrompt)
                         else Prompt(x) for x in children]
        self.children = children
        self.weights = weights

    def __repr__(self):
        return f"Blend:{self.children} | weights {self.weights}"
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
            #print("### making fragment for", x)
            if type(x) is str:
                return Fragment(x)
            elif type(x) is pp.ParseResults or type(x) is list:
                return Fragment(' '.join([s for s in x]))
            else:
                raise PromptParser.ParsingException("Cannot make fragment from " + str(x))

        fragment_inside_attention = pp.CharsNotIn(SPACE_CHARS+'()')\
            .set_parse_action(make_fragment)\
            .set_name("fragment_inside_attention")\
            .set_debug(False)

        attention = pp.Forward()
        attention_head = (number | pp.Word('+') | pp.Word('-'))\
            .set_name("attention_head")\
            .set_debug(False)

        attention_with_parens = pp.Forward()
        attention_with_parens_body = pp.nested_expr(content=pp.delimited_list((attention_with_parens | fragment_inside_attention), delim=SPACE_CHARS))
        attention_with_parens << (attention_head + attention_with_parens_body)
        attention_with_parens.set_parse_action(lambda x: Attention.from_parsing(x[0], x[1]))\
            .set_name("attention_with_parens")\
            .set_debug(False)

        attention_without_parens = ((pp.Word('+') | pp.Word('-')) + pp.CharsNotIn('()')
            .set_parse_action(lambda x: [[make_fragment(x)]]))\
            .set_name("attention_without_parens")
        attention_without_parens.set_parse_action(lambda x: Attention.from_parsing(x[0], x[1]))

        attention << (attention_with_parens | attention_without_parens)\
            .set_name("attention")
        attention.set_debug(False)



        #print("&&& parsing", attention.parse_string("1.5(trees)"))

        #cfg_scale_tail = (pp.Literal(".scale(") + number + ")")
        #cfg_guts = pp.CharsNotIn(')')
        #cfg_scale = (lparen + cfg_guts + rparen + cfg_scale_tail)

        word << pp.Word(pp.printables).set_parse_action(lambda x: Fragment(' '.join([s for s in x])))
        word.set_name("word")
        word.set_debug(False)
        prompt_part << (attention | word)
        prompt_part.set_debug(False)
        prompt_part.set_name("prompt_part")

        prompt = pp.Group(pp.OneOrMore(prompt_part))\
            .set_parse_action(lambda x: Prompt(x[0]))

        # cfg scale: (fragment).scale(number)
        blend_terms = pp.delimited_list(pp.dbl_quoted_string)
        blend_weights = number + pp.ZeroOrMore(pp.Suppress(",") + number)
        blend = pp.Group(lparen + pp.Group(blend_terms) + rparen + pp.Literal(".blend").suppress() + lparen + pp.Group(blend_weights) + rparen)

        self.root = pp.OneOrMore(blend | prompt)

        def make_blend_subprompt(x):
            weights = x[0][1]
            children = []
            for c in x[0][0]:
                c_unquoted = c[1:-1]
                if len(c_unquoted.strip()) == 0:
                    return [Prompt([Fragment('')])]
                c_parsed = prompt.parse_string(c_unquoted)
                print("blend part was parsed to", type(c_parsed),":", c_parsed)
                children.append(c_parsed[0])
            return Blend(children=children, weights=weights)

        blend.set_parse_action(make_blend_subprompt)



    def parse(self, prompt: str) -> [list]:
        '''
        :param prompt: The prompt string to parse
        :return: a tuple
        '''
        #print("parsing", prompt)

        if len(prompt.strip()) == 0:
            return Conjunction(parts=[FlattenedPrompt([('', 1.0)])])

        roots = self.root.parse_string(prompt)
        #fused = fuse_fragments(parts)
        #print("fused to", fused)

        result = []
        for x in roots:
            print("- Root:", x)
            if type(x) is Blend:
                blend_targets = []
                for child in x.children:
                    child_flattened = self.flatten(child)
                    blend_targets.extend(child_flattened)
                result.append(Blend(blend_targets, x.weights))
            else:
                result.extend(self.flatten(x))
        return Conjunction(parts=result)

    def flatten(self, root: Prompt):

        def flatten_internal(node, weight_scale, results, prefix):
            print(prefix + "flattening", node, "...")
            if type(node) is pp.ParseResults:
                for x in node:
                    results = flatten_internal(x, weight_scale, results, prefix+'pr')
                #print(prefix, " ParseResults expanded, results is now", results)
            elif type(node) is Fragment:
                results.append((node.text, float(weight_scale)))
            elif type(node) is Attention:
                if node.weight < 1:
                    print(prefix + "todo: add a blend when flattening attention with weight <1")
                for c in node.children:
                    results = flatten_internal(c, weight_scale*node.weight, results, prefix+'  ')
            elif type(node) is Blend:
                flattened_subprompts = []
                #print(" flattening blend with prompts", node.prompts)
                for prompt in node.prompts:
                    # prompt is a list
                    flattened_subprompt = [flatten_internal(p, weight_scale, [], prefix+'B ') for p in prompt]
                    #print(" blend flattened", prompt, "to", flattened_subprompt)
                    flattened_subprompts.append(flattened_subprompt)
                results += [Blend(prompts=fuse_fragments(flattened_subprompts), weights=node.weights)]
            elif type(node) is Prompt:
                print(prefix + "about to flatten Prompt with children", node.children)
                flattened_prompt = []
                for child in node.children:
                    flattened_prompt = flatten_internal(child, weight_scale, flattened_prompt, prefix+'P ')
                results += [FlattenedPrompt(parts=fuse_fragments(flattened_prompt))]
                print(prefix + "after flattening Prompt, results is", results)
            else:
                raise PromptParser.ParsingException(f"unhandled node type {type(node)} when flattening {node}")
            print(prefix + "-> after flattening", type(node), "results is", results)
            return results

        print("flattening", root)
        return flatten_internal(root, 1.0, [], '| ')
        #all_results = []
        #for c in root.children:
        #    flattened = flatten_internal(c, 1, [], '|')
        #    #print("| child", c, "flattened to", flattened)
        #    fused = fuse_fragments([x for x in flattened])
        #    #print("| child", c, "flattened and fused to", flattened)
        #    all_results += fused
        #return all_results






def fuse_fragments(items):
    #print("fusing fragments in ", items)
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
    #print(f"parsing '{prompt_string}'")
    parse_result = pp.parse(prompt_string)
    #print(f"-> parsed '{prompt_string}' to {parse_result}")
    return parse_result