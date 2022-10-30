
def build_parser_syntax_old(attention_plus_base: float, attention_minus_base: float):

    lparen = pp.Literal("(").suppress()
    rparen = pp.Literal(")").suppress()
    quotes = pp.Literal('"').suppress()
    comma = pp.Literal(",").suppress()

    # accepts int or float notation, always maps to float
    number = pp.pyparsing_common.real | \
             pp.Combine(pp.Optional("-")+pp.Word(pp.nums)).set_parse_action(pp.token_map(float))

    attention = pp.Forward()
    quoted_fragment = pp.Forward()
    parenthesized_fragment = pp.Forward()
    cross_attention_substitute = pp.Forward()

    def make_text_fragment(x):
        #print("### making fragment for", x)
        if type(x[0]) is Fragment:
            assert(False)
        if type(x) is str:
            return Fragment(x)
        elif type(x) is pp.ParseResults or type(x) is list:
            #print(f'converting {type(x).__name__} to Fragment')
            return Fragment(' '.join([s for s in x]))
        else:
            raise PromptParser.ParsingException("Cannot make fragment from " + str(x))

    def build_escaped_word_parser_charbychar(escaped_chars_to_ignore: str):
        escapes = []
        for c in escaped_chars_to_ignore:
            escapes.append(pp.Literal('\\'+c))
        return pp.Combine(pp.OneOrMore(
            pp.MatchFirst(escapes + [pp.CharsNotIn(
                string.whitespace + escaped_chars_to_ignore,
                exact=1
            )])
        ))



    def parse_fragment_str(x, in_quotes: bool=False, in_parens: bool=False):
        #print(f"parsing fragment string for {x}")
        fragment_string = x[0]
        #print(f"ppparsing fragment string \"{fragment_string}\"")

        if len(fragment_string.strip()) == 0:
            return Fragment('')

        if in_quotes:
            # escape unescaped quotes
            fragment_string = fragment_string.replace('"', '\\"')

        #fragment_parser = pp.Group(pp.OneOrMore(attention | cross_attention_substitute | (greedy_word.set_parse_action(make_text_fragment))))
        try:
            result = pp.Group(pp.MatchFirst([
                    pp.OneOrMore(quoted_fragment | attention | unquoted_word).set_name('pf_str_qfuq'),
                    pp.Empty().set_parse_action(make_text_fragment) + pp.StringEnd()
            ])).set_name('blend-result').set_debug(False).parse_string(fragment_string)
            #print("parsed to", result)
            return result
        except pp.ParseException as e:
            #print("parse_fragment_str couldn't parse prompt string:", e)
            raise

    quoted_fragment << pp.QuotedString(quote_char='"', esc_char=None, esc_quote='\\"')
    quoted_fragment.set_parse_action(lambda x: parse_fragment_str(x, in_quotes=True)).set_name('quoted_fragment')

    escaped_quote = pp.Literal('\\"')#.set_parse_action(lambda x: '"')
    escaped_lparen = pp.Literal('\\(')#.set_parse_action(lambda x: '(')
    escaped_rparen = pp.Literal('\\)')#.set_parse_action(lambda x: ')')
    escaped_backslash = pp.Literal('\\\\')#.set_parse_action(lambda x: '"')

    empty = (
            (lparen + pp.ZeroOrMore(pp.Word(string.whitespace)) + rparen) |
            (quotes + pp.ZeroOrMore(pp.Word(string.whitespace)) + quotes)).set_debug(False).set_name('empty')


    def not_ends_with_swap(x):
        #print("trying to match:", x)
        return not x[0].endswith('.swap')

    unquoted_word = (pp.Combine(pp.OneOrMore(
            escaped_rparen | escaped_lparen | escaped_quote | escaped_backslash |
            (pp.CharsNotIn(string.whitespace + '\\"()', exact=1)
    )))
            # don't whitespace when the next word starts with +, eg "badly +formed"
         + (pp.White().suppress() |
            # don't eat +/-
            pp.NotAny(pp.Word('+') | pp.Word('-'))
            )
                     )

    unquoted_word.set_parse_action(make_text_fragment).set_name('unquoted_word').set_debug(False)
    #print(unquoted_fragment.parse_string("cat.swap(dog)"))

    parenthesized_fragment << (lparen +
       pp.Or([
        (parenthesized_fragment),
        (quoted_fragment.copy().set_parse_action(lambda x: parse_fragment_str(x, in_quotes=True)).set_debug(False)).set_name('-quoted_paren_internal').set_debug(False),
        (pp.Combine(pp.OneOrMore(
            escaped_quote | escaped_lparen | escaped_rparen | escaped_backslash |
            pp.CharsNotIn(string.whitespace + '\\"()', exact=1) |
            pp.White()
        )).set_name('--combined').set_parse_action(lambda x: parse_fragment_str(x, in_parens=True)).set_debug(False)),
        pp.Empty()
       ]) + rparen)
    parenthesized_fragment.set_name('parenthesized_fragment')
    parenthesized_fragment.set_debug(False)

    debug_attention = False
    # attention control of the form (phrase)+ / (phrase)+ / (phrase)<weight>
    # phrase can be multiple words, can have multiple +/- signs to increase the effect or type a floating point or integer weight
    attention_with_parens = pp.Forward()
    attention_without_parens = pp.Forward()

    attention_with_parens_foot = (number | pp.Word('+') | pp.Word('-'))\
        .set_name("attention_foot")\
        .set_debug(False)
    attention_with_parens <<= pp.Group(
        lparen +
        pp.ZeroOrMore(quoted_fragment | attention_with_parens | parenthesized_fragment | cross_attention_substitute | attention_without_parens |
                      (pp.Empty() + build_escaped_word_parser_charbychar('()')).set_name('undecorated_word').set_debug(debug_attention)#.set_parse_action(lambda t: t[0])
                  )
        + rparen + attention_with_parens_foot)
    attention_with_parens.set_name('attention_with_parens').set_debug(debug_attention)

    attention_without_parens_foot = (pp.NotAny(pp.White()) + pp.Or([pp.Word('+'), pp.Word('-')]) + pp.FollowedBy(pp.StringEnd() | pp.White() | pp.Literal('(') | pp.Literal(')') | pp.Literal(',') | pp.Literal('"')) ).set_name('attention_without_parens_foots')
    attention_without_parens <<= pp.Group(pp.MatchFirst([
        quoted_fragment.copy().set_name('attention_quoted_fragment_without_parens').set_debug(debug_attention) + attention_without_parens_foot,
        pp.Combine(build_escaped_word_parser_charbychar('()+-')).set_name('attention_word_without_parens').set_debug(debug_attention)#.set_parse_action(lambda x: print('escapéd', x))
                                 + attention_without_parens_foot#.leave_whitespace()
    ]))
    attention_without_parens.set_name('attention_without_parens').set_debug(debug_attention)


    attention << pp.MatchFirst([attention_with_parens,
                  attention_without_parens
                  ])
    attention.set_name('attention')

    def make_attention(x):
        #print("entered make_attention with", x)
        children = x[0][:-1]
        weight_raw = x[0][-1]
        weight = 1.0
        if type(weight_raw) is float or type(weight_raw) is int:
            weight = weight_raw
        elif type(weight_raw) is str:
            base = attention_plus_base if weight_raw[0] == '+' else attention_minus_base
            weight = pow(base, len(weight_raw))

        #print("making Attention from", children, "with weight", weight)

        return Attention(weight=weight, children=[(Fragment(x) if type(x) is str else x) for x in children])

    attention_with_parens.set_parse_action(make_attention)
    attention_without_parens.set_parse_action(make_attention)

    #print("parsing test:", attention_with_parens.parse_string("mountain (man)1.1"))

    # cross-attention control
    empty_string = ((lparen + rparen) |
                    pp.Literal('""').suppress() |
                    (lparen + pp.Literal('""').suppress() + rparen)
                    ).set_parse_action(lambda x: Fragment(""))
    empty_string.set_name('empty_string')

    # cross attention control
    debug_cross_attention_control = False
    original_fragment = pp.MatchFirst([
                        quoted_fragment.set_debug(debug_cross_attention_control),
                        parenthesized_fragment.set_debug(debug_cross_attention_control),
                        pp.Combine(pp.OneOrMore(pp.CharsNotIn(string.whitespace + '.', exact=1))).set_parse_action(make_text_fragment) + pp.FollowedBy(".swap"),
                        empty_string.set_debug(debug_cross_attention_control),
               ])
    # support keyword=number arguments
    cross_attention_option_keyword = pp.Or([pp.Keyword("s_start"), pp.Keyword("s_end"), pp.Keyword("t_start"), pp.Keyword("t_end"), pp.Keyword("shape_freedom")])
    cross_attention_option = pp.Group(cross_attention_option_keyword + pp.Literal("=").suppress() + number)
    edited_fragment = pp.MatchFirst([
        (lparen + rparen).set_parse_action(lambda x: Fragment('')),
        lparen +
            (quoted_fragment | attention |
                pp.Group(pp.ZeroOrMore(build_escaped_word_parser_charbychar(',)').set_parse_action(make_text_fragment)))
            ) +
            pp.Dict(pp.ZeroOrMore(comma + cross_attention_option)) +
        rparen,
        parenthesized_fragment
    ])
    cross_attention_substitute << original_fragment + pp.Literal(".swap").set_debug(False).suppress() + edited_fragment

    original_fragment.set_name('original_fragment').set_debug(debug_cross_attention_control)
    edited_fragment.set_name('edited_fragment').set_debug(debug_cross_attention_control)
    cross_attention_substitute.set_name('cross_attention_substitute').set_debug(debug_cross_attention_control)

    def make_cross_attention_substitute(x):
        # print("making cacs for", x[0], "->", x[1], "with options", x.as_dict())
        # if len(x>2):
        cacs = CrossAttentionControlSubstitute(x[0], x[1], options=x.as_dict())
        # print("made", cacs)
        return cacs
    cross_attention_substitute.set_parse_action(make_cross_attention_substitute)


    # root prompt definition
    debug_root_prompt = False
    prompt = (pp.OneOrMore(pp.MatchFirst([cross_attention_substitute.set_debug(debug_root_prompt),
                                  attention.set_debug(debug_root_prompt),
                                  quoted_fragment.set_debug(debug_root_prompt),
                                  parenthesized_fragment.set_debug(debug_root_prompt),
                                  unquoted_word.set_debug(debug_root_prompt),
                                  empty.set_parse_action(make_text_fragment).set_debug(debug_root_prompt)])
                           ) + pp.StringEnd()) \
        .set_name('prompt') \
        .set_parse_action(lambda x: Prompt(x)) \
        .set_debug(debug_root_prompt)

    #print("parsing test:", prompt.parse_string("spaced eyes--"))
    #print("parsing test:", prompt.parse_string("eyes--"))

    # weighted blend of prompts
    # ("promptA", "promptB").blend(a, b) where "promptA" and "promptB" are valid prompts and a and b are float or
    # int weights.
    # can specify more terms eg ("promptA", "promptB", "promptC").blend(a,b,c)

    def make_prompt_from_quoted_string(x):
        #print(' got quoted prompt', x)

        x_unquoted = x[0][1:-1]
        if len(x_unquoted.strip()) == 0:
            # print(' b : just an empty string')
            return Prompt([Fragment('')])
        #print(f' b parsing \'{x_unquoted}\'')
        x_parsed = prompt.parse_string(x_unquoted)
        #print(" quoted prompt was parsed to", type(x_parsed),":", x_parsed)
        return x_parsed[0]

    quoted_prompt = pp.dbl_quoted_string.set_parse_action(make_prompt_from_quoted_string)
    quoted_prompt.set_name('quoted_prompt')

    debug_blend=False
    blend_terms = pp.delimited_list(quoted_prompt).set_name('blend_terms').set_debug(debug_blend)
    blend_weights = (pp.delimited_list(number) + pp.Optional(pp.Char(",").suppress() + "no_normalize")).set_name('blend_weights').set_debug(debug_blend)
    blend = pp.Group(lparen + pp.Group(blend_terms) + rparen
                     + pp.Literal(".blend").suppress()
                     + lparen + pp.Group(blend_weights) + rparen).set_name('blend')
    blend.set_debug(debug_blend)

    def make_blend(x):
        prompts = x[0][0]
        weights = x[0][1]
        normalize = True
        if weights[-1] == 'no_normalize':
            normalize = False
            weights = weights[:-1]
        return Blend(prompts=prompts, weights=weights, normalize_weights=normalize)

    blend.set_parse_action(make_blend)

    conjunction_terms = blend_terms.copy().set_name('conjunction_terms')
    conjunction_weights = blend_weights.copy().set_name('conjunction_weights')
    conjunction_with_parens_and_quotes = pp.Group(lparen + pp.Group(conjunction_terms) + rparen
                     + pp.Literal(".and").suppress()
                     + lparen + pp.Optional(pp.Group(conjunction_weights)) + rparen).set_name('conjunction')
    def make_conjunction(x):
        parts_raw = x[0][0]
        weights = x[0][1] if len(x[0])>1 else [1.0]*len(parts_raw)
        parts = [part for part in parts_raw]
        return Conjunction(parts, weights)
    conjunction_with_parens_and_quotes.set_parse_action(make_conjunction)

    implicit_conjunction = pp.OneOrMore(blend | prompt).set_name('implicit_conjunction')
    implicit_conjunction.set_parse_action(lambda x: Conjunction(x))

    conjunction = conjunction_with_parens_and_quotes | implicit_conjunction
    conjunction.set_debug(False)

    # top-level is a conjunction of one or more blends or prompts
    return conjunction, prompt

