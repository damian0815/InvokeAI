import unittest

from ldm.invoke.prompt_parser import parse_prompt, Fragment, Attention, Blend, CFGScale, Conjunction, Prompt

empty_conditioning = [('', 1)]

class PromptParserTestCase(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(Conjunction([('', 1)]), parse_prompt(''))

    def test_basic(self):
        self.assertEqual(Conjunction([[('fire (flames)', 1)]]), parse_prompt("fire (flames)"))
        self.assertEqual(Conjunction([[("fire flames", 1)]]), parse_prompt("fire flames"))
        self.assertEqual(Conjunction([[("fire, flames", 1)]]), parse_prompt("fire, flames"))
        self.assertEqual(Conjunction([[("fire, flames , fire", 1)]]), parse_prompt("fire, flames , fire"))

    def test_attention(self):
        self.assertEqual(Conjunction([[('flames', 0.5)]]), parse_prompt("0.5(flames)"))
        self.assertEqual(Conjunction([[('fire flames', 0.5)]]), parse_prompt("0.5(fire flames)"))
        self.assertEqual(Conjunction([[('flames', 1.1)]]), parse_prompt("+(flames)"))
        self.assertEqual(Conjunction([[('flames', 0.9)]]), parse_prompt("-(flames)"))
        self.assertEqual(Conjunction([[('fire', 1), ('flames', 0.5)]]), parse_prompt("fire 0.5(flames)"))
        self.assertEqual(Conjunction([[('flames', pow(1.1, 2))]]), parse_prompt("++(flames)"))
        self.assertEqual(Conjunction([[('flames', pow(0.9, 2))]]), parse_prompt("--(flames)"))
        self.assertEqual(Conjunction([[('flowers', pow(0.9, 3)), ('flames', pow(1.1, 3))]]), parse_prompt("---(flowers) +++flames"))
        self.assertEqual(Conjunction([[('flowers', pow(0.9, 3)), ('flames', pow(1.1, 3))]]), parse_prompt("---(flowers) +++flames"))
        self.assertEqual(Conjunction([[('flowers', pow(0.9, 3)), ('flames+', pow(1.1, 3))]]),
                         parse_prompt("---(flowers) +++flames+"))
        self.assertEqual(Conjunction([[('pretty flowers', 1.1)]]),
                         parse_prompt("+(pretty flowers)"))
        self.assertEqual(Conjunction([[('pretty flowers', 1.1), (', the flames are too hot', 1)]]),
                         parse_prompt("+(pretty flowers), the flames are too hot"))

        #self.assertEqual(pp.parse(prompt))
        #self.assertEqual(True, False)  # add assertion here



    def test_blend(self):
        self.assertEqual(Conjunction(
                            [Blend([[('fire', 1.0)], [('fire flames', 1.0)]], [0.7, 0.3])]),
                            parse_prompt("(\"fire\", \"fire flames\").blend(0.7, 0.3)")
        )
        self.assertEqual(Conjunction(
                            [Blend([[('fire', 1.0)], [('fire flames', 1.0)], [('hi', 1.0)]], [0.7, 0.3, 1.0])]),
                            parse_prompt("(\"fire\", \"fire flames\", \"hi\").blend(0.7, 0.3, 1.0)")
        )
        self.assertEqual(Conjunction(
                            [Blend([[('fire', 1.0)], [('fire flames', 1.0), ('hot', pow(1.1, 2))], [('hi', 1.0)]], [0.7, 0.3, 1.0])]),
                            parse_prompt("(\"fire\", \"fire flames ++(hot)\", \"hi\").blend(0.7, 0.3, 1.0)")
        )
        # blend a single entry is not a failure
        self.assertEqual(Conjunction(
                            [Blend([[('fire', 1.0)]], [0.7])]),
                            parse_prompt("(\"fire\").blend(0.7)")
        )
        # blend with empty
        self.assertEqual(Conjunction(
                            [Blend([[('fire', 1.0)], [('', 1.0)]], [0.7, 1.0])]),
                            parse_prompt("(\"fire\", \" \").blend(0.7, 1)")
        )
        self.assertEqual(Conjunction(
                            [Blend([[('fire', 1.0)], [('', 1.0)]], [0.7, 1.0])]),
                            parse_prompt("(\"fire\", \"\").blend(0.7, 1)")
        )

    def test_nested(self):
        self.assertEqual(Conjunction(
            [[('fire', 1.0), ('flames', 2.0), ('trees', 3.0)]]),
            parse_prompt('fire 2.0(flames 1.5(trees))'))
        self.assertEqual(Conjunction(
            [Blend(children=[[('fire', 1.0), ('flames', 1.2100000000000002)], [('mountain', 1.0), ('man', 2.0)]], weights=[1.0, 1.0])]),
            parse_prompt('("fire ++(flames)", "mountain 2(man)").blend(1,1)'))

if __name__ == '__main__':
    unittest.main()
