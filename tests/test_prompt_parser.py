import unittest

from ldm.dream.prompt_parser import parse_prompt, Word, Attention, Blend


empty_conditioning = [('', 1)]

class PromptParserTestCase(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(parse_prompt(''), [])

    def test_basic(self):
        self.assertEqual(parse_prompt("fire (flames)"), [Word("fire"), Word("(flames)")])
        self.assertEqual(parse_prompt("fire flames"), [Word("fire"), Word("flames")])
        self.assertEqual(parse_prompt("fire, flames"), [Word("fire,"), Word("flames")])
        self.assertEqual(parse_prompt("fire, flames , fire"), [Word("fire,"), Word("flames"), Word(","), Word("fire")])

    def test_attention(self):
        self.assertEqual(parse_prompt("fire 0.5(flames)"), [Word("fire"), Attention(0.5, "flames")])
        self.assertEqual(parse_prompt("0.5(flames)"), [Attention(0.5, "flames")])
        self.assertEqual(parse_prompt("+(flames)"), [Attention(1.1, "flames")])
        self.assertEqual(parse_prompt("-(flames)"), [Attention(0.9, "flames")])
        self.assertEqual(parse_prompt("++(flames)"), [Attention(pow(1.1,2), "flames")])
        self.assertEqual(parse_prompt("--(flames)"), [Attention(pow(0.9,2), "flames")])
        self.assertEqual(parse_prompt("---(flowers) +++flames"), [Attention(pow(0.9,3), "flowers"), Attention(pow(1.1,3), "flames")])
        self.assertEqual(parse_prompt("---flowers +++flames"), [Attention(pow(0.9,3), "flowers"), Attention(pow(1.1,3), "flames")])
        self.assertEqual(parse_prompt("---flowers +++flames+"), [Attention(pow(0.9,3), "flowers"), Attention(pow(1.1,3), "flames+")])

        #self.assertEqual(pp.parse(prompt))
        #self.assertEqual(True, False)  # add assertion here

    def test_attention_sml(self):
        self.assertEqual(parse_prompt("---(flowers) +++flames"), [Attention(pow(0.9,3), "flowers"), Attention(pow(1.1,3), "flames")])


    def test_blend(self):
        self.assertEqual(parse_prompt("(\"fire\", \"fire flames\", \"hi\").blend(0.7, 0.3, 1.0)"),
                         [Blend([[Word("fire")], [Word("fire"), Word("flames")], [Word("hi")]], [0.7, 0.3, 1.0])])

    def test_nested(self):
        self.assertEqual(parse_prompt('("fire ++(flames)", "mountain 2(man)").blend(1,1)'),
                        [Blend([[Word("fire"), Attention(pow(1.1,2), "flames")], [Word("mountain"), Attention(2.0, "man")]], [1.0,1.0])])

if __name__ == '__main__':
    unittest.main()
