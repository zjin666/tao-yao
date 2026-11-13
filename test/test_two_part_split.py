import unittest

from taoyao.two_part import split_approved_bubbles, split_clauses


class ClauseSplitTests(unittest.TestCase):
    def test_short_sentence_is_untouched(self):
        self.assertEqual(split_clauses("反正我现在闲着也是闲着，就勉强听听你的废话好了。"), ["反正我现在闲着也是闲着，就勉强听听你的废话好了。"])

    def test_long_sentence_is_cut_at_commas_into_one_idea_each(self):
        text = "既然要聊天，那我就勉为其难当个陪聊员好了，不过得先说好，话题不许绕回数学或者代码那些让人头秃的东西。"
        parts = split_clauses(text)
        self.assertGreaterEqual(len(parts), 2)
        self.assertTrue(all(len(p) <= 32 for p in parts))
        # 文字本身不变：只是在逗号处断开
        self.assertEqual("".join(parts).replace("，", ""), text.replace("，", ""))

    def test_never_cuts_inside_quotes_or_brackets(self):
        text = "哥哥这求生欲倒是强，看来是真把“禁止工作，包括想题目”刻进DNA里了呀，还一脸无辜的样子。"
        for part in split_clauses(text):
            self.assertEqual(part.count("“"), part.count("”"))

    def test_tiny_fragments_join_a_neighbour(self):
        parts = split_clauses("哼，我才不信你说的这些话呢，都是借口，你自己心里清楚得很，别想糊弄我。")
        self.assertTrue(all(len(p) >= 7 for p in parts))

    def test_tail_bubbles_split_further_and_overflow_only_merges_when_over_the_limit(self):
        tail = "刚才谁信誓旦旦说要给我惊喜的，现在倒嫌我严格啦。我对哥哥当然温柔呀，比如现在就没有把你那些贝果扔掉，这已经是最大的慈悲了嘛。少在那儿装可怜，把心思都花在让那条小金鱼变得闪闪发光上，到时候要是做得丑，我可是会连人带伞一起嫌弃的哦。"
        loose = split_approved_bubbles(tail, 8)
        tight = split_approved_bubbles(tail, 3)
        self.assertGreater(len(loose), len(tight))
        self.assertEqual(len(tight), 3)
        self.assertLessEqual(max(len(p) for p in loose), 40)           # 上限够时没有一个气泡挤了几层意思
        self.assertGreater(max(len(p) for p in tight), max(len(p) for p in loose))   # 超限才合并到最后一个

    def test_no_terminal_punctuation_and_empty_input(self):
        self.assertEqual(split_approved_bubbles("", 4), [])
        self.assertEqual(split_approved_bubbles("嗯嗯好的", 4), ["嗯嗯好的"])


if __name__ == "__main__":
    unittest.main()
