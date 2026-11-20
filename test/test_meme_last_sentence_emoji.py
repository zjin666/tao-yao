"""桃夭自己的表情包按“最后一句话”匹配：结尾单独一行 emoji 时，用它前面真正的话。"""
import unittest

from taoyao.memes import MemeLibrary


class LastSentenceTests(unittest.TestCase):
    def test_an_emoji_only_last_line_is_skipped(self):
        text = "胰岛：求求了，别把我的加班费算进幸福里呀！\n🥲"
        self.assertEqual(MemeLibrary._last_sentence(text), "胰岛：求求了，别把我的加班费算进幸福里呀")

    def test_normal_text_is_unchanged(self):
        self.assertEqual(MemeLibrary._last_sentence("哼。随便你！"), "随便你")
        self.assertEqual(MemeLibrary._last_sentence("第一句。\n第二句"), "第二句")

    def test_only_emoji_still_returns_something(self):
        self.assertEqual(MemeLibrary._last_sentence("🥲"), "🥲")
        self.assertEqual(MemeLibrary._last_sentence(""), "")


if __name__ == "__main__":
    unittest.main()
