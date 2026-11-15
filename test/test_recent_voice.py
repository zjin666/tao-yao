"""近期口吻提示：开头词扎堆、一直只回一句、原句重复时，提醒桃夭这次换一换。"""
import unittest

from taoyao.persona import recent_voice_note


def turns(*replies):
    out = []
    for text in replies:
        out.append({"role": "user", "content": "在吗"})
        out.append({"role": "assistant", "content": text})
    return out


class RecentVoiceTests(unittest.TestCase):
    def test_too_little_history_gives_no_note(self):
        self.assertEqual(recent_voice_note(turns("哼，来啦。", "好呀。")), "")

    def test_crowded_openers_are_named_and_alternatives_offered(self):
        note = recent_voice_note(turns("哼，大忙人终于想起我啦。让我看看。", "哼，才没有生气呢。真的没有。", "哼，谁稀罕你陪呀。走开啦。"))
        self.assertIn("近期开头已多次用“哼”", note)
        self.assertIn("呀、嘛、啦", note)

    def test_always_one_sentence_asks_for_a_bit_more(self):
        note = recent_voice_note(turns("来啦。", "嗯嗯。", "好哦。"))
        self.assertIn("只回了一句", note)
        self.assertNotIn("只回了一句", recent_voice_note(turns("来啦。顺便问一句。", "嗯嗯。我在听。", "好哦。")))

    def test_repeated_sentences_are_listed_to_avoid(self):
        note = recent_voice_note(turns("一大早就来戳我呆毛。", "一大早就来戳我呆毛。", "我在看书呢。"))
        self.assertIn("已重复说过，别原样或换个说法再说一遍", note)
        self.assertEqual(note.count("一大早就来戳我呆毛"), 1)             # 重复的只列一次

    def test_healthy_variety_adds_nothing(self):
        self.assertEqual(recent_voice_note(turns("诶，真的吗。我不信。", "嗯哼，你继续说。", "好呀。那我陪你。")), "")

    def test_note_is_not_leaked_when_the_model_copies_the_section_title(self):
        from taoyao.output_guard import guard_outgoing
        self.assertEqual(guard_outgoing("【近期口吻】\n好的。"), "")


if __name__ == "__main__":
    unittest.main()
