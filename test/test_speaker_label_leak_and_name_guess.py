"""群聊历史里的发言者标注不能抄进回复；认不出图里的角色时不乱猜名字。"""
import unittest

from taoyao.output_guard import contains_internal_notice, guard_outgoing
from taoyao.persona import build_system_prompt
from taoyao.vision import IMAGE_PROMPT


class SpeakerLabelLeakTests(unittest.TestCase):
    def test_copied_speaker_labels_and_unknown_code_like_tags_are_blocked(self):
        for text in ("[user] 蓝色大肥鱼守夜", "【user] 蓝色大肥鱼守夜", "哼\n[owner] 哥哥", "守夜#8552：你好",
                     "[system] x", "[id:12]", "【note】", "我觉得 [user] 这个词没什么", "说了[OK]吗"):
            self.assertTrue(contains_internal_notice(text), text)
            self.assertEqual(guard_outgoing(text), "", text)

    def test_our_own_tags_and_ordinary_bracketed_chat_words_pass(self):
        for text in ("明天#3号楼见", "第#1234名", "参考[1]和[2]", "哈哈[笑]", "【标题】开始", "（笑）", "正常的一句话",
                     "[[at]] 好", "好[[meme:teasing]]"):
            self.assertFalse(contains_internal_notice(text), text)
            self.assertEqual(guard_outgoing(text), text)


class PromptTests(unittest.TestCase):
    def test_group_prompt_says_speaker_labels_are_not_written_into_replies(self):
        prompt = build_system_prompt("人设", [], role="owner", is_group=True)
        self.assertIn("只是标出发言者，不写进回复", prompt)

    def test_prompt_tells_her_not_to_guess_character_names(self):
        prompt = build_system_prompt("人设", [], role="owner")
        self.assertIn("认不出就直说认不出", prompt)
        self.assertIn("不把猜测说成“太眼熟了”", prompt)

    def test_image_prompt_asks_for_identifying_details_but_not_a_guessed_name(self):
        for word in ("发色（含渐变）", "头饰和配饰的真实形状", "窗口标题栏或软件名", "不要猜具体的角色名或作品名"):
            self.assertIn(word, IMAGE_PROMPT)
        self.assertIn("160 字以内", IMAGE_PROMPT)


if __name__ == "__main__":
    unittest.main()
