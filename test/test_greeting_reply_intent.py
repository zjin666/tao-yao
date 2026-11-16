"""用户向桃夭道早安/晚安时，回复要带一句问候；与定时问候是否跳过无关。"""
import unittest

from taoyao.app import TaoYaoBot
from taoyao.persona import OWNER_RULES, build_system_prompt


class GreetingReplyIntentTests(unittest.TestCase):
    def setUp(self):
        self.bot = TaoYaoBot.__new__(TaoYaoBot)

    def test_greetings_ask_for_a_greeting_back(self):
        for text, word in (("妹妹早安", "早安"), ("早", "早安"), ("早啊", "早安"), ("Good morning", "早安"),
                           ("晚安", "晚安"), ("我先睡了", "晚安"), ("那我去睡啦，晚安哈", "晚安")):
            intent = self.bot._greeting_reply_intent(text)
            self.assertIn(f"不必一定回{word}", intent, text)
            self.assertIn("可以自然地带上一句", intent, text)

    def test_non_greetings_and_ambient_do_not(self):
        for text in ("你睡了吗", "怎么还不晚安？", "我昨晚没说晚安就睡了", "早点睡对身体好" * 6, "", "帮我查天气"):
            self.assertEqual(self.bot._greeting_reply_intent(text), "", text)
        self.assertEqual(self.bot._greeting_reply_intent("早安", ambient=True), "")

    def test_every_private_viewer_gets_the_care_rule_but_groups_do_not(self):
        for role, rel in (("owner", None), ("user", {"tier_key": "stranger"}),
                          ("user", {"tier_key": "acquaintance"}), ("user", {"tier_key": "friend"})):
            private = build_system_prompt("人设", [], role=role, relationship=rel)
            self.assertIn("对方自己道早安、晚安时可以回早安、晚安，不受此限", private, (role, rel))
            group = build_system_prompt("人设", [], role=role, is_group=True)
            self.assertNotIn("对方自己道早安、晚安时可以回早安", group, role)

    def test_owner_rules_no_longer_hold_the_care_sentence(self):
        self.assertNotIn("关心要有变化", OWNER_RULES)


if __name__ == "__main__":
    unittest.main()
