"""亲近行为的统一说明：只注入规则里出现的词，且保持简短。"""
import unittest

from taoyao.behavior_terms import BEHAVIORS, behavior_reference
from taoyao.persona import build_system_prompt
from taoyao.reply_flow import explicit_review_policy


class BehaviorTermsTests(unittest.TestCase):
    def test_every_definition_has_form_and_limit_and_is_short(self):
        for name, (what, limit) in BEHAVIORS.items():
            self.assertTrue(what and limit, name)
            self.assertLessEqual(len(what) + len(limit), 60, name)

    def test_owner_private_prompt_defines_only_the_terms_its_rules_use(self):
        prompt = build_system_prompt("人设", [], role="owner")
        self.assertIn("【行为参考】", prompt)
        for name in ("撒娇", "依赖", "捉弄", "唱反调", "关心陪伴", "偏袒护短"):
            self.assertIn(name + "：", prompt, name)
        self.assertNotIn("装熟：", prompt)

    def test_stranger_prompt_defines_what_it_forbids(self):
        prompt = build_system_prompt("人设", [], role="user", relationship={"tier_key": "stranger"})
        self.assertIn("撒娇：", prompt)
        self.assertIn("装熟：", prompt)
        self.assertIn("暧昧：", prompt)

    def test_whole_reference_block_stays_small(self):
        self.assertLessEqual(len(behavior_reference("撒娇依赖捉弄唱反调关心陪伴偏袒护短装熟暧昧")), 700)

    def test_review_policy_uses_the_same_definition(self):
        policy = explicit_review_policy("user", True, "stranger")
        self.assertTrue(any("软着语气" in item for item in policy["forbidden_intimacy"]))

    def test_romance_definition_does_not_forbid_sibling_closeness_for_everyone(self):
        what, limit = BEHAVIORS["暧昧"]
        self.assertNotIn("任何对象", limit)
        self.assertIn("兄妹亲近", limit)

    def test_no_terms_means_no_block(self):
        self.assertEqual(behavior_reference("称呼对方名字"), "")


if __name__ == "__main__":
    unittest.main()
