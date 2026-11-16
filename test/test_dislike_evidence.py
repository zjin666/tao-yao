"""dislikes 必须有用户亲口表达反感的原话；报备吃了什么不能变成“不喜欢”。"""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.personality import _dislike_is_evidenced, extract_basic_info_llm

RECENT = "我要健身的嘛，食物肯定就那几样\n鸡肉，西兰花和贝果啦，还有点橄榄油和坚果哦\n很健康的"


def llm_returning(data):
    return SimpleNamespace(chat=AsyncMock(return_value=json.dumps(data, ensure_ascii=False)))


class DislikeEvidenceTests(unittest.IsolatedAsyncioTestCase):
    def test_a_meal_report_is_not_evidence_but_an_explicit_dislike_is(self):
        for item in ("鸡肉", "西兰花", "贝果"):
            self.assertFalse(_dislike_is_evidenced(item, "很健康的", RECENT), item)
        self.assertTrue(_dislike_is_evidenced("西兰花", "我不喜欢西兰花", ""))
        self.assertTrue(_dislike_is_evidenced("香菜", "好烦", "我受不了香菜的味道"))
        self.assertFalse(_dislike_is_evidenced("香菜", "我受不了", "香菜还行"))        # 词和反感不在同一句
        self.assertFalse(_dislike_is_evidenced("", "我讨厌", ""))

    async def test_invented_dislikes_are_dropped_from_the_model_output(self):
        llm = llm_returning({"dislikes": ["鸡肉", "西兰花", "贝果", "橄榄油", "坚果"], "hobbies": ["健身"]})
        info = await extract_basic_info_llm("很健康的", llm, recent_text=RECENT)
        self.assertNotIn("dislikes", info)
        self.assertEqual(info["hobbies"], ["健身"])

    async def test_a_real_stated_dislike_is_kept(self):
        llm = llm_returning({"dislikes": ["西兰花"]})
        info = await extract_basic_info_llm("我真的不喜欢西兰花", llm, recent_text="")
        self.assertEqual(info["dislikes"], ["西兰花"])


if __name__ == "__main__":
    unittest.main()
