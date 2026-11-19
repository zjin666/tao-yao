"""整理摘要要用输出上限够大的模型；判断型辅助模型只给 220 token，会把摘要截断成“返回不完整”。"""
import unittest
from types import SimpleNamespace

from taoyao.app import TaoYaoBot


class SummaryModelTests(unittest.TestCase):
    def test_prefers_the_dedicated_summary_model(self):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.llm, bot.utility_llm, bot.summary_llm = (SimpleNamespace(n=n) for n in ("qwen", "helper", "summary"))
        self.assertEqual(bot._summary_model().n, "summary")

    def test_falls_back_to_the_helper_model_not_qwen(self):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.llm, bot.utility_llm = SimpleNamespace(n="qwen"), SimpleNamespace(n="helper")
        self.assertEqual(bot._summary_model().n, "helper")


if __name__ == "__main__":
    unittest.main()
