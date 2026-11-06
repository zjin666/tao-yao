import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.websearch import SearchResult, WebSearchClient


class SearchRoutingTests(unittest.IsolatedAsyncioTestCase):
    def make_bot(self, decision='{"search":false}'):
        bot = object.__new__(TaoYaoBot)
        bot.config = {'search': {'enabled': True, 'decision_llm': True}}
        bot.llm = SimpleNamespace(chat=AsyncMock(return_value=decision))
        bot.web_search = SimpleNamespace(search=AsyncMock(return_value=[]), format_results=WebSearchClient.format_results)
        return bot

    async def test_statement_without_question_mark_can_search_for_meme(self):
        bot = self.make_bot('{"search":true,"query":"奇变偶不变 符号看象限 梗 出处"}')
        query = await bot._search_query_for('奇变偶不变，符号看象限', 'user：接个暗号')
        self.assertIn('奇变偶不变', query)
        messages = bot.llm.chat.call_args.args[0]
        data = json.loads(messages[1]['content'])
        self.assertEqual(data['recent_context'], 'user：接个暗号')
        self.assertIn('没有问号', messages[0]['content'])

    async def test_unknown_expression_goes_to_decider_but_smalltalk_does_not(self):
        bot = self.make_bot()
        self.assertEqual(await bot._search_query_for('这是一句陌生固定说法'), '')
        bot.llm.chat.assert_awaited_once()
        bot.llm.chat.reset_mock()
        for text in ('嗯', '谢谢', '不要联网查这个梗', '/状态'):
            self.assertEqual(await bot._search_query_for(text), '')
        bot.llm.chat.assert_not_awaited()

    async def test_daily_time_words_never_trigger_retrieval(self):
        bot = self.make_bot('{"search":true,"query":"should not run"}')
        for text in ('今天好累', '我今天很开心', '最近想你', '你现在陪陪我', '现在有点委屈'):
            self.assertEqual(await bot._search_context_for(text), '', text)
        bot.llm.chat.assert_not_awaited()
        bot.web_search.search.assert_not_awaited()

    async def test_time_keywords_require_semantic_decision(self):
        bot = self.make_bot()
        for text in ('今天过得不顺利，想和你说说', '最近这个版本让我好烦', '帮我写一段今天的晚安文案'):
            self.assertEqual(await bot._search_context_for(text), '')
        self.assertEqual(bot.llm.chat.await_count, 3)
        bot.web_search.search.assert_not_awaited()
        self.assertIn('是否依赖外部知识', bot.llm.chat.call_args.args[0][0]['content'])

    async def test_factual_current_request_still_searches(self):
        bot = self.make_bot('{"search":true,"query":"北京 今天 天气"}')
        await bot._search_context_for('今天北京天气怎么样')
        bot.llm.chat.assert_awaited_once()
        bot.web_search.search.assert_awaited_once_with('北京 今天 天气')

    async def test_explicit_search_and_disabled_switch(self):
        bot = self.make_bot()
        self.assertEqual(await bot._search_query_for('帮我搜一下奇变偶不变出处'), '奇变偶不变出处')
        bot.llm.chat.assert_not_awaited()
        bot.config['search']['enabled'] = False
        self.assertEqual(await bot._search_query_for('帮我搜一下奇变偶不变出处'), '')

    async def test_capability_question_does_not_swallow_concrete_request(self):
        bot = self.make_bot('{"search":true,"query":"陌生句子 梗 含义"}')
        self.assertEqual(await bot._search_query_for('你可以联网吗？'), '')
        bot.llm.chat.assert_not_awaited()
        self.assertEqual(await bot._search_query_for('可以联网查这个梗吗'), '陌生句子 梗 含义')
        bot.llm.chat.assert_awaited_once()

    async def test_results_and_failure_status_reach_reply_context(self):
        bot = self.make_bot('{"search":true,"query":"示例语句 出处"}')
        bot.web_search.search.return_value = [SearchResult('相关语境', 'https://example.com/source', '某句的含义')]
        context = await bot._search_context_for('一段陌生说法')
        self.assertIn('https://example.com/source', context)
        bot.web_search.search.return_value = []
        context = await bot._search_context_for('一段陌生说法')
        self.assertIn('不要编造来源', context)
        self.assertIn('用户没有提问时，不主动说明不确定或没查到', context)
        bot.web_search.search.side_effect = RuntimeError('test transport failure')
        with self.assertLogs('taoyao.app', level='ERROR'):
            context = await bot._search_context_for('一段陌生说法')
        self.assertIn('搜索暂时失败', context)
        self.assertIn('用户只是说话、玩梗而没有提问时，不主动说不确定、没查到', context)


if __name__ == '__main__':
    unittest.main()
