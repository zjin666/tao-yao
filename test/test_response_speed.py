import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from taoyao.app import TaoYaoBot
from taoyao.llm import DeepSeekLLM


class ResponseSpeedTests(unittest.IsolatedAsyncioTestCase):
    async def test_small_decisions_use_utility_model_without_changing_body_model(self):
        bot = object.__new__(TaoYaoBot)
        bot.config = {'search': {'enabled': True, 'decision_llm': True}}
        bot.llm = SimpleNamespace(chat=AsyncMock())
        bot.utility_llm = SimpleNamespace(chat=AsyncMock(return_value='{"search":false}'))
        self.assertEqual(await bot._search_query_for('我刚忙完想喝杯茶'), '')
        bot.utility_llm.chat.assert_awaited_once()
        bot.llm.chat.assert_not_awaited()
        bot.utility_llm.chat.return_value = '{"allowed":true}'
        self.assertEqual(await bot._review_reply_boundary('你这脑洞真会拐弯。', [], {'identity': 'user'}), ('', ''))
        self.assertEqual(bot.utility_llm.chat.await_count, 2)
        bot.llm.chat.assert_not_awaited()

    async def test_thinking_and_token_limits_are_scoped_to_selected_client(self):
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / 'key'
            key.write_text('test-only', encoding='utf-8')
            for fast in (False, True):
                config = {'key_file': str(key), 'model': 'deepseek-chat'}
                if fast:
                    config.update(model='deepseek-flash', max_tokens=220, thinking=False)
                client = DeepSeekLLM(config)
                await client._client.aclose()
                requests = []
                async def handler(request):
                    body = json.loads(request.content)
                    requests.append(body)
                    if body['stream']:
                        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"好"}}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
                    return httpx.Response(200, json={'choices': [{'message': {'content': '好'}}]})
                client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
                self.assertEqual(await client.chat([]), '好')
                self.assertEqual(''.join([x async for x in client.stream_chat([])]), '好')
                for body in requests:
                    if fast:
                        self.assertEqual(body['model'], 'deepseek-flash')
                        self.assertEqual(body['max_tokens'], 220)
                        self.assertEqual(body['thinking'], {'type': 'disabled'})
                    else:
                        self.assertEqual(body['model'], 'deepseek-chat')
                        self.assertNotIn('thinking', body)
                await client.close()
