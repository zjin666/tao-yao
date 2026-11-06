import json
import tempfile
from pathlib import Path
import unittest
import httpx
from taoyao.llm import create_llm, QwenLLM, DeepSeekLLM, LLMError


class LLMProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_qwen_chat_stream_and_deepseek_explicit_override(self):
        with tempfile.TemporaryDirectory() as folder:
            key = Path(folder)/'key.txt'
            key.write_text('synthetic-test-key')
            config = {'text_provider': 'qwen', 'qwen': {'key_file': str(key), 'thinking': False},
                      'deepseek': {'key_file': str(key), 'thinking': False}}
            qwen, deepseek = create_llm(config), create_llm(config, provider='deepseek')
            self.assertIsInstance(qwen, QwenLLM)
            self.assertIsInstance(deepseek, DeepSeekLLM)
            observed = []
            def transport(request):
                data = json.loads(request.content)
                observed.append(data)
                if data['stream']:
                    body = 'data: '+json.dumps({'choices':[{'delta':{'reasoning_content':'PRIVATE'}}]})+'\n\n'
                    body += 'data: '+json.dumps({'choices':[{'delta':{'content':'你好'}}]})+'\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
                    return httpx.Response(200, text=body, headers={'content-type':'text/event-stream'})
                return httpx.Response(200, json={'choices':[{'message':{'content':'你好','reasoning_content':'PRIVATE'}}]})
            try:
                await qwen._client.aclose()
                qwen._client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
                self.assertEqual(await qwen.chat([{'role':'user','content':'你好'}]), '你好')
                self.assertEqual(''.join([p async for p in qwen.stream_chat([])]), '你好')
                for payload in observed:
                    self.assertEqual(payload['model'], 'qwen3.7-plus')
                    self.assertIs(payload['enable_thinking'], False)
                    self.assertNotIn('thinking', payload)
                self.assertEqual(deepseek.thinking_payload(), {'thinking':{'type':'disabled'}})
                with self.assertRaises(LLMError):
                    create_llm({**config,'text_provider':'unknown'})
            finally:
                await qwen.close()
                await deepseek.close()
