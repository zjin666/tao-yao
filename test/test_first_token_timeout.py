"""流式生成：一个字都没来就尽早放弃（交给调用方重试），不再干等整体超时。"""
import asyncio
import tempfile
import unittest
from pathlib import Path

import httpx

from taoyao.llm import LLMError, QwenLLM


def make(tmp, **extra):
    key = Path(tmp) / "key.txt"
    key.write_text("k", encoding="utf-8")
    return QwenLLM({"key_file": str(key), "timeout": 60, **extra})


class FakeResponse:
    status_code = 200

    async def aiter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"你好"}}]}'
        yield "data: [DONE]"

    async def aread(self):
        return b""


class FakeStream:
    def __init__(self, captured, error=None):
        self.captured, self.error = captured, error

    def __call__(self, *args, **kwargs):
        self.captured.update(kwargs)
        return self

    async def __aenter__(self):
        if self.error:
            raise self.error
        return FakeResponse()

    async def __aexit__(self, *exc):
        return False


class FirstTokenTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_requests_use_a_short_read_timeout_so_a_silent_stream_is_dropped_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = make(tmp)
            captured = {}
            llm._client.stream = FakeStream(captured)
            chunks = [part async for part in llm.stream_chat([{"role": "user", "content": "hi"}])]
            self.assertEqual("".join(chunks), "你好")
            timeout = captured["timeout"]
            self.assertEqual(timeout.read, 20)              # 默认 20 秒没有数据就放弃
            self.assertEqual(timeout.connect, 60)           # 其余超时不变
            await llm.close()

    async def test_the_limit_is_configurable_and_never_exceeds_the_overall_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(make(tmp, first_token_timeout=12).first_token_timeout, 12)
            self.assertEqual(make(tmp, timeout=15, first_token_timeout=40).first_token_timeout, 15)

    async def test_a_silent_stream_raises_a_clear_error_for_the_caller_to_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = make(tmp)
            llm._client.stream = FakeStream({}, error=httpx.ReadTimeout(""))
            with self.assertRaises(LLMError) as caught:
                async for _ in llm.stream_chat([{"role": "user", "content": "hi"}]):
                    pass
            self.assertIn("20 秒没有返回内容", str(caught.exception))
            await llm.close()


if __name__ == "__main__":
    unittest.main()
