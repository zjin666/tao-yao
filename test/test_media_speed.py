import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from PIL import Image

from taoyao.app import TaoYaoBot
from taoyao.sticker_memory import StickerMemory
from taoyao.vision import IMAGE_PROMPT, MediaDescriber


def make_describer(folder: Path) -> MediaDescriber:
    describer = MediaDescriber.__new__(MediaDescriber)
    describer.api_base = "https://example.invalid/v1"
    describer.model = "m"
    describer.max_upload_mb = 6
    describer.detail = "low"
    describer.max_tokens = 100
    describer.timeout = 5
    describer._api_key = "k"
    describer._api_ready = lambda: True
    return describer


def reply(text):
    return {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}


class RetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.picture = Path(self.folder.name) / "p.jpg"
        Image.new("RGB", (32, 32), (200, 100, 100)).save(self.picture)
        self.describer = make_describer(Path(self.folder.name))

    async def call(self, side_effect):
        with patch("taoyao.vision.collect_stream", AsyncMock(side_effect=side_effect)) as stream, \
                patch("taoyao.vision.asyncio.sleep", AsyncMock()):
            try:
                return await self.describer._api_describe([self.picture], "p"), stream
            except Exception as exc:
                return exc, stream

    async def test_incomplete_stream_is_retried_once(self):
        result, stream = await self.call([ValueError("Incomplete streaming response"), reply("一只猫")])
        self.assertEqual(result, "一只猫")
        self.assertEqual(stream.await_count, 2)

    async def test_gives_up_after_one_retry(self):
        result, stream = await self.call([ValueError("Incomplete streaming response")] * 3)
        self.assertIsInstance(result, ValueError)
        self.assertEqual(stream.await_count, 2)

    async def test_server_error_is_retried_but_client_error_and_timeout_are_not(self):
        def status(code):
            request = httpx.Request("POST", "https://example.invalid")
            return httpx.HTTPStatusError("x", request=request, response=httpx.Response(code, request=request))

        result, stream = await self.call([status(503), reply("好")])
        self.assertEqual(result, "好")
        result, stream = await self.call([status(400), reply("好")])
        self.assertIsInstance(result, httpx.HTTPStatusError)
        self.assertEqual(stream.await_count, 1)
        result, stream = await self.call([httpx.ReadTimeout("slow"), reply("好")])
        self.assertIsInstance(result, httpx.ReadTimeout)
        self.assertEqual(stream.await_count, 1)

    def test_description_limit_is_160_chars(self):
        self.assertIn("160 字以内", IMAGE_PROMPT)
        self.assertNotIn("200 字", IMAGE_PROMPT)


class ParallelTests(unittest.IsolatedAsyncioTestCase):
    def make_bot(self, delay=0.3):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.sticker_memory = StickerMemory({"enabled": False})
        bot._sticker_tasks = set()
        pictures = {}
        for name in ("a", "b", "c"):
            pictures[name] = root / f"{name}.png"
            Image.new("RGB", (40, 40), (10, 20, 30)).save(pictures[name])

        async def fetch(seg):
            return pictures[seg["data"]["name"]], False

        async def describe(path, kind, context=""):
            await asyncio.sleep(delay)
            if path.stem == "b":
                raise RuntimeError("boom")
            return f"画面{path.stem}"

        bot.vision = SimpleNamespace(enabled=True, fetch_segment=fetch, describe=describe)
        bot.user_memes = SimpleNamespace(maybe_store=lambda *a, **k: False)
        bot.memes = SimpleNamespace(_keyword_match=lambda text: "")
        bot._remember_private_image = lambda *a, **k: None
        bot._describe_kept_image = lambda *a, **k: None
        return bot

    async def test_images_are_described_concurrently_in_order_and_failures_are_isolated(self):
        bot = self.make_bot()
        data = {"message_type": "group", "group_id": 1, "user_id": 2, "message": [
            {"type": "image", "data": {"name": name}} for name in ("a", "b", "c")]}
        started = time.perf_counter()
        text = await bot._media_text(data)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.8)                       # 串行至少 0.9 秒
        lines = text.split("\n")
        self.assertEqual(len(lines), 3)
        self.assertIn("画面a", lines[0])
        self.assertIn("（未识别到文字）", lines[1])           # 识别出错的那张只影响自己
        self.assertIn("画面c", lines[2])


if __name__ == "__main__":
    unittest.main()
