import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import soundfile as sf

from taoyao.app import TaoYaoBot
from taoyao.voice import VoiceResult, VoiceTranscriber

MODEL = Path(__file__).resolve().parents[1] / "models" / "sensevoice"
SAMPLE = MODEL / "test_wavs" / "zh.wav"
import importlib.util

needs_model = unittest.skipUnless(
    (MODEL / "model.int8.onnx").exists() and SAMPLE.exists() and importlib.util.find_spec("sherpa_onnx") is not None,
    "没有本地语音模型或 sherpa-onnx")


def fake_transcriber(**result):
    transcriber = SimpleNamespace(
        ready=True, work_dir=Path(tempfile.gettempdir()),
        transcribe=AsyncMock(return_value=VoiceResult(**result)))
    return transcriber


class VoiceTranscriberTests(unittest.IsolatedAsyncioTestCase):
    @needs_model
    async def test_chinese_speech_is_transcribed_locally(self):
        voice = VoiceTranscriber({"model_dir": str(MODEL)})
        result = await voice.transcribe(SAMPLE)
        self.assertEqual(result.status, "ok")
        self.assertIn("9点", result.text)
        self.assertGreater(result.seconds, 3)

    @needs_model
    async def test_too_long_audio_is_not_transcribed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "long.wav"
            sf.write(str(path), np.zeros(16000 * 8, dtype="float32"), 16000)
            voice = VoiceTranscriber({"model_dir": str(MODEL), "max_seconds": 5})
            self.assertEqual((await voice.transcribe(path)).status, "too_long")

    @needs_model
    async def test_silence_gives_empty_not_garbage(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "silence.wav"
            sf.write(str(path), np.zeros(16000 * 2, dtype="float32"), 16000)
            voice = VoiceTranscriber({"model_dir": str(MODEL)})
            self.assertIn((await voice.transcribe(path)).status, ("empty", "ok"))

    async def test_missing_model_or_disabled_never_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(VoiceTranscriber({"model_dir": folder}).ready)
            self.assertEqual((await VoiceTranscriber({"model_dir": folder}).transcribe(Path("x.wav"))).status, "error")
            self.assertFalse(VoiceTranscriber({"enabled": False, "model_dir": str(MODEL)}).ready)

    async def test_unreadable_audio_returns_error(self):
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder) / "bad.silk"
            bad.write_bytes(b"not audio")
            voice = VoiceTranscriber({"model_dir": folder})
            voice.ready  # 模型不在
            self.assertEqual((await voice.transcribe(bad)).status, "error")


class VoiceMessageFlowTests(unittest.IsolatedAsyncioTestCase):
    def make_bot(self, **result):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.vision = SimpleNamespace(enabled=True)
        bot.voice = fake_transcriber(**result)
        bot.server = SimpleNamespace(call=AsyncMock(side_effect=RuntimeError("no api")))
        return bot

    async def test_local_file_is_transcribed_into_marked_text(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "a.wav"
            audio.write_bytes(b"x")
            bot = self.make_bot(status="ok", text="我今天作业写完了", seconds=2.0)
            data = {"message_type": "private", "user_id": 1,
                    "message": [{"type": "record", "data": {"file": str(audio)}}]}
            text = await bot._media_text(data)
            self.assertEqual(text, "[语音] 我今天作业写完了")
            bot.server.call.assert_not_awaited()              # 本地文件够用就不去问 OneBot

    async def test_falls_back_to_get_record_when_local_file_is_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "converted.wav"
            audio.write_bytes(b"x")
            bot = self.make_bot(status="ok", text="你好呀", seconds=1.0)
            bot.server.call = AsyncMock(return_value={"status": "ok", "data": {"file": str(audio)}})
            data = {"message_type": "private", "user_id": 1,
                    "message": [{"type": "record", "data": {"file": "ABC123.silk"}}]}
            self.assertEqual(await bot._media_text(data), "[语音] 你好呀")
            args = bot.server.call.await_args.args
            self.assertEqual(args[0], "get_record")
            self.assertEqual(args[1]["out_format"], "wav")

    async def test_failure_statuses_become_short_neutral_notes(self):
        for status, expect in (("too_long", "太长"), ("empty", "没有听到")):
            with tempfile.TemporaryDirectory() as folder:
                audio = Path(folder) / "a.wav"
                audio.write_bytes(b"x")
                bot = self.make_bot(status=status)
                data = {"message_type": "private", "user_id": 1,
                        "message": [{"type": "record", "data": {"path": str(audio)}}]}
                text = await bot._media_text(data)
                self.assertTrue(text.startswith("[语音]"))
                self.assertIn(expect, text)

    async def test_nothing_obtainable_gives_could_not_transcribe(self):
        bot = self.make_bot(status="error")
        data = {"message_type": "private", "user_id": 1,
                "message": [{"type": "record", "data": {"file": "gone.silk"}}]}
        self.assertEqual(await bot._media_text(data), "[语音]（没能转成文字）")

    async def test_voice_works_even_when_vision_is_off_and_other_media_is_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "a.wav"
            audio.write_bytes(b"x")
            bot = self.make_bot(status="ok", text="在吗", seconds=1.0)
            bot.vision = SimpleNamespace(enabled=False)
            data = {"message_type": "private", "user_id": 1, "message": [
                {"type": "image", "data": {}}, {"type": "record", "data": {"path": str(audio)}}]}
            self.assertEqual(await bot._media_text(data), "[语音] 在吗")

    def group_bot(self, **result):
        bot = self.make_bot(**result)
        bot.config = {"bot_name": "桃夭"}
        bot._roles = lambda: {"allow_all_groups": False}
        bot._allowed_groups = lambda: {100}
        return bot

    async def test_group_voice_that_says_the_bot_name_counts_as_calling_her(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "a.wav"
            audio.write_bytes(b"x")
            bot = self.group_bot(status="ok", text="桃夭你今天在干嘛", seconds=2.0)
            data = {"message_type": "group", "group_id": 100, "user_id": 5, "self_id": 9, "raw_message": "[CQ:record,file=a]",
                    "message": [{"type": "record", "data": {"path": str(audio)}}]}
            self.assertFalse(bot._is_at_bot(data))
            await bot._prepare_group_voice(data, 100)
            self.assertTrue(bot._is_at_bot(data))           # 说出口的名字被认出来了
            self.assertEqual(await bot._media_text(data), "[语音] 桃夭你今天在干嘛")
            self.assertEqual(bot.voice.transcribe.await_count, 1)   # 同一段语音只转一次

    async def test_group_voice_without_the_name_does_not_call_her(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "a.wav"
            audio.write_bytes(b"x")
            bot = self.group_bot(status="ok", text="今晚一起打游戏吗", seconds=2.0)
            data = {"message_type": "group", "group_id": 100, "user_id": 5, "self_id": 9, "raw_message": "[CQ:record,file=a]",
                    "message": [{"type": "record", "data": {"path": str(audio)}}]}
            await bot._prepare_group_voice(data, 100)
            self.assertFalse(bot._is_at_bot(data))

    async def test_voice_in_unregistered_group_is_not_transcribed(self):
        bot = self.group_bot(status="ok", text="桃夭", seconds=1.0)
        data = {"message_type": "group", "group_id": 999, "user_id": 5, "self_id": 9, "raw_message": "",
                "message": [{"type": "record", "data": {"path": "x"}}]}
        await bot._prepare_group_voice(data, 999)
        bot.voice.transcribe.assert_not_awaited()

    def test_group_voice_is_treated_like_other_media_for_group_context(self):
        self.assertTrue(TaoYaoBot._group_media_relevant({"message": [{"type": "record", "data": {}}]}, ""))
        self.assertFalse(TaoYaoBot._group_media_relevant({"message": [{"type": "text", "data": {}}]}, ""))

    async def test_voice_keeps_order_with_images_and_text_in_private_chat(self):
        from taoyao.image_order import private_image_order
        started = []

        class Probe:
            @private_image_order
            async def handle(self, data):
                started.append(data["tag"])
                await asyncio.sleep(0.05)

        probe = Probe()
        first = asyncio.create_task(probe.handle({"post_type": "message", "message_type": "private",
            "user_id": 1, "self_id": 9, "tag": "voice", "message": [{"type": "record", "data": {}}]}))
        await asyncio.sleep(0)
        second = asyncio.create_task(probe.handle({"post_type": "message", "message_type": "private",
            "user_id": 1, "self_id": 9, "tag": "text", "message": [{"type": "text", "data": {"text": "好"}}]}))
        await asyncio.gather(first, second)
        self.assertEqual(started, ["voice", "text"])


if __name__ == "__main__":
    unittest.main()
