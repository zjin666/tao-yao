import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock
from taoyao.app import TaoYaoBot
from taoyao.meme_tail import MemeTail


class MemeTailTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_model_emotion_wins_over_negated_keyword(self):
        bot=TaoYaoBot.__new__(TaoYaoBot)
        bot.config={'memes':{'enabled':True}}
        bot.memes=SimpleNamespace(categories={'happy':'开心','angry':'生气'},
            match_candidates=Mock(return_value=([('angry',.9)],('angry',.9))),
            _last_sentence=lambda text:text,infer_intensity=lambda text:.5,
            pick=Mock(return_value=Path('synthetic.png')))
        bot.user_memes=SimpleNamespace(use_probability=0)
        bot._meme_probability=lambda role:1
        bot._send_image=AsyncMock()
        self.assertTrue(await bot._maybe_send_meme('private',1,'我没生气，是真的替你高兴。','owner',preferred='happy'))
        self.assertEqual(bot.memes.pick.call_args.args[0],'happy')

    async def test_every_chunk_boundary_hides_metadata(self):
        raw='我没生气。是真的替你高兴。\n[[meme:happy]]'
        for split in range(len(raw)+1):
            async def source():
                yield raw[:split]
                yield raw[split:]
            tail=MemeTail()
            text=''.join([part async for part in tail.stream(source())])
            self.assertEqual(text,'我没生气。是真的替你高兴。\n')
            self.assertEqual(tail.category,'happy')

    async def test_first_prose_does_not_wait_for_tag(self):
        release=asyncio.Event()
        async def source():
            yield '好。'
            await release.wait()
            yield '[[meme:space out]]'
        tail=MemeTail();stream=tail.stream(source())
        self.assertEqual(await asyncio.wait_for(anext(stream),.5),'好。')
        self.assertEqual(tail.category,'')
        release.set()
        self.assertEqual([part async for part in stream],[])
        self.assertEqual(tail.category,'space out')

    async def test_bad_or_truncated_metadata_never_becomes_chat(self):
        for suffix in ('[[me','[[meme:shy','[[meme:shy]]多余内容','[[meme:bad\nvalue]]'):
            async def source():yield '好。'+suffix
            tail=MemeTail()
            self.assertEqual(''.join([s async for s in tail.stream(source())]),'好。')
            self.assertEqual(tail.category,'')

    async def test_normal_brackets_are_preserved(self):
        async def source():
            for c in '看看[这个]，还挺好。':yield c
        tail=MemeTail()
        self.assertEqual(''.join([s async for s in tail.stream(source())]),'看看[这个]，还挺好。')
