import asyncio,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock,patch
from taoyao.persona_reference import load_reference_description,with_reference
from taoyao.persona import build_system_prompt


class PersonaReferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_recognition_then_reuse_even_after_timestamp_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            image=Path(tmp)/'reference.png';image.write_bytes(b'image version 1')
            describe=AsyncMock(return_value='{"appearance":"银白长发，红色眼睛，桃花发饰。"}')
            first=await load_reference_description(tmp,describe)
            image.touch()
            self.assertEqual(await load_reference_description(tmp,describe),first)
            describe.assert_awaited_once()
            image.write_bytes(b'image version 2')
            await load_reference_description(tmp,describe)
            self.assertEqual(describe.await_count,2)
            for role in ('owner','user'):
                for group in (False,True):
                    prompt=build_system_prompt(with_reference('人设',first),[],role=role,is_group=group)
                    self.assertIn(first,prompt)

    async def test_failure_and_changed_image_do_not_load_stale_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            image=Path(tmp)/'reference.png';image.write_bytes(b'old')
            good=AsyncMock(return_value='{"appearance":"银白长发，桃花发饰。"}')
            await load_reference_description(tmp,good)
            previous=(Path(tmp)/'appearance.json').read_bytes()
            image.write_bytes(b'new')
            bad=AsyncMock(return_value='OCR文字，或者无效JSON')
            self.assertEqual(await load_reference_description(tmp,bad),'')
            self.assertEqual((Path(tmp)/'appearance.json').read_bytes(),previous)
            self.assertEqual(await load_reference_description(tmp,bad,allow_recognition=False),'')
            self.assertEqual(bad.await_count,1)

    async def test_replacement_during_recognition_is_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            image=Path(tmp)/'reference.png';image.write_bytes(b'old')
            async def describe(*a):
                image.write_bytes(b'new')
                return '{"appearance":"银白长发，桃花发饰。"}'
            self.assertEqual(await load_reference_description(tmp,describe),'')
            self.assertFalse((Path(tmp)/'appearance.json').exists())

    async def test_no_images_never_calls_vision(self):
        with tempfile.TemporaryDirectory() as tmp:
            describe=AsyncMock()
            self.assertEqual(await load_reference_description(tmp,describe),'')
            describe.assert_not_awaited()

    async def test_actual_start_loads_cached_description_without_model_call(self):
        from taoyao.app import TaoYaoBot
        from taoyao.config import load_config
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);persona=directory/'tao-yao.md';persona.write_text('桃夭人设',encoding='utf-8')
            (directory/'reference.png').write_bytes(b'image')
            await load_reference_description(directory,AsyncMock(return_value='{"appearance":"银白长发，红色眼睛。"}'))
            cfg=load_config();cfg['profile'].update(enabled=False,db=str(directory/'profiles.db'))
            cfg['memory']['db']=str(directory/'memory.db');cfg['privacy_index']['enabled']=False
            cfg['greetings']['enabled']=False;cfg['group_topics']['enabled']=False
            with patch('taoyao.persona.PERSONA_FILE',persona):
                bot=TaoYaoBot(cfg,mock=True)
                bot.classifier=SimpleNamespace(warmup=Mock())
                bot.server.start=AsyncMock();bot.vision._api_describe=AsyncMock()
                try:
                    await bot.start()
                    self.assertIn('银白长发，红色眼睛。',bot.persona)
                    bot.vision._api_describe.assert_not_awaited()
                finally:bot.memory.close();bot.profiles.close()
