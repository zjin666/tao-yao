import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.config import load_config
from taoyao.memory import MemoryStore
from taoyao.personality import ClassificationResult, DIMENSIONS
from taoyao.profiles import ProfileStore


class ReplyInputReuseTests(unittest.IsolatedAsyncioTestCase):
    async def test_bundled_context_matches_previous_reads_with_one_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory) / 'profiles.db')
            try:
                await store.ensure(1, '一号', 'owner')
                for uid, scene, gid, text, mid in [(1, 'private', 0, 'PRIVATE', 'p'),
                        (1, 'group', 10, 'GROUP', 'g'), (2, 'group', 20, 'OTHER_GROUP', 'o'),
                        (2, 'group', 10, 'PUBLIC', 'h'), (1, 'group', 10, 'CURRENT', 'current')]:
                    await store.add_message(uid, scene, gid, 'user', text, mid)
                await store.set_group_requirement(0, 'COMMON')
                await store.set_group_requirement(10, 'LOCAL')
                baseline = {
                    'history': await store.group_context(10, 12),
                    'group_summary': await store.get_group_summary(10),
                    'group_requirements': await store.get_group_requirements(10),
                    'user_memory': '',
                    'analysis_context': await store.annotation_context(1, 'group', 10, 6, 'current'),
                }
                store._run = AsyncMock(wraps=store._run)
                result = await store.reply_inputs(1, 'group', 10, 12, include_analysis=True, message_id='current')
                self.assertEqual(result, baseline)
                store._run.assert_awaited_once()
                self.assertNotIn('PRIVATE', str(result))
                self.assertNotIn('OTHER_GROUP', str(result))
                self.assertNotIn('CURRENT', str(result['analysis_context']))
                await store.set_group_requirement(10, 'NEW')
                self.assertIn('NEW', (await store.reply_inputs(1, 'group', 10))['group_requirements'])
            finally:
                store.close()

    async def test_owner_memory_bundle_preserves_values(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / 'memory.db')
            try:
                await store.set_summary('owner_private:1', 'SUMMARY')
                await store.add_fact('owner_private:1', 'FACT')
                expected = (await store.get_summary('owner_private:1'), await store.facts('owner_private:1'))
                store._run = AsyncMock(wraps=store._run)
                self.assertEqual(await store.reply_memory('owner_private:1'), expected)
                store._run.assert_awaited_once()
            finally:
                store.close()

    async def test_installed_handler_keeps_fresh_scores_without_duplicate_relationship_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config()
            config['memory']['db'] = str(Path(directory) / 'memory.db')
            config['response_speed']['speculative_reply'] = False
            config['response_speed']['defer_six_axis'] = False
            config['response_speed']['two_part_reply'] = False
            config['profile'].update(db=str(Path(directory) / 'profiles.db'), llm_extract=False)
            config['memes']['enabled'] = False
            config['user_memes']['enabled'] = False
            config['intro']['enabled'] = False
            bot = TaoYaoBot(config, mock=True)
            original_tone = bot._tone_hint
            def observable_tone(profile, compact=False):
                return original_tone(profile, compact) + f" [TEST HUM {float(profile['scores']['HUM']):.1f}]"
            bot._tone_hint = observable_tone
            try:
                generated_with = []
                async def stream(messages):
                    generated_with.append(messages)
                    yield '闻起来不错，那就慢慢喝。'
                fake = SimpleNamespace(stream_chat=stream, chat=AsyncMock(return_value='{"allowed":true}'))
                bot.llm = fake
                bot.utility_llm = fake
                labels = {dim: ClassificationResult('中性', .9, applied=False) for dim in DIMENSIONS}
                labels['HUM'] = ClassificationResult('轻松玩笑', .9)
                bot.analysis_pipeline = SimpleNamespace(config={'context_messages': 6}, analyze=AsyncMock(return_value=(labels, 'TEST_SAMPLE')))
                bot._profile_identity_allowed = AsyncMock(return_value=True)
                bot._maybe_send_reply_poke = AsyncMock()
                bot._search_context_for = AsyncMock(return_value='')
                bot._schedule_profile_summary = lambda profile: None
                bot.server.send_text = AsyncMock(return_value={'status': 'ok'})
                bot.profiles.get = AsyncMock(wraps=bot.profiles.get)
                bot.profiles.annotation_context = AsyncMock(wraps=bot.profiles.annotation_context)
                bot.profiles.reply_inputs = AsyncMock(wraps=bot.profiles.reply_inputs)
                await bot._handle_text('private', 900001, 'user_private:900001', 900001, '测试用户',
                    '这杯茶闻起来不错。', 'user', False, 0, 'private', from_batch=True, message_id='TEST')
                bot.server.send_text.assert_awaited()
                bot.profiles.get.assert_awaited_once()  # 更新后保留的一次最新画像回读。
                bot.profiles.annotation_context.assert_not_awaited()  # 已在同轮快照中取得。
                bot.profiles.reply_inputs.assert_awaited_once()
                current = await bot.profiles._run(bot.profiles._get, '900001')
                self.assertGreater(current['scores']['HUM'], 50)
                self.assertIn(f"HUM {current['scores']['HUM']:.1f}", generated_with[0][0]['content'])
            finally:
                bot.memory.close()
                bot.profiles.close()
