import asyncio
from copy import deepcopy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from taoyao.speculation import SpeculativeDraft
from taoyao.app import TaoYaoBot
from taoyao.config import load_config
from taoyao.personality import DIMENSIONS, ClassificationResult, describe_scores


class SpeculationTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_inputs_only_and_cancel_unused_tasks(self):
        request = {'messages': [{'role': 'user', 'content': 'hello'}], 'review_policy': {'role': 'user'},
                   'generation': {'model': 'test'}, 'scores': {'HUM': 50.}}
        for field, value in [('messages', []), ('review_policy', {}), ('generation', {}), ('scores', {'HUM': 50.01})]:
            entered = asyncio.Event()
            async def slow(messages):
                entered.set()
                await asyncio.Event().wait()
            draft = SpeculativeDraft(request, slow)
            await entered.wait()
            final = deepcopy(request)
            final[field] = value
            self.assertIsNone(await draft.resolve(final))
            self.assertTrue(draft.task.cancelled())
        generate = AsyncMock(return_value='draft')
        draft = SpeculativeDraft(request, generate)
        self.assertEqual(await draft.resolve(deepcopy(request)), 'draft')
        await draft.close()
        draft = SpeculativeDraft(request, AsyncMock(side_effect=RuntimeError('test')))
        self.assertIsNone(await draft.resolve(request))
        await draft.close()

    async def test_real_handler_reuse_invalidation_review_and_no_stale_send(self):
        for mode in ('reuse', 'scores', 'search', 'reject'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                config = load_config()
                config['response_speed']['speculative_reply'] = True
                config['response_speed']['defer_six_axis'] = False
                config['response_speed']['two_part_reply'] = False
                config['memory']['db'] = str(Path(directory)/'memory.db')
                config['profile'].update(db=str(Path(directory)/'profiles.db'), llm_extract=False)
                config['memes']['enabled'] = False
                config['user_memes']['enabled'] = False
                config['intro']['enabled'] = False
                bot = TaoYaoBot(config, mock=True)
                original_tone = bot._tone_hint
                def observable_tone(profile, compact=False):
                    return original_tone(profile, compact) + f" [TEST HUM {float(profile['scores']['HUM']):.1f}]"
                bot._tone_hint = observable_tone
                try:
                    await bot.profiles.ensure(900001, '测试用户', 'user')
                    initial = dict.fromkeys(DIMENSIONS, 50.)
                    await bot.profiles.update_scores(900001, initial, describe_scores(initial))
                    entered = asyncio.Event()
                    generations = []
                    async def stream(messages):
                        generations.append(deepcopy(messages))
                        entered.set()
                        yield ('哥哥，我在听。' if mode == 'reject' and len(generations) == 1
                               else '我在听，接着说。')
                    fake = SimpleNamespace(stream_chat=stream, chat=AsyncMock(return_value='{"allowed":true}'))
                    bot.llm = bot.utility_llm = fake
                    labels = {d: ClassificationResult('中性', .9, applied=False) for d in DIMENSIONS}
                    if mode == 'scores':
                        labels['HUM'] = ClassificationResult('轻松玩笑', .9)
                    async def analyze(*args, **kwargs):
                        await entered.wait()  # Generation must overlap preparation.
                        return labels, 'TEST'
                    bot.analysis_pipeline = SimpleNamespace(config={'context_messages': 6}, analyze=analyze)
                    bot._profile_identity_allowed = AsyncMock(return_value=True)
                    bot._maybe_send_reply_poke = AsyncMock()
                    bot._search_context_for = AsyncMock(return_value='检索内容' if mode == 'search' else '')
                    bot._schedule_profile_summary = lambda p: None
                    bot.server.send_text = AsyncMock(return_value={'status': 'ok'})
                    bot._review_reply_boundary = AsyncMock(return_value=('', ''))
                    await asyncio.wait_for(bot._handle_text('private', 900001, 'user_private:900001',
                        900001, '测试用户', '这杯茶闻起来不错。', 'user', False, 0, 'private',
                        from_batch=True, message_id='TEST'), 10)
                    self.assertEqual(len(generations), 1 if mode == 'reuse' else 2)
                    bot.server.send_text.assert_awaited()
                    self.assertNotIn('哥哥', str(bot.server.send_text.await_args_list))
                    bot._review_reply_boundary.assert_awaited_once()
                    if mode == 'scores':
                        profile = await bot.profiles._run(bot.profiles._get, '900001')
                        self.assertIn(f"HUM {profile['scores']['HUM']:.1f}", generations[-1][0]['content'])
                    if mode == 'search':
                        self.assertIn('检索内容', generations[-1][0]['content'])
                finally:
                    bot.memory.close()
                    bot.profiles.close()
