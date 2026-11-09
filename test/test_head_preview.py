import asyncio,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from taoyao.app import TaoYaoBot
from taoyao.config import load_config
from taoyao.two_part import ScopedReplyGuard
from taoyao.speculation import StreamingDraft

class HeadPreviewTests(unittest.IsolatedAsyncioTestCase):
    async def run_handler(self,changed):
        with tempfile.TemporaryDirectory() as directory:
            config=load_config();config['memory']['db']=str(Path(directory)/'memory.db')
            config['profile'].update(enabled=False,db=str(Path(directory)/'profiles.db'))
            config['intro']['enabled']=False;config['privacy_index']['enabled']=False
            config['memes']['enabled']=config['user_memes']['enabled']=False
            config['response_speed'].update(two_part_reply=True,speculative_reply=True,defer_six_axis=True)
            config['reply'].update(segment_gap_min_seconds=0,segment_gap_max_seconds=0)
            bot=TaoYaoBot(config,mock=True)
            checked=asyncio.Event();release_search=asyncio.Event();first_sent=asyncio.Event();requests=[]
            async def stream(messages):
                requests.append(messages)
                if len(requests)==1:
                    yield '旧草稿。' if changed else '我在听。'
                    await first_sent.wait()
                    yield '慢慢说。'
                else:
                    yield '新正文。接着说。'
            async def review(messages):checked.set();return '{"allowed":true}'
            bot.llm=SimpleNamespace(stream_chat=stream,chat=AsyncMock(return_value='{"allowed":true}'))
            bot.utility_llm=SimpleNamespace(chat=AsyncMock(side_effect=review))
            async def search(*a):
                await release_search.wait()
                return '新检索资料' if changed else ''
            bot._search_context_for=search;bot._maybe_send_reply_poke=AsyncMock()
            bot.server.send_text=AsyncMock(side_effect=lambda *a,**k:first_sent.set())
            bot._schedule_profile_summary=lambda p:None
            task=asyncio.create_task(bot._handle_text('private',900001,'user_private:900001',900001,
                '测试用户','有点累了','user',False,0,'private',from_batch=True,message_id='t'))
            try:
                await asyncio.wait_for(checked.wait(),3)
                bot.server.send_text.assert_not_awaited()  # No sending until complete input is validated.
                release_search.set()
                await asyncio.wait_for(task,3)
                self.assertEqual(bot.server.send_text.await_count,2)
                # Count only the reply checker; the parallel "is the sentence unfinished" judge is a separate call.
                reviews=[c for c in bot.utility_llm.chat.await_args_list
                         if c.args[0][0]['content'].startswith('发送前检查器')]
                self.assertEqual(len(reviews),3 if changed else 2)
                if changed:
                    self.assertEqual(len(requests),2)
                    self.assertNotIn('旧草稿',str(bot.server.send_text.await_args_list))
                    self.assertEqual(bot.server.send_text.await_args_list[0].args[2],'新正文')
                else:
                    self.assertEqual(len(requests),1)
                    self.assertEqual(bot.server.send_text.await_args_list[0].args[2],'我在听')
            finally:
                if not task.done():task.cancel()
                await asyncio.gather(task,return_exceptions=True)
                bot.memory.close();bot.profiles.close()

    async def test_check_starts_before_search_finishes_and_is_reused_once(self):
        await self.run_handler(False)

    async def test_changed_search_discards_prechecked_text_without_sending(self):
        await self.run_handler(True)

    async def test_negative_verdict_is_cached_but_changed_text_is_rechecked(self):
        llm=SimpleNamespace(chat=AsyncMock(side_effect=['{"allowed":false,"reason":"privacy"}','{"allowed":true}']))
        guard=ScopedReplyGuard(llm,role='user',current='测试',policy={'identity':'user'})
        self.assertEqual(await guard.check('','待检文本'),'privacy')
        self.assertEqual(await guard.check('','待检文本'),'privacy')
        self.assertEqual(llm.chat.await_count,1)
        self.assertEqual(await guard.check('','修正后的文本'),'')
        self.assertEqual(llm.chat.await_count,2)

    async def test_discard_cancels_pending_preview(self):
        entered=asyncio.Event();cancelled=asyncio.Event()
        async def stream(messages):yield '这是完整的一句。'
        async def preview(raw):
            entered.set()
            try:await asyncio.Event().wait()
            finally:cancelled.set()
        draft=StreamingDraft({'messages':[],'scores':{'HUM':50}},stream,preview=preview)
        await entered.wait()
        self.assertIsNone(await draft.resolve_stream({'messages':[],'scores':{'HUM':50.4}}))
        self.assertTrue(cancelled.is_set())
