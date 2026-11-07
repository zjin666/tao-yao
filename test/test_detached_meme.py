import asyncio,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock
from taoyao.app import TaoYaoBot
from taoyao.config import load_config
from taoyao.persona import build_system_prompt

class DetachedMemeTests(unittest.IsolatedAsyncioTestCase):
    def test_category_catalog_never_enters_body_prompt(self):
        prompt=build_system_prompt('PERSONA',['skeptical（SHOULD_NOT_ENTER_BODY_PROMPT）'])
        self.assertNotIn('SHOULD_NOT_ENTER_BODY_PROMPT',prompt)
        self.assertNotIn('【表情标签】',prompt)
        self.assertIn('不写分隔符、场景/身份/表情标签',prompt)

    async def test_first_send_and_tail_complete_before_independent_meme_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=load_config();cfg['profile'].update(enabled=False,db=str(Path(tmp)/'profiles.db'))
            cfg['memory']['db']=str(Path(tmp)/'memory.db')
            cfg['privacy_index']['enabled']=False;cfg['intro']['enabled']=False
            cfg['user_memes']['enabled']=False;cfg['memes']['enabled']=True
            cfg['response_speed'].update(speculative_reply=True,two_part_reply=True,defer_six_axis=False)
            cfg['reply'].update(segment_gap_min_seconds=0,segment_gap_max_seconds=0)
            bot=TaoYaoBot(cfg,mock=True)
            first_sent=asyncio.Event();finish_tail=asyncio.Event();events=[];inputs=[]
            async def stream(messages):
                inputs.append(messages)
                yield '我在听。'
                await finish_tail.wait()
                yield '现在脸红了。别看我。'
                yield '[[me'
                yield 'me:shy]]'
            bot.llm=bot.utility_llm=SimpleNamespace(stream_chat=stream,chat=AsyncMock(return_value='{"allowed":true}'))
            bot.memes.describe=Mock(side_effect=AssertionError('Body path must not read image catalog'))
            bot._search_context_for=AsyncMock(return_value='')
            bot._maybe_send_reply_poke=AsyncMock()
            async def send(*a,**k):events.append('text');first_sent.set();return {'status':'ok'}
            bot.server.send_text=AsyncMock(side_effect=send)
            async def meme(*a,**k):
                events.append('meme')
                self.assertIn('我在听',a[2]);self.assertIn('别看我',a[2]);return True
            bot._maybe_send_meme=AsyncMock(side_effect=meme)
            task=asyncio.create_task(bot._handle_text('private',900001,'user_private:900001',900001,
                '测试用户','继续说吧','user',False,0,'private',from_batch=True,message_id='detached'))
            try:
                await asyncio.wait_for(first_sent.wait(),3)
                bot._maybe_send_meme.assert_not_awaited()
                bot.memes.describe.assert_not_called()
                finish_tail.set()
                await asyncio.wait_for(task,3)
                await asyncio.gather(*list(bot._reply_extra_pending))
                self.assertEqual(events,['text','text','text','meme'])
                bot._maybe_send_meme.assert_awaited_once()
                self.assertEqual(bot._maybe_send_meme.await_args.kwargs['preferred'],'shy')
                self.assertIn('[[meme:ID]]',inputs[0][0]['content'])
                self.assertNotIn('meme:',bot._maybe_send_meme.await_args.args[2])
                stored=await bot.memory.recent('user_private:900001')
                self.assertTrue(any(row['role']=='assistant' for row in stored))
                self.assertTrue(all('meme:' not in row['content'] for row in stored))
                for call in bot.utility_llm.chat.await_args_list:
                    self.assertNotIn('meme:',str(call.args))
                self.assertNotIn('【表情标签】',inputs[0][0]['content'])
            finally:
                if not task.done():task.cancel()
                await asyncio.gather(task,return_exceptions=True)
                bot.memory.close();bot.profiles.close()
