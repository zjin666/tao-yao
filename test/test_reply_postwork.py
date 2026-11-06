import asyncio,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock
from taoyao.app import TaoYaoBot,PendingMessage
from taoyao.config import load_config
from taoyao.memory import MemoryStore


class ReplyPostworkTests(unittest.IsolatedAsyncioTestCase):
    def bot(self,tmp):
        cfg=load_config();cfg['memory']['db']=str(Path(tmp)/'memory.db')
        cfg['profile'].update(enabled=False,db=str(Path(tmp)/'profiles.db'))
        cfg['privacy_index']['enabled']=cfg['intro']['enabled']=False
        cfg['user_memes']['enabled']=False;cfg['memes']['enabled']=True
        cfg['reply'].update(segment_gap_min_seconds=0,segment_gap_max_seconds=0)
        bot=TaoYaoBot(cfg,mock=True);bot.batch_enabled=False
        bot.guard.allow_rate=Mock(return_value=True)
        async def stream(messages):yield '正文。'
        bot.llm=SimpleNamespace(stream_chat=stream,chat=AsyncMock(return_value='摘要'))
        bot.utility_llm=SimpleNamespace(chat=AsyncMock(return_value='{"allowed":true}'))
        bot._search_context_for=AsyncMock(return_value='')
        bot.server.send_text=AsyncMock(return_value={'status':'ok'})
        return bot

    async def turn(self,bot,index,role='user'):
        session=f'{role}_private:900100'
        await bot._enqueue(PendingMessage('private',900100,session,900100,'合成用户',
            '接着说吧',role,False,0,'private',message_id=str(index)))

    async def cleanup(self,bot):
        tasks=[*bot._reply_extra_pending,*bot._memory_tasks]
        for task in tasks:task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        bot.memory.close();bot.profiles.close()

    async def test_pending_meme_and_poke_do_not_hold_next_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot=self.bot(tmp);started={name:asyncio.Event() for name in ('meme','poke')}
            cancelled=set();release=asyncio.Event()
            async def pending(name):
                started[name].set()
                try:await release.wait()
                except asyncio.CancelledError:cancelled.add(name);raise
            async def meme(*a,**k):await pending('meme')
            async def poke(*a,**k):await pending('poke')
            bot._maybe_send_meme=AsyncMock(side_effect=meme)
            bot._maybe_send_reply_poke=AsyncMock(side_effect=poke)
            try:
                await asyncio.wait_for(self.turn(bot,1),3)
                await asyncio.wait_for(asyncio.gather(*(e.wait() for e in started.values())),1)
                await asyncio.wait_for(self.turn(bot,2),3)
                self.assertEqual(bot.server.send_text.await_count,2)
                self.assertEqual(cancelled,{'meme','poke'})
                release.set()
                await asyncio.gather(*list(bot._reply_extra_pending))
            finally:await self.cleanup(bot)

    async def test_ready_image_and_poke_survive_new_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot=self.bot(tmp);release=asyncio.Event();started=[asyncio.Event(),asyncio.Event()]
            sent=[];counts={'meme':0,'poke':0}
            async def image(*a,**k):started[0].set();await release.wait();sent.append('meme')
            async def call(*a,**k):started[1].set();await release.wait();sent.append('poke');return {'status':'ok'}
            bot.server.send_image=AsyncMock(side_effect=image);bot.server.call=AsyncMock(side_effect=call)
            bot.memes.remember=Mock()
            async def meme(*a,**k):
                counts['meme']+=1
                if counts['meme']==1:await bot._send_image('private',900100,Path('synthetic.png'))
            async def poke(*a,**k):
                counts['poke']+=1
                if counts['poke']==1:await bot._send_poke('private',900100,900100)
            bot._maybe_send_meme=AsyncMock(side_effect=meme);bot._maybe_send_reply_poke=AsyncMock(side_effect=poke)
            try:
                await asyncio.wait_for(self.turn(bot,1),3)
                await asyncio.wait_for(asyncio.gather(*(e.wait() for e in started)),1)
                await asyncio.wait_for(self.turn(bot,2),3)
                self.assertEqual(bot.server.send_text.await_count,2)
                self.assertEqual(sent,[])
                release.set();await asyncio.gather(*list(bot._reply_extra_pending))
                self.assertCountEqual(sent,['meme','poke'])
            finally:await self.cleanup(bot)

    async def test_owner_summary_does_not_hold_reply_and_summaries_keep_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot=self.bot(tmp);bot.config['memory']['summarize_every']=1
            bot._maybe_send_meme=AsyncMock();bot._maybe_send_reply_poke=AsyncMock()
            started=asyncio.Event();release=asyncio.Event();prompts=[]
            async def summarize(messages):
                prompts.append(messages[-1]['content'])
                if len(prompts)==1:started.set();await release.wait();return '第一份摘要'
                return '第二份摘要'
            bot.llm.chat=AsyncMock(side_effect=summarize)
            try:
                await asyncio.wait_for(self.turn(bot,1,'owner'),3)
                await asyncio.wait_for(started.wait(),1)
                await asyncio.wait_for(self.turn(bot,2,'owner'),3)
                self.assertEqual(bot.server.send_text.await_count,2)
                self.assertEqual(len(prompts),1)
                release.set();await asyncio.gather(*list(bot._memory_tasks))
                self.assertIn('第一份摘要',prompts[1])
                self.assertEqual(await bot.memory.get_summary('owner_private:900100'),'第二份摘要')
            finally:await self.cleanup(bot)

    async def test_deleted_memory_cannot_be_restored_by_inflight_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory=MemoryStore(Path(tmp)/'memory.db');started=asyncio.Event();release=asyncio.Event()
            async def summarize(*a):started.set();await release.wait();return '旧资料'
            try:
                await memory.add_message('owner_private:1','user','合成资料')
                task=asyncio.create_task(memory.maybe_summarize('owner_private:1',SimpleNamespace(chat=summarize),every=1))
                await asyncio.wait_for(started.wait(),1)
                await memory.forget_session('owner_private:1');release.set()
                self.assertEqual(await task,'')
                self.assertEqual(await memory.get_summary('owner_private:1'),'')
            finally:memory.close()

    async def test_pending_poke_reserves_cooldown_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot=self.bot(tmp);started=asyncio.Event();release=asyncio.Event()
            bot.config['poke_reply'].update(enabled=True,notify_poke_probability=1,
                cooldown_seconds=0,max_per_hour=4)
            async def call(*a,**k):started.set();await release.wait();return {'status':'ok'}
            bot.server.call=AsyncMock(side_effect=call)
            first=asyncio.create_task(bot._maybe_send_reply_poke('private',900100,900100,
                '记得休息。','接着说吧',False,0,False))
            try:
                await asyncio.wait_for(started.wait(),1)
                await bot._maybe_send_reply_poke('private',900100,900100,
                    '记得休息。','接着说吧',False,0,False)
                self.assertEqual(bot.server.call.await_count,1)
                release.set();await first
                self.assertEqual(len(bot._outgoing_poke_times[(0,900100)]),1)
            finally:
                first.cancel();await asyncio.gather(first,return_exceptions=True)
                await self.cleanup(bot)
