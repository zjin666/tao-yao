import asyncio
from collections import OrderedDict
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.config import load_config
from taoyao.impressions import ImpressionStore, describe_public_impression
from taoyao.memory import MemoryStore
from taoyao.profiles import ProfileStore


class ImpressionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=ImpressionStore(Path(self.tmp.name)/'impressions.db')

    async def asyncTearDown(self):self.tmp.cleanup()

    async def observe(self,*args,**kwargs):
        await self.store.observe_interaction(*args,**kwargs,replied=True)

    async def seed(self,uid=2,gid=10,name='蓝色测试守夜',count=8):
        texts=['哈哈，这个梗我接住了，我再编个反转。','那我也开个玩笑，想象一下鱼会跑步。',
            '这回换我来讲个有意思的。','我就顺着这个说法再逗两句。',
            '我喜欢把事情换个好玩的角度讲。','哈哈，我刚才那句故意拐了个弯。',
            '接下来这句是我的冷笑话。','我又想到一个好笑的比喻。']
        for i in range(count):
            await self.observe(uid,gid,[name],texts[i%len(texts)],f'{uid}:{gid}:{i}')

    def judge(self,description='幽默'):
        async def chat(messages):
            payload=json.loads(messages[-1]['content']);rows=payload['messages']
            return json.dumps({'traits':[{'trait':description,'subject_user_id':payload['subject_user_id'],
                'basis':'speaker_tone','evidence':[{'message_id':r['id'],'quote':r['content']} for r in rows[:2]],
                'confidence':.8}]},ensure_ascii=False)
        return SimpleNamespace(chat=AsyncMock(side_effect=chat))

    async def test_minimum_evidence_and_incremental_update(self):
        await self.seed(count=7)
        llm=self.judge()
        self.assertEqual(await self.store.update_due(llm),0)
        llm.chat.assert_not_awaited()
        await self.observe(2,10,['新群名片'],'第八句公开发言。','8')
        self.assertEqual(await self.store.update_due(llm),1)
        self.assertEqual(await self.store.update_due(llm),0)
        rows=await self.store.read(2,[10])
        self.assertEqual(rows[0]['traits'][0]['description'],'幽默')
        self.assertTrue(rows[0]['updated_at'].endswith('+08:00'))
        for i in range(5):await self.observe(2,10,[],f'后续发言{i}。',f'next{i}')
        self.assertEqual(await self.store.update_due(llm),1)
        names=(await self.store.catalogue([10]))[0]['names']
        self.assertIn('新群名片',names)

    async def test_group_cooldown_and_passive_observation_do_not_build_impressions(self):
        bot=TaoYaoBot.__new__(TaoYaoBot);bot.impressions=self.store
        bot.config={'allowed_groups':[10]};bot._seen_messages=OrderedDict()
        bot._group_reply_budget=SimpleNamespace(paused=lambda *args:True)
        event={'post_type':'message','message_type':'group','group_id':10,'user_id':2,
            'self_id':100,'message_id':'cooldown','sender':{'nickname':'公开昵称','card':'公开群名片'},
            'message':[{'type':'text','data':{'text':'这是本人在群里的公开发言。'}},
                       {'type':'forward','data':{'id':'NOT_OWN_SPEECH'}}]}
        await bot.on_event(event)
        self.assertEqual(await self.store.catalogue([10]),[])
        await self.store.observe_interaction(2,10,['旁观昵称'],'只看到但没有互动的发言。','passive')
        self.assertEqual(await self.store.catalogue([10]),[])
        event.update(group_id=20,message_id='unallowed')
        await bot.on_event(event)
        self.assertEqual(await self.store.catalogue([20]),[])

    async def test_names_are_scoped_and_ambiguity_is_not_guessed(self):
        await self.seed(2,10,'共同昵称',1);await self.seed(3,20,'共同昵称',1)
        self.assertEqual(self.store.resolve('共同昵称',await self.store.catalogue([10])),['2'])
        self.assertEqual(self.store.resolve('共同昵称',await self.store.catalogue([10,20])),['2','3'])
        await self.seed(4,10,'蓝色测试守夜',1)
        self.assertEqual(self.store.resolve('守夜',await self.store.catalogue([10])),['4'])
        self.assertEqual(self.store.resolve('完全不认识',await self.store.catalogue([10])),[])

    async def test_private_profile_and_messages_never_enter_bootstrap(self):
        profiles=ProfileStore(Path(self.tmp.name)/'profiles.db')
        try:
            await profiles.ensure(2,'公开昵称')
            await profiles.add_message(2,'private',0,'user','PRIVATE_CHAT_SECRET')
            await profiles.add_message(2,'group',10,'assistant','BOT_REPLY_NOT_EVIDENCE')
            await profiles.add_message(2,'group',10,'user','公开互动原话。','group-msg',routing={'state':'bot'})
            await profiles.add_message(2,'group',10,'assistant','桃夭确实回应了。',routing={'evidence':'bot_reply','target_user_id':'2'})
            await profiles.add_message(2,'group',20,'user','没和桃夭互动的旁观内容。','passive')
            await profiles._run(lambda:profiles._conn.execute("UPDATE profiles SET personality=?,summary=?,scores=? WHERE user_id='2'",('PRIVATE_PERSONALITY','PRIVATE_SUMMARY','{"HUM":99}')))
            before=await profiles.get(2)
            rows=await profiles.impression_seed()
            self.assertEqual(len(rows),1)
            self.assertNotIn('PRIVATE',str(rows))
            self.assertNotIn('BOT_REPLY',str(rows))
            self.assertNotIn('scores',str(rows))
            for row in rows:await self.observe(row['user_id'],row['group_id'],row['names'],row['content'],row['message_key'],row['sent_at'])
            self.assertEqual((await profiles.get(2))['personality'],before['personality'])
        finally:profiles.close()

    async def test_unfounded_or_sensitive_traits_are_not_published(self):
        rows=[{'id':1,'content':'哈哈，我来接这个梗，再编个反转。','sent_at':'2026-11-07T00:00:00+08:00'},
              {'id':2,'content':'那我也讲个冷笑话，鱼怎么会跑步呢。','sent_at':'2026-11-07T00:01:00+08:00'}]
        base={'trait':'幽默','subject_user_id':'self','basis':'speaker_tone',
            'evidence':[{'message_id':r['id'],'quote':r['content']} for r in rows],'confidence':.9}
        for override in ({'evidence':base['evidence'][:1]},
                         {'evidence':[base['evidence'][0],{'message_id':99,'quote':'伪造证据'}]},
                         {'trait':'疑似抑郁'},{'confidence':.4},{'subject_user_id':'另一个人'}):
            llm=SimpleNamespace(chat=AsyncMock(return_value=json.dumps({'traits':[{**base,**override}]},ensure_ascii=False)))
            self.assertEqual(await describe_public_impression(llm,rows),[])
        llm=SimpleNamespace(chat=AsyncMock(return_value=json.dumps({'traits':[{**base,'description':'不应传播的私人细节'}]},ensure_ascii=False)))
        traits=await describe_public_impression(llm,rows)
        self.assertEqual(traits[0]['description'],'幽默')
        self.assertNotIn('私人细节',str(traits))

    async def test_slow_analysis_does_not_block_reads_and_deletion_prevents_resurrection(self):
        await self.seed();started=asyncio.Event();release=asyncio.Event()
        async def chat(messages):
            started.set();await release.wait()
            return await self.judge().chat(messages)
        task=asyncio.create_task(self.store.update_due(SimpleNamespace(chat=chat)))
        await asyncio.wait_for(started.wait(),2)
        try:
            self.assertTrue(await asyncio.wait_for(self.store.catalogue([10]),2))
            await self.store.forget(2)
        finally:release.set()
        self.assertEqual(await task,0)
        await self.seed()
        self.assertEqual(await self.store.catalogue(),[])

    async def test_private_query_only_reads_shared_groups_and_never_other_profile(self):
        await self.seed(2,10);await self.seed(2,20)
        await self.store.update_due(self.judge())
        bot=TaoYaoBot.__new__(TaoYaoBot);bot.impressions=self.store
        bot.memory=MemoryStore(Path(self.tmp.name)/'memory.db')
        bot.utility_llm=SimpleNamespace(chat=AsyncMock(return_value='{"target":"守夜"}'))
        bot.profiles=SimpleNamespace(get=AsyncMock(side_effect=AssertionError('Do not read other profiles')))
        async def call(action,params,**kwargs):
            return {'status':'ok','data':{'user_id':1}} if params['group_id']==10 else {'status':'failed'}
        bot.server=SimpleNamespace(call=AsyncMock(side_effect=call))
        try:
            await bot.memory.add_message('owner_private:1','user','我刚才提到群里的守夜。')
            intent=await bot._impression_request('你对她的印象呢？',1,'owner_private:1',False,0)
            self.assertIn('幽默',intent)
            self.assertIn('"group_id": 10',intent)
            self.assertNotIn('"group_id": 20',intent)
            bot.profiles.get.assert_not_awaited()
            bot.utility_llm.chat.reset_mock();bot.server.call.reset_mock()
            self.assertEqual(await bot._impression_request('你好',1,'owner_private:1',False,0),'')
            bot.utility_llm.chat.assert_not_awaited();bot.server.call.assert_not_awaited()
            self.assertEqual(await bot._impression_request('你对我的印象呢？',1,'owner_private:1',False,0),'')
            bot.utility_llm.chat.assert_not_awaited();bot.server.call.assert_not_awaited()
        finally:bot.memory.close()

    async def test_ambiguous_names_require_explicit_group_then_resolve_followup(self):
        await self.seed(2,10,'同名用户');await self.seed(3,20,'同名用户')
        await self.store.update_due(self.judge())
        bot=TaoYaoBot.__new__(TaoYaoBot);bot.impressions=self.store
        bot.memory=MemoryStore(Path(self.tmp.name)/'memory.db')
        bot._impression_groups=AsyncMock(return_value=[10,20])
        bot._group_list=AsyncMock(return_value=[{'group_id':10,'group_name':'测试A群'},{'group_id':20,'group_name':'测试B群'}])
        bot.utility_llm=SimpleNamespace(chat=AsyncMock(side_effect=[
            '{"target":"同名用户"}',
            '{"target":"同名用户","group_id":10,"group_evidence":"测试A群"}']))
        try:
            first=await bot._impression_request('你对同名用户印象如何？',1,'s',False,0)
            self.assertIn('重名',first)
            second=await bot._impression_request('测试A群的',1,'s',False,0)
            self.assertIn('幽默',second)
            self.assertIn('"group_id": 10',second)
            self.assertNotIn('"group_id": 20',second)
        finally:bot.memory.close()

    async def test_real_reply_uses_original_persona_flow_with_public_impression(self):
        cfg=load_config();cfg['owner_qq']=1
        cfg['memory']['db']=str(Path(self.tmp.name)/'bot-memory.db')
        cfg['profile'].update(enabled=False,db=str(Path(self.tmp.name)/'bot-profiles.db'))
        cfg['impressions']['db']=str(self.store.path)
        cfg['intro']['enabled']=cfg['memes']['enabled']=cfg['user_memes']['enabled']=False
        cfg['response_speed'].update(speculative_reply=False,two_part_reply=False,defer_six_axis=False)
        cfg['reply'].update(segment_gap_min_seconds=0,segment_gap_max_seconds=0)
        await self.seed();await self.store.update_due(self.judge())
        bot=TaoYaoBot(cfg,mock=True);inputs=[]
        async def stream(messages):inputs.append(messages);yield '在群里给我的感觉是挺幽默的。'
        async def judge(messages):
            return '{"target":"守夜"}' if messages[0]['content'].startswith('只识别') else '{"allowed":true}'
        bot.llm=SimpleNamespace(stream_chat=stream)
        bot.utility_llm=SimpleNamespace(chat=AsyncMock(side_effect=judge))
        bot.server.call=AsyncMock(return_value={'status':'ok','data':{'user_id':1}})
        bot.server.send_text=AsyncMock(return_value={'status':'ok'})
        bot._schedule_reply_extras=lambda *a,**k:None;bot._schedule_owner_summary=AsyncMock()
        bot._tone_hint=lambda *a,**kw:'SIX_AXIS_TEST'
        try:
            await bot._handle_text('private',1,'owner_private:1',1,'测试owner','你对守夜的印象呢？','owner',False,0,'private',from_batch=True)
            self.assertIn('幽默',inputs[0][0]['content'])
            self.assertIn('SIX_AXIS_TEST',inputs[0][0]['content'])
            self.assertIn('【步骤二：确定关系边界】',inputs[0][0]['content'])
            bot.server.send_text.assert_awaited()
            await bot._handle_text('group',10,'group:10',3,'新互动者','这句是新用户与桃夭的互动。',
                'user',True,10,'group',from_batch=True,message_id='interacted')
            await asyncio.gather(*list(bot._impression_record_tasks))
            self.assertIn('3',{r['user_id'] for r in await bot.impressions.catalogue([10])})
            normal_stream=bot._stream_reply
            bot._stream_reply=AsyncMock(return_value=('','',False,False))
            await bot._handle_text('group',10,'group:10',4,'未接话用户','这句没有得到桃夭回应。',
                'user',True,10,'group',from_batch=True,message_id='no-response')
            self.assertNotIn('4',{r['user_id'] for r in await bot.impressions.catalogue([10])})
            bot._stream_reply=normal_stream
            entered=asyncio.Event();release=asyncio.Event()
            async def slow_record(*args):entered.set();await release.wait()
            bot._record_impression_interaction=AsyncMock(side_effect=slow_record)
            before=bot.server.send_text.await_count
            try:
                await asyncio.wait_for(bot._handle_text('group',10,'group:10',5,'并行测试用户','这是并行测试的一次互动。',
                    'user',True,10,'group',from_batch=True,message_id='parallel'),3)
                await asyncio.wait_for(entered.wait(),2)
                self.assertGreater(bot.server.send_text.await_count,before)
                self.assertTrue(any(not task.done() for task in bot._impression_record_tasks))
            finally:
                release.set()
                await asyncio.gather(*list(bot._impression_record_tasks))
        finally:bot.memory.close();bot.profiles.close()
