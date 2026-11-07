import asyncio,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from taoyao.app import TaoYaoBot
from taoyao.chat_history import ChatHistory
from taoyao.config import load_config
from taoyao.profiles import ProfileStore
from taoyao.quoted import quote_text,same_quote_scope


def event(scene='private',mid='incoming'):
    return {'post_type':'message','message_type':scene,'self_id':100,'user_id':200,
        'group_id':10 if scene=='group' else 0,'message_id':mid,
        'sender':{'nickname':'测试用户'},'message':[{'type':'reply','data':{'id':'quoted'}}]}


class QuotedMessageTests(unittest.IsolatedAsyncioTestCase):
    async def test_outgoing_bubble_lookup_scoped_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            history=ChatHistory(tmp)
            await history.record('private',200,'精确引用这一句。',{'status':'ok','data':{'message_id':'quoted'}},1700000000)
            await history.record('private',201,'他人的私聊。',{'status':'ok','data':{'message_id':'other'}},1700000000)
            bot=TaoYaoBot.__new__(TaoYaoBot);bot.profiles_enabled=False
            bot.server=SimpleNamespace(chat_history=history,call=AsyncMock(return_value={'status':'failed'}))
            quote=await bot._quoted_message(event(),'private',0,200)
            self.assertEqual(quote['content'],'精确引用这一句。')
            self.assertEqual(quote['nickname'],'桃夭')
            bot.server.call.assert_not_awaited()
            self.assertIsNone(await history.find_message('private',200,'other'))
            self.assertIsNone(await history.find_message('group',200,'quoted'))
            await history.flush()  # Windows cannot delete the temp dir while the background fsync holds the file.

    async def test_api_scope_cache_and_local_absence(self):
        bot=TaoYaoBot.__new__(TaoYaoBot);bot.profiles_enabled=False
        bot.server=SimpleNamespace(call=AsyncMock(return_value={'status':'ok','data':{
            'group_id':10,'sender':{'user_id':100},'message':[{'type':'text','data':{'text':'被引用的正文'}}]}}))
        data=event('group')
        route=await bot._group_message_route(data,10,200)
        self.assertEqual(route['state'],'bot')
        self.assertEqual((await bot._quoted_message(data,'group',10,200))['content'],'被引用的正文')
        bot.server.call.assert_awaited_once()
        bot.server.call.return_value['data']['group_id']=11
        self.assertIsNone(await bot._quoted_message(event('group','different'),'group',10,200))

    async def test_private_lookup_never_reads_same_users_group_quote(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ProfileStore(Path(tmp)/'profiles.db')
            try:
                await store.ensure(200,'测试用户')
                await store.add_message(200,'group',10,'user','群里的内容',message_id='quoted')
                self.assertIsNone(await store.find_message('quoted','private',0,200))
                self.assertEqual((await store.find_message('quoted','group',10,200))['content'],'群里的内容')
            finally:store.close()

    async def test_private_qq_quote_reaches_real_prompt_without_becoming_user_statement(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=load_config();cfg['memory']['db']=str(Path(tmp)/'memory.db')
            cfg['profile'].update(enabled=False,db=str(Path(tmp)/'profiles.db'))
            cfg['privacy_index']['enabled']=cfg['intro']['enabled']=False
            cfg['user_memes']['enabled']=cfg['memes']['enabled']=False
            cfg['reply'].update(segment_gap_min_seconds=0,segment_gap_max_seconds=0)
            bot=TaoYaoBot(cfg,mock=True);bot.batch_enabled=False
            bot._media_text=AsyncMock(return_value='');bot._forward_text=AsyncMock(return_value='')
            bot._search_context_for=AsyncMock(return_value='');bot._maybe_send_reply_poke=AsyncMock()
            bot.server.chat_history=ChatHistory(Path(tmp)/'outbound')
            await bot.server.chat_history.record('private',200,'我今年28岁，喜欢跑步。',
                {'status':'ok','data':{'message_id':'quoted'}},1700000000)
            inputs=[]
            async def stream(messages):
                inputs.append(messages)
                yield '是在说刚才那句话吗？'
            bot.llm=bot.utility_llm=SimpleNamespace(stream_chat=stream,chat=AsyncMock(return_value='{"allowed":true}'))
            bot.server.send_text=AsyncMock(return_value={'status':'ok'})
            bot.server.call=AsyncMock(side_effect=AssertionError('Local quote must not call QQ'))
            try:
                await asyncio.wait_for(bot.on_event(event()),5)
                self.assertTrue(inputs)
                self.assertIn('我今年28岁，喜欢跑步。',inputs[0][0]['content'])
                self.assertIn('【本条QQ引用】',inputs[0][0]['content'])
                self.assertNotIn('28岁',inputs[0][-1]['content'])
                for row in await bot.memory.recent('user_private:200'):
                    if row['role']=='user':self.assertNotIn('28岁',row['content'])
                bot.server.send_text.assert_awaited()
            finally:bot.memory.close();bot.profiles.close()


class QuoteScopeTests(unittest.TestCase):
    def test_cross_scene_and_private_recipient_evidence(self):
        self.assertFalse(same_quote_scope({'message_type':'group','group_id':10,'sender':{'user_id':200}},'private',0,200,100))
        self.assertFalse(same_quote_scope({'message_type':'private','sender':{'user_id':201}},'private',0,200,100))
        self.assertFalse(same_quote_scope({'message_type':'private','sender':{'user_id':100}},'private',0,200,100))
        self.assertTrue(same_quote_scope({'message_type':'private','sender':{'user_id':200}},'private',0,200,100))
        self.assertTrue(same_quote_scope({'message_type':'private','sender':{'user_id':100},'target_id':200},'private',0,200,100))
        self.assertEqual(quote_text('原话[CQ:image,file=secret]。'),'原话。')
