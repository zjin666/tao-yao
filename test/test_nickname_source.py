import unittest
from taoyao.personality import parse_tao_nickname_request,is_reported_tao_nickname
from taoyao.app import TaoYaoBot

class NicknameSourceTests(unittest.TestCase):
    def test_third_party_proposals_never_become_current_users_nickname_request(self):
        for text in ('小桃，守夜想给你换个称呼，你觉得什么比较合适',
                     '不是，是守夜想换个方式称呼你','守夜想叫你小夭',
                     '以后他叫你小夭','A让你记住B以后要叫你小夭',
                     '我替守夜说一下，他以后叫你小夭','他给你起了个名字叫小夭'):
            self.assertTrue(is_reported_tao_nickname(text),text)
            self.assertEqual(parse_tao_nickname_request(text),'',text)

    def test_direct_personal_requests_still_work(self):
        for text,name in [('我想叫你小夭','小夭'),('以后我叫你小夭','小夭'),
                          ('我能叫你小夭吗','小夭'),('桃夭，叫你小夭行吗','小夭'),
                          ('我给你起个名字叫小夭','小夭'),('我想叫你想想','想想'),
                          ('我想叫你“小夭”','小夭')]:
            self.assertFalse(is_reported_tao_nickname(text),text)
            self.assertEqual(parse_tao_nickname_request(text),name,text)

    def test_non_naming_requests_are_not_reported_nicknames(self):
        for text in ('他想叫你帮忙','他想叫你吃饭','我今天练了一个小时'):
            self.assertFalse(is_reported_tao_nickname(text),text)

    def test_profile_labels_nickname_direction_once(self):
        bot=TaoYaoBot.__new__(TaoYaoBot)
        summary=bot._profile_summary({'nickname':'用户昵称','basic_info':{'tao_nickname':'小桃'}})
        self.assertEqual(summary.count('小桃'),1)
        self.assertIn('此用户对桃夭的叫法',summary)
        self.assertIn('不是用户昵称',summary)


class NicknameSourceHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_delegated_memory_does_not_save_or_load_other_persons_profile(self):
        import asyncio,tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import AsyncMock,Mock
        from taoyao.config import load_config
        from taoyao.personality import DIMENSIONS,ClassificationResult
        with tempfile.TemporaryDirectory() as tmp:
            cfg=load_config();cfg['owner_qq']=900001
            cfg['memory']['db']=str(Path(tmp)/'memory.db')
            cfg['profile'].update(db=str(Path(tmp)/'profiles.db'),llm_extract=False)
            cfg['intro']['enabled']=False;cfg['privacy_index']['enabled']=False
            cfg['memes']['enabled']=cfg['user_memes']['enabled']=False
            cfg['reply'].update(segment_gap_min_seconds=0,segment_gap_max_seconds=0)
            bot=TaoYaoBot(cfg,mock=True)
            seen=[]
            async def stream(messages):
                seen.append(messages)
                yield '哥哥，让对方亲自来跟我说嘛。'
            bot.llm=bot.utility_llm=SimpleNamespace(stream_chat=stream,chat=AsyncMock(return_value='{"allowed":true}'))
            labels={d:ClassificationResult('中性',.9,applied=False) for d in DIMENSIONS}
            bot.analysis_pipeline=SimpleNamespace(config={'context_messages':6},analyze=AsyncMock(return_value=(labels,'test')))
            bot._profile_identity_allowed=AsyncMock(return_value=True)
            bot._handle_owner_group_task=AsyncMock(return_value=False)
            bot._handle_owner_qq_intent=AsyncMock(return_value=False)
            bot._memory_request_text=Mock(return_value='must not save this')
            bot._schedule_memory_update=Mock()
            bot._search_context_for=AsyncMock(return_value='')
            bot._maybe_send_reply_poke=AsyncMock()
            bot._schedule_profile_summary=lambda *a:None
            bot._schedule_profile_analysis=lambda *a:None
            bot.server.send_text=AsyncMock(return_value={'status':'ok'})
            try:
                await bot.profiles.ensure(900001,'测试owner','owner')
                await bot.profiles.update_basic_info(900001,{'tao_nickname':'小桃'},scene='private')
                await bot.profiles.ensure(900002,'测试B','user')
                bot.profiles.get=AsyncMock(wraps=bot.profiles.get)
                await asyncio.wait_for(bot._handle_text('private',900001,'owner_private:900001',900001,
                    '测试owner','A让你记住B以后要叫你小夭','owner',False,0,'private',from_batch=True,message_id='nickname'),5)
                bot._memory_request_text.assert_not_called()
                bot._schedule_memory_update.assert_not_called()
                system=seen[-1][0]['content']
                self.assertLess(system.index('【步骤一：理解意图与对象】'),system.index('【本轮答复意图】'))
                self.assertLess(system.index('【本轮答复意图】'),system.index('【步骤二：确定关系边界】'))
                bot.analysis_pipeline.analyze.assert_awaited_once()
                self.assertTrue(all(str(c.args[0])=='900001' for c in bot.profiles.get.await_args_list))
                a=await bot.profiles._run(bot.profiles._get,'900001')
                b=await bot.profiles._run(bot.profiles._get,'900002')
                self.assertEqual(a['basic_info']['tao_nickname'],'小桃')
                self.assertNotIn('tao_nickname',b['basic_info'])
                self.assertNotIn('小桃',str(bot.server.send_text.await_args_list))
            finally:bot.memory.close();bot.profiles.close()
