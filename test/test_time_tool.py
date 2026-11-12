import asyncio
from datetime import datetime,timezone
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock
from taoyao.time_tool import resolve_zone,clock_text,needs_clock,requested_zone,clock_only
from taoyao.app import TaoYaoBot
from taoyao.config import load_config
from taoyao.personality import extract_basic_info, explicit_user_timezone, extract_basic_info_llm

class TimeToolTests(unittest.TestCase):
    def test_default_applies_to_all_users_without_server_zone(self):
        for p in (None,{'role':'user'},{'role':'owner'},{'basic_info':{'timezone':'bad/zone'}}):
            self.assertEqual(resolve_zone(p),('Asia/Shanghai',True))
        self.assertEqual(resolve_zone({},'bad/zone'),('Asia/Shanghai',True))

    def test_explicit_and_legacy_private_zones(self):
        self.assertEqual(resolve_zone({'basic_info':{'timezone':'Asia/Tokyo'}}),('Asia/Tokyo',False))
        self.assertEqual(resolve_zone({'basic_info_private':{'timezone':'Europe/London'}}),('Europe/London',False))

    def test_date_rollover_and_weekday(self):
        now=datetime(2026,11,5,16,30,tzinfo=timezone.utc)
        self.assertIn('2026-11-06 00:30:00 星期五',clock_text(now=now))
        self.assertIn('2026-11-05 11:30:00 星期四',clock_text(target_zone='America/New_York',now=now))
        self.assertIn('UTC+0800',clock_text(now=now))

    def test_dst_follows_iana_not_fixed_offset(self):
        winter=datetime(2026,1,5,12,tzinfo=timezone.utc)
        summer=datetime(2026,7,5,12,tzinfo=timezone.utc)
        self.assertIn('07:00:00',clock_text(target_zone='America/New_York',now=winter))
        self.assertIn('08:00:00',clock_text(target_zone='America/New_York',now=summer))

    def test_reported_private_failure_uses_previous_day_in_new_york(self):
        profile={'basic_info':{'timezone':'America/New_York'}}
        result=clock_text(profile,now=datetime(2026,11,6,1,23,16,tzinfo=timezone.utc))
        self.assertIn('2026-11-05 20:23:16 星期四',result)
        self.assertIn('America/New_York',result)
        self.assertIn('UTC-0500',result)

    def test_question_gate_and_nonknowledge_chatter(self):
        for text in ('现在几点','今天几号','明天星期几','今天多少号','纽约现在几点','现在是早上还是晚上','还有2小时是几点','离周六还有几天','日期？','今天是周四吗','明天是不是星期五？','你能查到时间吗','离跨年还有多久'):
            self.assertTrue(needs_clock(text),text)
        for text in ('今天好累','最近想你','晚安','我讨厌时间复杂度','这个故事很有年代感','不要报时间'):
            self.assertFalse(needs_clock(text),text)
        self.assertTrue(needs_clock('东京呢','现在几点'))
        self.assertFalse(needs_clock('东京呢','今天好累'))

    def test_target_zone_does_not_mutate_user(self):
        profile={'basic_info':{'timezone':'Asia/Shanghai'}}
        self.assertEqual(requested_zone('伦敦现在几点'),'Europe/London')
        self.assertEqual(requested_zone('我在纽约，想问伦敦现在几点'),'Europe/London')
        self.assertEqual(requested_zone('America/New_York现在几点'),'America/New_York')
        self.assertEqual(requested_zone('UTC+80现在几点'),'')
        result=clock_text(profile,target_zone='Europe/London',now=datetime(2026,11,5,12,tzinfo=timezone.utc))
        self.assertIn('本次指定时区',result)
        self.assertIn('用户参考：',result)
        self.assertIn('UTC+0800',result)
        self.assertEqual(profile['basic_info']['timezone'],'Asia/Shanghai')

    def test_time_only_skips_search_but_not_external_facts(self):
        self.assertTrue(clock_only('请查一下现在几点'))
        for text in ('今天纽约股市几点开盘','现在几点，天气怎么样','今天是什么节假日'):
            self.assertFalse(clock_only(text))

    def test_private_time_question_and_correction_use_fresh_clock(self):
        for text in ('还记得我这边是几点吗','知道我这里现在几点吗？','纽约那边几点了呀','我这里现在是几点钟？'):
            self.assertTrue(needs_clock(text),text)
            self.assertTrue(clock_only(text),text)
        self.assertTrue(needs_clock('明明是晚上八点半','还记得我这边是几点吗'))
        self.assertFalse(needs_clock('明明是晚上八点半','我们聊的是电影里的台词'))

    def test_clock_query_does_not_change_user_timezone(self):
        for text in ('上海时间不应该是8点多吗','纽约现在几点','你是在哪个时区啊',
                     '我想问东京现在几点','我来自纽约','如果我在纽约，现在几点',
                     '他说我在纽约','我不在纽约','我现在不在纽约','我在纽约吗？','我在纽约的朋友说现在很晚了'):
            self.assertNotIn('timezone',extract_basic_info(text),text)
        self.assertEqual(requested_zone('上海时间不应该是8点多吗'),'Asia/Shanghai')

    def test_explicit_self_location_is_independent_of_queried_city(self):
        for text in ('我现在在纽约哦，这里是白天','我目前在纽约','我这边是纽约时间',
                     '我的时区是America/New_York',"I'm in New York",'我在纽约，想问伦敦现在几点'):
            self.assertEqual(explicit_user_timezone(text),'America/New_York',text)
            self.assertEqual(extract_basic_info(text)['timezone'],'America/New_York',text)
        self.assertEqual(requested_zone('我在纽约，想问伦敦现在几点'),'Europe/London')
        self.assertNotIn('timezone',extract_basic_info('看这个\n[最近图片记忆]\n我在纽约'))

class TimeToolIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_profile_extraction_cannot_store_a_queried_timezone(self):
        llm=SimpleNamespace(chat=AsyncMock(return_value='{"timezone":"Asia/Shanghai"}'))
        result=await extract_basic_info_llm('上海时间不应该是8点多吗',llm)
        self.assertNotIn('timezone',result)
        llm.chat.return_value='{"timezone":"Europe/London"}'
        result=await extract_basic_info_llm('我在纽约，想问伦敦现在几点',llm)
        self.assertEqual(result.get('timezone'),'America/New_York')

    async def test_clock_only_calls_no_search_decider(self):
        bot=TaoYaoBot.__new__(TaoYaoBot);bot.config={'search':{'enabled':True}}
        bot.llm=bot.utility_llm=SimpleNamespace(chat=AsyncMock())
        self.assertEqual(await bot._search_query_for('现在几点了'), '')
        bot.utility_llm.chat.assert_not_awaited()

    async def test_actual_handler_injects_clock_only_when_needed(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg=load_config();cfg['memory']['db']=str(Path(folder)/'memory.db')
            cfg['profile'].update(db=str(Path(folder)/'profiles.db'),enabled=False)
            cfg['intro']['enabled']=False;cfg['privacy_index']['enabled']=False
            cfg['response_speed'].update(two_part_reply=False,speculative_reply=False,defer_six_axis=False)
            cfg['memes']['enabled']=cfg['user_memes']['enabled']=False
            bot=TaoYaoBot(cfg,mock=True)
            calls=[]
            async def stream(messages):
                calls.append(messages);yield '现在是晚上。'
            bot.llm=SimpleNamespace(stream_chat=stream,chat=AsyncMock(return_value='{"allowed":true}'))
            bot.utility_llm=bot.llm
            bot._current_time_text=Mock(return_value='2026-11-05 20:30:00 星期四（Asia/Shanghai）')
            bot._search_context_for=AsyncMock(return_value='')
            bot._review_reply_boundary=AsyncMock(return_value=('',''))
            bot._maybe_send_reply_poke=AsyncMock()
            bot.server.send_text=AsyncMock(return_value={'status':'ok'})
            try:
                for n,text in enumerate(('我养了一只猫','现在几点了','还记得我这边是几点吗')):
                    await bot._handle_text('private',900001,'user_private:900001',900001,'测试用户',text,'user',False,0,'private',from_batch=True,message_id=str(n))
                    self.assertEqual(bot._current_time_text.call_count,n)
                self.assertNotIn('【当前时间】',calls[0][0]['content'])
                self.assertIn('2026-11-05 20:30:00',calls[1][0]['content'])
            finally:
                bot.memory.close();bot.profiles.close()
