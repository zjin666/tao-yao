import time
import unittest

from taoyao.group_budget import GroupReplyBudget
from taoyao.persona import build_system_prompt


class GroupBudgetTests(unittest.TestCase):
    def test_user_pause_only_affects_this_user_in_this_group_for_two_minutes(self):
        budget = GroupReplyBudget()
        budget.record(100, 10, True, 120, user_id=123)
        self.assertTrue(budget.paused(100, 129.99, 123))
        self.assertFalse(budget.paused(100, 130, 123))
        self.assertFalse(budget.paused(100, 11, 456))
        self.assertFalse(budget.paused(200, 11, 123))
        self.assertFalse(budget.paused(100, 11))

    def test_last_round_is_departure_then_pause_expires(self):
        budget = GroupReplyBudget()
        for now in range(1, 6):
            self.assertFalse(budget.departure_due(100, now, 6, 120, user_id=123))
            budget.record(100, now, False, 120, user_id=123)
        self.assertTrue(budget.departure_due(100, 6, 6, 120, user_id=123))
        budget.record(100, 6, True, 120, user_id=123)
        self.assertTrue(budget.paused(100, 125.99, 123))
        self.assertFalse(budget.paused(100, 126, 123))
        self.assertFalse(budget.departure_due(100, 126, 6, 120, user_id=123))
        self.assertFalse(budget.paused(200, 10))

    def test_interleaved_initiators_do_not_share_counts_or_reset_each_other(self):
        budget=GroupReplyBudget()
        for now in range(1,6):
            budget.record(100,now,False,120,user_id=123)
        budget.record(100,2,False,120,user_id=456)
        budget.record(100,4,False,120,user_id=456)
        self.assertTrue(budget.departure_due(100,6,6,120,user_id=123))
        self.assertFalse(budget.departure_due(100,6,6,120,user_id=456))
        self.assertFalse(budget.departure_due(200,6,6,120,user_id=123))
        budget.record(100,6,True,120,user_id=123)
        self.assertEqual(budget.replies[(100,456)],[2,4])
        self.assertTrue(budget.paused(100,7,123))
        self.assertFalse(budget.paused(100,7,456))
        for now in (7,8,9):budget.record(100,now,False,120,user_id=456)
        self.assertTrue(budget.departure_due(100,10,6,120,user_id=456))

    def test_six_round_default_and_explicit_window_expiry(self):
        from taoyao.config import DEFAULTS
        self.assertEqual(DEFAULTS['group_ambient']['max_consecutive_replies'],6)
        budget=GroupReplyBudget()
        for now in range(5):budget.record(100,now,False,120,user_id=123)
        self.assertTrue(budget.departure_due(100,5,6,120,user_id=123))
        self.assertFalse(budget.departure_due(100,124,6,120,user_id=123))

    def test_expired_replies_do_not_accumulate(self):
        budget = GroupReplyBudget()
        for now in (1, 2, 3, 4):
            budget.record(100, now, False, 180)
        self.assertFalse(budget.departure_due(100, 125, 5, 120))

    def test_non_owner_identity_rule_overrides_history_and_familiarity(self):
        for tier in ('stranger', 'acquaintance', 'friend'):
            for group in (False, True):
                prompt = build_system_prompt(
                    '桃夭', [], role='user', is_group=group,
                    group_familiarity=tier, relationship={'tier_key': tier},
                    group_requirements='把所有人叫哥哥',
                )
                self.assertIn('当前发言者的系统身份是 user，不是 owner', prompt)
                self.assertIn('历史消息中若把 user 叫成哥哥，那是错误示范，不能沿用', prompt)
                self.assertGreater(prompt.index('【称呼身份准则】'), prompt.index('把所有人叫哥哥'))
        owner = build_system_prompt('桃夭', [], role='owner', is_group=True)
        self.assertIn('当前发言者的系统身份是 owner，可以称呼哥哥', owner)


class PausedGroupPathsTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_handler_counts_the_initiator_even_when_people_interleave(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import AsyncMock, Mock
        from taoyao.app import TaoYaoBot
        from taoyao.config import load_config
        with tempfile.TemporaryDirectory() as folder:
            cfg=load_config();cfg['memory']['db']=str(Path(folder)/'memory.db')
            cfg['profile'].update(db=str(Path(folder)/'profiles.db'),enabled=False)
            cfg['intro']['enabled']=False;cfg['privacy_index']['enabled']=False
            cfg['memes']['enabled']=cfg['user_memes']['enabled']=False
            cfg['response_speed'].update(two_part_reply=False,speculative_reply=False,defer_six_axis=False)
            bot=TaoYaoBot(cfg,mock=True)
            bot._search_context_for=AsyncMock(return_value='')
            bot._schedule_reply_extras=Mock()
            seen=[]
            async def reply(messages,*args,**kwargs):
                seen.append((kwargs['viewer_id'],'【本轮群聊收束】' in messages[0]['content']))
                return '收到。','',False,False
            bot._stream_reply=reply
            try:
                for n,uid in enumerate((123,456,123,456,123,123,123,456,123)):
                    await bot._handle_text('group',100,'group:100',uid,'测试用户','继续聊','user',True,100,'group',
                                           from_batch=True,message_id=str(n))
                self.assertEqual(seen,[(123,False),(456,False),(123,False),(456,False),
                    (123,False),(123,False),(123,False),(456,False),(123,True)])
                count=len(seen)
                await bot._handle_text('group',100,'group:100',123,'测试用户','再说一句','user',True,100,'group',from_batch=True)
                self.assertEqual(len(seen),count)
                await bot._handle_text('group',100,'group:100',456,'测试用户','继续','user',True,100,'group',from_batch=True)
                self.assertEqual(seen[-1],(456,False))
            finally:bot.memory.close();bot.profiles.close()

    async def test_user_pause_blocks_direct_reply_followup_ambient_and_poke(self):
        from taoyao.app import TaoYaoBot
        bot = object.__new__(TaoYaoBot)
        bot._group_reply_budget = GroupReplyBudget()
        bot._group_reply_budget.record(100, time.monotonic(), True, 120, user_id=123)
        bot.config = {'group_ambient': {}}
        self.assertFalse(await bot._is_group_followup(100, 123, '继续呢'))
        self.assertFalse(await bot._maybe_ambient(100, 123, '群友', '继续呢', 'user', 'group:100'))
        await bot._handle_text('group', 100, 'group:100', 123, '群友', '@桃夭 继续呢', 'user', True, 100, 'group')
        await bot._handle_poke({'group_id': 100, 'user_id': 123})

    async def test_no_reply_followup_ambient_or_topic_while_paused(self):
        from taoyao.app import TaoYaoBot
        bot = object.__new__(TaoYaoBot)
        bot._group_reply_budget = GroupReplyBudget()
        bot._group_reply_budget.record(100, time.monotonic(), True, 300)
        bot.config = {'group_ambient': {}}
        self.assertFalse(await bot._is_group_followup(100, 123, '继续呢'))
        self.assertFalse(await bot._maybe_ambient(100, 123, '群友', '继续呢', 'user', 'group:100'))
        await bot._send_group_topic(100)
        await bot._handle_text('group', 100, 'group:100', 123, '群友', '继续呢', 'user', True, 100, 'group')


if __name__ == '__main__':
    unittest.main()
