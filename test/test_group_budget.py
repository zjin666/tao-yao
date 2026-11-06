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
        for now in range(1, 7):
            self.assertFalse(budget.departure_due(100, now, 7, 120))
            budget.record(100, now, False, 180)
        self.assertTrue(budget.departure_due(100, 7, 7, 120))
        budget.record(100, 7, True, 180)
        self.assertTrue(budget.paused(100, 186))
        self.assertFalse(budget.paused(100, 187))
        self.assertFalse(budget.departure_due(100, 187, 7, 120))
        self.assertFalse(budget.paused(200, 10))

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
