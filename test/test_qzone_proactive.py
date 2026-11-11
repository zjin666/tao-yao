import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from taoyao import qzone_proactive as qp
from taoyao.app import ReplyIntent, TaoYaoBot

OWNER = 111
SELF = 999
FRIEND = 222
NOW = time.time()


class StateTests(unittest.TestCase):
    def test_only_recent_visible_posts_with_something_to_talk_about_are_candidates(self):
        posts = [
            {'tid': 'a', 'content': '今天去爬山了', 'time': NOW - 3600},
            {'tid': 'b', 'content': '昨天的', 'time': NOW - 30 * 3600},                       # older than a day
            {'tid': 'c', 'content': '仅自己可见', 'time': NOW - 60, 'is_private': True},
            {'tid': 'd', 'content': '', 'time': NOW - 60, 'images': []},                      # nothing at all
            {'tid': 'e', 'content': '', 'time': NOW - 120, 'images': ['http://x/1.jpg']},     # picture only is fine
        ]
        self.assertEqual([p['tid'] for p in qp.recent_posts(posts, 24, NOW)], ['e', 'a'])

    def test_state_survives_a_restart_and_old_records_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 's.json'
            state = qp.ProactiveState(path)
            state.mark('t1', 'commented', FRIEND)
            state.posts['old'] = {'state': 'skipped', 'uin': FRIEND, 'at': NOW - 5 * 86400}
            state.save()
            again = qp.ProactiveState(path)
            self.assertEqual(again.get('t1')['state'], 'commented')
            self.assertIsNone(again.get('old'))
            self.assertEqual(again.count_commented(3600), 1)

    def test_unsure_means_skip_sixty_percent_and_ask_forty_percent(self):
        self.assertEqual(qp.decide_unsure(0.0), 'skip')
        self.assertEqual(qp.decide_unsure(0.59), 'skip')
        self.assertEqual(qp.decide_unsure(0.6), 'ask')
        self.assertEqual(qp.decide_unsure(0.99), 'ask')
        shares = sum(qp.decide_unsure(i / 1000) == 'skip' for i in range(1000)) / 1000
        self.assertAlmostEqual(shares, 0.6, places=2)


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def chat(self, messages, *a, **k):
        self.calls.append(messages)
        reply = self.reply
        if isinstance(reply, list):
            reply = reply.pop(0) if len(reply) > 1 else reply[0]
        return reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)


POSTS = [{'tid': 'p1', 'content': '今天终于把论文交了，松了口气', 'time': NOW - 600, 'comment_num': 0, 'images': []}]


def make_bot(reply, tmp, posts=None, config=None):
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot.config = {'owner_qq': OWNER, 'qzone_proactive_file': str(Path(tmp) / 'p.json'),
                  'qzone_state_file': str(Path(tmp) / 'seen.json'), **(config or {})}
    bot.utility_llm = FakeLLM(reply)
    bot.llm = bot.utility_llm
    bot.memory = SimpleNamespace(recent=AsyncMock(return_value=[]), add_message=AsyncMock())
    bot.profiles_enabled = True
    friend = {'user_id': str(FRIEND), 'role': 'user', 'nickname': '小红'}
    bot.profiles = SimpleNamespace(get=AsyncMock(side_effect=lambda uin: friend if int(uin) == FRIEND else None))
    bot.vision = SimpleNamespace(enabled=False, work_dir=Path(tmp))
    bot._send_segments = AsyncMock()
    bot._owner_qq = lambda: OWNER
    bot._onebot_ok = lambda r: bool(r and r.get('status') == 'ok')
    bot._friend_ids = AsyncMock(return_value=[OWNER, FRIEND, SELF])
    bot._qzone_view, bot._qzone_pending_reply, bot._qzone_seen = {}, {}, {}
    bot._qzone_auto_times, bot._qzone_counts, bot._self_uin = [], {}, SELF
    bot._qzone_pending_proactive = None
    bot._capture_public_draft = AsyncMock(return_value='恭喜呀，终于解放了')
    calls = []

    async def call(action, params, timeout=0):
        calls.append((action, params))
        if action == 'get_qzone_msg_list':
            return {'status': 'ok', 'data': {'msglist': list(posts if posts is not None else POSTS)
                                             if params.get('target_uin') == FRIEND else []}}
        if action == 'get_qzone_msg_detail':
            return {'status': 'ok', 'data': {'comments': []}}
        return {'status': 'ok', 'data': {}}
    bot.server = SimpleNamespace(call=call, calls=calls)
    return bot


def sent_comments(bot):
    return [c for c in bot.server.calls if c[0] == 'comment_qzone']


class CycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_post_worth_talking_about_gets_one_comment_with_the_friends_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'decision': 'comment'}, tmp)
            await bot._qzone_proactive_cycle()
            sent = sent_comments(bot)
            self.assertEqual(len(sent), 1)
            self.assertEqual((sent[0][1]['target_uin'], sent[0][1]['tid']), (FRIEND, 'p1'))
            intent = bot._capture_public_draft.await_args.args[0]
            self.assertEqual(intent.profile['nickname'], '小红')            # same identity handling as replies under her own post
            self.assertIn('不是初次见面', intent.facts)
            await bot._qzone_proactive_cycle()                             # commented once, never again by herself
            self.assertEqual(len(sent_comments(bot)), 1)

    async def test_the_owner_is_recognised_as_the_owner_when_commenting_on_their_post(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'decision': 'comment'}, tmp)
            bot.profiles.get = AsyncMock(return_value={'user_id': str(OWNER), 'role': 'owner', 'nickname': '哥哥'})
            profile, who = await bot._qzone_commenter({'uin': OWNER, 'name': ''})
            self.assertEqual(profile['role'], 'owner')
            self.assertIn('主人本人', who)

    async def test_a_post_with_nothing_to_talk_about_is_skipped_and_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'decision': 'skip'}, tmp)
            await bot._qzone_proactive_cycle()
            await bot._qzone_proactive_cycle()
            self.assertEqual(sent_comments(bot), [])
            self.assertEqual(len([c for c in bot.utility_llm.calls]), 1)    # decided once, remembered
            bot._send_segments.assert_not_awaited()

    async def test_an_unreadable_decision_means_no_comment(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot('不是json', tmp)
            await bot._qzone_proactive_cycle()
            self.assertEqual(sent_comments(bot), [])

    async def test_unsure_skips_or_asks_the_owner_by_chance(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'decision': 'unsure'}, tmp)
            with patch('taoyao.app.random.random', return_value=0.3):       # 60% branch: silently skip
                await bot._qzone_proactive_cycle()
            self.assertEqual(sent_comments(bot), [])
            bot._send_segments.assert_not_awaited()
            self.assertEqual(bot._proactive_state().get('p1')['state'], 'skipped')
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'decision': 'unsure'}, tmp)
            with patch('taoyao.app.random.random', return_value=0.9):       # 40% branch: ask the owner
                await bot._qzone_proactive_cycle()
            self.assertEqual(sent_comments(bot), [])
            intent = bot._send_segments.await_args.args[2]
            self.assertIn('论文', intent.facts)
            self.assertNotIn('222', intent.facts)
            self.assertEqual(bot._qzone_pending_proactive['tid'], 'p1')
            self.assertEqual(bot._proactive_state().get('p1')['state'], 'asked')

    async def test_only_one_question_to_the_owner_is_open_at_a_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            second = {'tid': 'p2', 'content': '心情不太好', 'time': NOW - 300, 'comment_num': 0, 'images': []}
            bot = make_bot({'decision': 'unsure'}, tmp, posts=[POSTS[0], second])
            with patch('taoyao.app.random.random', return_value=0.9):
                await bot._qzone_proactive_cycle()
            self.assertEqual(bot._send_segments.await_count, 1)
            self.assertEqual(bot._proactive_state().get('p1')['state'], 'skipped')    # the extra one is skipped, not asked

    async def test_the_hourly_cap_stops_commenting(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'decision': 'comment'}, tmp, config={'qzone': {'proactive_per_hour': 1, 'proactive_per_cycle': 5}})
            second = {'tid': 'p2', 'content': '去吃火锅了', 'time': NOW - 300, 'comment_num': 0, 'images': []}
            bot.server.call = bot.server.call
            bot2 = make_bot({'decision': 'comment'}, tmp, posts=[POSTS[0], second],
                            config={'qzone': {'proactive_per_hour': 1, 'proactive_per_cycle': 5}})
            await bot2._qzone_proactive_cycle()
            self.assertEqual(len(sent_comments(bot2)), 1)

    async def test_a_failed_send_is_not_recorded_as_commented(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'decision': 'comment'}, tmp)
            original = bot.server.call

            async def failing(action, params, timeout=0):
                if action == 'comment_qzone':
                    return {'status': 'failed'}
                return await original(action, params, timeout)
            bot.server.call = failing
            await bot._qzone_proactive_cycle()
            self.assertEqual(bot._proactive_state().get('p1')['state'], 'skipped')
            self.assertEqual(bot._proactive_state().count_commented(3600), 0)

    async def test_a_draft_that_fails_the_check_is_not_posted(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'decision': 'comment'}, tmp)
            bot._capture_public_draft = AsyncMock(return_value=None)
            await bot._qzone_proactive_cycle()
            self.assertEqual(sent_comments(bot), [])


class OwnerAnswerTests(unittest.IsolatedAsyncioTestCase):
    def waiting(self, bot):
        bot._qzone_pending_proactive = {'uin': FRIEND, 'tid': 'p1', 'name': '小红', 'post': '今天终于把论文交了',
                                        'image': '', 'at': time.monotonic()}
        bot._proactive_state().mark('p1', 'asked', FRIEND)

    async def test_advice_from_the_owner_is_used_for_the_comment(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'comment_advice', 'content': '先恭喜她，再问问答辩什么时候'}, tmp)
            self.waiting(bot)
            self.assertTrue(await bot._handle_owner_qq_intent('private', OWNER, '先恭喜她，再问问答辩什么时候'))
            intent = bot._capture_public_draft.await_args.args[0]
            self.assertIn('答辩', intent.facts)
            self.assertEqual(len(sent_comments(bot)), 1)
            self.assertEqual(bot._proactive_state().get('p1')['state'], 'commented')
            self.assertIsNone(bot._qzone_pending_proactive)
            self.assertIn('实际评论的内容', bot._send_segments.await_args.args[2].facts)

    async def test_an_answer_that_means_dont_bother_means_no_comment(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'comment_skip'}, tmp)
            self.waiting(bot)
            self.assertTrue(await bot._handle_owner_qq_intent('private', OWNER, '不用管她了，随便'))
            self.assertEqual(sent_comments(bot), [])
            self.assertEqual(bot._proactive_state().get('p1')['state'], 'skipped')
            self.assertIsNone(bot._qzone_pending_proactive)

    async def test_ordinary_chat_while_a_question_is_open_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'none'}, tmp)
            self.waiting(bot)
            self.assertFalse(await bot._handle_owner_qq_intent('private', OWNER, '今天吃什么好呢'))
            self.assertIsNotNone(bot._qzone_pending_proactive)             # still waiting

    async def test_without_an_open_question_advice_actions_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'comment_advice', 'content': '随便'}, tmp)
            self.assertFalse(await bot._handle_owner_qq_intent('private', OWNER, '评论一下吧'))
            self.assertEqual(sent_comments(bot), [])

    async def test_an_old_unanswered_question_counts_as_no_comment(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'decision': 'comment'}, tmp)
            self.waiting(bot)
            bot._qzone_pending_proactive['at'] -= 4 * 3600
            await bot._qzone_proactive_cycle()
            self.assertEqual(bot._proactive_state().get('p1')['state'], 'skipped')


class FollowUpTests(unittest.IsolatedAsyncioTestCase):
    DETAIL = {'comments': [
        {'uin': SELF, 'name': '桃夭', 'time': 1, 'content': '恭喜呀', 'comment_id': '0', 'replies': [
            {'uin': FRIEND, 'name': '小红', 'time': 2, 'content': '谢谢！下周答辩', 'comment_id': '7',
             'parent_comment_id': '0', 'parent_comment_uin': SELF}]},
        {'uin': 333, 'name': '路人', 'time': 3, 'content': '同喜', 'comment_id': '1', 'replies': []}]}

    async def test_only_a_reply_in_her_own_thread_is_followed_up_and_it_targets_the_friends_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'decision': 'reply'}, tmp)
            bot._proactive_state().mark('p1', 'commented', FRIEND)
            original = bot.server.call

            async def call(action, params, timeout=0):
                if action == 'get_qzone_msg_detail':
                    return {'status': 'ok', 'data': self.DETAIL}
                if action == 'get_qzone_msg_list':
                    return {'status': 'ok', 'data': {'msglist': [dict(POSTS[0], comment_num=2)]}} if params.get('target_uin') == FRIEND \
                        else {'status': 'ok', 'data': {'msglist': []}}
                return await original(action, params, timeout)
            bot.server.call = call
            bot._qzone_seen['p1'] = {'0'}
            await bot._qzone_proactive_cycle()
            sent = sent_comments(bot)
            self.assertEqual(len(sent), 1)                                  # 路人's top-level comment is not hers to answer
            params = sent[0][1]
            self.assertEqual((params['target_uin'], params['comment_id'], params['reply_uin']), (FRIEND, '0', FRIEND))
            intent = bot._capture_public_draft.await_args.args[0]
            self.assertEqual(intent.profile['nickname'], '小红')

    async def test_no_new_reply_means_no_follow_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'decision': 'comment'}, tmp)
            await bot._qzone_proactive_cycle()                              # comments once
            self.assertEqual(len(sent_comments(bot)), 1)
            await bot._qzone_proactive_cycle()
            await bot._qzone_proactive_cycle()
            self.assertEqual(len(sent_comments(bot)), 1)


if __name__ == '__main__':
    unittest.main()
