import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from taoyao.app import ReplyIntent, TaoYaoBot

OWNER = 111


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def chat(self, messages, *a, **k):
        self.calls.append(messages)
        reply = self.reply
        if isinstance(reply, list):                 # 依次返回；用完后重复最后一个
            reply = reply.pop(0) if len(reply) > 1 else reply[0]
        return reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)


def make_bot(reply, tmp, history=None):
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot.config = {'owner_qq': OWNER}
    bot.utility_llm = FakeLLM(reply)
    bot.llm = bot.utility_llm
    bot.memory = SimpleNamespace(recent=AsyncMock(return_value=history or []))
    bot.vision = SimpleNamespace(work_dir=Path(tmp))
    bot._send_segments = AsyncMock()
    bot._qzone_post = AsyncMock()
    bot._qzone_read = AsyncMock()
    bot._set_signature = AsyncMock()
    bot._qzone_view = {}
    bot._qzone_pending_reply = {}
    return bot


class IntentTests(unittest.IsolatedAsyncioTestCase):
    async def test_plain_chat_with_the_word_shuoshuo_is_not_a_post_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'none'}, tmp)
            self.assertFalse(await bot._handle_owner_qq_intent('private', OWNER, '说说你对这个的看法'))
            bot._qzone_post.assert_not_awaited()
            bot._send_segments.assert_not_awaited()

    async def test_messages_without_any_hint_cost_no_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'post_qzone', 'content': 'x'}, tmp)
            self.assertFalse(await bot._handle_owner_qq_intent('private', OWNER, '今天天气不错'))
            self.assertEqual(bot.utility_llm.calls, [])

    async def test_only_the_owners_private_chat_can_trigger(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'post_qzone', 'content': 'x'}, tmp)
            self.assertFalse(await bot._handle_owner_qq_intent('private', 999, '发说说：x'))
            self.assertFalse(await bot._handle_owner_qq_intent('group', OWNER, '发说说：x'))
            self.assertEqual(bot.utility_llm.calls, [])

    async def test_verbatim_content_is_posted_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'post_qzone', 'content': '今天的月亮很圆'}, tmp)
            self.assertTrue(await bot._handle_owner_qq_intent('private', OWNER, '发条说说：今天的月亮很圆'))
            args, kwargs = bot._qzone_post.await_args
            self.assertEqual(args[2], '今天的月亮很圆')
            self.assertEqual(kwargs['images'], [])

    async def test_a_composed_post_is_a_reply_intent_not_a_fixed_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'post_qzone', 'compose': True}, tmp,
                           history=[{'role': 'user', 'content': '这把伞好看吗'}])
            self.assertTrue(await bot._handle_owner_qq_intent('private', OWNER, '就按你的感想发说说'))
            content = bot._qzone_post.await_args.args[2]
            self.assertIsInstance(content, ReplyIntent)
            self.assertIn('这把伞好看吗', content.facts)

    async def test_missing_content_asks_instead_of_posting(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'post_qzone'}, tmp)
            self.assertTrue(await bot._handle_owner_qq_intent('private', OWNER, '发个说说'))
            bot._qzone_post.assert_not_awaited()
            self.assertIsInstance(bot._send_segments.await_args.args[2], ReplyIntent)

    async def test_the_picked_image_is_attached_by_number_not_just_the_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'post_qzone', 'content': '我的伞', 'image': 2}, tmp)
            await bot._handle_owner_qq_intent('private', OWNER, '把这张图发说说：我的伞')
            bot._qzone_post.assert_not_awaited()                       # no image on disk -> not posted, asks for it
            for name in ('粉白发少女', '笔记本屏幕', '羊角头像'):
                source = Path(tmp) / 'src.png'
                source.write_bytes(name.encode())
                bot._describe_kept_image(bot._remember_private_image(OWNER, source), name)
                time.sleep(0.02)
            await bot._handle_owner_qq_intent('private', OWNER, '把这张图发说说：我的伞')
            args, kwargs = bot._qzone_post.await_args
            self.assertEqual(len(kwargs['images']), 1)
            self.assertTrue(kwargs['images'][0].startswith('file:///'))
            from urllib.parse import unquote
            self.assertEqual(Path(unquote(kwargs['images'][0][8:])).read_bytes(), '笔记本屏幕'.encode())
            shown = bot.utility_llm.calls[-1][1]['content']
            self.assertIn('1. 粉白发少女', shown)
            self.assertIn('3. 羊角头像', shown)

    async def test_a_followup_without_keywords_is_judged_while_a_post_is_being_discussed(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot([{'action': 'post_qzone', 'content': '那就这段'}, {'ok': True, 'body': '那就这段'}], tmp,
                           history=[{'role': 'assistant', 'content': '这样贴在空间里，应该挺像我的吧：那就这段'}])
            self.assertTrue(await bot._handle_owner_qq_intent('private', OWNER, '你来哟'))
            bot._qzone_post.assert_awaited()
            self.assertEqual(bot._qzone_post.await_args.args[2], '那就这段')
            quiet = make_bot({'action': 'post_qzone', 'content': 'x'}, tmp, history=[{'role': 'assistant', 'content': '今天吃什么'}])
            self.assertFalse(await quiet._handle_owner_qq_intent('private', OWNER, '你来哟'))
            self.assertEqual(quiet.utility_llm.calls, [])

    async def test_a_request_taken_by_the_feature_is_kept_in_the_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'post_qzone', 'content': '好'}, tmp)
            bot.memory.add_message = AsyncMock()
            bot.profiles_enabled = False
            await bot._handle_owner_qq_intent('private', OWNER, '发说说：好')
            bot.memory.add_message.assert_awaited_with(f'owner_private:{OWNER}', 'user', '发说说：好')
            bot.memory.add_message.reset_mock()
            bot.utility_llm.reply = {'action': 'none'}
            await bot._handle_owner_qq_intent('private', OWNER, '说说你的看法')
            bot.memory.add_message.assert_not_awaited()                # ordinary chat is stored by the normal flow

    async def test_only_the_owners_recent_images_are_kept_and_pruned(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'none'}, tmp)
            source = Path(tmp) / 'src.jpg'
            source.write_bytes(b'x')
            self.assertIsNone(bot._remember_private_image(222, source))
            self.assertIsNone(bot._recent_private_image(222))
            for _ in range(7):
                bot._remember_private_image(OWNER, source)
            self.assertEqual(len(list((Path(tmp) / 'private' / str(OWNER)).glob('*'))), 5)

    async def test_unparseable_model_output_falls_back_to_normal_chat(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot('不是json', tmp)
            self.assertFalse(await bot._handle_owner_qq_intent('private', OWNER, '发说说吧'))


class SignatureTests(unittest.IsolatedAsyncioTestCase):
    async def test_candidate_from_the_conversation_becomes_the_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot([{'action': 'set_signature', 'content': '慢慢来，比较快'}, {'ok': True, 'body': '慢慢来，比较快'}], tmp,
                           history=[{'role': 'assistant', 'content': '签名可以写：慢慢来，比较快'}])
            self.assertTrue(await bot._handle_owner_qq_intent('private', OWNER, '就这个，改吧'))
            self.assertEqual(bot._set_signature.await_args.args[2], '慢慢来，比较快')

    async def test_a_composed_signature_is_a_reply_intent_with_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'set_signature', 'compose': True}, tmp,
                           history=[{'role': 'user', 'content': '今天过了卫戍协议20'}])
            self.assertTrue(await bot._handle_owner_qq_intent('private', OWNER, '你来想一句签名'))
            content = bot._set_signature.await_args.args[2]
            self.assertIsInstance(content, ReplyIntent)
            self.assertIn('卫戍协议20', content.facts)

    async def test_missing_signature_asks_and_does_not_claim_a_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'set_signature'}, tmp)
            self.assertTrue(await bot._handle_owner_qq_intent('private', OWNER, '改签名'))
            bot._set_signature.assert_not_awaited()
            self.assertIn('不要声称改好', bot._send_segments.await_args.args[2].situation)

    async def test_the_confirmation_states_the_real_signature_and_failure_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'none'}, tmp)
            bot._owner_qq = lambda: OWNER
            bot._onebot_ok = lambda result: bool(result and result.get('ok'))
            bot._set_signature = TaoYaoBot._set_signature.__get__(bot)
            bot.server = SimpleNamespace(call=AsyncMock(return_value={'ok': True}))
            await bot._set_signature('private', OWNER, '  慢慢来   比较快 ', owner_text='改')
            self.assertEqual(bot.server.call.await_args.args[1], {'longNick': '慢慢来 比较快'})
            self.assertIn('慢慢来 比较快', bot._send_segments.await_args.args[2].facts)
            bot.server.call = AsyncMock(return_value={'ok': False})
            await bot._set_signature('private', OWNER, '慢慢来', owner_text='改')
            self.assertIn('失败', bot._send_segments.await_args.args[2].situation)


class PublicBodyTests(unittest.IsolatedAsyncioTestCase):
    DRAFT = '图片我这边看准了，就是那个粉白发拿樱花的\n说说词给你：樱花掉下来的时候我伸手接了一下'

    async def test_only_the_verbatim_body_is_taken_from_a_chatty_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'ok': True, 'body': '樱花掉下来的时候我伸手接了一下'}, tmp)
            self.assertEqual(await bot._public_body(self.DRAFT, '说说'), '樱花掉下来的时候我伸手接了一下')

    async def test_a_body_that_is_not_in_the_draft_or_a_failed_check_means_no_post(self):
        with tempfile.TemporaryDirectory() as tmp:
            for reply in ({'ok': True, 'body': '模型自己改写的一句'}, {'ok': False, 'body': ''}, '不是json'):
                bot = make_bot(reply, tmp)
                self.assertIsNone(await bot._public_body(self.DRAFT, '说说'))
            bot = make_bot({'ok': True, 'body': 'x'}, tmp)
            self.assertIsNone(await bot._public_body('   ', '说说'))
            self.assertEqual(bot.utility_llm.calls, [])                      # empty draft: nothing to check

    async def test_a_candidate_taken_from_the_chat_is_checked_before_posting(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot([{'action': 'post_qzone', 'content': self.DRAFT}, {'ok': False, 'body': ''}], tmp)
            self.assertTrue(await bot._handle_owner_qq_intent('private', OWNER, '就这个，发说说'))
            bot._qzone_post.assert_not_awaited()
            self.assertIn('没有发布', bot._send_segments.await_args.args[2].situation)

    async def test_text_the_owner_wrote_in_the_message_is_not_second_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot([{'action': 'post_qzone', 'content': '今天的月亮很圆'}, {'ok': False}], tmp)
            await bot._handle_owner_qq_intent('private', OWNER, '发说说：今天的月亮很圆')
            self.assertEqual(bot._qzone_post.await_args.args[2], '今天的月亮很圆')
            self.assertEqual(len(bot.utility_llm.calls), 1)


class PostTests(unittest.IsolatedAsyncioTestCase):
    def bot(self, tmp, ok=True):
        bot = make_bot({'action': 'none'}, tmp)
        bot._qzone_post = TaoYaoBot._qzone_post.__get__(bot)
        bot._qzone_last_post = 0.0
        bot._owner_qq = lambda: OWNER
        bot._onebot_ok = lambda result: ok
        bot.server = SimpleNamespace(call=AsyncMock(return_value={'status': 'ok'}))
        return bot

    async def test_images_are_sent_and_the_confirmation_states_the_real_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = self.bot(tmp)
            await bot._qzone_post('private', OWNER, '我的伞', owner_text='发', images=['file:///a.png'])
            action, params = bot.server.call.await_args.args[:2]
            self.assertEqual(action, 'send_qzone_msg')
            self.assertEqual(params['images'], ['file:///a.png'])
            intent = bot._send_segments.await_args.args[2]
            self.assertIn('我的伞', intent.facts)
            self.assertIn('1 张图', intent.facts)

    async def test_failure_never_claims_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = self.bot(tmp, ok=False)
            await bot._qzone_post('private', OWNER, '我的伞', owner_text='发')
            self.assertIn('失败', bot._send_segments.await_args.args[2].situation)


if __name__ == '__main__':
    unittest.main()
