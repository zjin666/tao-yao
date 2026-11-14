import unittest
from types import SimpleNamespace

from taoyao import markers
from taoyao.app import ReplyStreamer, TaoYaoBot, note_quote_choice
from taoyao.onebot import OneBotServer
from taoyao.turn_timing import get_value, set_value, turn_timing_context


class QuoteMessageTests(unittest.IsolatedAsyncioTestCase):
    async def build(self, **kwargs):
        server = OneBotServer.__new__(OneBotServer)
        sent = []

        async def send_group(group_id, message):
            sent.append(("group", message))
            return {"status": "ok"}

        async def send_private(user_id, message):
            sent.append(("private", message))
            return {"status": "ok"}
        server.send_group, server.send_private = send_group, send_private
        await server.send_text(**kwargs)
        return sent[0][1]

    async def test_group_quote_uses_a_reply_segment_and_no_extra_at(self):
        message = await self.build(target="group", target_id=1, text="好呀", at_user_id=7, quote_id="123")
        self.assertEqual(message[0], {"type": "reply", "data": {"id": "123"}})
        self.assertFalse(any(seg["type"] == "at" for seg in message))
        self.assertEqual(message[-1]["data"]["text"], "好呀")

    async def test_without_a_quote_the_old_at_prefix_is_unchanged(self):
        message = await self.build(target="group", target_id=1, text="好呀", at_user_id=7)
        self.assertEqual(message[0], {"type": "at", "data": {"qq": "7"}})
        self.assertFalse(any(seg["type"] == "reply" for seg in message))

    async def test_private_chat_never_quotes(self):
        message = await self.build(target="private", target_id=1, text="好呀", quote_id="123")
        self.assertEqual([seg["type"] for seg in message], ["text"])


class QuoteChoiceTests(unittest.IsolatedAsyncioTestCase):
    def bot(self, quote_enabled=True):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = {"reply": {"quote_enabled": quote_enabled}}
        return bot

    async def test_quote_is_used_only_when_the_model_chose_quote_and_at_stays_at(self):
        @turn_timing_context
        async def run(bot, chose_quote, at):
            set_value("quote_id", "555")
            if chose_quote:
                note_quote_choice("[[quote]] 这个问题嘛")
            return bot._quote_target("group", at)
        self.assertEqual(await run(self.bot(), True, 9), (None, "555"))      # 回应那句话的内容：引用
        self.assertEqual(await run(self.bot(), False, 9), (9, None))        # 只是点名对象：仍然 @
        self.assertEqual(await run(self.bot(), True, None), (None, None))   # 没点明对象（比如自然插话）：都不用

    async def test_private_chat_missing_message_or_switch_off_fall_back_to_at(self):
        @turn_timing_context
        async def run(bot, target, quote):
            set_value("quote_id", quote)
            note_quote_choice("[[quote]]")
            return bot._quote_target(target, 9)
        self.assertEqual(await run(self.bot(), "private", "555"), (9, None))
        self.assertEqual(await run(self.bot(), "group", ""), (9, None))
        self.assertEqual(await run(self.bot(quote_enabled=False), "group", "555"), (9, None))


class QuoteTagTests(unittest.IsolatedAsyncioTestCase):
    def test_only_the_exact_tag_spelling_is_recognised(self):
        for text in ("[[quote]] 好呀", "[[at]]好呀"):
            self.assertTrue(markers.ADDRESS_RE.search(text), text)
        # 标记是代码名词：全角、大小写、空格、参数、单层括号都不是这个标记，而是写错的标记
        for text in ("【quote】好呀", "[[ QUOTE ]] 好呀", "[[Quote]] 好呀", "[[quote:123]] 好呀", "[quote] 好呀",
                     "[[ at ]] 好呀", "[[AT]] 好呀", "【at】好呀", "[[at:小明]] 好呀"):
            self.assertIsNone(markers.ADDRESS_RE.search(text), text)
            self.assertTrue(markers.malformed_tags(text), text)
        self.assertEqual(markers.split_pending_tag("好的 [[quo"), ("好的 ", "[[quo"))   # 流式时写了一半先扣住

    async def test_the_streamer_hides_the_tag_notes_the_choice_and_asks_for_an_at(self):
        @turn_timing_context
        async def run():
            streamer = ReplyStreamer(max_chars=70, max_segments=5)
            ready = streamer.feed("[[quo")
            ready += streamer.feed("te]] 这个问题我来答。")
            ready += streamer.finish()
            return streamer, ready, get_value("quote_requested")
        streamer, ready, chose_quote = await run()
        visible = "".join(piece for piece, *_ in ready)
        self.assertIn("这个问题我来答", visible)                  # 真的跑到了，不是空转
        self.assertNotIn("quote", visible)
        self.assertNotIn("[[", visible)
        self.assertTrue(streamer.at_requested)
        self.assertTrue(chose_quote)

    def test_boundary_checker_ignores_the_tag(self):
        from taoyao.reply_flow import reply_boundary_issue
        self.assertEqual(reply_boundary_issue("[[quote]] 好呀", "user"), reply_boundary_issue("好呀", "user"))


if __name__ == "__main__":
    unittest.main()
