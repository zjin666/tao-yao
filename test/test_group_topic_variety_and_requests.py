"""群话题要多样并参考群里聊过的内容；没点名的请求不靠抽签；讲笑话要有明确结尾。"""
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.persona import build_system_prompt


def topic_bot(rows, topics=()):
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot.config = {"group_topics": {}}
    bot.profiles_enabled = True
    bot.profiles = SimpleNamespace(
        recent_group_topics=AsyncMock(return_value=list(topics)),
        group_context=AsyncMock(return_value=rows),
        get_group_requirements=AsyncMock(return_value=[]))
    return bot


def row(text, role="user", nickname="群友"):
    return {"role": role, "content": text, "nickname": nickname,
            "created_at": datetime.now(timezone.utc).isoformat()}


class TopicTests(unittest.IsolatedAsyncioTestCase):
    async def test_older_group_chat_becomes_inspiration_but_images_and_bot_lines_do_not(self):
        bot = topic_bot([row("[群友] 昨晚把明日方舟的模拟打完了，十五回合就够"), row("桃夭的话", role="assistant"),
                         row("[群友] [图片内容] 画面是一张红叶照片"), row("[群友] 嗯")])
        text = await bot._group_topic_inspiration(1, {})
        self.assertIn("明日方舟", text)
        self.assertNotIn("桃夭的话", text)
        self.assertNotIn("红叶", text)
        self.assertNotIn("- 嗯", text)

    async def test_topic_prompt_uses_inspiration_and_bans_the_old_template(self):
        bot = topic_bot([row("[群友] 昨晚把明日方舟的模拟打完了")], topics=["刚把窗台那盆薄荷挪了个位置"])
        intent = await bot._group_topic_text(1)
        prompt = " ".join(str(getattr(intent, name, "")) for name in ("intent", "facts", "purpose", "reason"))
        blob = prompt + repr(vars(intent)) if hasattr(intent, "__dict__") else repr(intent)
        self.assertIn("明日方舟", blob)
        self.assertIn("你说它是不是", blob)          # 禁用句式被明确点名
        self.assertIn("这次的方式", blob)

    async def test_styles_rotate_and_do_not_repeat_within_three_picks(self):
        bot = topic_bot([])
        picks = [bot._pick_topic_style(1, True)[0] for _ in range(3)]
        self.assertEqual(len(set(picks)), 3)

    async def test_without_group_content_it_does_not_pick_styles_that_need_it(self):
        bot = topic_bot([])
        for _ in range(30):
            self.assertNotIn(bot._pick_topic_style(2, False)[0], ("接群友话头", "小知识"))
            bot.__dict__["_group_topic_styles"][2].clear()


class FirstMessageAfterTopicTests(unittest.TestCase):
    def bot(self):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot._ambient_config = lambda: {"topic_first_reply_seconds": 90}
        return bot

    def test_the_first_message_right_after_her_topic_counts_as_to_her(self):
        import time
        bot = self.bot()
        bot.__dict__["_group_fresh_topic"] = {1: time.monotonic() - 14}
        self.assertTrue(bot._take_fresh_topic(1, "能不能来个冷笑话"))
        self.assertFalse(bot._take_fresh_topic(1, "能不能来个冷笑话"))     # 只算第一条

    def test_too_late_other_groups_and_filler_do_not_count(self):
        import time
        bot = self.bot()
        bot.__dict__["_group_fresh_topic"] = {1: time.monotonic() - 200, 2: time.monotonic() - 5}
        self.assertFalse(bot._take_fresh_topic(1, "能不能来个冷笑话"))
        self.assertFalse(bot._take_fresh_topic(3, "能不能来个冷笑话"))
        self.assertFalse(bot._take_fresh_topic(2, "嗯"))
        self.assertFalse(bot._take_fresh_topic(2, "能不能来个冷笑话"))    # 第一条是“嗯”，已经用掉


class AddressedToOtherTests(unittest.IsolatedAsyncioTestCase):
    def bot(self, answer=None, error=False):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.profiles_enabled = False
        chat = AsyncMock(side_effect=RuntimeError("down")) if error else AsyncMock(return_value=answer)
        bot.utility_llm = SimpleNamespace(chat=chat)
        bot.llm = bot.utility_llm
        return bot

    async def test_a_message_aimed_at_someone_else_is_not_taken_as_to_her(self):
        self.assertTrue(await self.bot('{"to_other":true}')._addressed_to_other(1, "姐们，才周三"))

    async def test_a_plain_request_or_answer_is_to_her(self):
        self.assertFalse(await self.bot('{"to_other":false}')._addressed_to_other(1, "能不能来个冷笑话"))

    async def test_unknown_or_failed_judgement_does_not_force_a_reply(self):
        self.assertTrue(await self.bot("不是json")._addressed_to_other(1, "x"))
        self.assertTrue(await self.bot(error=True)._addressed_to_other(1, "x"))


class OpenRequestTests(unittest.TestCase):
    def test_unaddressed_requests_are_recognised(self):
        for text in ("能不能来个冷笑话", "[群友] 谁能帮我看看这个", "有没有人会这个", "来首歌", "陪我聊会儿", "给我推荐个游戏"):
            self.assertTrue(TaoYaoBot._looks_like_open_request(text), text)

    def test_ordinary_chatter_is_not(self):
        for text in ("今天好累", "哈哈哈这个梗太好笑了", "我昨天给我朋友来了一个惊喜" * 5, "https://x.com 帮我看看", "/帮我", "嗯"):
            self.assertFalse(TaoYaoBot._looks_like_open_request(text), text)


class JokeEndingPromptTests(unittest.TestCase):
    def test_prompt_asks_for_a_visible_ending_on_jokes_and_stories(self):
        prompt = build_system_prompt("人设", [], role="owner")
        self.assertIn("每次回复和起的话题，最后一句都要有明确的落点", prompt)
        self.assertNotIn("讲笑话、故事这类", prompt)       # 不是只对笑话和故事

    def test_group_topic_prompt_also_requires_a_clear_ending(self):
        prompt = build_system_prompt("人设", [], role="owner", is_group=True, ambient=True)
        self.assertIn("明确的落点", prompt)


if __name__ == "__main__":
    unittest.main()
