import asyncio
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot


def make_bot(reply, age=10, paused=False, config=None):
    bot = NS()
    bot._ambient_config = lambda: config or {}
    bot._group_last_bot_reply = {1: (time.monotonic() - age, 2, '雕塑这活儿给他干明白了', '哥哥的话')}
    bot._group_reply_budget = NS(paused=lambda *args, **kwargs: paused)
    bot.profiles_enabled = True
    bot.profiles = NS(group_context=AsyncMock(return_value=[
        {'role': 'assistant', 'content': '雕塑', 'nickname': ''}]))
    bot.llm = NS(chat=AsyncMock(return_value='{"refers":false}'))
    bot.utility_llm = NS(chat=AsyncMock(return_value=reply))
    return bot


class ReferencesBotWordsTests(unittest.TestCase):
    def check(self, bot, text='雕塑那句我不认，我可是会动的鱼'):
        return asyncio.run(TaoYaoBot._refers_to_bot_words(bot, 1, 3, text))

    def test_refers_when_model_confirms(self):
        self.assertTrue(self.check(make_bot('{"refers":true}')))

    def test_not_refers_when_model_denies_or_output_invalid(self):
        self.assertFalse(self.check(make_bot('{"refers":false}')))
        self.assertFalse(self.check(make_bot('不是json')))

    def test_too_old_skips_model(self):
        bot = make_bot('{"refers":true}', age=400)
        self.assertFalse(self.check(bot))
        bot.utility_llm.chat.assert_not_awaited()

    def test_paused_group_skips_model(self):
        bot = make_bot('{"refers":true}', paused=True)
        self.assertFalse(self.check(bot))
        bot.utility_llm.chat.assert_not_awaited()

    def test_can_be_disabled_by_config(self):
        bot = make_bot('{"refers":true}', config={'bot_words_enabled': False})
        self.assertFalse(self.check(bot))
        bot.utility_llm.chat.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
