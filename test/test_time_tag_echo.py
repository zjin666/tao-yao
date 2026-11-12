import unittest

from taoyao.output_guard import clean_reply_envelope, guard_outgoing


class TimeTagEchoTests(unittest.TestCase):
    def test_a_copied_history_time_tag_is_stripped_from_the_start_of_a_reply(self):
        for tag in ('[11-11 周三 19:40] ', '【11-11 周三 19:40】', '[2025-12-30 周二 08:05]'):
            self.assertEqual(clean_reply_envelope(tag + '诶，我算反了呀？'), '诶，我算反了呀？')
            self.assertEqual(guard_outgoing(tag + '诶，我算反了呀？'), '诶，我算反了呀？')

    def test_ordinary_brackets_and_times_in_the_body_stay(self):
        text = '晚上 19:40 见，[1] 这条是对的，【标题】也留着'
        self.assertEqual(clean_reply_envelope(text), text)


if __name__ == '__main__':
    unittest.main()
