import asyncio
import unittest

from taoyao.turn_timing import mark, timeline_once, turn_timing_context


class TurnTimingTests(unittest.TestCase):
    def test_marks_from_child_tasks_share_one_timeline(self):
        @turn_timing_context
        async def turn():
            async def child():
                mark('first_token')
                mark('first_sentence')
                mark('check_start')
                mark('check_end')
            mark('draft_start')
            await asyncio.create_task(child())
            mark('prep_done')
            mark('send_start')
            mark('send_end')
            return timeline_once(), timeline_once()

        first, second = asyncio.run(turn())
        for label in ('开始生成', '准备完成', '首字', '首句写完', '检查开始', '检查结束', '开始发送', '发送确认'):
            self.assertIn(label, first)
        self.assertEqual(second, '')

    def test_only_first_occurrence_of_a_stage_is_kept(self):
        @turn_timing_context
        async def turn():
            mark('check_end')
            await asyncio.sleep(0.02)
            mark('check_end')
            return timeline_once()

        line = asyncio.run(turn())
        self.assertEqual(line.count('检查结束'), 1)
        self.assertTrue(line.startswith('检查结束=0.00'))

    def test_outside_a_turn_everything_is_a_no_op(self):
        mark('send_start')
        self.assertEqual(timeline_once(), '')


if __name__ == '__main__':
    unittest.main()
