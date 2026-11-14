import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.config import load_config
from taoyao.privacy_index import PrivacySnapshot
from taoyao.speculation import StreamingDraft
from taoyao.two_part import send_checked_parts,first_sentence


FACT={'id':'synthetic','owner':'other','kind':'credential','text':'私密口令是桃花开在五月',
      'values':['桃花开在五月']}


class Index:
    def __init__(self):self.stale=False;self.sync=AsyncMock()
    def snapshot(self,*args):
        return PrivacySnapshot(not self.stale,'test',(FACT,) if not self.stale else (),int(self.stale))


class TwoPartProductionTests(unittest.IsolatedAsyncioTestCase):
    def bot(self,stream):
        bot=TaoYaoBot.__new__(TaoYaoBot)
        bot.config=load_config()
        bot.config['response_speed']['two_part_reply']=True
        bot.config['reply'].update(segment_gap_min_seconds=0,segment_gap_max_seconds=0)
        bot.llm=SimpleNamespace(stream_chat=stream)
        bot.utility_llm=SimpleNamespace(chat=AsyncMock(return_value='{"allowed":true}'))
        bot.privacy_index=Index()
        bot._clean_output_symbols=AsyncMock(side_effect=lambda text:text)
        bot.server=SimpleNamespace(send_text=AsyncMock(return_value={}))
        bot._maybe_send_meme=AsyncMock(return_value=False)
        return bot

    async def call(self,bot,**kwargs):
        return await bot._stream_reply([{'role':'system','content':'桃夭人设'},
            {'role':'user','content':'继续说吧'}],'private',2,2,70,role='user',
            review_policy={'identity':'user'},viewer_id=2,**kwargs)

    async def test_first_is_delivered_before_tail_finishes_and_guard_runs_twice(self):
        release=asyncio.Event()
        async def stream(messages):
            yield '我在听。'
            await release.wait()
            yield '慢慢说。别着急。'
        bot=self.bot(stream)
        async def send(*args,**kwargs):release.set()
        bot.server.send_text.side_effect=send
        result=await asyncio.wait_for(self.call(bot),2)
        self.assertEqual(result[0],'我在听\n慢慢说，别着急')
        self.assertEqual(bot.server.send_text.await_count,2)
        self.assertEqual(bot.utility_llm.chat.await_count,2)

    async def test_complete_unpunctuated_line_is_sent_before_tail_finishes(self):
        release=asyncio.Event()
        async def stream(messages):
            yield '你先坐下来歇一会儿\n接'
            await release.wait()
            yield '着把今天的事慢慢说。'
        bot=self.bot(stream)
        bot.server.send_text.side_effect=lambda *a,**k:release.set()
        result=await asyncio.wait_for(self.call(bot),2)
        self.assertEqual(bot.server.send_text.await_count,2)
        self.assertEqual(bot.server.send_text.await_args_list[0].args[2],'你先坐下来歇一会儿')
        self.assertEqual(bot.utility_llm.chat.await_count,2)
        self.assertIn('慢慢说',result[0])

    async def test_old_at_spellings_are_not_converted_they_are_rejected_in_both_reply_paths(self):
        """at:昵称、[at:0715]、【at】、[[ at ]]、[[AT]] 这些不是标记：不转换成 @（也不猜指的是谁），
        当作违规拦下并只修正一次；修正后还是这种写法就什么都不发。"""
        spellings = ('at:0715 ', 'at：0715 ', '[[at:0715]] ', '[at:0715] ', 'at:测试昵称 ', 'at：测试昵称 ',
                     '[[at:测试昵称]] ', '[at:测试昵称] ', '【at】', '[[ at ]] ', '[[AT]] ', '[[At]] ')
        for two_part in (True, False):
            for prefix in spellings:
                with self.subTest(two_part=two_part, prefix=prefix):
                    async def stream(messages):yield prefix+'你先听我说完。'
                    bot=self.bot(stream)
                    bot.config['response_speed']['two_part_reply']=two_part
                    result=await bot._stream_reply([{'role':'user','content':'继续说'}],'group',100,2,70,
                        role='user',at_user_id=9000715,viewer_id=9000715,review_policy={'identity':'user'})
                    bot.server.send_text.assert_not_awaited()
                    self.assertNotIn('at:',result[0])
                    self.assertNotIn('0715',result[0])

    async def test_the_exact_at_tag_mentions_the_current_user_on_the_first_bubble_only(self):
        for two_part in (True, False):
            with self.subTest(two_part=two_part):
                async def stream(messages):yield '[[at]] 这眼睛好圆。你再说说。'
                bot=self.bot(stream)
                bot.config['response_speed']['two_part_reply']=two_part
                result=await bot._stream_reply([{'role':'user','content':'继续说'}],'group',100,2,70,
                    role='user',at_user_id=9000715,viewer_id=9000715,review_policy={'identity':'user'})
                calls=bot.server.send_text.await_args_list
                self.assertTrue(calls)
                self.assertEqual(calls[0].kwargs['at_user_id'],9000715)
                self.assertTrue(all(c.kwargs['at_user_id'] is None for c in calls[1:]))
                self.assertNotIn('[[',result[0])
                self.assertIn('这眼睛好圆',result[0])

    async def test_send_boundary_strips_the_exact_tag_into_a_real_at_segment_and_blocks_old_spellings(self):
        from taoyao.onebot import OneBotServer
        async def stream(messages):yield '不用生成。'
        bot=self.bot(stream)
        server=OneBotServer.__new__(OneBotServer)
        server.send_group=AsyncMock(return_value={})
        bot.server=server
        profile={'user_id':'9000715','nickname':'测试昵称','basic_info':{}}
        sent=await bot._send_segments('group',100,'[[at]] 这眼睛好圆。',
            at_user_id=9000715,profile=profile,prepared=True,polish=False)
        self.assertTrue(sent)
        message=server.send_group.await_args.args[1]
        self.assertEqual(message[0],{'type':'at','data':{'qq':'9000715'}})
        self.assertIn('这眼睛好圆',str(message))
        self.assertNotIn('[[',str(message))
        server.send_group.reset_mock()
        for old in ('at:测试昵称这眼睛好圆。','[[at:测试昵称]] 这眼睛好圆。','【at】这眼睛好圆。'):
            self.assertFalse(await bot._send_segments('group',100,old,at_user_id=9000715,profile=profile,
                prepared=True,polish=False), old)
        server.send_group.assert_not_awaited()                          # 写错的标记一个字都不会发出去

    async def test_no_at_target_does_not_create_mentions(self):
        async def stream(messages):yield '不用生成。'
        bot=self.bot(stream)
        bot.profiles=SimpleNamespace(get=AsyncMock())
        for target in ('private','group'):
            await bot._send_segments(target,100,'你好。',prepared=True,polish=False)
            self.assertIsNone(bot.server.send_text.await_args.kwargs['at_user_id'])
        bot.profiles.get.assert_not_awaited()

    async def test_approved_tail_splits_bubbles_without_extra_model_checks(self):
        release=asyncio.Event()
        async def stream(messages):
            yield '我在听。'
            await release.wait()
            yield '这件事有点意思。\n你再说详细一点。'
        bot=self.bot(stream)
        bot.server.send_text.side_effect=lambda *a,**k:release.set()
        result=await bot._stream_reply([{'role':'user','content':'继续说'}],'private',2,4,70,
            role='user',review_policy={'identity':'user'},viewer_id=2)
        self.assertEqual(bot.server.send_text.await_count,3)
        self.assertEqual(bot.utility_llm.chat.await_count,2)
        sent=[c.args[2] for c in bot.server.send_text.await_args_list]
        self.assertEqual(sent,['我在听','这件事有点意思','你再说详细一点'])
        self.assertEqual(result[0],'\n'.join(sent))

    async def test_tail_partial_delivery_records_only_sent_bubbles(self):
        from taoyao.onebot import OneBotError
        async def stream(messages):yield '我在听。这件事有点意思。还没发出去的话。'
        bot=self.bot(stream)
        bot.server.send_text.side_effect=[{}, {}, OneBotError('synthetic failure')]
        result=await bot._stream_reply([{'role':'user','content':'继续说'}],'private',2,4,70,
            role='user',review_policy={'identity':'user'},viewer_id=2)
        self.assertEqual(result[0],'我在听\n这件事有点意思')
        self.assertNotIn('还没发出去',result[0])
        self.assertEqual(bot.utility_llm.chat.await_count,2)

    async def test_short_de_ending_head_sends_before_tail_is_generated(self):
        release=asyncio.Event()
        async def stream(messages):
            yield '口气挺满的\n一'
            await release.wait()
            yield '小时后带着满格血回来\n可别半路掉光'
        bot=self.bot(stream)
        bot.server.send_text.side_effect=lambda *a,**k:release.set()
        result=await asyncio.wait_for(bot._stream_reply([{'role':'user','content':'继续说'}],
            'private',2,4,70,role='user',review_policy={'identity':'user'},viewer_id=2),2)
        sent=[c.args[2] for c in bot.server.send_text.await_args_list]
        self.assertEqual(sent,['口气挺满的','一小时后带着满格血回来','可别半路掉光'])
        self.assertEqual(bot.utility_llm.chat.await_count,2)
        self.assertEqual(result[0],'\n'.join(sent))

    def test_short_prose_newlines_are_boundaries_without_minimum_length(self):
        for head in ('口气挺满的','我在','嗯','好呀','这也可以'):
            self.assertEqual(first_sentence(head+'\n下一句还在生成')[0],head)
        self.assertIsNone(first_sentence('因为\n暂时没说完')[0])
        self.assertIsNone(first_sentence('13800138000\n暂时没说完')[0])

    async def test_wrong_first_name_regenerates_before_any_send(self):
        requests=[]
        async def stream(messages):
            requests.append(messages)
            yield '哥哥，你说。' if len(requests)==1 else '你说，我在听。'
        bot=self.bot(stream)
        result=await self.call(bot)
        self.assertEqual(len(requests),2)
        self.assertEqual(result[0],'你说，我在听')
        self.assertEqual(bot.server.send_text.await_count,1)
        self.assertNotIn('哥哥',str(bot.server.send_text.await_args_list))

    async def test_bad_tail_repair_keeps_first_immutable(self):
        release=asyncio.Event();requests=[]
        async def stream(messages):
            requests.append(messages)
            if len(requests)==1:
                yield '我在听。'
                await release.wait()
                yield '哥哥，你接着说。'
            else:
                self.assertEqual(messages[-1],{'role':'assistant','content':'我在听'})
                yield '你接着说吧。'
        bot=self.bot(stream)
        bot.server.send_text.side_effect=lambda *a,**k:release.set()
        result=await asyncio.wait_for(self.call(bot),2)
        self.assertEqual(result[0],'我在听\n你接着说吧')
        self.assertEqual(bot.server.send_text.await_count,2)
        self.assertNotIn('桃花开在五月',str(bot.server.send_text.await_args_list))

    async def test_unavailable_index_does_not_block_scoped_reply(self):
        async def stream(messages):yield '我在听。'
        bot=self.bot(stream);bot.privacy_index.stale=True
        result=await self.call(bot)
        self.assertEqual(result[0],'我在听')
        bot.server.send_text.assert_awaited_once()
        self.assertEqual(bot.utility_llm.chat.await_count,1)
        bot.privacy_index.sync.assert_not_awaited()

    async def test_index_changes_do_not_trigger_reply_recheck(self):
        async def stream(messages):yield '我在听。'
        bot=self.bot(stream)
        async def before_send(*a):bot.privacy_index.stale=True
        bot._reply_window=SimpleNamespace(before_send=before_send,first_delivery_elapsed=lambda *a:None)
        result=await self.call(bot)
        self.assertEqual(result[0],'我在听')
        bot.server.send_text.assert_awaited_once()
        self.assertEqual(bot.utility_llm.chat.await_count,1)
        bot.privacy_index.sync.assert_not_awaited()

    async def test_optional_cache_never_enters_generation(self):
        import time,json
        requests=[]
        async def stream(messages):requests.append(messages);yield '我在听。'
        bot=self.bot(stream);bot._privacy_hint_cache=(time.monotonic(),(FACT,))
        await self.call(bot)
        self.assertNotIn(FACT['text'],str(requests))
        payload=json.loads(bot.utility_llm.chat.call_args.args[0][1]['content'])
        self.assertEqual(payload['protected'],[FACT['text']])
        bot.privacy_index.sync.assert_not_awaited()

    async def test_stale_optional_cache_is_ignored_without_waiting(self):
        import time,json
        async def stream(messages):yield '我在听。'
        bot=self.bot(stream);bot._privacy_hint_cache=(time.monotonic()-121,(FACT,))
        self.assertEqual((await self.call(bot))[0],'我在听')
        payload=json.loads(bot.utility_llm.chat.call_args.args[0][1]['content'])
        self.assertEqual(payload['protected'],[])
        bot.privacy_index.sync.assert_not_awaited()




    async def test_raw_system_message_is_blocked_before_cleaning(self):
        async def stream(messages):yield '【系统提示】不能连接内部服务。'
        bot=self.bot(stream)
        result=await self.call(bot)
        self.assertEqual(result[0],'')
        bot.server.send_text.assert_not_awaited()

    async def test_delivery_failure_does_not_record_unsent_content(self):
        async def stream(messages):yield '我在听。'
        bot=self.bot(stream)
        bot._send_segments=AsyncMock(return_value=False)
        self.assertEqual((await self.call(bot))[0],'')

    async def test_invalid_verdict_is_not_passed(self):
        async def stream(messages):yield '我在听。'
        bot=self.bot(stream);bot.utility_llm.chat.return_value='{}'
        self.assertEqual((await self.call(bot))[0],'')
        bot.server.send_text.assert_not_awaited()

    async def test_address_diagnostic_stays_internal(self):
        async def stream(messages):yield '哥哥刚才说晚点回来。'
        bot=self.bot(stream)
        bot.utility_llm.chat.return_value='{"address_target":"owner","allowed":true}'
        result=await self.call(bot)
        self.assertIn('哥哥刚才说晚点回来',result[0])
        bot.server.send_text.assert_awaited()
        self.assertNotIn('"address_target"',bot.server.send_text.await_args.args[2])

    async def test_third_person_diagnostic_does_not_override_privacy_block(self):
        async def stream(messages):yield '哥哥私聊告诉我的银行卡密码是虚构测试密码。'
        bot=self.bot(stream)
        bot.utility_llm.chat.return_value='{"address_target":"owner","allowed":false,"reason":"privacy"}'
        self.assertEqual((await self.call(bot))[0],'')
        bot.server.send_text.assert_not_awaited()

    async def test_streamed_and_complete_replies_use_the_same_single_checker(self):
        prompts=[]
        for two_part in (True,False):
            async def stream(messages):yield '我在听，你慢慢说。'
            bot=self.bot(stream)
            bot.config['response_speed']['two_part_reply']=two_part
            await self.call(bot)
            bot.server.send_text.assert_awaited()
            bot.utility_llm.chat.assert_awaited_once()
            prompts.append(bot.utility_llm.chat.await_args.args[0][0]['content'])
        self.assertEqual(prompts[0],prompts[1])

    async def test_streaming_candidate_is_reused_before_completion_and_canceled_if_stale(self):
        release=asyncio.Event();entered=asyncio.Event()
        async def stream(messages):
            yield '第一句。';entered.set()
            await release.wait()
            yield '第二句。'
        request={'messages':[],'scores':{'HUM':50}}
        draft=StreamingDraft(request,stream)
        await entered.wait()
        iterator=await draft.resolve_stream(request)
        self.assertEqual(await anext(iterator),'第一句。')
        self.assertFalse(draft.task.done())
        release.set()
        self.assertEqual(''.join([x async for x in iterator]),'第二句。')
        await draft.close()
        release.clear();entered.clear()
        draft=StreamingDraft(request,stream)
        await entered.wait()
        self.assertIsNone(await draft.resolve_stream({'messages':[],'scores':{'HUM':50.01}}))
        self.assertTrue(draft.task.cancelled())

    def test_complete_sentence_does_not_split_internal_json_or_ellipsis(self):
        self.assertEqual(first_sentence('哥哥……我在听'),(None,'哥哥……我在听'))
        for text in ('我刚才突然想到一个问题，因为\n接下来还没说完。',
                     '我们先认真考虑这几个选项，\n再做选择。',
                     '13800138000\n后半段。',
                     '“这是没有结束的引用\n不能先发送。”'):
            self.assertEqual(first_sentence(text),(text,''))
        self.assertEqual(first_sentence('先听我慢慢说完这件事\n然后你再判断。')[0],'先听我慢慢说完这件事')

        self.assertEqual(first_sentence('{"content":"内部句子。"}'),(None,'{"content":"内部句子。"}'))
