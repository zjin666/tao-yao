import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import unittest

from taoyao.name_preferences import rejected_user_names
from taoyao.profiles import ProfileStore
from taoyao.app import TaoYaoBot
from taoyao.config import load_config


class NameRequestTests(unittest.TestCase):
    def test_explicit_own_requests_only(self):
        for text,names in [('别叫我小明',['小明']),('桃夭，以后不要再叫我“小明”了',['小明']),
            ('不许称呼我笨蛋',['笨蛋']),('我不喜欢你叫我小明',['小明']),
            ('别叫我小明、笨蛋了',['小明','笨蛋']),('我跟你说别叫我小明',['小明'])]:
            self.assertEqual(rejected_user_names(text),names,text)
        for text in ('他说别叫我小明','他说“别叫我小明”','我跟小明说别叫我小明',
            '我让小明别叫我笨蛋','比如别叫我小明','不是说别叫我小明','我不叫你小明了',
            '别叫他小明','他不喜欢你叫他小明','别叫我起床','> 别叫我小明','小明别叫我笨蛋','别人别叫我笨蛋'):
            self.assertEqual(rejected_user_names(text),[],text)
        self.assertEqual(rejected_user_names('小桃别叫我小明',('桃夭','小桃')),['小明'])


class NameStorageTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_removal_keeps_other_names_and_resists_stale_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            store=ProfileStore(Path(folder)/'profiles.db')
            try:
                await store.ensure(1,'小明');await store.ensure(2,'另一个人')
                for uid in (1,2):
                    await store.update_basic_info(uid,{'name':'小明','aliases':['小明','阿明','王小明'],
                        'tao_nickname':'小桃'},scene='private')
                    await store.update_basic_info(uid,{'name':'小明'},scene='group',group_id=8)
                await store.reject_user_names(1,['小明'])
                row=await store.get(1)
                for key in ('basic_info','basic_info_private'):
                    self.assertNotIn('name',row[key])
                    self.assertEqual(row[key]['aliases'],['阿明','王小明'])
                    self.assertEqual(row[key]['tao_nickname'],'小桃')
                self.assertNotIn('name',row['basic_info_group']['8'])
                self.assertEqual((await store.get(2))['basic_info']['name'],'小明')
                # QQ identity metadata is retained, but no longer a usable form of address.
                self.assertEqual(row['nickname'],'小明')
                for scene,gid in (('private',0),('group',9)):
                    await store.update_basic_info(1,{'name':'小明','aliases':['小明','阿明']},scene,gid)
                row=await store.get(1)
                self.assertNotIn('name',row['basic_info'])
                self.assertNotIn('name',row['basic_info_group']['9'])
                bot=TaoYaoBot.__new__(TaoYaoBot)
                summary=bot._profile_summary(row)
                self.assertNotIn('昵称：小明；',summary)
                self.assertIn('明确拒绝的称呼 小明',summary)
            finally:store.close()

    async def test_handler_requires_current_user_request_to_bot(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg=load_config();cfg['memory']['db']=str(Path(folder)/'memory.db')
            cfg['profile']['db']=str(Path(folder)/'profiles.db')
            cfg['intro']['enabled']=False;cfg['privacy_index']['enabled']=False
            bot=TaoYaoBot(cfg,mock=True)
            bot._profile_identity_allowed=AsyncMock(return_value=True)
            bot._maybe_intro=AsyncMock();bot._send_segments=AsyncMock()
            try:
                await bot.profiles.ensure(1,'用户')
                await bot.profiles.ensure(2,'其他用户')
                await bot.profiles.update_basic_info(2,{'name':'小明'},'private')
                for index,(state,evidence,targeted,ambient,should_remove) in enumerate([
                    ('other','explicit_at',True,False,False),
                    ('unknown','unresolved_reply',True,False,False),
                    ('bot','name_or_direct_context',False,False,False),
                    ('bot','explicit_at',True,True,False),
                    ('bot','explicit_at',True,False,True),
                    ('bot','reply',False,False,True),
                    ('bot','name_or_direct_context',True,False,True)]):
                    # Use another stored alias each iteration to avoid bans from a preceding case.
                    name='称呼'+str(index)
                    await bot.profiles.update_basic_info(1,{'name':name},'private')
                    mid='m'+str(index)
                    bot._group_routes[(8,mid)]={'state':state,'evidence':evidence}
                    bot._send_segments.reset_mock()
                    await bot._handle_text('group',8,'group:8',1,'用户','别叫我'+name,'user',True,8,'group',
                        from_batch=True,message_id=mid,nickname_targeted=targeted,ambient=ambient)
                    row=await bot.profiles.get(1)
                    self.assertEqual(row['basic_info'].get('name')!=name,should_remove)
                    self.assertEqual(bot._send_segments.await_count,1 if should_remove else 0)
                await bot.profiles.update_basic_info(1,{'name':'私聊称呼'},'private')
                await bot._handle_text('private',1,'user_private:1',1,'用户','别叫我私聊称呼','user',False,0,'private',from_batch=True)
                self.assertNotIn('name',(await bot.profiles.get(1))['basic_info'])
                self.assertEqual((await bot.profiles.get(2))['basic_info']['name'],'小明')
            finally:bot.memory.close();bot.profiles.close()


if __name__=='__main__':unittest.main()
