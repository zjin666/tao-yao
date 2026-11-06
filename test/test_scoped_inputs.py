import tempfile,unittest
from pathlib import Path
from taoyao.profiles import ProfileStore
from taoyao.app import TaoYaoBot

class ScopedInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_cannot_opt_in_to_private_or_other_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ProfileStore(Path(tmp)/'profiles.db')
            try:
                for uid,scene,gid,text in [(1,'private',0,'PRIVATE_ONE'),(2,'private',0,'PRIVATE_TWO'),(1,'group',10,'GROUP_TEN'),(1,'group',20,'GROUP_TWENTY'),(2,'group',10,'PUBLIC_OTHER')]:
                    await store.add_message(uid,scene,gid,'user',text)
                for flag in (False,True):
                    data=await store.reply_inputs(1,'group',10,20,include_private=flag)
                    self.assertNotIn('PRIVATE',str(data));self.assertNotIn('GROUP_TWENTY',str(data))
                    self.assertIn('GROUP_TEN',str(data));self.assertIn('PUBLIC_OTHER',str(data))
                private=await store.reply_inputs(1,'private',0,20)
                self.assertIn('PRIVATE_ONE',str(private));self.assertIn('GROUP_TEN',str(private));self.assertIn('GROUP_TWENTY',str(private))
                self.assertNotIn('PRIVATE_TWO',str(private));self.assertNotIn('PUBLIC_OTHER',str(private))
            finally:store.close()

    async def test_group_profile_uses_only_target_public_fields_in_this_group(self):
        bot=TaoYaoBot.__new__(TaoYaoBot)
        profile={'nickname':'target','basic_info':{'name':'PRIVATE_NAME','location':'PRIVATE_LOCATION'},
            'basic_info_private':{'occupation':'PRIVATE_JOB'},'summary':'PRIVATE_SUMMARY','personality':'PRIVATE_TRAIT',
            'basic_info_group':{'10':{'name':'PUBLIC_NAME'},'20':{'name':'OTHER_GROUP_NAME'}}}
        result=bot._profile_summary(profile,is_group=True,group_id=10)
        self.assertIn('PUBLIC_NAME',result)
        self.assertNotIn('PRIVATE',result);self.assertNotIn('OTHER_GROUP_NAME',result)
