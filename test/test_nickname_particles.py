import tempfile
import unittest
from pathlib import Path
from taoyao.personality import parse_tao_nickname_request,validate_tao_nickname
from taoyao.profiles import ProfileStore


class NicknameParticleTests(unittest.TestCase):
    def test_particles_and_unspecified_calls_are_not_naming(self):
        for message in ('叫你啦','我叫你啦','我想叫你啦','叫你呢','叫你一声','叫你一下',
                        '能叫你什么','桃夭，叫你啥','不叫你啦','以后不叫你啦'):
            self.assertEqual(parse_tao_nickname_request(message),'',message)
        for name in ('啦','呢','吧','哇','呗','一声','一下','什么','你'):
            self.assertEqual(validate_tao_nickname(name,True),'')
            self.assertEqual(validate_tao_nickname(name,False),'')

    def test_real_names_and_particle_suffixes(self):
        for message,name in (('我想叫你小夭啦','小夭'),('我能叫你小夭吗','小夭'),
                             ('叫你桃吧','桃'),('我想叫你“小哇”','小哇'),
                             ('我想叫你“啦啦”','啦啦'),('我想叫你想想','想想')):
            self.assertEqual(parse_tao_nickname_request(message),name,message)
            self.assertEqual(validate_tao_nickname(name,False),name)


class NicknameStorageTests(unittest.IsolatedAsyncioTestCase):
    async def test_particle_cannot_replace_valid_name_in_any_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ProfileStore(Path(tmp)/'profiles.db')
            try:
                await store.ensure(1,'测试用户')
                for scene in ('private','group'):
                    await store.update_basic_info(1,{'tao_nickname':'小夭'},scene,group_id=9)
                    await store.update_basic_info(1,{'tao_nickname':'啦','occupation':'学生'},scene,group_id=9)
                row=await store.get(1)
                self.assertEqual(row['basic_info']['tao_nickname'],'小夭')
                self.assertEqual(row['basic_info_private']['tao_nickname'],'小夭')
                self.assertEqual(row['basic_info_group']['9']['tao_nickname'],'小夭')
                self.assertEqual(row['basic_info']['occupation'],'学生')
            finally:store.close()
