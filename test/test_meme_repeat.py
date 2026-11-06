import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from taoyao.memes import MemeLibrary


class MemeRepeatTests(unittest.TestCase):
    def test_multiple_images_cooldown_and_expiry_and_conversation_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'happy').mkdir()
            for n in (1, 2):
                (root/'happy'/f'{n:02}_test.png').write_bytes(bytes([n]))
            m = MemeLibrary(root)
            with patch('taoyao.memes.time.monotonic', return_value=10):
                one = m.pick('happy', conversation='private:1')
                two = m.pick('happy', conversation='private:1')
                self.assertNotEqual(one, two)
                self.assertIsNone(m.pick('happy', conversation='private:1'))
                self.assertIsNotNone(m.pick('happy', conversation='group:1'))
            with patch('taoyao.memes.time.monotonic', return_value=310):
                self.assertIsNotNone(m.pick('happy', conversation='private:1'))

    def test_same_image_under_different_filenames_is_single_and_can_repeat(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'shy').mkdir()
            for n in (1, 2):
                (root/'shy'/f'{n:02}_test.png').write_bytes(b'same-image')
            m = MemeLibrary(root)
            for _ in range(5):
                self.assertIsNotNone(m.pick('shy'))

    def test_shared_image_under_another_category_stays_in_cooldown(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'happy').mkdir()
            (root/'shy').mkdir()
            a = root/'happy'/'01_test.png'
            a.write_bytes(b'shared')
            (root/'shy'/'01_test.png').write_bytes(b'shared')
            b = root/'shy'/'02_test.png'
            b.write_bytes(b'other')
            m = MemeLibrary(root)
            m.remember(a, 'private:1')
            self.assertEqual(m.pick('shy', conversation='private:1'), b)
