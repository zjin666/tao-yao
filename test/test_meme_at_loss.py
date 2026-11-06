import unittest
from pathlib import Path

from taoyao.memes import MemeLibrary


class AtLossMemeTests(unittest.TestCase):
    def setUp(self):
        self.library = MemeLibrary(Path(__file__).resolve().parents[1] / 'memes')

    def test_new_category_and_moved_images_are_available(self):
        self.assertEqual(self.library.counts()['confused'], 2)
        self.assertEqual(self.library.counts()['at_loss'], 2)
        self.assertEqual(self.library.total(), 110)
        for name in ('03_晕了.png', '04_晕头转向.png'):
            self.assertTrue((self.library.root / 'at_loss' / name).exists())
            self.assertFalse((self.library.root / 'confused' / name).exists())
        self.assertTrue(any(s.startswith('at_loss（') for s in self.library.describe()))

    def test_confusion_embarrassment_resignation_and_being_at_loss_stay_distinct(self):
        cases = [
            ('我不知道该怎么做，脑子都打结了。', 'at_loss'),
            ('这下真的有点手足无措。', 'at_loss'),
            ('你这是什么意思，我没听懂？', 'confused'),
            ('我刚才说错了，太社死了。', 'awkward'),
            ('没办法，只能摊手了。', 'helpless'),
        ]
        for text, category in cases:
            self.assertEqual(self.library.match(text)[0], category, text)

    def test_model_tag_and_chinese_alias_pick_only_new_category_images(self):
        for category in ('at_loss', '无措', '不知所措', '手足无措'):
            self.assertTrue(self.library.has_category(category))
            self.assertEqual(self.library.pick(category, conversation=category).parent.name, 'at_loss')
        self.assertEqual(self.library.match('这下有点无措了。', preferred='无措')[0], 'at_loss')
