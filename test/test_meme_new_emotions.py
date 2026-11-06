import hashlib,unittest
from pathlib import Path
from PIL import Image
from taoyao.memes import MemeLibrary

CASES=[('skeptical','怀疑','我怎么不太信，你确定？'),
       ('disappointed','失望','白期待了，原来没有啊。'),
       ('admiring','佩服','居然坚持一年了，真有你的！'),
       ('competitive','较劲','再来一次，这次绝对赢你！'),
       ('jealous','吃醋','怎么只夸别人，为什么不先找我？'),
       ('hesitant','犹豫','这个嘛……我还没想好。'),
       ('conflicted','纠结','一方面想去，一方面又不想去，好难选啊。'),
       ('content','满足','这样就很好了，已经很满足了。'),
       ('blissful','幸福','今天真的好幸福，感觉一切都刚刚好。'),
       ('irritated','烦躁','怎么还没好，真的有点烦了。'),
       ('impatient','不耐烦','快一点嘛，还要等多久？'),
       ('envious','羡慕','好羡慕啊，我也想要。'),
       ('guilty','内疚','都是我的问题，我不应该那样做。'),
       ('respectful','尊敬','我很尊重你的决定，这点确实值得学习。'),
       ('suspicious','警惕','你是不是有什么企图，总觉得哪里不对。'),
       ('determined','坚定','这次我一定要做到，绝对不会放弃。')]
class NewEmotionPackTests(unittest.TestCase):
    def setUp(self):self.lib=MemeLibrary(Path(__file__).resolve().parents[1]/'memes')
    def test_all_sixteen_categories_have_two_distinct_valid_tiles(self):
        self.assertEqual(self.lib.total(),110)
        self.assertEqual(len(self.lib.categories),52)
        for cat,alias,_ in CASES:
            with self.subTest(category=cat):
                files=list((self.lib.root/cat).glob('*.png'))
                self.assertEqual(len(files),2)
                self.assertEqual(len({hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),2)
                for p in files:
                    with Image.open(p) as im:
                        self.assertIn(im.width,(313,314));self.assertIn(im.height,(313,314))
                        im.verify()
                self.assertTrue(self.lib.has_category(alias))
                self.assertTrue(self.lib.pick(alias,conversation='new:'+cat).exists())
    def test_each_typical_expression_selects_corresponding_category(self):
        for cat,_,text in CASES:
            with self.subTest(category=cat):self.assertEqual(self.lib.match(text)[0],cat)
        self.assertEqual(self.lib.match('这个嘛……')[0],'hesitant')
    def test_latest_emotion_outweighs_previous_new_category(self):
        self.assertEqual(self.lib.match('我很羡慕。原来没有啊，白期待了。')[0],'disappointed')
        self.assertEqual(self.lib.match('真有你的。总觉得你有什么企图。')[0],'suspicious')
        self.assertEqual(self.lib.match('刚才有点犹豫。下定决心了，这次我一定要做到。')[0],'determined')
    def test_new_two_image_categories_obey_existing_repeat_window(self):
        first=self.lib.pick('skeptical',conversation='packtest')
        second=self.lib.pick('怀疑',conversation='packtest')
        self.assertNotEqual(first,second)
        self.assertIsNone(self.lib.pick('skeptical',conversation='packtest'))
