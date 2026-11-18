"""表情指纹除了形状还要比色彩：形状一样、颜色不同的不是同一张。"""
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from taoyao.sticker_memory import StickerMemory


def make(path: Path, body, bg=(240, 240, 240)) -> Path:
    img = Image.new("RGB", (128, 128), bg)
    ImageDraw.Draw(img).ellipse((20, 20, 100, 100), fill=body)
    img.save(path)
    return path


class ColorTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.memory = StickerMemory({"db": str(self.root / "s.db")})

    def tearDown(self):
        self.memory.close()
        self.folder.cleanup()

    def test_same_shape_but_different_colors_are_different_pictures(self):
        red = self.memory.fingerprint(make(self.root / "r.png", (200, 30, 30)))
        blue = self.memory.fingerprint(make(self.root / "b.png", (30, 30, 200)))
        self.assertLessEqual(self.memory._distance(red, blue), self.memory.near_distance)   # 形状指纹分不开
        self.assertIsNone(self.memory.same_picture(red, blue))
        self.memory.remember("g:1", 1, red, "红")
        self.assertIsNone(self.memory.lookup("g:1", blue))

    def test_recompressed_picture_is_still_recognised(self):
        first = make(self.root / "r.png", (200, 30, 30))
        again = self.root / "r.jpg"
        Image.open(first).save(again, quality=60)
        fp = self.memory.fingerprint(first)
        self.memory.remember("g:1", 1, fp, "红")
        self.assertIsNotNone(self.memory.lookup("g:1", self.memory.fingerprint(again)))

    def test_old_shape_only_record_matches_and_gets_color_added(self):
        fp = self.memory.fingerprint(make(self.root / "r.png", (200, 30, 30)))
        sid = self.memory.remember("g:1", 1, fp[:16], "旧记录")
        self.assertEqual(self.memory.lookup("g:1", fp)["id"], sid)
        self.assertEqual(self.memory.lookup("g:1", fp)["fp"], fp)


if __name__ == "__main__":
    unittest.main()
