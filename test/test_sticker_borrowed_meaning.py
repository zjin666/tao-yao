"""owner 在私聊里说明过的表情，群里再出现同一张时沿用含义；别人说明的不借。"""
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from taoyao.sticker_memory import StickerMemory


def make_image(path: Path) -> Path:
    img = Image.new("RGB", (128, 128), (240, 240, 240))
    ImageDraw.Draw(img).ellipse((20, 20, 100, 100), fill=(30, 30, 30))
    img.save(path)
    return path


class BorrowedMeaningTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        self.memory = StickerMemory({"db": str(root / "s.db")})
        self.fp = self.memory.fingerprint(make_image(root / "a.png"))

    def tearDown(self):
        self.memory.close()
        self.folder.cleanup()

    def teach(self, scope, teacher):
        sid = self.memory.remember(scope, teacher, self.fp, "吐彩虹的小人")
        self.memory.set_meaning(sid, teacher, "吃瓜看热闹", "原话")

    def test_owner_meaning_from_private_is_used_in_group(self):
        self.teach("private:1", 1)
        found = self.memory.borrowed_meaning("group:9", self.fp, {1})
        self.assertEqual(found["meaning"], "吃瓜看热闹")

    def test_meaning_taught_by_someone_else_is_not_borrowed(self):
        self.teach("private:2", 2)
        self.assertIsNone(self.memory.borrowed_meaning("group:9", self.fp, {1}))

    def test_other_images_and_same_scope_do_not_match(self):
        self.teach("private:1", 1)
        self.assertIsNone(self.memory.borrowed_meaning("private:1", self.fp, {1}))
        self.assertIsNone(self.memory.borrowed_meaning("group:9", "0000000000000000", {1}))
        self.assertIsNone(self.memory.borrowed_meaning("group:9", self.fp, {0}))


if __name__ == "__main__":
    unittest.main()
