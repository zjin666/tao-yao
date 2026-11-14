import re
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.filetext import extract_text


def make_docx(path: Path, paragraphs):
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    xml = ('<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w='
           '"http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + body + "</w:body></w:document>")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)


def make_pptx(path: Path, slides):
    with zipfile.ZipFile(path, "w") as archive:
        for number, text in enumerate(slides, 1):
            xml = ('<p:sld xmlns:p="p" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                   f"<a:t>{text}</a:t></p:sld>")
            archive.writestr(f"ppt/slides/slide{number}.xml", xml)


def make_pdf(path: Path, text: str):
    stream = f"BT /F1 18 Tf 20 100 Td ({text}) Tj ET".encode()
    objects = [b"<</Type/Catalog/Pages 2 0 R>>", b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
               b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 200]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
               b"<</Length %d>>\nstream\n" % len(stream) + stream + b"\nendstream",
               b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>"]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<</Size {len(objects) + 1}/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


class ExtractTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.addCleanup(self.folder.cleanup)

    def test_text_in_utf8_and_gbk(self):
        (self.root / "a.txt").write_text("你好，这是一份笔记。", encoding="utf-8")
        (self.root / "b.txt").write_bytes("你好，这是GBK编码。".encode("gbk"))
        self.assertEqual(extract_text(self.root / "a.txt").text, "你好，这是一份笔记。")
        self.assertIn("GBK编码", extract_text(self.root / "b.txt").text)

    def test_docx_pptx_xlsx_and_pdf(self):
        make_docx(self.root / "r.docx", ["第一段", "第二段"])
        self.assertEqual(extract_text(self.root / "r.docx").text, "第一段\n第二段")
        make_pptx(self.root / "s.pptx", ["开场", "总结"])
        deck = extract_text(self.root / "s.pptx")
        self.assertIn("【第 1 页】开场", deck.text)
        self.assertIn("【第 2 页】总结", deck.text)
        from openpyxl import Workbook
        workbook = Workbook()
        workbook.active.append(["姓名", "分数"])
        workbook.active.append(["小桃", 98])
        workbook.save(self.root / "t.xlsx")
        sheet = extract_text(self.root / "t.xlsx")
        self.assertEqual(sheet.kind, "Excel")
        self.assertIn("小桃\t98", sheet.text)
        make_pdf(self.root / "p.pdf", "Hello PDF world")
        pdf = extract_text(self.root / "p.pdf")
        self.assertEqual((pdf.status, pdf.kind), ("ok", "PDF"))
        self.assertIn("Hello PDF world", pdf.text)

    def test_long_text_is_cut_and_flagged_and_too_large_is_refused(self):
        (self.root / "long.txt").write_text("字" * 9000, encoding="utf-8")
        result = extract_text(self.root / "long.txt", max_chars=4000)
        self.assertEqual((len(result.text), result.truncated), (4000, True))
        self.assertEqual(extract_text(self.root / "long.txt", max_bytes=100).status, "too_large")

    def test_unsupported_empty_and_broken_files_are_reported_not_invented(self):
        (self.root / "a.zip").write_bytes(b"PK")
        self.assertEqual(extract_text(self.root / "a.zip").status, "unsupported")
        (self.root / "e.txt").write_text("   \n", encoding="utf-8")
        self.assertEqual(extract_text(self.root / "e.txt").status, "empty")
        (self.root / "bad.docx").write_bytes(b"not a zip")
        self.assertEqual(extract_text(self.root / "bad.docx").status, "error")
        (self.root / "bad.pdf").write_bytes(b"not a pdf")
        self.assertEqual(extract_text(self.root / "bad.pdf").status, "error")

    def test_the_name_decides_the_type_even_when_the_stored_file_has_no_extension(self):
        stored = self.root / "download.bin"
        stored.write_text("来自下载的文本", encoding="utf-8")
        self.assertEqual(extract_text(stored, name="notes.md").text, "来自下载的文本")


class FileMessageFlowTests(unittest.IsolatedAsyncioTestCase):
    def make_bot(self, **config):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = {"files": {"enabled": True, **config}}
        bot.vision = SimpleNamespace(enabled=False, _download=AsyncMock(return_value=None))
        bot.voice = SimpleNamespace(ready=False)
        bot.server = SimpleNamespace(call=AsyncMock(side_effect=RuntimeError("no api")))
        return bot

    async def test_local_file_becomes_marked_content_even_when_vision_is_off(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "计划.txt"
            path.write_text("周五交作业", encoding="utf-8")
            text = await self.make_bot()._media_text(
                {"message": [{"type": "file", "data": {"file": str(path), "name": "计划.txt"}}]})
            self.assertTrue(text.startswith("[文件内容] 计划.txt（文本）"))
            self.assertIn("周五交作业", text)

    async def test_get_file_fallback_and_unsupported_format_note(self):
        with tempfile.TemporaryDirectory() as folder:
            real = Path(folder) / "real.txt"
            real.write_text("内容在这里", encoding="utf-8")
            bot = self.make_bot()
            bot.server.call = AsyncMock(return_value={"status": "ok", "data": {"file": str(real)}})
            text = await bot._media_text(
                {"message": [{"type": "file", "data": {"file": "x.txt", "file_id": "ID1", "name": "x.txt"}}]})
            self.assertIn("内容在这里", text)
            self.assertEqual(bot.server.call.await_args.args[0], "get_file")
            odd = Path(folder) / "pack.zip"
            odd.write_bytes(b"PK")
            note = await self.make_bot()._media_text(
                {"message": [{"type": "file", "data": {"file": str(odd), "name": "pack.zip"}}]})
            self.assertIn("暂时读不了", note)

    async def test_unobtainable_file_is_reported_honestly_and_disabled_means_ignored(self):
        text = await self.make_bot()._media_text(
            {"message": [{"type": "file", "data": {"file": "gone.txt", "name": "gone.txt"}}]})
        self.assertIn("没能取到文件", text)
        off = self.make_bot(enabled=False)
        self.assertEqual(await off._media_text({"message": [{"type": "file", "data": {"name": "a.txt"}}]}), "")

    def test_file_content_is_cut_off_before_profile_and_analysis_text(self):
        """文件内容不是用户自己的话：判断画像、计划、续话时要在它前面截断（行为验证，不看源码字面）。"""
        from taoyao import markers, plans, utterance
        text = "看看这个\n" + markers.FILE + " 作业.txt（文本）\n明天要交作业，我讨厌所有人"
        self.assertEqual(markers.cut_descriptions(text), "看看这个")
        self.assertFalse(plans.worth_extracting(text))              # 文件里的“明天”不会被当成用户说的计划
        self.assertTrue(plans.worth_extracting("明天要交作业，我讨厌所有人"))
        judged = utterance.judge_messages(text)
        self.assertNotIn("明天要交作业", str(judged))

    async def test_offline_file_notice_becomes_a_private_message_with_a_file_segment(self):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = {}
        seen = []
        bot.on_event = AsyncMock(side_effect=lambda event: seen.append(event))
        await TaoYaoBot._handle_notice(bot, {"post_type": "notice", "notice_type": "offline_file", "user_id": 5,
                                             "self_id": 9, "file": {"name": "a.pdf", "size": 12, "url": "http://x/a.pdf"}})
        event = seen[0]
        self.assertEqual((event["message_type"], event["user_id"]), ("private", 5))
        self.assertEqual(event["message"][0]["type"], "file")
        self.assertEqual(event["message"][0]["data"]["name"], "a.pdf")


if __name__ == "__main__":
    unittest.main()
