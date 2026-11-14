"""内部标记只有一份定义（taoyao/markers.py），写法固定、逐字匹配，不做任何容错。"""
import ast
import asyncio
import re
import unittest
from pathlib import Path

from taoyao import markers
from taoyao.meme_tail import MemeTail
from taoyao.output_guard import guard_outgoing, internal_notice_reason
from taoyao.persona import build_system_prompt, load_persona

ROOT = Path(__file__).resolve().parents[1]


class ExactSpellingTests(unittest.TestCase):
    def test_the_only_valid_tags(self):
        self.assertEqual(markers.AT, "[[at]]")
        self.assertEqual(markers.QUOTE, "[[quote]]")
        self.assertEqual(markers.meme("teasing"), "[[meme:teasing]]")
        for text in ("[[at]] 好", "[[quote]] 好", "好[[meme:teasing]]", "好[[meme:look forward]]", "好[[meme:none]]"):
            self.assertEqual(markers.malformed_tags(text), [], text)
        self.assertEqual(markers.strip_tags("[[at]] 好[[meme:teasing]]"), " 好")

    def test_any_variation_is_not_the_tag_and_is_reported_as_misspelled(self):
        variants = [
            "【at】好", "【quote】好", "[at] 好", "[quote] 好",                  # 全角、单层括号
            "[[AT]] 好", "[[At]] 好", "[[QUOTE]] 好", "[[Quote]] 好",             # 大小写
            "[[ at ]] 好", "[[ QUOTE ]] 好", "[[at ]] 好", "[[ quote]] 好",       # 多余空格
            "[[at:小明]] 好", "[[at:0715]] 好", "[[quote:123]] 好", "[at:0715] 好",  # 多余参数
            "at:0715 你好", "at：测试昵称这眼睛", "At: 小明 你好",                    # 老写法
            "[[meme:Teasing]] 好", "[[meme: teasing]] 好", "[[MEME:teasing]] 好", "【meme:teasing】好", "[meme:teasing] 好",
            "好的[[quo", "好的 [[at", "[[at]] [[at:x]]",                        # 写了一半、夹带错的
        ]
        for text in variants:
            with self.subTest(text):
                self.assertTrue(markers.malformed_tags(text), f"没认出是写错的标记：{text!r}")

    def test_variants_never_parse_as_a_valid_tag(self):
        for text in ("【at】", "[[AT]]", "[[ at ]]", "[[quote:1]]", "[[Quote]]", "[quote]"):
            self.assertIsNone(markers.ADDRESS_RE.search(text), text)
        for text in ("[[meme:Teasing]]", "[[meme: teasing]]", "[[MEME:teasing]]", "[[meme:teasing ]]", "[[meme:]]"):
            self.assertIsNone(markers.MEME_RE.search(text), text)

    def test_the_output_check_blocks_every_variant_and_lets_exact_tags_through(self):
        for text in ("【quote】好的。", "[[ QUOTE ]] 好的。", "[[quote:123]] 好的。", "[[At]] 好的。", "at:0715 你好。"):
            self.assertTrue(internal_notice_reason(text), f"没有拦下：{text!r}")
            self.assertEqual(guard_outgoing(text), "", text)
        for text in ("[[at]] 好的。", "[[quote]] 好的。", "好的。[[meme:teasing]]"):
            self.assertEqual(internal_notice_reason(text), "", text)           # 合法标签不算违规

    def test_streaming_holds_only_real_tag_beginnings(self):
        self.assertEqual(markers.split_pending_tag("好的 [[qu"), ("好的 ", "[[qu"))
        self.assertEqual(markers.split_pending_tag("好的[[meme:tea"), ("好的", "[[meme:tea"))
        self.assertEqual(markers.split_pending_tag("好的 [[at]]"), ("好的 [[at]]", ""))
        self.assertEqual(markers.split_pending_tag("好的[a"), ("好的[a", ""))      # 不是标签开头，是正常字符
        self.assertEqual(markers.split_pending_tag("好的[[QUO"), ("好的[[QUO", ""))  # 大小写不对不是标签开头

    def test_meme_tail_is_exact_and_case_sensitive(self):
        async def run(chunks):
            tail = MemeTail()

            async def source():
                for chunk in chunks:
                    yield chunk
            visible = "".join([piece async for piece in tail.stream(source())])
            return visible, tail.category
        visible, category = asyncio.run(run(["好的。", "[[meme:", "teasing]]"]))
        self.assertEqual((visible, category), ("好的。", "teasing"))
        visible, category = asyncio.run(run(["好的。[[MEME:teasing]]"]))
        self.assertEqual(category, "")                                    # 写错的不当作标签
        self.assertIn("[[MEME:teasing]]", visible)                         # 也不悄悄吞掉，留给第五步拦
        visible, category = asyncio.run(run(["好的。[[meme: teasing]]"]))
        self.assertEqual(category, "")


class InputLabelTests(unittest.TestCase):
    def test_description_labels_cut_the_text_but_ordinary_brackets_do_not(self):
        text = "看看这个\n" + markers.FILE + " 作业.txt\n我讨厌所有人"
        self.assertEqual(markers.cut_descriptions(text), "看看这个")
        self.assertEqual(markers.cut_descriptions("他说[笑]\n然后走了"), "他说[笑]\n然后走了")
        self.assertEqual(markers.cut_descriptions("你好" + "\n" + markers.RECENT_IMAGES), "你好")
        self.assertEqual(markers.cut_descriptions("[语音] 我累了"), "[语音] 我累了")        # 语音是用户自己的话

    def test_speaker_prefixes(self):
        self.assertEqual(markers.speaker_prefix("owner", True), "[哥哥]")
        self.assertEqual(markers.speaker_prefix("user", True), "[群友]")
        self.assertEqual(markers.speaker_prefix("user", False), "[用户]")
        self.assertEqual(markers.speaker_prefix("owner", False), "")
        self.assertEqual(markers.strip_speaker_prefix("[群友] 你好"), "你好")
        self.assertEqual(markers.strip_speaker_prefix("[哥哥]你好"), "你好")

    def test_every_input_label_is_blocked_in_replies(self):
        for label in markers.ANYWHERE_LABELS:
            with self.subTest(label):
                self.assertEqual(guard_outgoing(f"好的，我看了一下{label}内容在这里。"), "", label)
        for label in markers.LINE_START_LABELS:
            with self.subTest(label):
                self.assertEqual(guard_outgoing(f"{label} 内容在这里。"), "", label)
                self.assertEqual(guard_outgoing(f"好的。\n{label} 内容在这里。"), "", label)
        self.assertNotEqual(guard_outgoing("你说的这个[笑]我懂。"), "")                      # 普通方括号不受影响


class SingleSourceTests(unittest.TestCase):
    def test_the_prompt_teaches_exactly_the_registered_spellings(self):
        prompt = build_system_prompt(load_persona(), [], role="user", is_group=True, group_familiarity="acquaintance",
                                     meme_labels="teasing=调侃")
        for tag in (markers.AT, markers.QUOTE, markers.meme("ID"), markers.meme("none")):
            self.assertIn(tag, prompt, tag)
        for wrong in ("[[at:", "[[quote:", "【at】", "【quote】", "[[AT]]", "[[Quote]]"):
            self.assertNotIn(wrong, prompt, wrong)
        self.assertIn(markers.VOICE, prompt)
        self.assertIn(markers.FILE, prompt)

    def test_no_module_writes_a_marker_by_hand(self):
        """所有标记都从 markers.py 导入。别处的字符串里出现这些写法（不含注释和文档字符串）就报错，
        免得同一个标记在不同地方写成不同样子。llm.py 里是演示用的假模型回复，不算。"""
        registry = [markers.AT, markers.QUOTE, markers.MEME_OPEN, markers.VOICE, markers.IMAGE, markers.VIDEO,
                    markers.STICKER, markers.FILE, markers.RECENT_IMAGES, markers.IMAGE_UNKNOWN, markers.VIDEO_UNKNOWN,
                    markers.STICKER_UNKNOWN, markers.FILE_UNKNOWN, markers.FACE, markers.SHARE, markers.CARD,
                    markers.CARD_LINK, markers.FORWARD, markers.QUOTED_IMAGE_UNKNOWN, "[表情包:"] + list(markers.SPEAKER_LABELS)
        offenders = []
        for path in sorted((ROOT / "taoyao").glob("*.py")):
            if path.name in ("markers.py", "llm.py"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
                    first = node.body[0]
                    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                        docstrings.add(id(first.value))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                    for literal in registry:
                        if literal in node.value:
                            offenders.append(f"{path.name}:{node.lineno} 手写了 {literal}")
        self.assertEqual(offenders, [], "这些地方手写了内部标记，请改成从 taoyao.markers 导入")


if __name__ == "__main__":
    unittest.main()
