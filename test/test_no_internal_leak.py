"""桃夭发出去的回复里不能出现内部指令、标记、提示词分节、识图或判断用的原始格式。

两条真实的发送路径都要过：两段式（_prepare_reply_part）和最终发送前的清洗（guard_outgoing + sanitize_outgoing）。
"""
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import ReplyStreamer, TaoYaoBot, _clean_segment, sanitize_outgoing
from taoyao.output_guard import SECTION_TITLES, guard_outgoing
from taoyao.turn_timing import turn_timing_context
from taoyao.two_part import RawPartBlocked

ROOT = Path(__file__).resolve().parents[1]

# 每一项：模型可能写出来的、不该让用户看到的东西
LEAKS = {
    "标记 at": "[[at]] 你好呀",
    "标记 quote": "[[quote]] 这个问题我来答。",
    "标记 quote 全角": "【quote】好的。",
    "标记 大小写空格": "[[ QUOTE ]] 好的。",
    "标记 quote 带参数": "[[quote:123]] 好的。",
    "标记 at 带昵称": "[[at:小明]] 好的。",
    "标记 meme": "哼，才不是呢。[[meme:teasing]]",
    "标记 中途": "好的[[at]]，我知道了。",
    "语音标注": "[语音] 我听到你说今天很累。",
    "图片标注": "[图片内容] 一只猫趴在桌上。好可爱。",
    "表情包标注": "[表情包内容] 形象：猫；神态：哭。我懂你。",
    "文件标注": "[文件内容] 作业.txt（文本）\n周五交作业",
    "视频标注": "[视频内容] 画面里有一条鱼。",
    "最近图片记忆": "[最近图片记忆]\n- 发送者 1：一只猫",
    "历史前缀 群友": "[群友] 你们好呀。",
    "历史前缀 哥哥": "[哥哥] 在吗。",
    "历史前缀 用户": "[用户] 在吗。",
    "时间标签": "[11-13 周三 19:39] 你好呀。",
    "表情内部注记": "（内部参考，不要向对方提起：这张表情已经认识，不要当成第一次见到）好呀。",
    "识图原始格式": "形象：灰色圆脸猫；神态：流泪；可能想表达：委屈。我懂。",
    "搜索失败状态": "搜索暂时失败，尚未核实含义。",
    "搜索无结果状态": "没有查到可用搜索结果，未确认含义或出处。",
    "检索状态": "这是内部检索状态，不要复述。",
    "判断用 JSON": '{"decision":"approve","request_id":1,"evidence":"同意"}',
    "分类用 JSON": '{"category":"teasing"}',
    "内部备注": "内部备注：这个人是陌生人。",
    "给模型的指令": "给模型的指令：不要提起记忆。",
    "思考过程": "思考过程：先看上下文。",
    "CQ 码": "[CQ:at,qq=123] 你好。",
}
SECTION_SAMPLES = {
    "当前时间": "【当前时间】2026-11-13 20:30 星期五",
    "未发出的话": "【未发出的话】你对上一条消息本来准备好的回复",
    "群回复指向": "【群回复指向】消息对象备注供内部判断",
    "图片/视频回复": "【图片/视频回复】识别已完成",
    "聊天规则": "【聊天规则】中文QQ短句",
    "本轮答复意图": "【本轮答复意图】把申请的事实告诉哥哥",
    "联网搜索结果": "【联网搜索结果】查询：xxx",
    "六维的具体表达方向": "【六维的具体表达方向】ACT：轻微偏向",
}

# 经过清理或拦截之后，下面这些东西一个都不该留在可见文字里
SUSPECT = re.compile(
    r"\[\[|\]\]|\[CQ|quote|\[(?:语音|图片内容|表情包内容|文件内容|视频内容|群友|哥哥|用户)\]|最近图片记忆|\[\d\d-\d\d|"
    r"内部参考|形象：|神态：|可能想表达|搜索暂时|没有查到可用|内部检索|\"decision\"|\"category\"|内部备注|给模型的指令|思考过程|"
    r"【(?:" + SECTION_TITLES + r")】")


def make_bot():
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot.config = {}
    bot.llm = SimpleNamespace(chat=AsyncMock(side_effect=lambda m: m[-1]["content"].split("\n\n", 1)[-1]))
    return bot


class NoInternalLeakTests(unittest.IsolatedAsyncioTestCase):
    async def check_both_paths(self, name: str, raw: str):
        bot = make_bot()
        try:
            visible, _ = await bot._prepare_reply_part(raw, "group", 9)
        except RawPartBlocked:
            visible = ""
        self.assertIsNone(SUSPECT.search(visible), f"两段式路径漏出：{name} -> {visible!r}")
        final = _clean_segment(sanitize_outgoing(guard_outgoing(raw))) or ""
        self.assertIsNone(SUSPECT.search(final), f"最终发送路径漏出：{name} -> {final!r}")

    async def test_internal_markers_never_reach_the_user_on_either_path(self):
        for name, raw in {**LEAKS, **SECTION_SAMPLES}.items():
            with self.subTest(name):
                await self.check_both_paths(name, raw)

    async def test_ordinary_replies_are_not_swallowed_by_the_new_rules(self):
        for text in ("哼，才不是呢。", "刚才那个[表情]好好笑呀。", "你说的「图片内容」我没看懂。", "今天形象管理失败了，神态也不太好。",
                     "我觉得这个决定不错，原因是证据很充分。", "哥哥在吗？我在群友那边看到了。"):
            with self.subTest(text):
                bot = make_bot()
                visible, _ = await bot._prepare_reply_part(text, "group", 9)
                self.assertTrue(visible, f"正常回复被拦了：{text}")
                self.assertTrue(guard_outgoing(text), f"正常回复被最终检查拦了：{text}")

    async def test_tags_split_across_streamed_chunks_never_become_visible(self):
        @turn_timing_context
        async def run(chunks):
            streamer = ReplyStreamer(max_chars=70, max_segments=5)
            ready = []
            for chunk in chunks:
                ready += streamer.feed(chunk)
            ready += streamer.finish()
            return "".join(piece for piece, *_ in ready), streamer
        for chunks in (["[[", "quo", "te]] 好的。"], ["[", "[at", "]] 好的。"], ["好的[[mem", "e:teasing]]。"],
                       ["[[qu", "ote]]", " 好的。"], ["好的。[[me", "me:look for", "ward]]"]):
            with self.subTest(chunks):
                visible, _ = await run(chunks)
                self.assertIsNone(re.search(r"\[\[|\]\]|quote|meme|【|】", visible), f"漏出：{visible!r}")
                self.assertIn("好的", visible)

    async def test_misspelled_tags_are_not_swallowed_by_the_streamer_they_are_left_for_step_five(self):
        """写错的标记不猜：流式切句不会把它当标签吞掉，原样留在文字里，由第五步（发送前检查）当作违规拦下。"""
        @turn_timing_context
        async def run(chunks):
            streamer = ReplyStreamer(max_chars=70, max_segments=5)
            ready = []
            for chunk in chunks:
                ready += streamer.feed(chunk)
            ready += streamer.finish()
            return "".join(piece for piece, *_ in ready)
        for chunks in (["[[quote:12", "3]] 好的。"], ["【qu", "ote】好的。"], ["[[ at ]] 好的。"], ["[[AT]] 好的。"]):
            with self.subTest(chunks):
                visible = await run(chunks)
                self.assertTrue(guard_outgoing(visible) == "" or not visible, f"写错的标记没被检查拦下：{visible!r}")

    def test_every_prompt_section_title_in_the_source_is_guarded(self):
        """代码里给模型的每个【分节标题】都要在拦截清单里；新增分节没登记，这里会报错。"""
        pattern = re.compile(r"(?:^|[\"'\n]|\\n)【([^】\n]{2,14})】(?=\\n|\n|\{)")
        guard = re.compile(r"(?:" + SECTION_TITLES + r")")
        missing = set()
        for name in ("persona.py", "app.py", "reply_flow.py", "personality.py"):
            source = (ROOT / "taoyao" / name).read_text(encoding="utf-8")
            for title in pattern.findall(source):
                if not guard.fullmatch(title) and title not in ("标题",):
                    missing.add(f"{name}:{title}")
        self.assertEqual(missing, set(), "这些提示词分节标题没有登记到 output_guard.SECTION_TITLES")


if __name__ == "__main__":
    unittest.main()
