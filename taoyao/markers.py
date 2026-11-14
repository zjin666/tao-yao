"""所有内部标记的唯一定义。

内部标记是给程序用的“代码名词”，不是给人读的话：写法固定，只认逐字匹配——区分大小写、只认半角方括号、
不允许多余的空格或参数。差一个字符就不是这个标记，而是“写错的标记”，由第五步（发送前检查）当作违规拦下，
带着修正说明重新生成一次。不要在别处另写这些字符串或正则，也不要做任何容错识别，一律从这里导入。

两类标记：
  1. 模型写在回复里、由程序读取并在发送前去掉的标签：[[at]]  [[quote]]  [[meme:ID]]
  2. 程序写进给模型的输入里的标注：[语音] [图片内容] [视频内容] [表情包内容] [文件内容] [最近图片记忆]
     和发言人前缀 [群友] [哥哥] [用户] [群聊] [私聊]。它们只在输入里出现，绝不能出现在发给用户的回复里。
"""
from __future__ import annotations

import re

# ---------- 1. 回复里的标签 ----------
AT = "[[at]]"                      # 只是点名对象：发送时 @ 对方
QUOTE = "[[quote]]"                # 回应对方那句话的具体内容：发送时引用那条消息
MEME_OPEN = "[[meme:"              # [[meme:ID]]：配一张表情，ID 是表情分类名；不配写 [[meme:none]]
MEME_CLOSE = "]]"
MEME_ID = r"[a-z][a-z0-9_]*(?: [a-z0-9_]+)*"   # 全小写，单词之间最多一个空格（如 look forward）

AT_RE = re.compile(re.escape(AT))
QUOTE_RE = re.compile(re.escape(QUOTE))
ADDRESS_RE = re.compile(re.escape(AT) + "|" + re.escape(QUOTE))    # 两者都表示“要点明对象”
MEME_RE = re.compile(re.escape(MEME_OPEN) + "(" + MEME_ID + ")" + re.escape(MEME_CLOSE))
TAG_PREFIXES = (AT, QUOTE, MEME_OPEN)


def meme(category: str) -> str:
    return f"{MEME_OPEN}{category}{MEME_CLOSE}"


def split_pending_tag(text: str) -> tuple[str, str]:
    """流式输出时，末尾可能是某个标签写了一半（如 "[[qu"）：返回 (可以先发的部分, 要扣住等下一块的部分)。
    只扣住确实是某个标签开头的那一截；"[" 之类单独的字符是正常字符。"""
    longest = 0
    for tag in TAG_PREFIXES:
        for size in range(min(len(text), len(tag) - 1), 0, -1):
            if tag.startswith(text[-size:]):
                longest = max(longest, size)
                break
    # 完整的 [[meme: 已经开始、但 ID 还没写完：整段扣住直到闭合
    open_meme = text.rfind(MEME_OPEN)
    if open_meme >= 0 and MEME_CLOSE not in text[open_meme:]:
        return text[:open_meme], text[open_meme:]
    return (text[:-longest], text[-longest:]) if longest else (text, "")


def strip_tags(text: str) -> str:
    """去掉合法的标签（只认上面那三种）。"""
    return MEME_RE.sub("", ADDRESS_RE.sub("", text or ""))


# 看起来像标签、但写法不对的东西：全角括号、单层括号、大小写不对、多余空格或参数、at:昵称 之类的老写法
_TAG_LIKE = re.compile(r"[\[【]{1,2}[ \t]*(?:at|quote|meme)(?![A-Za-z])[^\]】\n]{0,60}[\]】]{1,2}", re.IGNORECASE)
_LEGACY_AT = re.compile(r"(?:^|\n)[ \t]*at[ \t]*[:：]", re.IGNORECASE)


def malformed_tags(text: str) -> list[str]:
    """写错的标记（不含合法标签）。有就说明模型没按固定写法来，要当作违规重新生成，而不是猜它的意思。"""
    rest = strip_tags(text)
    found = [m.group(0) for m in _TAG_LIKE.finditer(rest)]
    if "[[" in rest or "]]" in rest:                      # 去掉合法标签后还剩双括号：写了一半或写错了
        found.append("[[/]]")
    if _LEGACY_AT.search(rest):
        found.append("at:")
    return found


# ---------- 2. 输入里的标注 ----------
VOICE = "[语音]"                   # 语音转成的文字，算对方自己的话
IMAGE = "[图片内容]"
VIDEO = "[视频内容]"
STICKER = "[表情包内容]"
FILE = "[文件内容]"
RECENT_IMAGES = "[最近图片记忆]"
# 兜底标注（识别不了时）
IMAGE_UNKNOWN = "[图片]"
VIDEO_UNKNOWN = "[视频]"
STICKER_UNKNOWN = "[表情包]"
FILE_UNKNOWN = "[文件]"
VOICE_UNKNOWN = "[语音]"

# 消息里其他内容的占位和标注（程序拼进输入文字里）
FACE = "[表情]"                    # QQ 小黄脸
SHARE = "[分享]"
CARD = "[卡片]"
CARD_LINK = "[卡片链接]"
FORWARD = "[转发内容]"
QUOTED_IMAGE_UNKNOWN = "[图片或表情包]"      # 被引用的图片还没识别出来时的占位


def sticker_named(summary: str) -> str:
    """QQ 表情包消息自带的名字：[表情包:名字]"""
    return f"[表情包:{summary}]"


SPEAKER_GROUP_MEMBER = "[群友]"
SPEAKER_OWNER = "[哥哥]"
SPEAKER_USER = "[用户]"
SPEAKER_GROUP = "[群聊]"
SPEAKER_PRIVATE = "[私聊]"
SPEAKER_LABELS = (SPEAKER_GROUP_MEMBER, SPEAKER_OWNER, SPEAKER_USER, SPEAKER_GROUP, SPEAKER_PRIVATE)

# 这些标注后面跟的是识别出来的内容，不是用户自己的话：判断画像、六维、计划时要在它们之前截断
DESCRIPTION_LABELS = (RECENT_IMAGES, IMAGE, VIDEO, STICKER, FILE)
CONTENT_LABELS = DESCRIPTION_LABELS + (VOICE,)

_DESCRIPTION_CUT = re.compile(r"\n(?:" + "|".join(re.escape(label) for label in DESCRIPTION_LABELS) + ")")


def cut_descriptions(text: str) -> str:
    """去掉消息后面附带的图片/视频/表情包/文件描述，只留用户自己写的话。"""
    return _DESCRIPTION_CUT.split(str(text or ""), maxsplit=1)[0]


def label_names(labels: tuple[str, ...]) -> str:
    """给正则用：把 [图片内容] 这类标注的方括号去掉，用 | 连起来。"""
    return "|".join(re.escape(label[1:-1]) for label in labels)


def speaker_prefix(role: str, is_group: bool) -> str:
    if is_group:
        return SPEAKER_OWNER if role == "owner" else SPEAKER_GROUP_MEMBER
    return SPEAKER_USER if role == "user" else ""


# 所有程序拼进输入的标注（含占位），输出检查要拦：媒体和卡片类在任何位置都不行，其余只在行首查
ANYWHERE_LABELS = DESCRIPTION_LABELS + (SHARE, CARD_LINK, CARD, FORWARD)
LINE_START_LABELS = (VOICE, IMAGE_UNKNOWN, VIDEO_UNKNOWN, STICKER_UNKNOWN, FILE_UNKNOWN, QUOTED_IMAGE_UNKNOWN) + SPEAKER_LABELS
SPEAKER_PREFIX_RE = re.compile(r"^(?:" + "|".join(re.escape(label) for label in (SPEAKER_GROUP_MEMBER, SPEAKER_OWNER)) + r")\s*")


def strip_speaker_prefix(text: str) -> str:
    """去掉历史里 [群友]/[哥哥] 这样的发言人前缀。"""
    return SPEAKER_PREFIX_RE.sub("", text or "", count=1)


def label_leak_pattern() -> str:
    """输出检查用：回复里不该出现的输入标注。"""
    anywhere = "|".join(re.escape(label) for label in ANYWHERE_LABELS)
    line_start = "|".join(re.escape(label) for label in LINE_START_LABELS)
    return rf"(?:{anywhere})|(?:^|\n)\s*(?:{line_start})"
