"""判断用户刚发的这句话是不是还没说完（意思可能分两条发）。

纯函数：提示词、是否值得判断、解析。判断由辅助模型和生成回复并行完成，只是内部分类，
结果只用来决定“要不要多等一会儿看同一个人有没有接着发”，不会直接生成任何面向用户的文字。
"""
from __future__ import annotations

import json
import re

MAX_JUDGE_CHARS = 80
JUDGE_PROMPT_HEAD = "判断用户这句话是不是明显还没说完"


def worth_judging(text: str, max_chars: int = MAX_JUDGE_CHARS) -> bool:
    """命令、空消息、很长的消息和纯符号不用判断；长消息通常已经是完整的意思。"""
    value = re.split(r"\n\[(?:最近图片记忆|图片内容|视频内容|表情包内容)", str(text or ""), maxsplit=1)[0].strip()
    if not value or value.startswith("/") or len(value) > max_chars:
        return False
    return bool(re.search(r"[\w一-鿿]", value))


def judge_messages(text: str) -> list[dict]:
    value = re.split(r"\n\[(?:最近图片记忆|图片内容|视频内容|表情包内容)", str(text or ""), maxsplit=1)[0].strip()
    return [
        {"role": "system", "content": (
            JUDGE_PROMPT_HEAD + "，下一条消息会接着补充。只输出JSON：{\"incomplete\":true} 或 {\"incomplete\":false}。\n"
            "只有话明显在中途断开才算没说完：以逗号、顿号、省略号结尾；以“但是、然后、而且、因为、所以、如果、还有、"
            "就是、比如、不过、其实、我觉得、那个”这类需要后文的词结尾；或者是缺了主干的半句"
            "（如“我昨天去了”“你觉得这个”）。\n"
            "完整的句子、问句、感叹、问候、“好的/嗯/哈哈”这类独立的短回应、只发了一个独立的词或表情，都算说完了。\n"
            "拿不准一律输出 {\"incomplete\":false}。")},
        {"role": "user", "content": value},
    ]


def parse_incomplete(raw: str) -> bool:
    """只有明确的 true 才算没说完；格式不对、缺字段、其他值一律当作说完了。"""
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        return False
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return False
    return isinstance(data, dict) and data.get("incomplete") is True
