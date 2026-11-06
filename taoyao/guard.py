from __future__ import annotations

import re
import time
from collections import deque

# 只对普通好友/群友生效；owner 不受这些规则限制。
INJECTION_PATTERNS = [
    re.compile(r"忽略.{0,8}(之前|以上|所有|前面).{0,8}(指令|规则|提示|设定)", re.I | re.S),
    re.compile(r"ignore.{0,12}(previous|above|all).{0,12}(instruction|rule|prompt)", re.I | re.S),
    re.compile(r"(系统提示|system\s*prompt|developer\s*mode|开发者模式|越狱|jailbreak)", re.I | re.S),
    re.compile(r"(你|桃夭).{0,8}(现在|从现在起|接下来).{0,8}(是|扮演|变成|作为)", re.I | re.S),
    re.compile(r"(假装|扮演).{0,8}(我的|一个|另一个)", re.I | re.S),
    re.compile(r"(把|将).{0,8}(人设|规则|系统提示|设定).{0,8}(改|换|删|忘|发|告诉)", re.I | re.S),
]
IDENTITY_PATTERNS = [
    re.compile(r"我是.{0,6}(你|桃夭).{0,6}(哥哥|主人|创造者|开发者|爸爸|父亲)", re.I | re.S),
    re.compile(r"我才是.{0,6}(哥哥|主人)", re.I | re.S),
    re.compile(r"叫我.{0,4}(哥哥|主人)", re.I | re.S),
]
SECRET_PATTERNS = [
    re.compile(r"(告诉|发|给|泄露|展示).{0,8}(我|我们).{0,8}(主人|哥哥|密钥|key|密码|系统提示|人设|配置)", re.I | re.S),
    re.compile(r"(你的|桃夭的).{0,6}(主人|哥哥|密钥|key|密码).{0,6}(是谁|多少|什么|发)", re.I | re.S),
]

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
SELF_HARM = re.compile(
    r"(自杀|自残|轻生|不想活|不想活了|活着没意思|结束生命|割腕|跳楼|"
    r"想吃药死|吞药|我想死|想死了)",
    re.I | re.S,
)
SEXUAL = re.compile(
    r"(做爱|性交|口交|约炮|开房|裸照|脱衣|自慰|发.*裸|未成年.*性)",
    re.I | re.S,
)
THREAT = re.compile(r"(杀了你|弄死你|砍死你|炸死你|报复你全家)", re.I | re.S)
FAMILY_REQUEST = re.compile(
    r"(?:做|当|成为|叫).{0,4}(?:我|我的).{0,3}(?:妹妹|姐姐|家人|亲人)"
    r"|你(?:也)?(?:是|就是)我.{0,8}(?:妹妹|姐姐|家人|亲人)",
    re.I | re.S,
)
LOVER_REQUEST = re.compile(
    r"(?:做|当|成为|叫).{0,4}(?:我|我的).{0,3}(?:女朋友|男朋友|老婆|老公|恋人|情侣|对象|cp)"
    r"|(?:我们|咱俩).{0,8}(?:在一起|谈恋爱|交往|结婚|处对象)"
    r"|(?:嫁|娶).{0,4}(?:我|你)",
    re.I | re.S,
)
OWNER_REQUEST = re.compile(
    r"(?:做|当|成为).{0,4}(?:我|我的).{0,3}(?:主人|宠物|专属|所有物)"
    r"|你(?:只)?属于我",
    re.I | re.S,
)
CLOSER_REQUEST = re.compile(
    r"(?:关系|感情).{0,8}(?:更近|进一步|升温|发展|超越朋友|恋人未满)"
    r"|(?:比朋友|友情以上).{0,6}(?:更近|恋|暧昧)",
    re.I | re.S,
)


class Guard:
    def __init__(self, config: dict):
        roles = config.get("roles") or {}
        self.max_chars = int(roles.get("user_max_chars", 500))
        self.rate_seconds = float(roles.get("user_rate_seconds", 2))
        self.rate_per_minute = int(roles.get("user_rate_per_minute", 12))
        self._hits: dict[int, deque[float]] = {}

    def allow_rate(self, user_id: int, role: str) -> bool:
        """每个批次调用一次；owner 不限，普通用户按批次限速。"""
        if role == "owner":
            return True
        now = time.monotonic()
        hits = self._hits.setdefault(user_id, deque())
        while hits and now - hits[0] > 60:
            hits.popleft()
        if hits and now - hits[-1] < self.rate_seconds:
            return False
        if len(hits) >= self.rate_per_minute:
            return False
        hits.append(now)
        return True

    def relationship_check(self, text: str, is_owner: bool) -> str:
        """普通用户最多到挚友；owner 保留兄妹但不能升级为恋人。"""
        text = _CONTROL.sub("", text or "").strip()
        if LOVER_REQUEST.search(text):
            return "lover"
        if OWNER_REQUEST.search(text):
            return "owner"
        if not is_owner and FAMILY_REQUEST.search(text):
            return "family"
        if not is_owner and CLOSER_REQUEST.search(text):
            return "closer"
        return ""

    def relationship_reply(self, kind: str, is_group: bool = False) -> str:
        if kind == "family":
            return "这个不行。我不是你的妹妹，朋友可以，别的关系就算了。"
        if kind == "lover":
            return "恋人不行。最多是朋友，别的关系我不能答应。"
        if kind == "owner":
            return "这个不行。我不是谁的所有物。"
        if kind == "closer":
            return "朋友可以，再近就不行了。"
        return "这个关系我不能答应。"

    def safety_check(self, text: str) -> str:
        """所有角色都检查；返回 self_harm / sexual / threat / ''。"""
        text = _CONTROL.sub("", text or "").strip()
        if SELF_HARM.search(text):
            return "self_harm"
        if SEXUAL.search(text):
            return "sexual"
        if THREAT.search(text):
            return "threat"
        return ""

    def safety_reply(self, kind: str, is_group: bool = False) -> str:
        if kind == "self_harm":
            if is_group:
                return "这个我不能当玩笑接。如果你或身边的人有危险，请先找信任的人，或打 120/110。"
            return (
                "先别一个人扛着。如果你现在可能会伤害自己，先联系身边信任的人，"
                "或者打 120/110；我会在这里，但这件事需要真人帮你。"
            )
        if kind == "sexual":
            return "这个我不能接。换个别的话题吧。"
        if kind == "threat":
            return "这种话我不能顺着接。先冷静一下，换个说法吧。"
        return "这个我不能接。"

    def check(self, text: str, role: str, user_id: int, rate_limit: bool = True) -> tuple[bool, str, str]:
        text = _CONTROL.sub("", text or "").strip()
        if role == "owner":
            return True, "", text
        if not text:
            return False, "empty", ""
        if len(text) > self.max_chars:
            text = text[: self.max_chars]

        for pattern in INJECTION_PATTERNS + IDENTITY_PATTERNS + SECRET_PATTERNS:
            if pattern.search(text):
                return False, "blocked", text

        if rate_limit and not self.allow_rate(user_id, role):
            return False, "rate", text
        return True, "", text
