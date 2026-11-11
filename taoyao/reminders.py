"""提醒、定时任务、生日和纪念日的纯逻辑：把“规则”算成下一次触发时间、校验模型解析出的结果。不碰数据库和网络。

时间都按对方的时区算（zoneinfo），库里存 UTC 的 epoch 秒。重复规则：
- once     只提醒一次，规则里是本地时间 "YYYY-MM-DD HH:MM"
- daily    每天 hour:minute
- weekly   每周 weekday（周一=0）hour:minute
- monthly  每月 day 号 hour:minute（没有这一天的月份取月底）
- yearly   每年 month 月 day 日 hour:minute（2 月 29 日在平年取 2 月 28 日）
"""
from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

REPEATS = ("none", "daily", "weekly", "monthly", "yearly")
KINDS = ("reminder", "birthday", "date")
WEEKDAYS = "一二三四五六日"
MAX_CONTENT = 200
DEFAULT_HOUR = {"reminder": 9, "birthday": 9, "date": 9}


def _zone(tz: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz)
    except Exception:
        return ZoneInfo("Asia/Shanghai")


def _clamp_day(year: int, month: int, day: int) -> int:
    return min(day, calendar.monthrange(year, month)[1])


def _at(zone: ZoneInfo, year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, _clamp_day(year, month, day), hour, minute, tzinfo=zone)


def next_occurrence(repeat: str, spec: dict, tz: str, after: float) -> float | None:
    """严格晚于 after（epoch 秒）的下一次触发时间；once 在已过去时返回 None。"""
    zone = _zone(tz)
    base = datetime.fromtimestamp(after, zone)
    hour = int(spec.get("hour", 9))
    minute = int(spec.get("minute", 0))
    if repeat == "none":
        try:
            when = datetime.strptime(str(spec.get("at")), "%Y-%m-%d %H:%M").replace(tzinfo=zone)
        except (TypeError, ValueError):
            return None
        return when.timestamp() if when.timestamp() > after else None
    if repeat == "daily":
        cand = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return (cand if cand.timestamp() > after else cand + timedelta(days=1)).timestamp()
    if repeat == "weekly":
        weekday = int(spec.get("weekday", 0)) % 7
        cand = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        cand += timedelta(days=(weekday - cand.weekday()) % 7)
        return (cand if cand.timestamp() > after else cand + timedelta(days=7)).timestamp()
    if repeat == "monthly":
        day = int(spec.get("day", 1))
        year, month = base.year, base.month
        for _ in range(3):
            cand = _at(zone, year, month, day, hour, minute)
            if cand.timestamp() > after:
                return cand.timestamp()
            month += 1
            if month > 12:
                year, month = year + 1, 1
        return None
    if repeat == "yearly":
        month, day = int(spec.get("month", 1)), int(spec.get("day", 1))
        for year in (base.year, base.year + 1, base.year + 2):
            cand = _at(zone, year, month, day, hour, minute)
            if cand.timestamp() > after:
                return cand.timestamp()
    return None


def validate(parsed: dict, now: float, tz: str) -> tuple[dict | None, str]:
    """检查模型解析出的“创建提醒”结果。返回（规范化后的条目，问题说明）；有问题时条目为 None。"""
    kind = parsed.get("kind") if parsed.get("kind") in KINDS else "reminder"
    repeat = parsed.get("repeat") if parsed.get("repeat") in REPEATS else "none"
    content = re.sub(r"\s+", " ", str(parsed.get("content") or "")).strip()[:MAX_CONTENT]
    subject = re.sub(r"\s+", " ", str(parsed.get("subject") or "")).strip()[:60]

    def num(key, lo, hi, default=None):
        try:
            value = int(parsed.get(key))
        except (TypeError, ValueError):
            return default
        return value if lo <= value <= hi else default

    hour = num("hour", 0, 23, DEFAULT_HOUR[kind])
    minute = num("minute", 0, 59, 0)
    spec: dict = {"hour": hour, "minute": minute}
    if kind in ("birthday", "date"):
        month, day = num("month", 1, 12), num("day", 1, 31)
        if month is None or day is None:
            return None, "没有说清是几月几号"
        if day > calendar.monthrange(2024, month)[1]:        # 闰年里都不存在的日子，比如 2 月 30 日
            return None, "这个日期不存在"
        repeat, spec = "yearly", {"hour": hour, "minute": minute, "month": month, "day": day}
        subject = subject or ("我的生日" if kind == "birthday" else "纪念日")
        content = content or subject
    else:
        if not content:
            return None, "没有说清要提醒什么"
        if repeat == "none":
            at = str(parsed.get("at") or "").strip()
            try:
                datetime.strptime(at, "%Y-%m-%d %H:%M")
            except ValueError:
                return None, "没有说清是什么时候"
            spec = {"at": at}
        elif repeat == "weekly":
            weekday = num("weekday", 0, 6)
            if weekday is None:
                return None, "没有说清是每周几"
            spec["weekday"] = weekday
        elif repeat == "monthly":
            day = num("day", 1, 31)
            if day is None:
                return None, "没有说清是每月几号"
            spec["day"] = day
    due = next_occurrence(repeat, spec, tz, now + 5)
    if due is None:
        return None, "这个时间已经过去了"
    lead = num("lead_days", 0, 30, 3 if kind == "date" else 0) if kind == "date" else 0
    return {"kind": kind, "repeat": repeat, "spec": spec, "content": content, "subject": subject,
            "due_at": due, "lead_days": lead, "tz": tz}, ""


def self_birthday(parsed: dict) -> bool:
    """“我的生日”这类是自己的，当天祝福；别人的（妈妈生日）才提前提醒。"""
    subject = str(parsed.get("subject") or "我的生日")
    return bool(re.match(r"^(我|自己|本人)", subject)) or "我的" in subject[:3]


def describe(item: dict, tz: str = "Asia/Shanghai") -> str:
    """一条提醒的简短说明，给模型当事实、也用于列表。不含内部编号以外的敏感信息。"""
    zone = _zone(item.get("tz") or tz)
    due = datetime.fromtimestamp(float(item["due_at"]), zone)
    when = due.strftime("%Y-%m-%d %H:%M")
    spec = item.get("spec") or {}
    repeat = item.get("repeat")
    if repeat == "daily":
        rule = f"每天 {spec.get('hour', 0):02d}:{spec.get('minute', 0):02d}"
    elif repeat == "weekly":
        rule = f"每周{WEEKDAYS[int(spec.get('weekday', 0)) % 7]} {spec.get('hour', 0):02d}:{spec.get('minute', 0):02d}"
    elif repeat == "monthly":
        rule = f"每月{spec.get('day')}号 {spec.get('hour', 0):02d}:{spec.get('minute', 0):02d}"
    elif repeat == "yearly":
        rule = f"每年{spec.get('month')}月{spec.get('day')}日"
    else:
        rule = "一次"
    label = item.get("subject") or item.get("content")
    lead = f"，提前{item['lead_days']}天提醒" if item.get("lead_days") else ""
    return f"{label}（{rule}{lead}，下次 {when}）"


def parse_prompt(now_local: datetime, tz: str, active: list[str]) -> str:
    return (
        "你是提醒意图解析器，只输出 JSON，不要解释。格式："
        '{"action":"create|list|cancel|none","kind":"reminder|birthday|date","content":"","subject":"",'
        '"repeat":"none|daily|weekly|monthly|yearly","at":"","hour":0,"minute":0,"weekday":0,"day":0,"month":0,'
        '"lead_days":0,"index":0}。'
        f"现在是 {now_local.strftime('%Y-%m-%d %H:%M')}（周{WEEKDAYS[now_local.weekday()]}），用户时区 {tz}。"
        "只有用户“当前这条消息”明确要求设置提醒/闹钟/定时任务、记录生日或纪念日、查看自己的提醒、取消提醒时才返回对应 action；"
        "普通聊天、讨论、询问能不能做、回顾以前的提醒、否定（不用提醒了不算取消，除非明确指某条）一律 none。"
        "create：reminder=让桃夭到点提醒用户某件事；birthday=用户告诉桃夭他自己或别人的生日；date=纪念日或别人的重要日子。"
        "content 写提醒的事（如“交周报”“吃药”），不要带时间；生日/纪念日写 subject（如“我的生日”“妈妈的生日”“结婚纪念日”）。"
        "once（repeat=none）时 at 填具体的本地时间 YYYY-MM-DD HH:MM，由“现在”推算出来（“明天8点”“半小时后”“周五下午3点”都要算成具体日期）；"
        "每天/每周/每月/每年的提醒填 hour、minute，每周再填 weekday（周一=0……周日=6），每月填 day，生日/纪念日填 month、day。"
        "时间含糊（只说“晚上”“待会儿”）时按常识选一个合理的整点；实在无法确定就把 action 设为 none。"
        "lead_days 只用于别人的生日/纪念日：用户说了提前几天就填，没说填 3；自己的生日填 0。"
        "cancel：index 填要取消的提醒编号（对照下面的列表，用户可能说第几个、或说提醒的内容）；对应不上填 0。"
        "用户现有的提醒：\n" + ("\n".join(active) or "（没有）"))
