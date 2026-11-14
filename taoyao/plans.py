"""近期计划记忆：对方亲口说过的、带日期的计划/安排/状态（“代码大后天才写”“下周都在出差”）。

和长期知识、画像不同：每条都带到期日，过期自动失效，之后不再注入提示。纯逻辑，不碰数据库和网络。
抽取由辅助模型在后台做；这里只负责“值不值得抽”“怎么问”“结果是否合法”“怎么展示给回复用的模型”。
"""
from __future__ import annotations

import re
from . import markers
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

WEEKDAYS = "一二三四五六日"
MAX_HORIZON_DAYS = 90        # 只记三个月内的安排
GRACE_DAYS = 2               # 日期过后还留两天，让桃夭能自然问一句“后来怎么样了”
MAX_TEXT = 80
MAX_ACTIVE = 10

# 粗筛：消息里要有时间或安排的痕迹才值得让模型看一眼，省得每句闲聊都调用模型。
_HINT = re.compile(
    r"(今天|今晚|明天|明晚|后天|大后天|昨天|下周|这周|本周|周[一二三四五六日天末]|星期|礼拜|周末|月底|月初|下个月|这个月|过几天|这几天|"
    r"最近|等会|待会|晚点|之后|以后才|才会|才能|才有|要到|得到|[0-9一二三四五六七八九十]+[号日]|[0-9一二三四五六七八九十]+点|"
    r"出差|考试|面试|开会|旅行|旅游|休假|放假|请假|生病|住院|搬家|加班|比赛|答辩|回家|出门)")


def worth_extracting(text: str) -> bool:
    value = (text or "").strip()
    if not 4 <= len(value) <= 300 or value.startswith("/"):
        return False
    value = markers.cut_descriptions(value)
    return bool(_HINT.search(value))


def extract_prompt(now_local: datetime, tz: str, existing: list[str]) -> str:
    return (
        "你是“近期计划抽取器”，只输出 JSON，不要解释。格式："
        '{"plans":[{"text":"","until":"YYYY-MM-DD","hedged":false}],"cancel":[0]}。'
        f"现在是 {now_local.strftime('%Y-%m-%d %H:%M')}（周{WEEKDAYS[now_local.weekday()]}），用户时区 {tz}。"
        "只从用户“当前这条消息”里抽取【用户自己】明确说出的、带时间范围的近期计划、安排或状态，"
        "例如“代码大后天才写”“下周都在出差”“明天去医院”“这周末不在家”“周五考试”。"
        "不抽：闲聊、玩笑、假设、别人的计划、对桃夭的要求或指令（设提醒、别再提某事等）、没有时间的长期习惯和喜好、已经发生的事。"
        "text 用第三人称、客观的短句写清楚，保留“才/要到/在那之前不”这类关键限定"
        "（如“写代码要到周六才开始，在那之前不写”）。"
        "until 填这个计划或状态涉及的最后一天，把“明天/后天/大后天/下周/周五”按“现在”换算成具体日期；"
        "一个时间点就填那一天，一段时间填最后一天；换算不了就不要抽。hedged=用户用了“估计/可能/大概”这类不确定的说法。"
        "cancel：用户这句话明确改变、取消或推翻了下面某条现有计划，填它的编号。没有就都留空。"
        "现有计划：\n" + ("\n".join(existing) or "（没有）"))


def validate(parsed: dict, today: date) -> list[dict]:
    """检查模型给出的新计划：日期合法、在今天到三个月内、文字不空。"""
    items = []
    for raw in (parsed or {}).get("plans") or []:
        if not isinstance(raw, dict):
            continue
        text = re.sub(r"\s+", " ", str(raw.get("text") or "")).strip()[:MAX_TEXT]
        try:
            until = date.fromisoformat(str(raw.get("until") or "").strip())
        except ValueError:
            continue
        if not text or until < today or until > today + timedelta(days=MAX_HORIZON_DAYS):
            continue
        items.append({"text": text, "until": until.isoformat(), "hedged": raw.get("hedged") is True})
    return items[:3]


def cancel_numbers(parsed: dict) -> list[int]:
    found = []
    for value in (parsed or {}).get("cancel") or []:
        try:
            found.append(int(value))
        except (TypeError, ValueError):
            continue
    return found


def describe_existing(plans: list[dict]) -> list[str]:
    return [f"{i}. {p['until']}前后：{p['text']}" for i, p in enumerate(plans, 1)]


def render_block(plans: list[dict], today: date, tz: str = "Asia/Shanghai") -> str:
    """给回复用的模型看的短块：只含还没过期（含宽限期）的计划。"""
    lines = []
    for plan in plans:
        try:
            until = date.fromisoformat(plan["until"])
            said = datetime.fromisoformat(str(plan.get("created_at") or "").replace("Z", "+00:00"))
            said_day = said.astimezone(ZoneInfo(tz)).strftime("%m-%d")
        except (ValueError, KeyError, TypeError):
            continue
        if today > until + timedelta(days=GRACE_DAYS):
            continue
        hedge = "，他说的是“估计”" if plan.get("hedged") else ""
        if today > until:
            status = "日期已过，可以自然地问问后来怎么样了，别当成还没发生"
        else:
            status = f"涉及到 {until.strftime('%m-%d')} 周{WEEKDAYS[until.weekday()]}，在那之前按这个安排理解他"
        lines.append(f"- {plan['text']}（{said_day} 说的{hedge}；{status}）")
    if not lines:
        return ""
    return ("\n\n【对方近期说过的计划和安排】\n" + "\n".join(lines) +
            "\n这些是对方亲口说的，优先于画像里的习惯和旧印象：到日期之前，不要假设他在做相反的事，"
            "也不要说和它矛盾的话；日期过后自然翻篇。需要时可以顺着它开启或延续话题，但别硬提、别逐条复述。")
