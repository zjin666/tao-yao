"""On-demand local clock. Defaults are effective settings, never inferred locations."""
from datetime import datetime, timezone
import re
from zoneinfo import ZoneInfo
from .personality import _TIMEZONE_HINTS


def valid_zone(value):
    try:
        ZoneInfo(str(value or '').strip())
        return str(value).strip()
    except (ValueError, KeyError, TypeError):
        return ''


def resolve_zone(profile=None, default='Asia/Shanghai'):
    profile = profile or {}
    for source in ('basic_info', 'basic_info_private'):
        zone = valid_zone((profile.get(source) or {}).get('timezone'))
        if zone:
            return zone, False
    return valid_zone(default) or 'Asia/Shanghai', True


def requested_zone(text):
    # Explicit target city/timezone affects this query only, not the user's profile.
    for name in re.findall(r'[A-Za-z_]+/[A-Za-z_]+(?:/[A-Za-z_]+)?', text or ''):
        if valid_zone(name):
            return name
    candidates = []
    for alias, name in _TIMEZONE_HINTS.items():
        for match in re.finditer(re.escape(alias)+(r'(?![\d:])' if alias.upper().startswith(('UTC','GMT')) else ''), text or '', re.I):
            candidates.append((match.start(), len(alias), name))
    return max(candidates)[2] if candidates else ''


def needs_clock(text, context=''):
    value = (text or '').strip()
    if re.search(r'不要(?:查|报|告诉).*时间|不用(?:查|报).*时间', value):
        return False
    if re.search(r'(?:现在|目前|当前|此刻|这会儿|当地|本地|北京时间|东京|纽约|伦敦).{0,12}(?:几点|日期|星期|周几|时间|几号|哪年)|'
                 r'(?:今天|明天|后天|昨天|今年).{0,8}(?:几号|几月|日期|星期几|周几|哪年|哪一天)|'
                 r'几点了|(?:报|看|查|告诉|知道|给).{0,8}(?:现在的?时间|当前时间|今天的?日期)|'
                 r'(?:现在|今天)(?:是)?(?:星期几|周几|几号|哪年)|(?:再过|还有).{0,8}(?:小时|分钟|天).{0,8}(?:几点|几号|日期|星期|周几)|'
                 r'时差|/时间|/日期|/时区|今天.{0,6}(?:多少号|什么日期)|现在.{0,6}(?:几月|早上还是晚上)|(?:距离|离).{0,16}还有几天', value):
        return True
    if re.search(r'(?:几点|几号|星期几|周几|日期是多少|日期是什么|什么时间|什么日期)[？?]?$|^(?:时间|日期)[？?]$', value):
        return True
    if re.search(r'(?:今天|明天|后天|昨天).{0,8}(?:星期|周)[一二三四五六日天1-7].{0,3}(?:吗|么|？|\?)|(?:能|可以).{0,8}(?:查|获取|知道).{0,5}(?:日期|时间)|(?:离|距离).{0,16}(?:还有多久|还剩多久)', value):
        return True
    # A short followup to an actual time question may specify another place.
    return bool(requested_zone(value) and re.search(r'呢|那边|这边|当地', value)
                and context and needs_clock(context[-600:]))


def clock_only(text):
    return needs_clock(text) and not re.search(
        r'天气|新闻|价格|汇率|股|节假日|放假|农历|日出|日落|航班|开盘|闭市|比赛|发售|发布|截止|开门|关门|预约|电影|营业|课程|直播', text)


def clock_text(profile=None, default='Asia/Shanghai', *, target_zone='', now=None):
    zone, assumed = resolve_zone(profile, default)
    base_zone, base_assumed = zone, assumed
    if valid_zone(target_zone):
        zone, assumed = target_zone, False
        source = '本次指定时区'
    else:
        source = '默认时区，未推断用户所在地' if assumed else '用户已记录时区'
    instant = now if now is not None else datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise ValueError('Clock requires an aware datetime')
    local = instant.astimezone(ZoneInfo(zone))
    weekday = '一二三四五六日'[local.weekday()]
    result = f'{local:%Y-%m-%d %H:%M:%S} 星期{weekday}（{zone}，{source}，UTC{local:%z}）'
    if target_zone and zone != base_zone:
        base = instant.astimezone(ZoneInfo(base_zone))
        label = '默认时区，未推断用户所在地' if base_assumed else '用户已记录时区'
        result += f'\n用户参考：{base:%Y-%m-%d %H:%M:%S} 星期{"一二三四五六日"[base.weekday()]}（{base_zone}，{label}，UTC{base:%z}）'
    return result
