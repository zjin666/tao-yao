"""Exact user-name rejection requests; not requests to rename the bot."""
import re
from . import markers

NAME_FIELDS = ('name', 'preferred_name', 'nicknames', 'aliases')


def name_key(value):
    return str(value or '').strip(' \t\r\n“”"\'「」『』').casefold()


def rejected_user_names(text, bot_names=('桃夭',)):
    # Only the current speaker's own request. Do not execute examples, quotes,
    # reported speech, actions, or another person's naming preferences.
    text = markers.cut_descriptions(text)
    pattern = re.compile(r'(?:不要|别|不许|不准)(?:再)?(?:叫|喊|称呼)我\s*|'
                         r'(?:别|不要)(?:再)?管我叫\s*|我不喜欢你(?:叫|喊|称呼)我\s*')
    out=[]
    for match in pattern.finditer(text):
        prefix=re.split(r'[，,。！？!?；;\n]',text[:match.start()])[-1]
        if re.search(r'我(?:让|要求|请)(?!你$).+$',prefix):
            continue
        direct_prefix=prefix.strip()
        for bot_name in sorted((n for n in bot_names if n),key=len,reverse=True):
            if direct_prefix.startswith(bot_name):
                direct_prefix=direct_prefix[len(bot_name):]
                break
        if direct_prefix and not re.fullmatch(r'(?:你|请|拜托|以后|以后都|从现在起|我想|我希望|我说了|我说|我再说|我刚才说|我跟你说|我对你说|我是说|记住|麻烦你|我不喜欢)*',direct_prefix):
            continue
        probe=re.sub(r'我(?:跟|对|向)(?:你|桃夭)(?:说|讲)|我(?:刚才|之前|再|是)?说','',prefix)
        if re.search(r'说|转告|告诉|提醒|跟|向|替|帮',probe):
            continue
        if re.search(r'他说|她说|他说的|她说的|他说过|她说过|转述|举例|比如|假如|如果|不是说|并不是|别说|不要说|[“"「『>]',prefix):
            continue
        tail=text[match.end():]
        quoted=re.match(r'[“"「『\']([^”"」』\'\n]{1,24})[”"」』\']',tail)
        if quoted:
            candidate=quoted[1]
        else:
            plain=re.match(r'[^\s，,。！？!?；;：:\n]{1,30}',tail)
            if not plain:continue
            candidate=re.sub(r'(?:好不好|可以吗|好吗|行吗|了|啦|呀|哦)+$','',plain[0])
            if candidate in ('起床','吃饭','睡觉','干活','上班') or re.match(r'去|来|做|帮|陪|替',candidate):
                continue
        for name in candidate.split('、'):
            name=name.strip()
            if 1<=len(name)<=24 and name_key(name) not in {name_key(n) for n in out}:
                out.append(name)
    return out


def remove_disallowed_names(info, rejected=()):
    data=dict(info or {})
    blocked=list(dict.fromkeys([*(data.get('disallowed_names') or []),*rejected]))
    keys={name_key(n) for n in blocked}
    if blocked:data['disallowed_names']=blocked
    for field in NAME_FIELDS:
        value=data.get(field)
        if isinstance(value,list):
            kept=[v for v in value if name_key(v) not in keys]
            if kept:data[field]=kept
            else:data.pop(field,None)
        elif isinstance(value,str):
            # Explicit alias separators only, never substring deletion (小明 != 王小明).
            parts=re.split(r'[、,，/]',value)
            kept=[v for v in parts if name_key(v) not in keys]
            if not kept:data.pop(field,None)
            elif len(kept)!=len(parts):data[field]='、'.join(kept)
    return data
