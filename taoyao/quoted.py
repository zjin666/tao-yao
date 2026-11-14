"""Read only visible quoted content, and verify the OneBot conversation scope."""
import html
import re
from . import markers
import urllib.parse


def meme_label(data):
    """桃夭自己发的表情包：文件在 memes/<分类>/<序号_名字>.png，名字就是它的内容。"""
    raw=urllib.parse.unquote(str(data.get('file') or data.get('path') or '')).replace('\\','/')
    match=re.search(r'/memes/([^/]+)/(?:\d+_)?([^/]+?)\.[A-Za-z0-9]+$',raw)
    return f'[桃夭发的表情包：{match.group(2)}（{match.group(1)}类）]' if match else ''


def quote_images(message):
    """被引用消息里需要识别内容的图片片段（桃夭自己的表情包已知含义，不用再识别）。"""
    return [seg for seg in message if isinstance(seg,dict) and seg.get('type') in ('image','mface')
            and not meme_label(seg.get('data') or {})] if isinstance(message,list) else []


def quote_text(message,image_texts=None):
    """image_texts：按顺序对应 quote_images 的识别结果；没有就保持“[图片或表情包]”占位。"""
    if isinstance(message,str):
        return html.unescape(re.sub(r'\[CQ:[^\]]*\]','',message)).strip()[:2000]
    parts=[];pending=list(image_texts or [])
    for segment in message if isinstance(message,list) else []:
        if not isinstance(segment,dict):continue
        kind=segment.get('type');data=segment.get('data') or {}
        if kind=='text':parts.append(str(data.get('text') or ''))
        elif kind in ('image','mface'):
            label=meme_label(data)
            if label:parts.append(label)
            elif pending:
                text=str(pending.pop(0) or '').strip()
                parts.append(f'{markers.IMAGE} {text}' if text else markers.QUOTED_IMAGE_UNKNOWN)
            else:parts.append(markers.QUOTED_IMAGE_UNKNOWN)
        elif kind=='face':parts.append(markers.FACE)
        elif kind=='video':parts.append(markers.VIDEO_UNKNOWN)
        elif kind=='at':parts.append('@'+str(data.get('name') or '被提及的人')+' ')
    return ''.join(parts).strip()[:2000]


def same_quote_scope(message,scene,group_id,user_id,self_id):
    kind=message.get('message_type')
    if scene=='group':
        return kind in (None,'group') and str(message.get('group_id') or '')==str(group_id)
    if kind!='private' or message.get('group_id') not in (None,0,'0',''):
        return False
    sender=str((message.get('sender') or {}).get('user_id') or message.get('user_id') or '')
    if sender==str(user_id):return True
    # A bot-sent private message needs explicit recipient evidence. Otherwise use
    # the scoped local send history, never infer the recipient from the sender.
    return sender==str(self_id) and any(str(message.get(key) or '')==str(user_id)
        for key in ('peer_id','target_id','target_user_id','user_id'))
