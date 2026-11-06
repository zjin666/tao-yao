"""Read only visible quoted content, and verify the OneBot conversation scope."""
import html
import re


def quote_text(message):
    if isinstance(message,str):
        return html.unescape(re.sub(r'\[CQ:[^\]]*\]','',message)).strip()[:2000]
    parts=[]
    for segment in message if isinstance(message,list) else []:
        if not isinstance(segment,dict):continue
        kind=segment.get('type');data=segment.get('data') or {}
        if kind=='text':parts.append(str(data.get('text') or ''))
        elif kind in ('image','mface'):parts.append('[图片或表情包]')
        elif kind=='face':parts.append('[表情]')
        elif kind=='video':parts.append('[视频]')
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
