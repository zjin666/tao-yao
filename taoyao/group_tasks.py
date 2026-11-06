"""Conservative gates for owner-requested cross-chat sends."""
import re


SEND_VERB = re.compile(r'发送|转发|发|说|问|讲|通知|转告|回复|回应|提醒|告诉|接话|接一句|聊')
REQUEST_START = re.compile(
    r'^(?:桃夭|妹妹)?\s*(?:'
    r'请(?:你)?|麻烦(?:你)?|帮我|替我|劳烦(?:你)?|'
    r'(?:你)?(?:能不能|可不可以|可以)(?:帮我|替我|去|到|往|向|在|发到|转发到)|'
    r'你(?:现在|这就|立刻|马上|再)?(?:去|到|往|向|给)|'
    r'(?:现在|这就|马上|立刻)?(?:去|到|往|向|发到|转发到)|把|在)')


def send_request_candidate(text):
    # Quoted instructions are data, never the owner's current authorization.
    plain = re.sub(r'“[^”]*”|「[^」]*」|『[^』]*』|"[^"\n]*"|‘[^’]*’|`[^`]*`', '', text or '')
    for sentence in re.split(r'[。！？!?；;\n]', plain):
        if re.match(r'\s*(?:如果|假如|要是|以后|下次|等到|假设)',sentence): continue
        for clause in re.split(r'[，,：:]',sentence):
            clause=clause.strip()
            if not REQUEST_START.match(clause): continue
            if '群' not in clause: continue
            verb=SEND_VERB.search(clause)
            if not verb: continue
            if re.match(r'了|过|的|错',clause[verb.end():]): continue
            if re.search(r'不要|别|不用|不必|不许|先不|暂时不',clause[:verb.start()]): continue
            return True
    return False


def target_only_reply(text, groups):
    value=(text or '').strip().rstrip('。.!！')
    prefix=r'(?:(?:就|是|发到|到|在|去)\s*)?'
    if re.fullmatch(prefix+r'(?:所有群|每个群|各个群|全部群)(?:吧|就行)?',value):
        return groups
    matches=[]
    for group in groups:
        names=[str(group['group_id'])]
        if group.get('group_name'): names.append(group['group_name'])
        alternatives='|'.join(re.escape(name) for name in names)
        if re.fullmatch(prefix+r'(?:群号[：:\s]*)?(?:'+alternatives+r')(?:吧|就行)?',value):
            matches.append(group)
    return matches
