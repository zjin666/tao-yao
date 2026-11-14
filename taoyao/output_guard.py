"""阻止机器人内部异常与提示块进入聊天；不屏蔽普通技术讨论。"""
import json
import re
from . import markers

NATURAL_RETRY = "……刚才没接上，你再发一次吧。"


HISTORY_TIME_TAG = re.compile(r'^\s*[\[【]\s*(?:\d{4}-)?\d{1,2}-\d{1,2}(?:\s*周[一二三四五六日])?\s+\d{1,2}:\d{2}\s*[\]】]\s*')


def _clean_prefix(text: str) -> str:
    """Remove presentation wrappers only; retain malformed/other internal blocks for rejection."""
    text = text or ""
    # Models sometimes echo the group message's routing envelope before the reply.
    # Decode JSON instead of using a greedy regex that could eat ordinary content.
    for _ in range(4):
        previous = text
        text = re.sub(r'^\s*assistant\s*[:：]\s*', '', text, count=1, flags=re.I)
        match = re.match(r'^\s*\[消息对象\s*(?=\{)', text)
        if match:
            try:
                obj, end = json.JSONDecoder().raw_decode(text[match.end():])
                tail = text[match.end()+end:]
                if isinstance(obj, dict) and tail.lstrip().startswith(']'):
                    text = tail.lstrip()[1:].lstrip()
            except ValueError:
                pass
        # 历史记录每行开头带的“[11-10 周二 22:50]”只是给模型看的时间标注，模型照着学会写在自己回复开头时要去掉。
        text = HISTORY_TIME_TAG.sub('', text, count=1)
        if previous == text:
            break
    return text


def clean_reply_envelope(text: str) -> str:
    text = _clean_prefix(text or '')
    # A wrapper may be repeated at each paragraph, not just once at the start.
    return '\n'.join(_clean_prefix(line) for line in text.split('\n'))


# 发给模型的提示词分节标题：这些不会出现在给用户的回复里。新增分节时在这里加一个（测试会核对）。
SECTION_TITLES = r'本轮群聊收束|称呼身份准则|当前对话对象|六维的具体表达方向|联网搜索结果|当前用户画像|系统提示|内部状态|回复执行流程|步骤[一二三四五]：[^】]+|当前时间|未发出的话|群回复指向|群聊隐私边界|图片/视频回复|聊天规则|日常接话|本条QQ引用|本轮答复意图|群聊自然插话|关系与称呼|关系参考|关于哥哥的事实|用户要求记住的事|群公共要求|长期记忆摘要|输出格式|桃夭已学知识参考|最近聊天[^】]*'

def is_internal_only(text: str) -> bool:
    """Explicit self-directed material, not ordinary user-facing text with an envelope.

    Routing metadata describes the INPUT recipient. Never infer that an output
    should be suppressed just because that metadata says bot/桃夭/self.
    """
    text = clean_reply_envelope(text).strip()
    if not text:
        return True
    # Explicit public-body delimiters mean potentially mixed output: rewrite it
    # instead of guessing how much following text belongs to the internal block.
    if re.search(r'(?:^|\n)\s*(?:回复正文|对用户的回复|最终回复|final)\s*[:：]|<final>', text, re.I):
        return False
    return bool(re.match(
        r'(?:(?:system|developer|tool|analysis)\s*[:：]|'
        rf'【(?:{SECTION_TITLES})】|'
        r'(?:这是内部检索状态|此为内部状态)|'
        r'(?:给模型的指令|给助手的指令|仅供模型阅读|内部备注|思考过程)\s*[:：])',
        text, re.I))

# 输入里的标注、识图和判断用的原始格式：模型有时会照抄进回复。和下面的内部通知分开，便于单独归类和修正。
_MARKER_RE = re.compile(
    markers.label_leak_pattern() + r"|"
    r"形象[:：][^\n]{0,60}(?:神态|可能想表达)[:：]|"
    r'\{\s*"(?:decision|category|allowed|reason|evidence|search|query|followup|guess|meaning|request_id|incomplete)"\s*:'
)

_INTERNAL_NOTICE = re.compile(
    r"Traceback\s*\(most recent call last\)|"
    r"(?:^|\n)\s*File\s+\"[^\"]+\",\s*line\s+\d+|"
    r"(?:^|\n)\s*(?:httpx\.|httpcore\.|requests\.)?(?:ConnectError|ReadTimeout|ConnectTimeout|RemoteProtocolError|HTTPStatusError)\b|"
    r"(?:连不上\s*(?:DeepSeek|Qwen|DashScope)|(?:DeepSeek|Qwen|DashScope)\s*(?:返回\s*\d{3}|key\s*不对|余额不够|余额不足|返回格式不对))|"
    r"Authorization\s*:\s*Bearer\s+\S+|\bsk-[A-Za-z0-9]{20,}\b|"
    r"(?:^|\n)\s*(?:ERROR|WARNING|INFO)\s+(?:taoyao|httpx|websockets)\b|"
    r"(?:^|\n)\s*\d{4}-\d\d-\d\d\s+\d\d:\d\d:\d\d[,\.]\d+\s+(?:ERROR|WARNING|INFO)\b|"
    rf"【(?:{SECTION_TITLES})】|"
    r"内部参考[，,]\s*不要向对方提起|这张表情已经认识|发送者(?:之前)?说明过(?:的表情含义|它的意思)|"
    r"\[消息对象\s*\{|这是内部检索状态|此为内部状态|搜索暂时失败，尚未核实|没有查到可用搜索结果，未确认|"
    r"(?:^|\n)\s*(?:system|developer|assistant|tool|analysis)\s*[:：]\s*|"
    r"(?:^|\n)\s*(?:给模型的指令|给助手的指令|仅供模型阅读|内部备注|思考过程)\s*[:：]",
    re.IGNORECASE,
)


def contains_internal_notice(text: str) -> bool:
    text = text or ""
    # 写错的标记（全角、大小写、多余空格或参数、at:昵称 等）和输入标注、原始格式一样算违规；合法标签不算。
    return bool(_INTERNAL_NOTICE.search(text) or _MARKER_RE.search(text) or markers.malformed_tags(text))


def internal_notice_reason(text: str) -> str:
    """Diagnostic categories only: never log the captured secret/prompt content."""
    text = text or ''
    if not contains_internal_notice(text):
        return ''
    if re.search(r'Authorization\s*:|\bsk-[A-Za-z0-9]{20,}', text, re.I):
        return 'credential'
    if '【' in text or '[消息对象' in text or '内部状态' in text or '内部检索' in text or '内部参考' in text:
        return 'internal_prompt_or_routing'
    if re.search(r'^\s*(system|developer|assistant|tool)\s*:', text, re.I|re.M):
        return 'role_prefix'
    if _MARKER_RE.search(text) or markers.malformed_tags(text):
        return 'internal_marker'
    return 'exception_or_log'


def guard_outgoing(text: str) -> str:
    text = clean_reply_envelope(text)
    # Final send boundary must never turn internal-only material into a visible
    # apology, meme, poke, or "wait" notice. Generation handles repair upstream.
    return '' if is_internal_only(text) or contains_internal_notice(text) else text
