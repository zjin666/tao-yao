"""Conservative allowlist: unfamiliar wording always keeps model search judgment."""
import re


def clearly_daily_message(text: str, context: str = '') -> bool:
    value = re.sub(r'[\s，,。！!？?～~呀啊啦呢哦]+$', '', text.strip())
    # A short acknowledgment may refer to a term or a fact in recent dialogue.
    if re.search(r'梗|出处|台词|歌词|术语|什么意思|搜索|查一下|最新|新闻|天气|股价|汇率|https?://', context):
        return False
    if re.search(r'https?://|[“”「」『』"\[\]【】]|梗|出处|台词|歌词|什么意思|查|搜|联网|最新|天气|价格|新闻', value):
        return False
    # Strip time/adverb framing only for a complete known daily expression.
    value = re.sub(r'^(?:我|你)?(?:今天|现在|最近|这几天|刚才)[，, ]*', '', value)
    value = re.sub(r'^(?:真的|实在|突然|感觉)[，, ]*', '', value)
    return bool(re.fullmatch(
        r'(?:早上好|下午好|晚上好|早安|晚安|你好|嗨|在吗|谢谢你?|好的?|嗯嗯?|哈哈哈?|'
        r'(?:我)?(?:有点|好|很|太)?(?:累了?|困了?|饿了?|难过|开心|委屈|想你了?)|'
        r'(?:我)?(?:想|想要|需要)?(?:你)?(?:陪陪我|陪我聊聊天|安慰我|抱抱我)|'
        r'我(?:刚)?(?:忙完了?|下班了|起床了|到家了|吃完饭了)|'
        r'我(?:先)?(?:去睡觉了?|去吃饭了?|喝杯茶|休息一下)|'
        r'陪我聊(?:聊|一会儿)|抱抱|辛苦了)', value))
