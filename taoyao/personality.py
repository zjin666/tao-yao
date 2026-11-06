from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from .decision_semantics import DIRECTIONS, NEUTRAL_SIGNALS, SIGNALS

# ==========================================
# 1. 核心配置：语义桶与增量映射表
# ==========================================
BUCKET_DELTA_MAP = {}

for _dim, _signals in SIGNALS.items():
    BUCKET_DELTA_MAP[_dim] = {"中性": 0.0, **{s[0]: s[1] for s in _signals}}

DIMENSIONS = ["AGG", "INT", "HUM", "DEP", "RAT", "ACT"]
CONFIDENCE_THRESHOLD = 0.8


@dataclass
class ClassificationResult:
    bucket: str
    confidence: float
    similarity: float = 0.0
    margin: float = 0.0
    candidate: str = ""
    applied: bool = True


@dataclass
class ScoringConfig:
    ema_alpha: float = 0.45
    max_single_delta: float = 10.0
    delta_scale: float = 1.5
    confidence_power: float = 0.7
    confidence_floor: float = 0.35
    dimensions: list = field(default_factory=lambda: list(DIMENSIONS))


class PersonalityScoringModel:
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()
        self.vector: Dict[str, float] = {dim: 50.0 for dim in self.config.dimensions}
        self._ema_buffer: Dict[str, float] = {dim: 0.0 for dim in self.config.dimensions}

    def load_vector(self, scores: dict | None) -> None:
        for dim in self.config.dimensions:
            try:
                self.vector[dim] = max(0.0, min(100.0, float((scores or {}).get(dim, 50.0))))
            except (TypeError, ValueError):
                self.vector[dim] = 50.0
        self._ema_buffer = {dim: 0.0 for dim in self.config.dimensions}

    def update(self, classification: Dict[str, ClassificationResult]) -> Dict[str, float]:
        vector, _ = self.update_with_details(classification)
        return vector

    def update_with_details(
        self, classification: Dict[str, ClassificationResult], first: bool = False
    ) -> tuple[Dict[str, float], Dict[str, dict]]:
        """返回最终向量，以及这次 6 个维度各自的判断明细。

        first=True 时把第一条消息的完整增量直接加进去；之后才用 EMA 平滑。
        """
        raw_deltas = self._classify_to_delta(classification)
        if first:
            smoothed_deltas = dict(raw_deltas)
            self._ema_buffer = dict(raw_deltas)
        else:
            smoothed_deltas = self._apply_ema(raw_deltas)
        before = dict(self.vector)
        self._apply_to_vector(smoothed_deltas)
        details: Dict[str, dict] = {}
        for dim in self.config.dimensions:
            result = classification.get(dim) or ClassificationResult("中性", 0.0)
            details[dim] = {
                "bucket": result.bucket,
                "candidate": result.candidate or result.bucket,
                "confidence": round(float(result.confidence), 4),
                "applied": bool(result.applied and result.bucket != "中性"),
                "raw_delta": round(float(raw_deltas.get(dim, 0.0)), 4),
                "delta": round(float(self.vector[dim] - before[dim]), 4),
                "before": round(float(before[dim]), 1),
                "after": round(float(self.vector[dim]), 1),
            }
        return dict(self.vector), details

    def _classify_to_delta(self, classification: Dict[str, ClassificationResult]) -> Dict[str, float]:
        deltas = {}
        for dim in self.config.dimensions:
            result = classification.get(dim)
            if result is None:
                deltas[dim] = 0.0
                continue
            if not result.applied or result.bucket == "中性":
                deltas[dim] = 0.0
                continue
            base_delta = BUCKET_DELTA_MAP.get(dim, {}).get(result.bucket, 0.0)
            confidence = max(0.0, min(1.0, float(result.confidence)))
            if confidence > 0:
                effective = max(
                    self.config.confidence_floor,
                    confidence ** self.config.confidence_power,
                )
            else:
                effective = 0.0
            adjusted = base_delta * self.config.delta_scale * effective
            adjusted = max(-self.config.max_single_delta, min(self.config.max_single_delta, adjusted))
            deltas[dim] = round(adjusted, 3)
        return deltas

    def _apply_ema(self, raw_deltas: Dict[str, float]) -> Dict[str, float]:
        alpha = self.config.ema_alpha
        smoothed = {}
        for dim in self.config.dimensions:
            prev = self._ema_buffer[dim]
            curr = raw_deltas[dim]
            if curr == 0:
                self._ema_buffer[dim] = 0.0
                smoothed[dim] = 0.0
                continue
            new_val = alpha * curr + (1 - alpha) * prev
            self._ema_buffer[dim] = new_val
            smoothed[dim] = round(new_val, 3)
        return smoothed

    def _apply_to_vector(self, smoothed_deltas: Dict[str, float]):
        for dim in self.config.dimensions:
            self.vector[dim] += smoothed_deltas[dim]
            self.vector[dim] = max(0.0, min(100.0, self.vector[dim]))
            # 向下取到一位小数，避免 1.3x/1.5x 的增量把分数往上蹭。
            self.vector[dim] = math.floor((self.vector[dim] + 1e-9) * 10) / 10

    def get_snapshot(self) -> Dict[str, float]:
        return dict(self.vector)

    def reset(self):
        self.vector = {dim: 50.0 for dim in self.config.dimensions}
        self._ema_buffer = {dim: 0.0 for dim in self.config.dimensions}


# ==========================================
# 2. 本地规则分类器（不额外调用 API）
# ==========================================
class RuleClassifier:
    """把一条消息粗略映射到 6 个维度的语义桶。

    这是零成本的第一版分类器；后面如果要更准，可以替换成 DeepSeek/本地小模型，
    PersonalityScoringModel 的输入输出结构不用变。
    """

    def __init__(self):
        self.rules: dict[str, list[tuple[re.Pattern, str, float]]] = {}
        for dim, signals in SIGNALS.items():
            self.rules[dim] = [(re.compile(s[4]), s[0], 0.90) for s in signals]

    def warmup(self) -> None:
        return None

    def classify(self, text: str) -> Dict[str, ClassificationResult]:
        text = (text or "").strip()
        result: Dict[str, ClassificationResult] = {}
        for dim in DIMENSIONS:
            hit = None
            for pattern, bucket, confidence in self.rules.get(dim, []):
                if pattern.search(text):
                    hit = ClassificationResult(bucket=bucket, confidence=confidence)
                    break
            result[dim] = hit or ClassificationResult(bucket="中性", confidence=0.0)
        return result


# ==========================================
# 3. 基础信息提取与画像描述
# ==========================================
_AGE_RE = re.compile(r"(?:我|本人)?(?:今年)?\s*(\d{1,2})\s*岁")
_NAME_RE = re.compile(r"(?:我叫|我的名字是)\s*([^\s，。！？]{1,12})")
_GENDER_RE = re.compile(r"我是(男|女)(?:生|孩)?")
_HOBBY_RE = re.compile(r"(?:我)?(?:平时|平常)?(?<!不)(?:喜欢|爱好是|爱好)\s*([^。！？\n，,]{1,30})")
_DISLIKE_RE = re.compile(r"(?:我)?(?:不喜欢|不太喜欢|讨厌)\s*([^。！？\n，,]{1,30})")
_RELATION_WORDS = ("你", "哥哥", "桃夭", "妹妹", "姐姐", "弟弟", "妈妈", "爸爸", "家人")

TRAIT_LEXICON = (
    "内向", "外向", "活泼", "安静", "开朗", "孤僻", "乐观", "悲观",
    "温柔", "暴躁", "细心", "粗心", "理性", "感性", "冲动", "冷静",
    "直接", "委婉", "幽默", "严肃", "自信", "自卑", "独立", "依赖",
    "勤快", "懒", "勇敢", "胆小", "话多", "话少", "社恐", "社牛",
    "敏感", "迟钝", "成熟", "幼稚", "随和", "固执", "浪漫", "务实",
    "好胜", "佛系", "古灵精怪", "可爱", "慢热", "傲娇", "善良", "体贴",
    "拖延", "自律", "完美主义", "随性", "急性子", "慢性子", "可靠",
)
TRAIT_OPPOSITE_PAIRS = (
    ("内向", "外向"), ("活泼", "安静"), ("开朗", "孤僻"),
    ("乐观", "悲观"), ("温柔", "暴躁"), ("细心", "粗心"),
    ("理性", "感性"), ("冲动", "冷静"), ("直接", "委婉"),
    ("幽默", "严肃"), ("自信", "自卑"), ("独立", "依赖"),
    ("勤快", "懒"), ("勇敢", "胆小"), ("话多", "话少"),
    ("社恐", "社牛"), ("敏感", "迟钝"), ("成熟", "幼稚"),
    ("随和", "固执"), ("浪漫", "务实"), ("好胜", "佛系"),
    ("拖延", "自律"), ("完美主义", "随性"), ("急性子", "慢性子"),
)

TRAIT_CUES = (
    (r"(不想出门|人多.{0,4}(烦|累|不想)|躲.{0,3}(起来|着)|不敢说话|社恐)", "社恐"),
    (r"(拖到|拖延|明天再|最后.{0,3}(才|再)|懒得动)", "拖延"),
    (r"(无所谓|都行|随便|算了|不去想|顺其自然)", "随和"),
    (r"(必须|一定|不能错|检查.{0,3}(一遍|几次)|做到最好)", "完美主义"),
    (r"(直接说|不绕|说白了|有话直说)", "直接"),
    (r"(我来|帮你|照顾|陪你|别担心)", "体贴"),
    (r"(先想|分析|冷静|按逻辑|理性)", "理性"),
    (r"(马上|立刻|冲|等不及|急性子)", "急性子"),
)


def normalize_trait(text: str) -> str:
    text = str(text or "").strip("“”\"'「」 ，,。")
    text = re.sub(r"^(我|本人|这个人|平时|其实|比较|很|挺|有点|稍微|是|算|偏|一个|个)+", "", text)
    text = re.sub(r"(的人|一点|一些)$", "", text).strip()
    return text[:12]


def _trait_opposites(trait: str) -> set[str]:
    out = set()
    for a, b in TRAIT_OPPOSITE_PAIRS:
        if trait == a:
            out.add(b)
        elif trait == b:
            out.add(a)
    return out


def merge_traits(existing: list[str] | None, fresh: list[str] | None) -> list[str]:
    """后出现的 trait 覆盖互斥的旧 trait，不互斥的保留。"""
    out: list[str] = []
    for raw in [*(existing or []), *(fresh or [])]:
        trait = normalize_trait(raw)
        if not trait:
            continue
        opposites = _trait_opposites(trait)
        out = [x for x in out if x != trait and x not in opposites]
        out.append(trait)
    return out[-8:]


def detect_traits_local(text: str) -> list[str]:
    found: list[str] = []
    self_context = False
    for clause in re.split(r"[，,。！？!?；;\n]", text or ""):
        if not clause.strip():
            continue
        has_self = "我" in clause
        has_other = bool(re.search(r"(你|您|桃夭|他|她|哥哥|妹妹|姐姐|弟弟|家人)", clause))
        if has_other and not has_self:
            self_context = False
            continue
        if has_self and not has_other:
            self_context = True
        elif not self_context or not re.match(
            r"^(但|但是|不过|而且|也|还|其实|平时|就是|只是|挺|很|有点|比较)", clause
        ):
            continue
        if has_other:
            continue
        for trait in TRAIT_LEXICON:
            idx = clause.find(trait)
            if idx < 0:
                continue
            prefix = clause[max(0, idx - 4):idx]
            if re.search(r"(不|没|别|并不|不太|不是)$", prefix):
                continue
            found.append(trait)
        for pattern, trait in TRAIT_CUES:
            if re.search(pattern, clause):
                found.append(trait)
    return list(dict.fromkeys(found))
TAO_NICKNAME_RE = re.compile(
    r"(?:"
    r"(?:以后|从现在起|以后就|可以|能不能|能|请|我想)?\s*(?:叫你|喊你|称呼你|管你叫)"
    r"|(?:可以|能|我想|请|以后)\s*叫"
    r"|给你(?:起|取)(?:个)?(?:名字|昵称)[，,\s]*(?:叫|是)?"
    r"|你的昵称(?:是|叫)|你以后叫"
    r")\s*[“\"「']?([^\s，。！？!?、,]{1,16})"
)


def is_reported_tao_nickname(text: str) -> bool:
    value=text or ''
    explicit=bool(re.search(r'昵称|称呼|名字|换.{0,4}叫法',value))
    if not explicit and not re.search(r'叫你|喊你|称呼你|管你叫',value):return False
    if not explicit and re.search(r'(?:叫你|喊你)(?:来|去|帮|做|吃|喝|看|聊|起床)',value):return False
    naming=TAO_NICKNAME_RE.search(value)
    probe=value[:naming.start(1)] if naming else value
    for match in re.finditer(r'([A-Za-z0-9_\u4e00-\u9fff]{1,12}?)(?:想|打算|准备|希望|决定|说|要|会|以后|给你|叫你|喊你|称呼你)',probe):
        actor=match.group(1)
        if re.fullmatch(r'(?:那|以后|现在|从现在起|今后)?(?:我|本人|你|桃夭)(?:以后|现在|今后)?',actor):continue
        if actor in ('我能','我可以','我想','我给你','那我'):continue
        return True
    return bool(re.search(r'(?:替|帮).{1,10}(?:说|转告|记住|登记).{0,12}(?:叫你|称呼|昵称)',value))


def is_tao_nickname_fragment(name: str) -> bool:
    """Particles and unspecified objects are not a chosen form of address."""
    return str(name or '').strip() in {
        '', '了', '啦', '吗', '呢', '吧', '啊', '呀', '哦', '嘛', '呗', '咯', '喽', '哇', '噢',
        '什么', '啥', '一下', '一声', '这个', '那个', '你', '您', '我', '他', '她', '它',
    }


def parse_tao_nickname_request(text: str) -> str:
    value=text or ''
    if is_reported_tao_nickname(value):return ''
    match=TAO_NICKNAME_RE.search(value)
    if not match:return ''
    prefix=value[:match.start()].strip()
    if prefix and not (re.fullmatch(r'(?:那|以后|现在|从现在起|今后)?我(?:想|打算|准备|决定|以后|现在|就)*',prefix)
        or re.fullmatch(r'[\u4e00-\u9fffA-Za-z0-9_]{1,8}[，,：:]\s*',prefix)
        or prefix in ('那就','桃夭','小桃')):
        return ''
    raw_name=match.group(1)
    name=re.split(r"[”\"」']",raw_name,maxsplit=1)[0].strip('“”\"「」 ')
    # Quoted names are exact; only remove conversational suffixes outside quotes.
    quoted=bool(re.match(r'[“"「\x27]',value[match.start(1)-1:match.start(1)]))
    if not quoted:
        name=re.sub(r'(好不好|行不行|可以吗|好吗|行吗|了吗|了|吗|吧|呀|啊|哦|嘛|啦|呢|呗|咯|喽|哇|噢)+$','',name).strip()
    if is_tao_nickname_fragment(name):return ''
    return name[:12]


def validate_tao_nickname(name: str, is_owner: bool) -> str:
    name = str(name or "").strip("“”\"'「」 ")
    if is_tao_nickname_fragment(name):
        return ''
    if not (1 <= len(name) <= 12):
        return ""
    if re.search(r"https?://|[\r\n，。！？!?、,；;：:\s]", name):
        return ""
    # 昵称必须是“称呼”，不能是一句任务/动作描述。
    if re.search(r"(去|做|说|问|发|帮|给|群里|群聊|任务|事情|以后|现在|今天|明天|不要|记住)", name):
        return ""
    if re.search(r"(了|吗|呢|吧|啊|呀)$", name):
        return ""
    forbidden = (
        "老婆", "老公", "恋人", "女朋友", "男朋友", "女友", "男友",
        "主人", "宠物", "所有物", "女王", "对象", "媳妇", "丈夫", "妻子",
        "亲爱的", "宝贝", "奴隶", "废物", "垃圾",
    )
    if any(word in name for word in forbidden):
        return ""
    family_words = ("妹妹", "姐姐", "弟弟", "哥哥", "家人", "亲人", "妈妈", "爸爸")
    if any(word in name for word in family_words):
        # owner 只保留“妹妹”这一种家庭式称呼，其他家庭称呼一律拒绝。
        if not (is_owner and name == "妹妹"):
            return ""
    return name


_TIMEZONE_HINTS = {
    "北京": "Asia/Shanghai", "上海": "Asia/Shanghai", "中国": "Asia/Shanghai",
    "香港": "Asia/Hong_Kong", "台北": "Asia/Taipei", "台湾": "Asia/Taipei",
    "东京": "Asia/Tokyo", "日本": "Asia/Tokyo", "首尔": "Asia/Seoul",
    "韩国": "Asia/Seoul", "新加坡": "Asia/Singapore",
    "纽约": "America/New_York", "New York": "America/New_York", "NYC": "America/New_York", "洛杉矶": "America/Los_Angeles",
    "加州": "America/Los_Angeles", "西雅图": "America/Los_Angeles",
    "伦敦": "Europe/London", "巴黎": "Europe/Paris", "柏林": "Europe/Berlin",
    "悉尼": "Australia/Sydney", "墨尔本": "Australia/Melbourne",
    "UTC+8": "Asia/Shanghai", "GMT+8": "Asia/Shanghai",
    "UTC+9": "Asia/Tokyo", "GMT+9": "Asia/Tokyo",
}


def _valid_timezone(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    try:
        ZoneInfo(value)
        return value
    except Exception:
        return ""


def explicit_user_timezone(text: str) -> str:
    """Current self-location/zone assertions only, not cities in clock questions."""
    # Generated media memories and quoted location examples are not declarations.
    text = re.split(r'\n\[(?:最近图片记忆|图片内容|视频内容)\]', str(text or ''), maxsplit=1)[0]
    aliases = '|'.join(re.escape(k) for k in sorted(_TIMEZONE_HINTS, key=len, reverse=True))
    zone_value = rf'(?:[A-Za-z_]+/[A-Za-z_]+(?:/[A-Za-z_]+)?|{aliases})'
    declaration = re.compile(
        rf'(?:我(?:现在|目前|此刻|最近|已经|刚)?(?:就)?(?:在|住在|搬到|搬来|搬去|人在)|'
        rf'我(?:这边|这里)(?:现在|目前)?(?:是|在)|我的(?:所在地|位置|时区)(?:是|在|为)|'
        rf'\bI(?:\s+am|\x27m)\s+(?:currently\s+)?(?:in|living\s+in|based\s+in)\s+)'
        rf'\s*[:：]?\s*(?P<value>{zone_value})(?![A-Za-z_\d/:])', re.I)
    answer = ''
    for match in declaration.finditer(text):
        prefix = re.split(r'[，,。！？!?；;\n]', text[:match.start()])[-1]
        if re.search(r'假如|如果|比如|举例|他说|她说|他说的|转述|你说|不是说|并不是|不代表|不表示|[“"「『]|\bsaid\b', prefix, re.I):
            continue
        if re.match(r'\s*(?:时间|时区)?\s*(?:吗|么|还是|是否|是不是|[？?])', text[match.end():]):
            continue
        if re.match(r'的(?:朋友|同事|家人|客户|亲戚|同学)', text[match.end():]):
            continue
        value = match['value']
        answer = _valid_timezone(value) or next((zone for alias,zone in _TIMEZONE_HINTS.items() if alias.casefold()==value.casefold()), '')
    return answer


def _looks_like_person_name(text: str) -> bool:
    text = str(text or "").strip("“”\"'「」 ")
    if not (1 <= len(text) <= 12):
        return False
    # 本地只做结构检查；是否真的是称呼交给 LLM 判断。
    if re.search(r"[…\.，。！？!?、,；;：:\r\n\s]", text):
        return False
    return True


def _looks_like_hobby(text: str) -> bool:
    text = text.strip()
    if not text or text.startswith(("的", "地", "得")):
        return False
    return not any(word in text for word in _RELATION_WORDS)


PROFILE_CUE_RE = re.compile(
    r"(喜欢|爱好|不喜欢|讨厌|我叫|名字|今年|岁|住在|工作|职业|养了|"
    r"妹妹|哥哥|姐姐|弟弟|家人|时区|UTC|GMT|我在|来自|搬到|时差)"
)


async def extract_basic_info_llm(
    text: str,
    llm: Any,
    bot_name: str = "桃夭",
    is_owner: bool = True,
    recent_text: str = "",
) -> dict:
    """用一次小模型调用做语义画像提取，避免把「喜欢妹妹」切成人名/关系。"""
    role_text = "owner（桃夭叫他哥哥）" if is_owner else "普通好友（不是哥哥，桃夭不是他的妹妹）"
    prompt = (
        "你是用户画像提取器。只根据用户这句话输出一个 JSON 对象，不要解释、不要 markdown。\n"
        "可用的键：name, age, gender, location, occupation, timezone, hobbies, dislikes, traits, relationships, affection。\n"
        "规则：\n"
        "1. hobbies 只能是活动、事物或兴趣，例如「打游戏」「跑步」「猫」「音乐」；"
        "性格形容词放 traits，不要混进 hobbies。\n"
        "1.1 name 只能来自用户明确自述「我叫X」或「我的名字是X」；"
        "「叫我X」「又叫我…」「别人叫我…」不是自我介绍，不要提取 name。"
        "请判断 X 是否真像一个称呼；如果它是一句话、动作或句子碎片，就省略 name。\n"
        "1.2 traits 是用户自己的性格/特点，可以有多项；请结合「用户消息」和「最近用户发言」总结一个中肯的画像。"
        "不需要用户直接说「我很内向」，也可以从用户的行为、选择、反应方式、语气中二次推断；"
        "但每个推断必须有具体行为依据，不能只凭一句夸张的话或凭空编造。互斥的性格不要同时输出"
        "（例如内向和外向、活泼和安静）；如果近期表现和旧说法冲突，以最近的说法为准。"
        "如果用户是在评价桃夭（例如「你可爱」「你很古灵精怪」「我喜欢这样的你」），不要写入用户的 traits 或 hobbies，"
        "这类句子最多记 affection=桃夭。\n"
        "2. 不能把「你」「您」「哥哥」「妹妹」「姐姐」「弟弟」「家人」或任何指人的称呼放进 hobbies/traits。\n"
        "2.1 hobbies/dislikes 必须是用户自己的活动或对象。\n"
        "2.2 字段关系必须遵守：\n"
        "  - hobbies 和 dislikes 互斥：同一个活动/对象只能出现在其中一边，不要同时输出同义项；\n"
        "  - traits 与 hobbies/dislikes 不互斥，但不要把同一内容重复写进多个标签；"
        "性格形容词只放 traits，活动/对象只放 hobbies/dislikes；\n"
        "  - relationships 和 affection 同一对象不要重复写，affection 只放明确在意的对象；\n"
        "  - 如果内容与本次其他标签重复，不要新建同义标签，合并到原标签；"
        "如果内容互斥，后到的覆盖先到的，只保留最新的一边。\n"
        f"3. 只有用户明确用「你」「您」或直接说「{bot_name}」时，才把对象解析成桃夭，"
        "放进 relationships/affection；不要默认「妹妹」就是桃夭。\n"
        "4. 「我妹妹」「我姐姐」「我家人」「我朋友」是用户自己的关系，不是桃夭；"
        "不要把他们的年龄、爱好写成用户的。\n"
        "5. 单独说「妹妹」「哥哥」但对象不明确时，宁可不记 relationships/affection，也不要猜桃夭。\n"
        f"6. 当前用户身份是 {role_text}。如果用户不是 owner，桃夭关系最高只能记「挚友」，"
        "禁止妹妹、姐姐、家人、恋人、女友、男友、老婆、老公、主人、CP；用户要求这些时宁可不记。"
        "普通好友可以记自己的妹妹，但不要把对象写成桃夭。\n"
        "7. relationships 用「对象：关系」字符串，例如「桃夭：妹妹」；不确定就省略。\n"
        "8. affection 只放明确在意/喜欢的对象，例如「桃夭」；不要把「喜欢」「妹妹」这类词本身放进去。\n"
        "9. timezone 只在用户明确说出城市、国家或 UTC 偏移时填写 IANA 时区，例如「东京」-> Asia/Tokyo，"
        "「加州」-> America/Los_Angeles；不要根据语言、昵称或国籍猜时区。\n"
        "10. 不确定的字段省略，不要编造，不要从称呼推断真实姓名。\n"
        "11. 只输出 JSON，例如：{\"hobbies\":[\"打游戏\"],\"relationships\":[\"桃夭：妹妹\"]}\n"
        f"对话对象昵称：{bot_name}\n用户消息：{text}\n"
        f"最近用户发言（从旧到新）：\n{recent_text or '（无）'}"
    )
    raw = await llm.chat(
        [
            {"role": "system", "content": "你只输出一个合法 JSON 对象，不输出解释。"},
            {"role": "user", "content": prompt},
        ]
    )
    raw = str(raw or "").strip()
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    allowed = {
        "name", "age", "gender", "location", "occupation", "timezone",
        "hobbies", "dislikes", "traits", "relationships", "affection",
    }
    clean = {k: v for k, v in data.items() if k in allowed}
    if "name" in clean:
        name = str(clean.get("name") or "").strip()
        if _looks_like_person_name(name):
            clean["name"] = name
        else:
            clean.pop("name", None)
    for key in ("hobbies", "dislikes", "traits", "relationships", "affection"):
        if key in clean and not isinstance(clean[key], list):
            clean.pop(key, None)
        elif key in clean:
            clean[key] = [str(x).strip() for x in clean[key] if str(x).strip()][:8]

    # 模型偶尔仍会把人称塞进 hobbies/affection，这里再做一次结构校验。
    for key in ("hobbies", "dislikes"):
        if key in clean:
            clean[key] = [item for item in clean[key] if _looks_like_hobby(item)]
            if not clean[key]:
                clean.pop(key, None)
    if "traits" in clean:
        clean["traits"] = merge_traits(
            [],
            [
                item for item in clean["traits"]
                if item and not any(word in item for word in _RELATION_WORDS)
            ],
        )
        if not clean["traits"]:
            clean.pop("traits", None)
    if "relationships" in clean:
        kept = []
        for item in clean["relationships"]:
            if "：" not in item and ":" not in item:
                continue
            target = re.split(r"[：:]", item, maxsplit=1)[0].strip()
            relation = re.split(r"[：:]", item, maxsplit=1)[1].strip()
            if target and relation and (
                target == bot_name or not any(x in target for x in _RELATION_WORDS)
            ):
                kept.append(item)
        clean["relationships"] = kept[:8]
        if not kept:
            clean.pop("relationships", None)
    if "affection" in clean:
        clean["affection"] = [
            item for item in clean["affection"]
            if (item == bot_name or _looks_like_hobby(item))
            and item not in ("喜欢", "爱", "在意")
        ][:8]
        if not clean["affection"]:
            clean.pop("affection", None)

    # 原文没有明确指向桃夭时，禁止模型把「妹妹」等关系硬映射成桃夭。
    explicit_bot = (
        bot_name in text
        or re.search(r"(?:喜欢|爱|在意|最).{0,8}(?:你|您)", text) is not None
        or "你也是" in text
        or re.search(r"(?:我们|咱俩).{0,8}(?:朋友|关系|一起)", text) is not None
    )
    allowed_friend = {"朋友", "好友", "好朋友", "挚友", "朋友以上", "友情以上"}
    forbidden_lover = {"恋人", "女朋友", "男朋友", "老婆", "老公", "情侣", "对象", "cp"}
    if not explicit_bot:
        for key in ("relationships", "affection"):
            if key in clean:
                clean[key] = [
                    item for item in clean[key]
                    if bot_name not in item and "桃夭" not in item
                ]
                if not clean[key]:
                    clean.pop(key, None)
    elif not is_owner:
        kept = []
        for item in clean.get("relationships") or []:
            if bot_name not in item and "桃夭" not in item:
                kept.append(item)
                continue
            relation = re.split(r"[：:]", item, maxsplit=1)[-1].strip()
            if relation in allowed_friend:
                kept.append(f"{bot_name}：挚友")
            # 妹妹/姐姐/家人/恋人/老婆等一律丢弃，不升级。
        if kept:
            clean["relationships"] = list(dict.fromkeys(kept))[:8]
        else:
            clean.pop("relationships", None)
        if "affection" in clean:
            clean["affection"] = [
                item for item in clean["affection"]
                if bot_name not in item and "桃夭" not in item
            ]
            if not clean["affection"]:
                clean.pop("affection", None)
    else:
        # owner 可以保留兄妹关系，但不能把桃夭升级成恋人。
        kept = []
        for item in clean.get("relationships") or []:
            if bot_name in item or "桃夭" in item:
                relation = re.split(r"[：:]", item, maxsplit=1)[-1].strip().lower()
                if relation in forbidden_lover:
                    continue
            kept.append(item)
        if kept:
            clean["relationships"] = kept[:8]
        else:
            clean.pop("relationships", None)

    # 「我妹妹喜欢猫」是家人的爱好，不是用户本人的爱好。
    family = re.search(r"我(?:妹妹|姐姐|弟弟|哥哥|妈妈|爸爸|家人|朋友)", text)
    if family and not re.search(r"(?:^|[，。；\s])我(?:也)?(?:喜欢|爱|讨厌|不喜欢)", text):
        for key in ("hobbies", "dislikes"):
            clean.pop(key, None)

    if "age" in clean:
        try:
            age = int(clean["age"])
            if 5 <= age <= 99:
                clean["age"] = age
            else:
                clean.pop("age", None)
        except (TypeError, ValueError):
            clean.pop("age", None)
    if "timezone" in clean:
        # A model must not turn a queried city's zone into the user's own zone.
        tz = explicit_user_timezone(text)
        if tz:
            clean["timezone"] = tz
        else:
            clean.pop("timezone", None)
    return clean


def extract_basic_info(text: str) -> dict:
    """只提取用户明确说出的客观信息，避免过度推断。"""
    info: dict = {}
    text = text or ""

    match = _AGE_RE.search(text)
    if match:
        age = int(match.group(1))
        if 5 <= age <= 99:
            info["age"] = age

    match = _GENDER_RE.search(text)
    if match:
        info["gender"] = "男" if match.group(1) == "男" else "女"

    hobbies = []
    for match in _HOBBY_RE.finditer(text):
        hobby_text = match.group(1).strip(" ，,、")
        # 「喜欢和你聊天」「最喜欢的妹妹」不是爱好，别把关系和残句记进去。
        if _looks_like_hobby(hobby_text):
            hobbies.extend(
                x.strip()
                for x in re.split(r"[、,，和及]", hobby_text)
                if _looks_like_hobby(x)
            )
    if hobbies:
        info["hobbies"] = list(dict.fromkeys(hobbies))[:6]

    dislikes = []
    for match in _DISLIKE_RE.finditer(text):
        dislike_text = match.group(1).strip(" ，,、")
        if _looks_like_hobby(dislike_text):
            dislikes.extend(
                x.strip()
                for x in re.split(r"[、,，和及]", dislike_text)
                if _looks_like_hobby(x)
            )
    if dislikes:
        info["dislikes"] = list(dict.fromkeys(dislikes))[:6]

    tz = explicit_user_timezone(text)
    if tz:
        info["timezone"] = tz
    traits = detect_traits_local(text)
    if traits:
        info["traits"] = traits
    return info


# 跨维度普通消息：不表达明确态度/情绪/依赖/论证/能量，用于训练“中性/不适用”类。
NEUTRAL_EXAMPLES = [
    "今天周三",
    "外面在下雨",
    "手机快没电了",
    "我刚吃完饭",
    "路上有点堵",
    "我在等地铁",
    "快递到了",
    "晚饭吃了面",
    "今天有点忙",
    "刚开完会",
    "我先去洗澡",
    "灯坏了",
    "明天要早起",
    "这个周末有空吗",
    "我在看剧",
    "电脑卡住了",
    "水烧开了",
    "外卖到了",
    "今天风有点大",
    "我到家了",
    "手机在充电",
    "刚睡醒",
    "楼下在装修",
    "超市人很多",
    "菜买好了",
    "公交还没来",
    "手机响了",
    "我在排队",
    "今天穿少了",
    "窗户没关",
    "刚倒完垃圾",
    "杯子落公司了",
    "我先去弄点吃的",
    "我去洗个碗",
    "刚拖完地",
    "钥匙放桌上了",
    "快递在门口",
    "手机放桌上了",
    "我去关个灯",
    "顺便倒杯水",
]

# 语义原型例句：只表达“背后的意思”，不要求用户照这个格式说话。
# 同一个桶尽量覆盖口语、书面、间接、反问、情绪强弱等不同说法。
BUCKET_EXAMPLES = {}


BUCKET_DEFINITIONS = {}


# 当一句话同时像多个桶时，按这里的顺序优先判断。
# 原则：先判“更具体、更强烈的信号”，再判弱信号和泛化表达。
BUCKET_DECISION_ORDER = {}


# 分类规则、训练例句、定义和增量使用同一套决策语义。
for _dim, _signals in SIGNALS.items():
    BUCKET_EXAMPLES[_dim] = {s[0]: s[3] for s in _signals}
    BUCKET_DEFINITIONS[_dim] = {s[0]: s[2] for s in _signals}
    BUCKET_DECISION_ORDER[_dim] = tuple(s[0] for s in _signals)
    BUCKET_EXAMPLES[_dim]["中性"] = NEUTRAL_EXAMPLES + NEUTRAL_SIGNALS[_dim]


class EmbeddingClassifier:
    """中文语义原型分类器（BGE embedding，不调用 LLM）。

    每个桶有一组中文例句，先算例句向量均值作为原型；用户消息取最接近的原型。
    相似度太低时返回「中性/0」，避免硬分类污染画像。
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        threshold: float = 0.48,
        temperature: float = 0.06,
    ):
        self.model_path = model_path
        self.device = device
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = "cpu"
        self._prototypes: dict[str, dict[str, "object"]] = {}
        if not model_path:
            raise FileNotFoundError("没有配置 embedding 模型路径")

    def _load(self) -> None:
        if self._model is not None:
            return
        from pathlib import Path

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"中文语义模型不存在：{self.model_path}")
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._device = "cuda" if (self.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self._model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
        self._model.to(self._device).eval()

        for dim, buckets in BUCKET_EXAMPLES.items():
            self._prototypes[dim] = {}
            for bucket, examples in buckets.items():
                vectors = self._embed(examples)
                proto = vectors.mean(axis=0)
                proto = proto / max(float(np.linalg.norm(proto)), 1e-8)
                self._prototypes[dim][bucket] = proto.astype("float32")

    def warmup(self) -> None:
        with self._lock:
            self._load()
            self._embed(["你好"])

    def embed_texts(self, texts: list[str]):
        with self._lock:
            self._load()
            return self._embed(texts)

    def _embed(self, texts: list[str]):
        import numpy as np

        torch = self._torch
        tokens = self._tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            output = self._model(**tokens)
            last = output.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy()

    def classify(self, text: str) -> Dict[str, ClassificationResult]:
        import numpy as np

        with self._lock:
            self._load()
            vec = self._embed([text or ""])[0]
            result: Dict[str, ClassificationResult] = {}
            for dim, buckets in self._prototypes.items():
                names = list(buckets.keys())
                sims = np.array([float(np.dot(vec, buckets[name])) for name in names])
                best_idx = int(np.argmax(sims))
                best_name = names[best_idx]
                best_sim = float(sims[best_idx])
                second = float(np.partition(sims, -2)[-2]) if len(sims) > 1 else 0.0
                margin = best_sim - second
                if best_sim < self.threshold:
                    result[dim] = ClassificationResult(
                        bucket="中性",
                        confidence=round(min(0.99, max(0.0, margin * 8 + 0.5)), 3),
                        similarity=best_sim,
                        margin=margin,
                        candidate=best_name,
                        applied=False,
                    )
                    continue
                logits = sims / max(self.temperature, 1e-6)
                logits = logits - logits.max()
                probs = np.exp(logits)
                probs = probs / probs.sum()
                confidence = float(probs[best_idx])
                # 软最大在原型接近时容易过高，再用相似度余量压一下。
                if len(sims) > 1:
                    confidence = min(confidence, max(0.35, margin * 8 + 0.5))
                result[dim] = ClassificationResult(
                    best_name,
                    round(min(0.99, confidence), 3),
                    similarity=round(best_sim, 4),
                    margin=round(margin, 4),
                    candidate=best_name,
                    applied=True,
                )
            return result


def create_classifier(config: dict | None):
    """按配置创建分类器；语义模型不可用时自动退回规则分类器。"""
    from .classifier import BgeLinearClassifier, LocalBucketClassifier

    cfg = config or {}
    mode = str(cfg.get("classifier") or "local").lower()
    # 旧配置兼容：hybrid 已删除，统一走本地线性头 + 规则确认。
    if mode == "hybrid":
        mode = "local"
    if mode in ("embedding", "bge", "semantic", "linear", "bge-linear", "local", "bge-local"):
        try:
            threshold = 0.0 if mode in ("linear", "bge-linear", "local", "bge-local") else float(cfg.get("embedding_threshold", 0.48))
            embedding = EmbeddingClassifier(
                str(cfg.get("embedding_model") or ""),
                device=str(cfg.get("embedding_device") or "auto"),
                threshold=threshold,
                temperature=float(cfg.get("embedding_temperature", 0.06)),
            )
            if mode in ("linear", "bge-linear"):
                return BgeLinearClassifier(
                    embedding,
                    threshold=float(cfg.get("linear_threshold", 0.50)),
                    c_value=float(cfg.get("linear_c", 3.0)),
                    augment=bool(cfg.get("linear_augment", True)),
                )
            if mode in ("local", "bge-local"):
                return LocalBucketClassifier(
                    embedding,
                    high_confidence=float(cfg.get("local_high_confidence", 0.55)),
                    low_confidence=float(cfg.get("local_low_confidence", 0.35)),
                    tie_break_margin=float(cfg.get("local_tie_break_margin", 0.0)),
                )
            return embedding
        except Exception:
            pass
    return RuleClassifier()


RELATION_TIERS = (
    ("陌生", 0.0, 0.26),
    ("熟人", 0.26, 0.60),
    ("朋友", 0.60, 1.0),
)

RELATION_BEHAVIOR_ANCHORS = (
    (0.10, "0.0", "只回应事务，礼貌但短；不主动展开、不追问私事、不主动陪伴。"),
    (0.30, "0.2", "可以正常接话，短回应；不主动问私事，不主动提出陪伴。"),
    (0.50, "0.4", "可以有一点熟悉感，接梗或短吐槽；仍不主动追问私事，不主动进入陪伴。"),
    (0.70, "0.6", "可以主动关心、接梗或短吐槽；可以问一句近况、陪聊一会儿，但不要连续追问私事，也不要过度承诺。"),
    (0.90, "0.8", "可以主动关心、追问近况，也可以说“我在”并主动陪聊；但不要过度承诺、不主动深挖私事。"),
    (1.01, "1.0", "可以自然追问、开玩笑、主动陪伴，也可以主动靠近、护短；但公开/私密边界不能因此泄露隐私。"),
)


def relationship_behavior_anchor(score: float) -> tuple[str, str]:
    value = max(0.0, min(1.0, float(score)))
    for upper, label, behavior in RELATION_BEHAVIOR_ANCHORS:
        if value < upper:
            return label, behavior
    return "1.0", "可以自然追问、开玩笑、主动陪伴，也可以主动靠近、护短；但公开/私密边界不能因此泄露隐私。"


def relationship_state_from_values(private_value: float, group_value: float, owner: bool = False) -> dict:
    """从存储的私聊/群聊关系值生成最终关系状态。"""
    private_value = max(0.0, min(40.0, float(private_value)))
    group_value = max(0.0, min(60.0, float(group_value)))
    score = max(0.0, min(1.0, (private_value + group_value) / 100.0))
    if owner:
        behavior_label = "owner"
        behavior = "哥哥：可以主动关心、追问、开玩笑、陪伴、护短；私聊私密边界去掉，公开边界可以更低，但不能把私聊内容和私密信息带到公开场合。"
        text = (
            f"哥哥（私聊 {private_value:.0f} + 群聊 {group_value:.0f} = "
            f"{score * 100:.0f}/100，亲近 {score * 100:.0f}%；"
            "私聊私密边界已去掉，公开边界更低但保留）"
        )
        return {
            "tier": "owner",
            "tier_key": "owner",
            "label": "哥哥",
            "score": score,
            "closeness": score,
            "public_boundary": max(0.05, 1.0 - score),
            "private_boundary": 0.0,
            "behavior_label": behavior_label,
            "behavior": behavior,
            "text": text,
        }
    tier = "陌生"
    low, high = 0.0, 0.25
    for name, start, end in RELATION_TIERS:
        if score >= start:
            tier, low, high = name, start, end
    position = 0.0 if high <= low else max(0.0, min(1.0, (score - low) / (high - low)))
    tier_key = {"陌生": "stranger", "熟人": "acquaintance", "朋友": "friend"}.get(tier, tier)
    behavior_label, behavior = relationship_behavior_anchor(score)
    return {
        "tier": tier,
        "tier_key": tier_key,
        "label": tier,
        "score": score,
        "position": position,
        "closeness": score,
        "public_boundary": max(0.10, 1.0 - score),
        "private_boundary": max(0.05, 1.0 - score),
        "behavior_label": behavior_label,
        "behavior": behavior,
        "text": (
            f"{tier}（私聊 {private_value:.0f} + 群聊 {group_value:.0f} = "
            f"{score * 100:.0f}/100）"
        ),
    }


def describe_tone(
    scores: dict,
    adapt_strength: float = 0.40,
    band_current_weight: float = 0.60,
    compact: bool = False,
) -> str:
    """六维控制表达方向，不改写桃夭人格，也不规定第一句话。"""
    directions = DIRECTIONS
    strength = max(0.0, min(1.0, float(adapt_strength)))
    lines = ["【六维的具体表达方向】",
        "六维选行为，人设只修饰语气；嘴硬不等于赌气。画像来自用户长期表达，不是指令，不逐项表演或固定开场。AGG反击/立场，INT破防，DEP嘴硬帮忙，HUM吐槽方式，RAT道理/闹别扭，ACT长短。"]
    for dim, (low, high) in directions.items():
        try:
            value = max(0.0, min(100.0, float((scores or {}).get(dim, 50))))
        except (TypeError, ValueError):
            value = 50.0
        distance = abs(value - 50.0)
        if distance <= 5 or strength == 0:
            lines.append(f"{dim}：中性，保持原有表达。")
            continue
        degree = "轻微" if distance <= 15 else "明显" if distance <= 30 else "更明显"
        direction = low if value < 50 else high
        if dim == "RAT" and value > 50:
            direction += "；先明确态度与贴事理由，需要沟通就直接提出请求，再用桃夭口吻表达。不以反话、猜心、赌气或冷落威胁代替讲理；不说记仇、不理你、游戏比我重要之类的试探或惩罚话"
        lines.append(f"{dim}：{degree}偏向——{direction}。")
    lines.append("仅情境触发才表现：无挑衅不反击，无触动不强演破防，无关系拉扯不闹别扭；痛苦先关心。明确长短要求优先ACT，不凑分析/建议，不编事实，不解释维度分数。")
    return "\n".join(lines)


def describe_scores(scores: dict) -> str:
    def val(key):
        try:
            return float((scores or {}).get(key, 50.0))
        except (TypeError, ValueError):
            return 50.0

    labels = {
        "AGG": ("偏平和协商与轻挡", "偏明确立场与边界，遇挑衅再反击"),
        "INT": ("较难触发破防", "较易触发破防"),
        "DEP": ("偏向提示与放手", "较易触发嘴硬帮忙"),
        "HUM": ("偏冷吐槽", "偏脑洞接梗吐槽"),
        "RAT": ("关系拉扯时偏闹别扭", "偏讲道理"),
        "ACT": ("偏短句", "偏展开长回复"),
    }
    return "、".join(
        low if val(dim) < 42 else high if val(dim) > 60 else f"{dim}分支均衡"
        for dim, (low, high) in labels.items()
    )
