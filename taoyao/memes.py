from __future__ import annotations

import json
import random
import hashlib
import re
import time
from pathlib import Path
from typing import Any


# Caring speech acts, rather than merely mentioning food, tiredness or greetings.
CARE_REMINDER_RE = re.compile(
    r'(?:记得|别忘了|要记着).{0,8}(?:吃饭|吃点|喝水|补水|补点能量|补充(?:能量|体力)|休息|保暖|添衣)|'
    r'(?:别|不要|不许).{0,5}(?:空腹|硬撑|撑着|逞强|饿着|饿坏|累坏|熬夜|着凉|冻着)|'
    r'(?:练完|运动完|忙完|睡前).{0,8}(?:补点能量|补充(?:能量|体力)|喝点水|喝水|吃点东西)'
)

CONTEXT_CUE_RULES: tuple[tuple[str, tuple[str, ...], float], ...] = (
    (r"(哈哈|笑死|好耶|太棒|开心|高兴|成功了|通过了|太好了)", ("happy",), 0.10),
    (r"(害羞|脸红|捂脸|别看我|低头|脑袋低|就一下|不要太久|别戳|(?:把)?脸埋(?:进|在|到)?(?:枕头|被子|被窝|怀)里?|耳朵红|脸发烫)", ("shy",), 0.14),
    (r"(生气|气死|烦死|讨厌|不理你|过分|欺负|哼)", ("angry",), 0.10),
    (r"(难过|伤心|委屈|想哭|失落|不开心)", ("sad",), 0.10),
    (r"(不懂|没听懂|什么意思|疑惑|问号|为什么|啊？)", ("confused",), 0.10),
    (r"(无措|手足无措|不知所措|不知道该怎么做|不知道该怎么办|不知道该怎么回应|慌了神|脑子打结|晕头转向|晕了)", ("at_loss",), 0.14),
    (r"(居然|竟然|真的吗|没想到|惊讶|吃惊)", ("surprised",), 0.10),
    (r"(困|睡觉|晚安|打哈欠|熬夜|睁不开眼)", ("sleepy",), 0.10),
    (r"(喜欢你|想抱|贴贴|亲亲|摸|陪你|心动)", ("affection",), 0.10),
    (r"(坏笑|调侃|逗你|得意|小恶魔|故意)", ("teasing",), 0.10),
    (r"(吃|饭|饿|好吃|奶茶|馋|夜宵)", ("food",), 0.08),
    (r"(害怕|吓|不敢|恐怖|紧张)", ("afraid",), 0.10),
    (r"(可爱|猫猫|卖萌|比耶|软乎乎)", ("cute",), 0.10),
    (r"(嫌弃|无语|恶心|看不下去|服了|受不了)", ("disgust",), 0.10),
    (r"(期待|等着|惊喜|马上|迫不及待)", ("look forward",), 0.10),
    (r"(嗯|哦|好|知道|淡定|普通|随便)", ("plain",), 0.08),
    (r"(舒服|放松|听歌|惬意|松了口气)", ("relief",), 0.10),
    (r"(发呆|放空|走神|空白|愣住)", ("space out",), 0.10),
    (r"(对不起|抱歉|我错|不好意思|愧疚)", ("sorry",), 0.10),
    (r"(安慰|别难过|休息|温柔|我在|陪你)", ("gentle",), 0.10),
    (r"(谢谢|感谢|多亏|感动)", ("grateful",), 0.10),
    (r"(担心|没事吧|还好吗|注意安全|小心)", ("worried",), 0.10),
    (r"(孤单|一个人|想你|陪我|寂寞)", ("lonely",), 0.10),
    (r"(激动|兴奋|等不及|太好了|冲)", ("excited",), 0.10),
    (r"(自豪|骄傲|厉害|被夸|牛)", ("proud",), 0.10),
    (r"(尴尬|社死|说错|当我没说|汗)", ("awkward",), 0.10),
    (r"(无奈|没办法|帮不了|摊手|无力)", ("helpless",), 0.10),
    (r"(累|困|没力气|想躺|撑不住)", ("tired",), 0.10),
    (r"(无聊|没劲|好闲|没意思)", ("bored",), 0.10),
    (r"(好奇|想知道|然后呢|追问|为什么)", ("curious",), 0.10),
    (r"(认真|正经|重要|严肃|别闹)", ("serious",), 0.08),
    (r"(赌气|傲娇|嘴硬|才不|哼|不理你|你赔|都怪你|不要太久|别戳)", ("sulky",), 0.14),
    (r"(怀念|想起以前|回忆|以前)", ("nostalgic",), 0.10),
    (r"(安心|松口气|那就好|放心|还好)", ("relieved",), 0.10),
    (r"(心疼|可怜|受苦|抱抱)", ("sympathy",), 0.10),
    (r"(加油|你可以|别放弃|打气|相信你)", ("encouraging",), 0.10),
)


class MemeLibrary:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest: dict = {}
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        self.by_category: dict[str, list[Path]] = {}
        for path in sorted(self.root.glob("*/*.png")):
            self.by_category.setdefault(path.parent.name, []).append(path)

        # manifest 可以给一个分类映射多张已有图片，方便新增“抱歉/温柔”等细分类。
        self.categories: dict[str, str] = {}
        self.category_aliases: dict[str,str] = {}
        self.intent_rules: list[tuple[str,list]] = []
        for cat, meta in (self.manifest.get("categories") or {}).items():
            cat = str(cat)
            desc = str(meta.get("description", "")) if isinstance(meta, dict) else str(meta)
            images = meta.get("images") if isinstance(meta, dict) else None
            if images:
                paths = []
                for rel in images:
                    path = self.root / str(rel)
                    if path.exists():
                        paths.append(path)
                if paths:
                    self.by_category[cat] = paths
            if self.by_category.get(cat):
                self.categories[cat] = desc
                if isinstance(meta,dict):
                    self.category_aliases.update({str(alias):cat for alias in meta.get('aliases',[])})
                    patterns=[re.compile(pattern,re.I) for pattern in meta.get('patterns',[])]
                    if patterns:self.intent_rules.append((cat,patterns))
        for cat in self.by_category:
            self.categories.setdefault(cat, cat)

        self._last: dict[str, list[Path]] = {}
        self._recent: list[Path] = []
        self._used_at: dict[tuple[str, str], float] = {}
        self.repeat_window_seconds = 300.0
        self._hash_cache: dict[Path, str] = {}
        self._embedder: Any = None
        self._cat_vectors = None
        self._cat_vector_names: list[str] = []

        self.keyword_rules = [
            ("at_loss", ("无措", "手足无措", "不知所措", "不知道该怎么做", "不知道该怎么办", "不知道该怎么回应", "慌了神", "脑子打结", "晕头转向", "晕了")),
            ("grateful", ("谢谢", "感谢", "感动", "多亏")),
            ("worried", ("担心", "没事吧", "还好吗", "注意安全")),
            ("lonely", ("孤单", "想你", "一个人", "陪我")),
            ("excited", ("激动", "兴奋", "太好了", "等不及")),
            ("proud", ("自豪", "骄傲", "厉害", "被夸")),
            ("awkward", ("尴尬", "社死", "说错话", "当我没说")),
            ("helpless", ("无奈", "没办法", "帮不上", "摊手")),
            ("tired", ("疲惫", "累死", "好累", "很累", "累了", "没力气", "想躺", "躺一会儿")),
            ("bored", ("无聊", "没劲", "好闲")),
            ("curious", ("好奇", "然后呢", "想知道", "追问")),
            ("serious", ("认真", "正经", "重要", "郑重")),
            ("sulky", ("赌气", "傲娇", "嘴硬", "才不是")),
            ("nostalgic", ("怀念", "想起以前", "回忆", "以前")),
            ("relieved", ("安心", "松口气", "那就好", "放心")),
            ("sympathy", ("心疼", "怜惜", "受苦")),
            ("encouraging", ("加油", "你可以的", "别放弃", "打气")),
            ("sorry", ("抱歉", "对不起", "不好意思", "愧疚", "抱歉啦")),
            ("gentle", ("歇一会儿", "休息", "慢慢来", "别太累", "没关系", "我在", "陪着", "温柔", "别急")),
            ("sad", ("难过", "不开心", "哭", "委屈", "伤心", "抽泣")),
            ("angry", ("生气", "哼", "讨厌", "不理你", "可恶", "闹脾气", "赌气", "不满")),
            ("afraid", ("害怕", "紧张", "担心", "不敢", "吓")),
            ("shy", ("害羞", "脸红", "捂脸", "别看我", "脸埋枕头", "脸埋进枕头", "脸埋被子", "耳朵红", "脸发烫")),
            ("affection", ("抱抱", "贴贴", "想你", "喜欢", "心动", "送花")),
            ("teasing", ("得意", "坏笑", "小恶魔", "调侃", "唱反调")),
            ("food", ("吃", "喝", "团子", "茶", "饿", "馋")),
            ("sleepy", ("困", "睡", "晚安", "打瞌睡", "趴睡")),
            ("surprised", ("惊讶", "真的吗", "诶", "居然", "吃惊")),
            ("look forward", ("期待", "惊喜", "眼睛发亮", "等着")),
            ("disgust", ("嫌弃", "无语", "恶心", "看不下去")),
            ("relief", ("舒服", "放松", "听歌", "惬意")),
            ("space out", ("发呆", "放空", "走神", "空白")),
            ("happy", ("开心", "高兴", "哈哈", "笑", "好耶", "太棒", "恭喜", "加油", "自信")),
            ("cute", ("可爱", "猫猫", "比耶", "卖萌")),
            ("confused", ("疑惑", "不懂", "什么意思", "为什么", "？", "?")),
            ("plain", ("嗯", "哦", "好", "知道", "认真", "淡定")),
        ]

    def set_embedder(self, embedder: Any) -> None:
        self._embedder = embedder
        self._cat_vectors = None

    def describe(self) -> list[str]:
        return [f"{cat}（{desc or cat}）" for cat, desc in self.categories.items()]

    def has_category(self, category: str) -> bool:
        category=self.category_aliases.get(category,category)
        category = {"无措": "at_loss", "不知所措": "at_loss", "手足无措": "at_loss"}.get(category, category)
        return category in self.by_category

    def _keyword_match(self, text: str) -> str | None:
        if CARE_REMINDER_RE.search(text) and not re.search(r'担心|没事吧|还好吗|注意安全',text):
            return 'gentle'
        for cat,patterns in self.intent_rules:
            if any(pattern.search(text) for pattern in patterns):return cat
        for cat, words in self.keyword_rules:
            if any(w in text for w in words):
                return cat
        return None

    @staticmethod
    def _last_sentence(text: str) -> str:
        parts = [part.strip() for part in re.split(r"[。！？!?；;\n]+", text or "") if part.strip()]
        return parts[-1] if parts else (text or "").strip()

    def _semantic_scores(self, text: str) -> dict[str, float]:
        if self._embedder is None:
            return {}
        if self._cat_vectors is None:
            names = list(self.categories.keys())
            texts = [f"{cat}：{self.categories[cat]}" for cat in names]
            self._cat_vectors = self._embedder.embed_texts(texts)
            self._cat_vector_names = names
        vec = self._embedder.embed_texts([text])[0]
        sims = self._cat_vectors @ vec
        return {
            name: float(score)
            for name, score in zip(self._cat_vector_names, sims)
        }

    def match_candidates(
        self, text: str, preferred: str | None = None, top_k: int = 3, margin: float = 0.04
    ) -> tuple[list[tuple[str, float]], tuple[str, float] | None]:
        """末句为主、整段消歧；全部文字已生成后选择表情。"""
        preferred=self.category_aliases.get(preferred,preferred)
        text = (text or "").strip()
        if not text:
            return [], None
        last = self._last_sentence(text)
        full_scores: dict[str, float] = {}
        last_scores: dict[str, float] = {}
        keyword_full = self._keyword_match(text)
        keyword_last = self._keyword_match(last)
        try:
            full_scores.update(self._semantic_scores(text))
            last_scores.update(self._semantic_scores(last))
        except Exception:
            pass
        if keyword_full:
            full_scores[keyword_full] = max(full_scores.get(keyword_full, 0.0), 0.8)
        if keyword_last:
            last_scores[keyword_last] = max(last_scores.get(keyword_last, 0.0), 0.8)
        categories = set(full_scores) | set(last_scores)
        scores = {
            cat: last_scores.get(cat, 0.0) * 0.75 + full_scores.get(cat, 0.0) * 0.25
            for cat in categories
        }
        # 所有分类都做本地语境修正；命中线索才加分，不命中不影响原分数。
        for pattern, cats, boost in CONTEXT_CUE_RULES:
            weight = 1.0 if re.search(pattern, last) else 0.25
            if re.search(pattern, text):
                for cat in cats:
                    scores[cat] = min(0.95, max(scores.get(cat, 0.0), 0.0) + boost * weight)
        for cat,patterns in self.intent_rules:
            weight=1.0 if any(pattern.search(last) for pattern in patterns) else 0.25
            if any(pattern.search(text) for pattern in patterns):
                scores[cat]=min(.95,scores.get(cat,0)+.12*weight)
        if CARE_REMINDER_RE.search(last) and not re.search(r'担心|没事吧|还好吗|注意安全',last):
            scores['gentle']=max(scores.get('gentle',0),0.90)
            for neutral in ('plain','food'):
                if neutral in scores:scores[neutral]=min(scores[neutral],0.55)
            if preferred in ('plain','food','淡定','认真'):preferred=None
        if not scores:
            return [], None

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:max(1, top_k)]
        if not ranked:
            return [], None
        strong_keyword = keyword_last or keyword_full
        if ranked[0][1] < 0.55 and ranked[0][0] != strong_keyword:
            return ranked, None
        # 本地已经很自信时，直接采用本地结果，不让 LLM 建议改写。
        if ranked[0][1] >= 0.65 and (
            len(ranked) == 1 or ranked[0][1] - ranked[1][1] >= 0.05
        ):
            return ranked, ranked[0]

        # LLM 的 [[meme:分类]] 是明确建议：本地判断不够确定，
        # 或建议与最高分很接近时，允许它修正本地结果。
        preferred_aliases = {
            "开心": "happy", "笑": "happy", "害羞": "shy", "脸红": "shy",
            "生气": "angry", "不满": "angry", "难过": "sad", "哭": "sad",
            "疑惑": "confused", "疑问": "confused", "惊讶": "surprised",
            "无措": "at_loss", "不知所措": "at_loss", "手足无措": "at_loss", "晕": "at_loss",
            "困": "sleepy", "抱抱": "affection", "贴贴": "affection",
            "得意": "teasing", "嫌弃": "disgust", "无语": "disgust",
            "淡定": "plain", "认真": "plain", "期待": "look forward",
            "害怕": "afraid", "可爱": "cute", "舒服": "relief",
            "发呆": "space out", "抱歉": "sorry", "温柔": "gentle",
            "安慰": "gentle", "赌气": "sulky", "傲娇": "sulky",
        }
        preferred = preferred_aliases.get(preferred or "", preferred)
        if preferred and preferred in scores:
            preferred_score = scores[preferred]
            top_score = ranked[0][1]
            if preferred_score >= 0.55 and (
                top_score < 0.60 or top_score - preferred_score <= 0.08
            ):
                return ranked, (preferred, preferred_score)

        chosen = ranked[0]
        if len(ranked) >= 2 and ranked[0][1] - ranked[1][1] <= margin:
            similar = [item for item in ranked if ranked[0][1] - item[1] <= margin]
            # 二次比较：模型标签命中相似组时优先标签，否则取相似组最高分。
            preferred_hit = next((item for item in similar if item[0] == preferred), None)
            chosen = preferred_hit or max(similar, key=lambda item: item[1])
            # 如果二次比较后仍与最高分同级，保留最高分那个。
            if chosen[1] < ranked[0][1] - margin:
                chosen = ranked[0]
        return ranked, chosen

    def match(self, text: str, preferred: str | None = None) -> tuple[str | None, float]:
        """返回 (表情分类, 匹配分)。"""
        ranked, chosen = self.match_candidates(text, preferred=preferred)
        if not chosen:
            return None, 0.0
        return chosen[0], round(chosen[1], 4)

    @staticmethod
    def _number(path: Path) -> int:
        match = re.match(r"^(\d+)_", path.stem)
        return int(match.group(1)) if match else 1

    def infer_intensity(self, text: str) -> float:
        """从回复文字估计情绪强度，0=很淡，1=很强烈。"""
        text = (text or "").strip()
        if not text:
            return 0.0
        score = 0.35
        strong = (
            "非常", "特别", "太", "超级", "最", "极", "死了", "爆", "疯", "崩",
            "哭死", "笑死", "气死", "累死", "吓死", "爱死", "好想", "真的",
        )
        score += min(0.35, sum(text.count(w) for w in strong) * 0.07)
        score += min(0.2, (text.count("！") + text.count("!")) * 0.05)
        if re.search(r"(哈哈|呜呜|啊啊|嘿嘿|嘻嘻){2,}", text):
            score += 0.15
        if len(text) > 40:
            score += 0.05
        return max(0.0, min(1.0, score))

    def _fingerprint(self, path: Path) -> str:
        cached = self._hash_cache.get(path)
        if cached:
            return cached
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            digest = str(path)
        self._hash_cache[path] = digest
        return digest

    def remember(self, path: Path, conversation: str = '') -> None:
        now = time.monotonic()
        self._used_at = {k: t for k, t in self._used_at.items() if now-t < self.repeat_window_seconds}
        self._used_at[(conversation, self._fingerprint(path))] = now

    def can_use(self, path: Path, category: str | None, conversation: str = '') -> bool:
        unique = {self._fingerprint(p) for p in self.by_category.get(category, [])}
        if len(unique) == 1 and self._fingerprint(path) in unique:
            return True
        used = self._used_at.get((conversation, self._fingerprint(path)))
        return used is None or time.monotonic()-used >= self.repeat_window_seconds

    def pick(self, category: str | None, intensity: float = 0.5, conversation: str = '') -> Path | None:
        """按文件名前缀数字选图：数字越大，情绪越明显。"""
        aliases = {
            "开心": "happy", "笑": "happy", "害羞": "shy", "脸红": "shy",
            "生气": "angry", "不满": "angry", "难过": "sad", "哭": "sad",
            "疑惑": "confused", "疑问": "confused", "晕": "at_loss",
            "无措": "at_loss", "不知所措": "at_loss", "手足无措": "at_loss",
            "惊讶": "surprised", "吃惊": "surprised", "困": "sleepy",
            "困倦": "sleepy", "睡觉": "sleepy", "抱抱": "affection",
            "贴贴": "affection", "心动": "affection", "得意": "teasing",
            "嫌弃": "disgust", "无语": "disgust", "吃": "food",
            "喝茶": "food", "团子": "food", "淡定": "plain",
            "认真": "plain", "reply": "plain", "期待": "look forward",
            "惊喜": "look forward", "害怕": "afraid", "可爱": "cute",
            "舒服": "relief", "听歌": "relief", "发呆": "space out",
            "抱歉": "sorry", "对不起": "sorry", "温柔": "gentle",
            "安慰": "gentle", "休息": "gentle",
        }
        if not category:
            return None
        category = self.category_aliases.get(category,aliases.get(category, category))
        if category not in self.by_category:
            return None
        raw_paths = list(dict.fromkeys(self.by_category.get(category) or []))
        paths = []
        seen_hashes = set()
        for item in raw_paths:
            fingerprint = self._fingerprint(item)
            if fingerprint not in seen_hashes:
                seen_hashes.add(fingerprint)
                paths.append(item)
        if not paths:
            return None
        available = [p for p in paths if self.can_use(p, category, conversation)]
        if not available:
            return None  # 多图分类全在冷却时宁可不发，不换错情绪或强行重复。
        paths = available
        recent = self._last.setdefault(category, [])
        recent_hashes = {self._fingerprint(item) for item in self._recent}
        candidates = [
            p for p in paths
            if p not in recent and self._fingerprint(p) not in recent_hashes
        ]
        if not candidates:
            candidates = [p for p in paths if p not in recent] or paths

        numbers = [self._number(p) for p in candidates]
        max_num = max(numbers) if numbers else 1
        target = 1.0 + max(0.0, min(1.0, float(intensity))) * (max_num - 1)
        ranked = sorted(candidates, key=lambda p: abs(self._number(p) - target))
        # 不只取“距离完全相同”的一张，前几张都能用，避免固定重复。
        pool = ranked[: min(5, len(ranked))]
        chosen = random.choice(pool)
        recent.append(chosen)
        del recent[:-3]
        self._recent.append(chosen)
        del self._recent[:-8]
        self.remember(chosen, conversation)
        return chosen

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.by_category.items()}

    def total(self) -> int:
        return len({path for paths in self.by_category.values() for path in paths})

    def category_for_text(self, text: str) -> str | None:
        cat, _ = self.match(text)
        return cat
