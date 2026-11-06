from __future__ import annotations

import threading
from typing import Dict
from .decision_semantics import NEUTRAL_SIGNALS

from .personality import (
    BUCKET_DECISION_ORDER,
    BUCKET_EXAMPLES,
    DIMENSIONS,
    NEUTRAL_EXAMPLES,
    ClassificationResult,
    EmbeddingClassifier,
    RuleClassifier,
)


def augment_bucket_examples(examples: list[str]) -> list[str]:
    """生成不改变语义的口语外壳变体，扩大线性头的训练覆盖。"""
    prefixes = ("", "其实，", "说真的，", "刚刚，")
    suffixes = ("。", "……")
    out: list[str] = []
    for raw in examples:
        base = str(raw).strip().rstrip("。！？!?…，, ")
        if not base:
            continue
        for prefix in prefixes:
            for suffix in suffixes:
                item = f"{prefix}{base}{suffix}"
                if item not in out:
                    out.append(item)
    return out


class BgeLinearClassifier:
    """冻结 BGE，只用例句训练每个维度的轻量线性分类头。"""

    def __init__(
        self,
        embedding: EmbeddingClassifier,
        threshold: float = 0.50,
        c_value: float = 3.0,
        augment: bool = True,
        tie_break_margin: float = 0.0,
    ):
        self.embedding = embedding
        self.threshold = float(threshold)
        self.c_value = float(c_value)
        self.augment = bool(augment)
        self.tie_break_margin = float(tie_break_margin)
        self._models: dict[str, object] = {}
        self._centroids: dict[str, dict[str, object]] = {}
        self._classes: dict[str, list[str]] = {}
        self._loaded = False
        self._lock = threading.Lock()

    def warmup(self) -> None:
        with self._lock:
            self._load()

    def _load(self) -> None:
        if self._loaded:
            return
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        self.embedding.warmup()
        for dim, buckets in BUCKET_EXAMPLES.items():
            texts: list[str] = []
            labels: list[str] = []
            examples_by_bucket = dict(buckets)
            examples_by_bucket["中性"] = NEUTRAL_EXAMPLES + NEUTRAL_SIGNALS.get(dim, [])
            for bucket, examples in examples_by_bucket.items():
                expanded = augment_bucket_examples(examples) if self.augment else examples
                for example in expanded:
                    texts.append(example)
                    labels.append(bucket)
            vectors = self.embedding.embed_texts(texts)
            model = LogisticRegression(
                max_iter=500,
                C=self.c_value,
                class_weight="balanced",
                solver="lbfgs",
            )
            model.fit(vectors, labels)
            self._models[dim] = model
            self._classes[dim] = list(model.classes_)
            self._centroids[dim] = {
                bucket: vectors[[i for i, label in enumerate(labels) if label == bucket]].mean(axis=0)
                for bucket in examples_by_bucket
            }
        self._loaded = True

    def classify(self, text: str) -> Dict[str, ClassificationResult]:
        import numpy as np

        self.warmup()
        vector = self.embedding.embed_texts([text or ""])[0]
        result: Dict[str, ClassificationResult] = {}
        for dim, buckets in BUCKET_EXAMPLES.items():
            names = [*buckets, "中性"]
            model = self._models[dim]
            classes = self._classes[dim]
            probabilities = model.predict_proba([vector])[0]
            ranked = np.argsort(probabilities)[::-1]
            top_idx = int(ranked[0])
            if (
                self.tie_break_margin > 0
                and len(ranked) > 1
                and float(probabilities[top_idx] - probabilities[int(ranked[1])])
                <= self.tie_break_margin
            ):
                near = {
                    classes[int(index)]
                    for index in ranked
                    if float(probabilities[top_idx] - probabilities[int(index)])
                    <= self.tie_break_margin
                }
                ordered = [name for name in BUCKET_DECISION_ORDER.get(dim, ()) if name in near]
                if ordered:
                    top_idx = classes.index(ordered[0])
            bucket = classes[top_idx]
            confidence = float(probabilities[top_idx])
            sims = {
                name: float(np.dot(vector, self._centroids[dim][name]))
                for name in names
            }
            ordered = sorted(sims.values(), reverse=True)
            best_sim = ordered[0]
            second_sim = ordered[1] if len(ordered) > 1 else 0.0
            result[dim] = ClassificationResult(
                bucket=bucket,
                confidence=round(min(0.99, confidence), 4),
                similarity=round(best_sim, 4),
                margin=round(best_sim - second_sim, 4),
                candidate=bucket,
                applied=confidence >= self.threshold,
            )
        return result


class LocalBucketClassifier:
    """BGE 线性头为主，规则只做低置信度确认，边界按桶顺序 tie-break。"""

    def __init__(
        self,
        embedding: EmbeddingClassifier,
        high_confidence: float = 0.55,
        low_confidence: float = 0.35,
        tie_break_margin: float = 0.0,
    ):
        self.embedding = embedding
        self.linear = BgeLinearClassifier(
            embedding,
            threshold=0.0,
            augment=True,
            tie_break_margin=tie_break_margin,
        )
        self.rules = RuleClassifier()
        self.high_confidence = float(high_confidence)
        self.low_confidence = float(low_confidence)

    def warmup(self) -> None:
        self.linear.warmup()

    def classify(self, text: str) -> Dict[str, ClassificationResult]:
        linear = self.linear.classify(text)
        rules = self.rules.classify(text)
        out: Dict[str, ClassificationResult] = {}
        for dim in DIMENSIONS:
            top = linear[dim]
            rule = rules[dim]
            if top.bucket == "中性":
                out[dim] = ClassificationResult(
                    bucket="中性",
                    confidence=top.confidence,
                    similarity=top.similarity,
                    margin=top.margin,
                    candidate="中性",
                    applied=False,
                )
                continue
            if top.confidence >= self.high_confidence:
                out[dim] = ClassificationResult(
                    bucket=top.bucket,
                    confidence=top.confidence,
                    similarity=top.similarity,
                    margin=top.margin,
                    candidate=top.bucket,
                    applied=True,
                )
                continue
            # 低置信度：规则只在“同名确认”时帮忙，不强行改判。
            if (
                rule.bucket == top.bucket
                and rule.bucket != "中性"
                and rule.confidence >= 0.60
            ):
                out[dim] = ClassificationResult(
                    bucket=top.bucket,
                    confidence=max(top.confidence, min(0.85, rule.confidence)),
                    similarity=top.similarity,
                    margin=top.margin,
                    candidate=top.bucket,
                    applied=True,
                )
            elif top.confidence >= self.low_confidence:
                out[dim] = ClassificationResult(
                    bucket="中性",
                    confidence=top.confidence,
                    similarity=top.similarity,
                    margin=top.margin,
                    candidate=top.bucket,
                    applied=False,
                )
            else:
                out[dim] = ClassificationResult(
                    bucket="中性",
                    confidence=top.confidence,
                    similarity=top.similarity,
                    margin=top.margin,
                    candidate=top.bucket,
                    applied=False,
                )
        return out
