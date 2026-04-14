"""Official-style answer normalization and EM/F1 metrics for QA benchmarks."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple


OFFICIAL_ANSWER_METRIC_DATASETS = {
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "multihoprag",
}


def _normalize_answer(text: str) -> str:
    """HotpotQA-style answer normalization."""
    text = str(text or "").lower()

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text)))


def _f1_single(prediction: str, ground_truth: str) -> float:
    pred = _normalize_answer(prediction)
    gold = _normalize_answer(ground_truth)

    special = {"yes", "no", "maybe", "insufficient information"}
    if pred in special or gold in special:
        return 1.0 if pred == gold else 0.0

    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _em_single(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def compute_answer_em_f1(
    prediction: str,
    gold_answer: str,
    aliases: Optional[Sequence[str]] = None,
) -> Tuple[float, float]:
    """Return max EM/F1 across the gold answer and any alias variants."""
    candidates: List[str] = [str(gold_answer or "")]
    candidates.extend(str(a or "") for a in (aliases or []))
    candidates = [c for c in candidates if c.strip()]
    if not candidates:
        return 0.0, 0.0

    em = max(_em_single(prediction, candidate) for candidate in candidates)
    f1 = max(_f1_single(prediction, candidate) for candidate in candidates)
    return em, f1


def supports_official_answer_metrics(dataset_name: str) -> bool:
    return str(dataset_name or "").strip().lower() in OFFICIAL_ANSWER_METRIC_DATASETS
