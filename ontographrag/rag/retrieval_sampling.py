import hashlib
from typing import Any, Callable, List, Optional, Sequence

import numpy as np


def compute_candidate_limit(
    max_chunks: int,
    retrieval_temperature: float,
    shortlist_factor: int,
    *,
    hard_cap: int = 200,
) -> int:
    """Return the candidate-pool size for stochastic final-stage selection."""
    base = max(1, int(max_chunks))
    if float(retrieval_temperature or 0.0) <= 0.0:
        return base
    factor = max(1, int(shortlist_factor or 1))
    return max(base, min(base * factor, int(hard_cap)))


def stable_sample_seed(*parts: Any) -> int:
    raw = "||".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def select_ranked_subset(
    items: Sequence[Any],
    *,
    max_items: int,
    retrieval_temperature: float,
    shortlist_factor: int,
    sample_id: int = 0,
    seed_parts: Optional[Sequence[Any]] = None,
    score_getter: Optional[Callable[[Any], float]] = None,
) -> List[Any]:
    """Sample a top-k subset from a ranked list while preserving original order.

    When `retrieval_temperature <= 0`, returns the first `max_items` unchanged.
    Otherwise, samples from the top shortlist using a Gumbel-top-k perturbation
    over the item scores. The selected items are returned in their original
    ranked order so we isolate set changes from ordering changes.
    """
    ranked = list(items)
    k = max(1, int(max_items))
    if len(ranked) <= k:
        return ranked[:k]

    temp = float(retrieval_temperature or 0.0)
    if temp <= 0.0:
        return ranked[:k]

    candidate_limit = min(
        len(ranked),
        compute_candidate_limit(k, temp, shortlist_factor),
    )
    candidates = ranked[:candidate_limit]
    if len(candidates) <= k:
        return candidates[:k]

    get_score = score_getter or (lambda item: float(item.get("score", 0.0)))
    scores = np.asarray([float(get_score(item)) for item in candidates], dtype=float)
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    temp = max(temp, 1e-6)
    logits = scores / temp
    rng = np.random.default_rng(stable_sample_seed(*(seed_parts or ()), sample_id))
    gumbels = rng.gumbel(size=len(candidates))
    sampled_order = np.argsort(-(logits + gumbels))
    selected = np.sort(sampled_order[:k])
    return [candidates[idx] for idx in selected]
