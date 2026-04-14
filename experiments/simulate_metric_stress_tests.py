"""
Synthetic stress tests for uncertainty/hallucination metrics.

Goal:
  Run fast simulations that highlight *different evaluation aspects* beyond AUROC/AUREC,
  including calibration quality (ECE/MCE), imbalance behavior (PR-AUC), and subgroup
  robustness (grounding strata).

Usage:
  python experiments/simulate_metric_stress_tests.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception as e:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for this simulation script") from e


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 1e-6, 1.0 - 1e-6)


def _ece_mce(y_true: np.ndarray, confidence: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """ECE/MCE for confidence as estimated P(correct)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidence >= lo) & (confidence < hi if i < n_bins - 1 else confidence <= hi)
        if not np.any(mask):
            continue
        acc_bin = float(np.mean(y_true[mask]))
        conf_bin = float(np.mean(confidence[mask]))
        gap = abs(acc_bin - conf_bin)
        ece += (np.sum(mask) / n) * gap
        mce = max(mce, gap)
    return {"ece": float(ece), "mce": float(mce)}


def _aurec(y_true: np.ndarray, uncertainty: np.ndarray) -> float:
    """Area Under Rejection Error Curve (lower is better)."""
    order = np.argsort(-uncertainty)  # reject most uncertain first
    errors = 1.0 - y_true[order]
    suffix = np.cumsum(errors[::-1])[::-1]
    denom = np.arange(len(y_true), 0, -1, dtype=float)
    rejection_errors = suffix / denom
    return float(np.mean(rejection_errors))


def _evaluate(y_true: np.ndarray, confidence: np.ndarray) -> Dict[str, float]:
    confidence = _clip01(confidence)
    uncertainty = 1.0 - confidence
    cal = _ece_mce(y_true, confidence)
    return {
        "auroc": float(roc_auc_score(y_true, confidence)),
        "pr_auc": float(average_precision_score(y_true, confidence)),
        "aurec": _aurec(y_true, uncertainty),
        "brier": float(np.mean((confidence - y_true) ** 2)),
        "ece": cal["ece"],
        "mce": cal["mce"],
    }


def _test_calibration_vs_ranking(rng: np.random.Generator) -> Dict[str, Dict[str, float]]:
    n = 5000
    latent = rng.normal(0, 1, n)
    p_true = _sigmoid(1.2 * latent)
    y = rng.binomial(1, p_true).astype(float)

    rank_signal = _clip01(p_true + rng.normal(0, 0.08, n))
    overconfident = _sigmoid(5.0 * (rank_signal - 0.5))  # monotonic transform => similar ranking
    underconfident = _clip01(0.5 + 0.35 * (rank_signal - 0.5))
    random_conf = rng.uniform(0.0, 1.0, n)

    return {
        "ranked_baseline": _evaluate(y, rank_signal),
        "overconfident_monotonic": _evaluate(y, overconfident),
        "underconfident_monotonic": _evaluate(y, underconfident),
        "random": _evaluate(y, random_conf),
    }


def _test_imbalance_effect(rng: np.random.Generator) -> Dict[str, Dict[str, float]]:
    n = 7000
    latent = rng.normal(0, 1, n)
    # strong class imbalance: mostly correct answers
    p_true = _sigmoid(2.1 + 0.9 * latent)
    y = rng.binomial(1, p_true).astype(float)

    good_metric = _clip01(p_true + rng.normal(0, 0.08, n))
    medium_metric = _clip01(0.65 * p_true + 0.35 * rng.uniform(0, 1, n))
    random_conf = rng.uniform(0.0, 1.0, n)

    return {
        "positive_rate_correct": {"value": float(np.mean(y))},
        "good_metric": _evaluate(y, good_metric),
        "medium_metric": _evaluate(y, medium_metric),
        "random": _evaluate(y, random_conf),
    }


def _test_grounding_strata(rng: np.random.Generator) -> Dict[str, Dict[str, float]]:
    n = 6000
    grounding = rng.binomial(1, 0.5, n)  # 1=high grounding, 0=low grounding

    latent = rng.normal(0, 1, n)
    p_high = _sigmoid(1.4 * latent + 0.2)
    p_low = _sigmoid(0.7 * latent - 0.3)
    p_true = np.where(grounding == 1, p_high, p_low)
    y = rng.binomial(1, p_true).astype(float)

    # Semantic-style metric: good in high-grounding, collapses in low-grounding
    semantic_conf = np.where(
        grounding == 1,
        _clip01(p_true + rng.normal(0, 0.09, n)),
        _clip01(0.55 + rng.normal(0, 0.02, n)),
    )

    # Structural-style metric: keeps useful signal in low-grounding too
    structural_conf = np.where(
        grounding == 1,
        _clip01(p_true + rng.normal(0, 0.10, n)),
        _clip01(0.35 + 0.55 * p_true + rng.normal(0, 0.09, n)),
    )

    out = {
        "semantic_overall": _evaluate(y, semantic_conf),
        "structural_overall": _evaluate(y, structural_conf),
    }

    for g_name, g_val in (("high_grounding", 1), ("low_grounding", 0)):
        mask = grounding == g_val
        out[f"semantic_{g_name}"] = _evaluate(y[mask], semantic_conf[mask])
        out[f"structural_{g_name}"] = _evaluate(y[mask], structural_conf[mask])

    return out


def _pretty_print(title: str, rows: Dict[str, Dict[str, float]], keys: List[str]):
    print(f"\n=== {title} ===")
    for name, vals in rows.items():
        if "value" in vals:
            print(f"{name:28s} | {vals['value']:.4f}")
            continue
        payload = " | ".join(f"{k}={vals[k]:.4f}" for k in keys if k in vals)
        print(f"{name:28s} | {payload}")


def main():
    rng = np.random.default_rng(42)

    report = {
        "calibration_vs_ranking": _test_calibration_vs_ranking(rng),
        "imbalance_effect": _test_imbalance_effect(rng),
        "grounding_strata": _test_grounding_strata(rng),
    }

    _pretty_print(
        "Test 1: Calibration vs Ranking",
        report["calibration_vs_ranking"],
        ["auroc", "aurec", "ece", "mce", "brier"],
    )
    _pretty_print(
        "Test 2: Class Imbalance (AUROC vs PR-AUC)",
        report["imbalance_effect"],
        ["auroc", "pr_auc", "aurec", "ece"],
    )
    _pretty_print(
        "Test 3: Grounding-Stratified Robustness",
        report["grounding_strata"],
        ["auroc", "aurec", "ece"],
    )

    out_path = Path("results") / "simulated_metric_stress_tests.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved simulation report to: {out_path}")


if __name__ == "__main__":
    main()
