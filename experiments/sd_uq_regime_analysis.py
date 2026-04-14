"""
sd_uq_regime_analysis.py — compare SD-UQ vs VN-Entropy by regime.

This is a post-hoc analysis over saved per-question artifacts. It is intended
to answer a narrow ablation question:

    When does projecting response embeddings into the question-orthogonal
    subspace actually help, relative to plain VN-Entropy?

The script reports AUROC / AUREC for SD-UQ and VN-Entropy under several
regimes:
  - all rows
  - high vs low grounding
  - binary vs free-text
  - 2-hop vs 4-hop

It emits both pooled summaries and per-dataset summaries.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = REPO_ROOT / "results" / "analyses"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_DATASET_FILES: Dict[str, Path] = {
    "bioasq": REPO_ROOT / "results" / "runs" / "20260402-172730-bioasq-n100-full-metrics-evaluation-subset" / "questions" / "bioasq_n100_seed42_hcbd4a4111e21_thr0.1_k10_questions.json",
    "realmedqa": REPO_ROOT / "results" / "runs" / "20260402-155138-realmedqa-n100-full-metrics-evaluation-subset" / "questions" / "realmedqa_n100_seed0_h3100790161ec_thr0.1_k10_questions.json",
    "2wikimultihopqa": REPO_ROOT / "results" / "runs" / "20260402-185601-2wikimultihopqa-n100-full-metrics-evaluation-subset" / "questions" / "2wikimultihopqa_n100_seed42_h7767a87f44f4_thr0.1_k10_questions.json",
    "pubmedqa": REPO_ROOT / "results" / "runs" / "20260401-074100-pubmedqa-n100-full-metrics-evaluation-subset-rebuildkg" / "questions" / "pubmedqa_n100_seed42_hefb402b9dd8c_thr0.1_k10_questions.json",
}


@dataclass
class RegimeResult:
    dataset: str
    system: str
    regime: str
    n: int
    positives: int
    negatives: int
    vn_auroc: float
    sd_auroc: float
    delta_auroc: float
    vn_aurec: float
    sd_aurec: float
    delta_aurec: float


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("questions", "rows", "details", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported record format: {path}")


def _aurec(y_true: np.ndarray, uncertainty: np.ndarray) -> float:
    order = np.argsort(-uncertainty)
    errors = (1 - y_true[order]).astype(float)
    suffix_errors = np.cumsum(errors[::-1])[::-1]
    n_remaining = np.arange(len(y_true), 0, -1, dtype=float)
    return float((suffix_errors / n_remaining).mean())


def _evaluate(records: Sequence[Dict[str, Any]], system: str) -> RegimeResult | None:
    answered = [r for r in records if not r.get(f"{system}_generation_failed", False)]
    if len(answered) < 2:
        return None

    y_true = np.array([1.0 if r.get(f"{system}_correct", False) else 0.0 for r in answered], dtype=float)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return None

    vn = np.array([float(r.get(f"{system}_vn_entropy", 0.0)) for r in answered], dtype=float)
    sd = np.array([float(r.get(f"{system}_sd_uq", 0.0)) for r in answered], dtype=float)

    vn_auroc = float(roc_auc_score(y_true, -vn))
    sd_auroc = float(roc_auc_score(y_true, -sd))
    vn_aurec = _aurec(y_true, vn)
    sd_aurec = _aurec(y_true, sd)

    return RegimeResult(
        dataset="",
        system=system,
        regime="",
        n=len(answered),
        positives=positives,
        negatives=negatives,
        vn_auroc=vn_auroc,
        sd_auroc=sd_auroc,
        delta_auroc=sd_auroc - vn_auroc,
        vn_aurec=vn_aurec,
        sd_aurec=sd_aurec,
        delta_aurec=vn_aurec - sd_aurec,
    )


def _regimes(grounding_threshold: float) -> Dict[str, Callable[[Dict[str, Any]], bool]]:
    return {
        "all": lambda r: True,
        "grounding_low": lambda r: float(r.get("grounding_quality", 0.0) or 0.0) < grounding_threshold,
        "grounding_high": lambda r: float(r.get("grounding_quality", 0.0) or 0.0) >= grounding_threshold,
        "binary": lambda r: r.get("task_type") == "binary",
        "free_text": lambda r: r.get("task_type") == "free_text",
        "hop_2": lambda r: r.get("hop_count") == 2,
        "hop_4": lambda r: r.get("hop_count") == 4,
    }


def _print_table(title: str, rows: Iterable[RegimeResult]) -> None:
    rows = list(rows)
    print(f"\n{title}")
    if not rows:
        print("  (no valid rows)")
        return
    header = (
        f"{'dataset':<18} {'system':<8} {'regime':<15} {'n':>4} "
        f"{'VN_AUROC':>8} {'SD_AUROC':>8} {'dAUROC':>8} "
        f"{'VN_AUREC':>8} {'SD_AUREC':>8} {'dAUREC':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.dataset:<18} {row.system:<8} {row.regime:<15} {row.n:>4} "
            f"{row.vn_auroc:>8.3f} {row.sd_auroc:>8.3f} {row.delta_auroc:>8.3f} "
            f"{row.vn_aurec:>8.3f} {row.sd_aurec:>8.3f} {row.delta_aurec:>8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SD-UQ vs VN-Entropy by regime.")
    parser.add_argument(
        "--grounding-threshold",
        type=float,
        default=0.5,
        help="Threshold for high vs low grounding split (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ANALYSIS_DIR / "sd_uq_regime_analysis.json",
        help="Where to save the JSON summary.",
    )
    args = parser.parse_args()

    regime_fns = _regimes(args.grounding_threshold)

    dataset_records: Dict[str, List[Dict[str, Any]]] = {}
    for dataset, path in DEFAULT_DATASET_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing default analysis input for {dataset}: {path}")
        records = _load_records(path)
        for record in records:
            record["_dataset"] = dataset
        dataset_records[dataset] = records

    per_dataset: List[RegimeResult] = []
    pooled: List[RegimeResult] = []

    for dataset, records in dataset_records.items():
        for regime_name, regime_fn in regime_fns.items():
            subset = [r for r in records if regime_fn(r)]
            for system in ("vanilla", "kg"):
                result = _evaluate(subset, system)
                if result is None:
                    continue
                result.dataset = dataset
                result.regime = regime_name
                per_dataset.append(result)

    all_records = [r for records in dataset_records.values() for r in records]
    for regime_name, regime_fn in regime_fns.items():
        subset = [r for r in all_records if regime_fn(r)]
        for system in ("vanilla", "kg"):
            result = _evaluate(subset, system)
            if result is None:
                continue
            result.dataset = "pooled"
            result.regime = regime_name
            pooled.append(result)

    payload = {
        "grounding_threshold": args.grounding_threshold,
        "inputs": {dataset: str(path) for dataset, path in DEFAULT_DATASET_FILES.items()},
        "pooled": [asdict(r) for r in pooled],
        "per_dataset": [asdict(r) for r in per_dataset],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    pooled_sorted = sorted(pooled, key=lambda r: (r.system, r.regime))
    per_dataset_sorted = sorted(per_dataset, key=lambda r: (r.dataset, r.system, r.regime))
    _print_table("Pooled SD-UQ vs VN-Entropy by regime", pooled_sorted)
    _print_table("Per-dataset SD-UQ vs VN-Entropy by regime", per_dataset_sorted)
    print(f"\nSaved JSON summary to {args.output}")


if __name__ == "__main__":
    main()
