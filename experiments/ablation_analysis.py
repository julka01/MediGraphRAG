"""
ablation_analysis.py — post-hoc analysis and figure generation for paper ablations.

Produces:
  figures/ece_reliability.pdf     — reliability diagrams + ECE for all metrics
  figures/n_sweep.pdf             — AUROC vs N for vanilla vs KG-RAG
  figures/temp_sweep.pdf          — AUROC vs temperature (2WikiMHQA)
  figures/hop_depth.pdf           — AUROC by hop depth (0-hop, 1-hop, 2-hop)
  figures/grounding_threshold.pdf — routed AUROC vs grounding threshold g*
  figures/retrieval_overlap.pdf   — Jaccard overlap of retrieved chunks across samples
                                    (direct proof of context determinism)
  figures/grounding_split.pdf     — AUROC by grounding quality bin (high-g vs low-g)
                                    for structural vs generative metrics

Usage:
    python experiments/ablation_analysis.py                                    # all figures
    python experiments/ablation_analysis.py --figures retrieval_overlap grounding_split
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
ABLATIONS_DIR = RESULTS_DIR / "ablations"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Metrics reported in the main paper (generative + structural)
GENERATIVE_METRICS = [
    "semantic_entropy", "discrete_semantic_entropy", "sre_uq",
    "p_true", "selfcheckgpt", "vn_entropy", "sd_uq",
]
STRUCTURAL_METRICS = ["graph_path_support", "subgraph_perturbation_stability"]
ALL_METRICS = GENERATIVE_METRICS + STRUCTURAL_METRICS

METRIC_LABELS = {
    "semantic_entropy": "SE",
    "discrete_semantic_entropy": "DSE",
    "sre_uq": "SRE-UQ",
    "p_true": "P(True)",
    "selfcheckgpt": "SCG",
    "vn_entropy": "VN-Entropy",
    "sd_uq": "SD-UQ",
    "graph_path_support": "GPS",
    "subgraph_perturbation_stability": "SPS-UQ",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_details(results_path: Path) -> List[Dict]:
    """Load per-question detail records from a dataset results JSON."""
    with open(results_path) as f:
        data = json.load(f)
    # Handle both direct list and nested {"config_results": [...]} structures
    if isinstance(data, list):
        blocks = data
    elif "config_results" in data:
        blocks = data["config_results"]
    elif "results" in data:
        # summary file — flatten
        details = []
        for ds in data["results"]:
            for cr in ds.get("config_results", []):
                details.extend(cr.get("details", []))
        return details
    else:
        blocks = [data]

    details = []
    for block in blocks:
        if isinstance(block, dict):
            details.extend(block.get("details", []))
    return details


def _get_scores_labels(
    details: List[Dict],
    metric: str,
    system: str = "kg",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (uncertainty_scores, correctness_labels) for one metric+system."""
    prefix = f"{system}_"
    scores, labels = [], []
    for d in details:
        s = d.get(f"{prefix}{metric}")
        correct = d.get(f"{prefix}_correct", d.get(f"{system}_correct"))
        if s is None or correct is None:
            continue
        # orientation: high score = high uncertainty = likely wrong
        # p_true is confidence, so invert
        if metric == "p_true":
            s = 1.0 - s
        scores.append(float(s))
        labels.append(int(bool(correct)))
    return np.array(scores), np.array(labels)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC (higher uncertainty → incorrect)."""
    if len(scores) < 2 or labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(labels, -scores))
    except Exception:
        return float("nan")


def _ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error.

    Bins queries by confidence (1 - uncertainty). For each bin, computes
    |mean_confidence - mean_accuracy|, weighted by bin size.
    """
    if len(scores) < n_bins:
        return float("nan")
    confidence = 1.0 - scores
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = confidence[mask].mean()
        ece += mask.mean() * abs(conf - acc)
    return float(ece)


# ---------------------------------------------------------------------------
# Figure 1: Reliability diagrams + ECE
# ---------------------------------------------------------------------------

def make_reliability_diagrams(results_path: Path, out_path: Path):
    details = _load_details(results_path)
    if not details:
        print(f"  No details found in {results_path}, skipping reliability diagrams.")
        return

    systems = ["vanilla", "kg"]
    metrics = [m for m in ALL_METRICS if m != "graph_path_support"]  # GPS is binary
    n_metrics = len(metrics)
    n_bins = 10

    fig, axes = plt.subplots(
        n_metrics, 2,
        figsize=(8, n_metrics * 2.2),
        sharex=True,
    )
    fig.suptitle("Reliability Diagrams — Vanilla RAG vs KG-RAG", fontsize=11, y=1.002)

    for row, metric in enumerate(metrics):
        for col, system in enumerate(systems):
            ax = axes[row, col]
            scores, labels = _get_scores_labels(details, metric, system)
            if len(scores) < n_bins:
                ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="grey")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                continue

            confidence = 1.0 - scores
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_means, _, binnumber = binned_statistic(
                confidence, confidence, statistic="mean", bins=bin_edges
            )
            acc_means, _, _ = binned_statistic(
                confidence, labels.astype(float), statistic="mean", bins=bin_edges
            )
            counts, _ = np.histogram(confidence, bins=bin_edges)

            valid = ~np.isnan(bin_means) & ~np.isnan(acc_means) & (counts > 0)
            ece_val = _ece(scores, labels, n_bins)
            auroc_val = _auroc(scores, labels)

            ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Perfect")
            ax.bar(
                bin_edges[:-1][valid], acc_means[valid],
                width=1 / n_bins, align="edge",
                alpha=0.6,
                color="#e07b54" if system == "kg" else "#5b8db8",
                label=f"ECE={ece_val:.3f}",
            )
            # Gap shading
            for lo, acc, conf in zip(
                bin_edges[:-1][valid], acc_means[valid], bin_means[valid]
            ):
                ax.bar(lo, abs(conf - acc), bottom=min(conf, acc),
                       width=1 / n_bins, align="edge", alpha=0.25, color="red")

            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.legend(fontsize=6, loc="upper left")
            label = METRIC_LABELS.get(metric, metric)
            if col == 0:
                ax.set_ylabel(label, fontsize=7)
            title = f"{'Vanilla' if system == 'vanilla' else 'KG-RAG'}"
            if row == 0:
                ax.set_title(title, fontsize=8)
            ax.text(0.97, 0.05, f"AUROC={auroc_val:.3f}", ha="right", va="bottom",
                    transform=ax.transAxes, fontsize=6)

    for ax in axes[-1]:
        ax.set_xlabel("Confidence", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: N-samples sweep
# ---------------------------------------------------------------------------

def make_n_sweep(ablations_dir: Path, out_path: Path):
    sweep_dir = ablations_dir / "n_sweep"
    if not sweep_dir.exists():
        print(f"  N-sweep results not found at {sweep_dir}. Run run_ablations.py first.")
        return

    n_values = sorted(
        int(p.name.lstrip("N")) for p in sweep_dir.iterdir() if p.is_dir()
    )
    if not n_values:
        print("  No N-sweep subdirectories found.")
        return

    # Collect AUROC for best generative metric and SPS-UQ per N
    dataset = "2wikimultihopqa"
    metrics_to_plot = ["sre_uq", "vn_entropy", "subgraph_perturbation_stability"]
    results: Dict[str, Dict[int, Dict[str, float]]] = {
        "vanilla": {m: {} for m in metrics_to_plot},
        "kg": {m: {} for m in metrics_to_plot},
    }

    for n in n_values:
        path = sweep_dir / f"N{n}" / f"mirage_{dataset}_results.json"
        if not path.exists():
            continue
        details = _load_details(path)
        for system in ["vanilla", "kg"]:
            for metric in metrics_to_plot:
                scores, labels = _get_scores_labels(details, metric, system)
                results[system][metric][n] = _auroc(scores, labels)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    colors = {"sre_uq": "#e07b54", "vn_entropy": "#5b8db8",
              "subgraph_perturbation_stability": "#2ca02c"}
    labels_map = {"sre_uq": "SRE-UQ", "vn_entropy": "VN-Entropy",
                  "subgraph_perturbation_stability": "SPS-UQ"}

    for ax, system, title in zip(axes, ["vanilla", "kg"], ["Vanilla RAG", "KG-RAG"]):
        for metric in metrics_to_plot:
            ns = sorted(results[system][metric].keys())
            aurocs = [results[system][metric][n] for n in ns]
            valid = [(n, a) for n, a in zip(ns, aurocs) if not np.isnan(a)]
            if not valid:
                continue
            ns_v, aurocs_v = zip(*valid)
            ax.plot(ns_v, aurocs_v, "o-", color=colors[metric],
                    label=labels_map[metric], lw=1.5, ms=5)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("N samples", fontsize=9)
        ax.set_xticks(n_values)
        ax.set_ylim(0.3, 1.0)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("AUROC", fontsize=9)
    fig.suptitle(
        "AUROC vs Number of Samples — 2WikiMultiHopQA\n"
        "Vanilla RAG improves with N; KG-RAG is flat (context determinism)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Temperature sweep
# ---------------------------------------------------------------------------

def make_temp_sweep(ablations_dir: Path, out_path: Path):
    sweep_dir = ablations_dir / "temp_sweep"
    if not sweep_dir.exists():
        print(f"  Temp-sweep results not found at {sweep_dir}. Run run_ablations.py first.")
        return

    temp_dirs = {
        0.0: sweep_dir / "T0p0",
        0.5: sweep_dir / "T0p5",
        1.0: sweep_dir / "T1p0",
    }
    dataset = "2wikimultihopqa"
    metrics_to_plot = ["sre_uq", "vn_entropy", "subgraph_perturbation_stability"]

    results: Dict[str, Dict[float, float]] = {
        "vanilla": {m: {} for m in metrics_to_plot},
        "kg": {m: {} for m in metrics_to_plot},
    }

    for temp, d in temp_dirs.items():
        path = d / f"mirage_{dataset}_results.json"
        if not path.exists():
            continue
        details = _load_details(path)
        for system in ["vanilla", "kg"]:
            for metric in metrics_to_plot:
                scores, labels = _get_scores_labels(details, metric, system)
                results[system][metric][temp] = _auroc(scores, labels)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    colors = {"sre_uq": "#e07b54", "vn_entropy": "#5b8db8",
              "subgraph_perturbation_stability": "#2ca02c"}
    labels_map = {"sre_uq": "SRE-UQ", "vn_entropy": "VN-Entropy",
                  "subgraph_perturbation_stability": "SPS-UQ"}
    temps = [0.0, 0.5, 1.0]

    for ax, system, title in zip(axes, ["vanilla", "kg"], ["Vanilla RAG", "KG-RAG"]):
        for metric in metrics_to_plot:
            aurocs = [results[system][metric].get(t, float("nan")) for t in temps]
            valid = [(t, a) for t, a in zip(temps, aurocs) if not np.isnan(a)]
            if not valid:
                continue
            ts_v, aurocs_v = zip(*valid)
            ax.plot(ts_v, aurocs_v, "o-", color=colors[metric],
                    label=labels_map[metric], lw=1.5, ms=6)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Temperature", fontsize=9)
        ax.set_xticks(temps)
        ax.set_xticklabels(["0.0\n(greedy)", "0.5\n(SE-optimal)", "1.0\n(paper)"])
        ax.set_ylim(0.3, 1.0)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("AUROC", fontsize=9)
    fig.suptitle(
        "AUROC vs Temperature — 2WikiMultiHopQA\n"
        "Vanilla RAG sensitive to temperature; KG-RAG flat (context determinism)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Hop-depth AUROC split
# ---------------------------------------------------------------------------

def make_hop_depth(results_path: Path, out_path: Path):
    """Split AUROC by hop depth using hop_score metadata if available,
    falling back to grounding_quality bins."""
    details = _load_details(results_path)
    if not details:
        print(f"  No details found in {results_path}, skipping hop depth figure.")
        return

    # Check what hop metadata is available
    has_hop_score = any("kg_avg_hop_score" in d or "hop_score" in d for d in details)
    has_grounding = any("grounding_quality" in d or "kg_grounding_quality" in d
                        for d in details)

    if not has_hop_score and not has_grounding:
        print("  No hop_score or grounding_quality in results — skipping hop depth figure.")
        return

    # Use grounding_quality as proxy for hop depth:
    # high g(q) → entities matched → likely 0/1-hop; low g → 2-hop or fallback
    def get_g(d):
        return d.get("grounding_quality", d.get("kg_grounding_quality", None))

    bins = {
        "high-g\n(g≥0.7)": lambda d: get_g(d) is not None and get_g(d) >= 0.7,
        "mid-g\n(0.3≤g<0.7)": lambda d: get_g(d) is not None and 0.3 <= get_g(d) < 0.7,
        "low-g\n(g<0.3)": lambda d: get_g(d) is not None and get_g(d) < 0.3,
    }

    metrics_to_plot = ["sre_uq", "vn_entropy", "subgraph_perturbation_stability"]
    colors = {"sre_uq": "#e07b54", "vn_entropy": "#5b8db8",
              "subgraph_perturbation_stability": "#2ca02c"}
    labels_map = {"sre_uq": "SRE-UQ", "vn_entropy": "VN-Entropy",
                  "subgraph_perturbation_stability": "SPS-UQ"}

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)

    for ax, system, title in zip(axes, ["vanilla", "kg"], ["Vanilla RAG", "KG-RAG"]):
        bin_names = list(bins.keys())
        x = np.arange(len(bin_names))
        width = 0.25

        for i, metric in enumerate(metrics_to_plot):
            aurocs = []
            ns = []
            for bin_name, filter_fn in bins.items():
                subset = [d for d in details if filter_fn(d)]
                scores, lbs = _get_scores_labels(subset, metric, system)
                aurocs.append(_auroc(scores, lbs))
                ns.append(len(subset))

            offset = (i - 1) * width
            bars = ax.bar(
                x + offset, aurocs, width,
                label=labels_map[metric],
                color=colors[metric], alpha=0.8,
            )
            # Annotate n
            for bar, n in zip(bars, ns):
                if not np.isnan(bar.get_height()):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"n={n}", ha="center", va="bottom", fontsize=5,
                    )

        ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_names, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.2, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("AUROC", fontsize=9)
    fig.suptitle("AUROC by Grounding Quality (proxy for hop depth)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: Grounding-threshold sweep (routing protocol validation)
# ---------------------------------------------------------------------------

def make_grounding_threshold(results_path: Path, out_path: Path):
    """For each grounding threshold g*, route: use SPS-UQ if g≥g*, else VN-Entropy.
    Plot routed AUROC vs g* vs using either metric alone."""
    details = _load_details(results_path)
    if not details:
        print(f"  No details in {results_path}, skipping grounding threshold figure.")
        return

    def get_g(d):
        return d.get("grounding_quality", d.get("kg_grounding_quality", None))

    if not any(get_g(d) is not None for d in details):
        print("  No grounding_quality in results — skipping grounding threshold figure.")
        return

    thresholds = np.linspace(0.0, 1.0, 21)
    routed_aurocs, n_structural = [], []

    # Baseline: always SPS-UQ, always VN-Entropy
    scores_sps, labels_sps = _get_scores_labeled_kg(details, "subgraph_perturbation_stability")
    scores_vn, labels_vn = _get_scores_labeled_kg(details, "vn_entropy")
    baseline_sps = _auroc(scores_sps, labels_sps)
    baseline_vn = _auroc(scores_vn, labels_vn)

    for g_thresh in thresholds:
        routed_scores, routed_labels = [], []
        n_struct = 0
        for d in details:
            g = get_g(d)
            correct = d.get("kg_correct")
            if g is None or correct is None:
                continue
            if g >= g_thresh:
                s = d.get("kg_subgraph_perturbation_stability")
                n_struct += 1
            else:
                s = d.get("kg_vn_entropy")
                if s is not None:
                    s = 1.0 - s  # orientation: high vn_entropy = high uncertainty
            if s is None:
                continue
            routed_scores.append(float(s))
            routed_labels.append(int(bool(correct)))
        ra = _auroc(np.array(routed_scores), np.array(routed_labels))
        routed_aurocs.append(ra)
        n_structural.append(n_struct)

    fig, ax1 = plt.subplots(figsize=(6, 3.8))
    ax2 = ax1.twinx()

    ax1.plot(thresholds, routed_aurocs, "o-", color="#2ca02c",
             lw=1.8, ms=4, label="Routed (SPS-UQ if g≥g*, else VN-Entropy)")
    ax1.axhline(baseline_sps, color="#e07b54", lw=1.2, ls="--",
                label=f"Always SPS-UQ (AUROC={baseline_sps:.3f})")
    ax1.axhline(baseline_vn, color="#5b8db8", lw=1.2, ls="--",
                label=f"Always VN-Entropy (AUROC={baseline_vn:.3f})")
    ax1.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.5)

    ax2.bar(thresholds, n_structural, width=0.04, alpha=0.15,
            color="grey", label="n routed to SPS-UQ")
    ax2.set_ylabel("Queries routed to SPS-UQ", fontsize=8, color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")

    ax1.set_xlabel("Grounding threshold g*", fontsize=9)
    ax1.set_ylabel("AUROC (KG-RAG)", fontsize=9)
    ax1.set_ylim(0.3, 1.0)
    ax1.legend(fontsize=7, loc="lower right")
    ax1.set_title("Routing Protocol: AUROC vs Grounding Threshold g*\n(2WikiMultiHopQA, KG-RAG)",
                  fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _get_scores_labeled_kg(details, metric):
    """Convenience wrapper for kg system."""
    return _get_scores_labels(details, metric, "kg")


# ---------------------------------------------------------------------------
# Ablation A: Retrieval overlap (context determinism proof)
# ---------------------------------------------------------------------------

def make_retrieval_overlap(results_path: Path, out_path: Path) -> None:
    """Violin plot of mean pairwise Jaccard retrieval overlap — vanilla vs KG-RAG.

    KG-RAG overlap should cluster near 1.0 (same subgraph every sample).
    Vanilla RAG overlap should be lower (stochastic chunk retrieval).
    This is direct empirical proof of context determinism.
    """
    if not results_path.exists():
        print(f"  [skip] {results_path} not found")
        return

    details = _load_details(results_path)
    vanilla_overlaps = [
        float(d["vanilla_retrieval_overlap"])
        for d in details
        if d.get("vanilla_retrieval_overlap") is not None
    ]
    kg_overlaps = [
        float(d["kg_retrieval_overlap"])
        for d in details
        if d.get("kg_retrieval_overlap") is not None
    ]

    if not vanilla_overlaps or not kg_overlaps:
        print("  [skip] retrieval_overlap fields missing from results")
        return

    fig, ax = plt.subplots(figsize=(4, 4))
    data = [vanilla_overlaps, kg_overlaps]
    parts = ax.violinplot(data, positions=[1, 2], showmedians=True, showextrema=True)

    for pc in parts["bodies"]:
        pc.set_alpha(0.7)
    parts["bodies"][0].set_facecolor("#4C72B0")
    parts["bodies"][1].set_facecolor("#DD8452")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Vanilla RAG", "KG-RAG"])
    ax.set_ylabel("Mean pairwise Jaccard (retrieved chunks)")
    ax.set_ylim(-0.05, 1.10)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("Retrieval Overlap Across Samples")

    # Annotate medians
    for pos, vals in zip([1, 2], [vanilla_overlaps, kg_overlaps]):
        med = float(np.median(vals))
        ax.text(pos, med + 0.03, f"{med:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    print(f"  Vanilla median Jaccard: {np.median(vanilla_overlaps):.3f}")
    print(f"  KG-RAG median Jaccard:  {np.median(kg_overlaps):.3f}")


# ---------------------------------------------------------------------------
# Ablation B: Grounding quality split
# ---------------------------------------------------------------------------

def make_grounding_split(results_path: Path, out_path: Path, threshold: float = 0.7) -> None:
    """Grouped bar chart: AUROC by grounding quality bin for structural vs generative metrics.

    Questions with grounding_quality >= threshold are 'high-g'; others are 'low-g'.
    Hypothesis: structural metrics (GPS, SPS) outperform SE on high-g questions;
    SE is more reliable on low-g questions (where retrieval fell back to vector search).
    """
    if not results_path.exists():
        print(f"  [skip] {results_path} not found")
        return

    details = _load_details(results_path)

    high_g = [d for d in details if d.get("grounding_quality") is not None and d["grounding_quality"] >= threshold]
    low_g  = [d for d in details if d.get("grounding_quality") is not None and d["grounding_quality"] < threshold]

    if len(high_g) < 5 or len(low_g) < 5:
        print(f"  [skip] insufficient data: high-g={len(high_g)}, low-g={len(low_g)}")
        return

    focus_metrics = {
        "semantic_entropy": "SE",
        "sre_uq": "SRE-UQ",
        "vn_entropy": "VN-Entropy",
        "graph_path_support": "GPS",
        "subgraph_perturbation_stability": "SPS-UQ",
    }

    x = np.arange(len(focus_metrics))
    width = 0.35

    high_aurocs, low_aurocs = [], []
    for metric in focus_metrics:
        s_h, l_h = _get_scores_labels(high_g, metric, "kg")
        s_l, l_l = _get_scores_labels(low_g,  metric, "kg")
        high_aurocs.append(_auroc(s_h, l_h))
        low_aurocs.append(_auroc(s_l, l_l))

    fig, ax = plt.subplots(figsize=(7, 4))
    bars_high = ax.bar(x - width / 2, high_aurocs, width, label=f"High grounding (g≥{threshold})", color="#4C72B0", alpha=0.85)
    bars_low  = ax.bar(x + width / 2, low_aurocs,  width, label=f"Low grounding (g<{threshold})",  color="#DD8452", alpha=0.85)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("AUROC (KG-RAG)")
    ax.set_xticks(x)
    ax.set_xticklabels(list(focus_metrics.values()))
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=9)
    ax.set_title(f"AUROC by Grounding Quality (KG-RAG, g*={threshold})")

    # Add value labels on bars
    for bar in list(bars_high) + list(bars_low):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    print(f"  High-g n={len(high_g)}, Low-g n={len(low_g)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate ablation figures for the paper.")
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=["ece", "n_sweep", "temp_sweep", "hop_depth", "grounding_threshold",
                 "retrieval_overlap", "grounding_split", "all"],
        default=["all"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="2wikimultihopqa",
        help="Dataset to use for per-dataset figures (default: 2wikimultihopqa)",
    )
    args = parser.parse_args()

    run_all = "all" in args.figures
    dataset = args.dataset
    main_results = RESULTS_DIR / f"mirage_{dataset}_results.json"

    print(f"Output directory: {FIGURES_DIR}")

    if run_all or "ece" in args.figures:
        print("\n[1/7] Reliability diagrams + ECE")
        make_reliability_diagrams(
            main_results,
            FIGURES_DIR / "ece_reliability.pdf",
        )

    if run_all or "n_sweep" in args.figures:
        print("\n[2/7] N-samples sweep")
        make_n_sweep(ABLATIONS_DIR, FIGURES_DIR / "n_sweep.pdf")

    if run_all or "temp_sweep" in args.figures:
        print("\n[3/7] Temperature sweep")
        make_temp_sweep(ABLATIONS_DIR, FIGURES_DIR / "temp_sweep.pdf")

    if run_all or "hop_depth" in args.figures:
        print("\n[4/7] Hop-depth / grounding-quality AUROC split")
        make_hop_depth(main_results, FIGURES_DIR / "hop_depth.pdf")

    if run_all or "grounding_threshold" in args.figures:
        print("\n[5/7] Grounding-threshold routing sweep")
        make_grounding_threshold(
            main_results,
            FIGURES_DIR / "grounding_threshold.pdf",
        )

    if run_all or "retrieval_overlap" in args.figures:
        print("\n[6/7] Retrieval overlap (context determinism proof)")
        make_retrieval_overlap(main_results, FIGURES_DIR / "retrieval_overlap.pdf")

    if run_all or "grounding_split" in args.figures:
        print("\n[7/7] Grounding quality AUROC split")
        make_grounding_split(main_results, FIGURES_DIR / "grounding_split.pdf")

    print("\nAll done.")


if __name__ == "__main__":
    main()
