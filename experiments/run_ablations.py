"""
run_ablations.py — orchestrate all ablation experiments.

Ablations implemented:
  1. N-samples sweep      : N in {1, 3, 5, 10, 20} on all datasets
  2. Temperature sweep    : T in {0.0, 0.5, 1.0} on 2WikiMultiHopQA only
  3. Hop-depth analysis   : post-hoc from existing results (no new runs needed)
  4. Grounding-threshold  : post-hoc from existing results (no new runs needed)

Usage:
    python experiments/run_ablations.py --ablations n_sweep temp_sweep
    python experiments/run_ablations.py --ablations n_sweep               # N sweep only
    python experiments/run_ablations.py --ablations temp_sweep            # temperature only
    python experiments/run_ablations.py --ablations all                   # everything

Results land in results/ablations/{ablation_name}/.
Post-hoc ablations (3 & 4) are handled by ablation_analysis.py.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

EXPERIMENT_SCRIPT = Path(__file__).parent / "experiment.py"
RESULTS_BASE = Path(__file__).parent.parent / "results" / "ablations"

# ---------------------------------------------------------------------------
# Ablation 1: N-samples sweep
# ---------------------------------------------------------------------------
N_SWEEP_VALUES = [1, 3, 5, 10, 20]
N_SWEEP_DATASETS = ["2wikimultihopqa", "pubmedqa", "bioasq", "musique"]

# ---------------------------------------------------------------------------
# Ablation 2: Temperature sweep (2WikiMHQA only for cost)
# ---------------------------------------------------------------------------
TEMP_SWEEP_VALUES = [0.0, 0.5, 1.0]
TEMP_SWEEP_DATASET = "2wikimultihopqa"


def _run(cmd: list[str], label: str) -> int:
    print(f"\n{'='*60}")
    print(f"RUNNING: {label}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"WARNING: {label} exited with code {result.returncode}", file=sys.stderr)
    return result.returncode


def run_n_sweep(dry_run: bool = False):
    """Ablation 1: vary entropy_samples, all datasets, T=1.0."""
    for n in N_SWEEP_VALUES:
        out_dir = RESULTS_BASE / "n_sweep" / f"N{n}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(EXPERIMENT_SCRIPT),
            "--entropy-samples", str(n),
            "--temperature", "1.0",
            "--num-samples", "30",
            "--datasets", *N_SWEEP_DATASETS,
            "--output-dir", str(out_dir),
        ]
        if dry_run:
            print(f"[DRY RUN] {' '.join(cmd)}")
        else:
            _run(cmd, f"N sweep N={n}")


def run_temp_sweep(dry_run: bool = False):
    """Ablation 2: vary temperature, 2WikiMHQA only, N=10."""
    for temp in TEMP_SWEEP_VALUES:
        label = f"T{temp:.1f}".replace(".", "p")
        out_dir = RESULTS_BASE / "temp_sweep" / label
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(EXPERIMENT_SCRIPT),
            "--temperature", str(temp),
            "--entropy-samples", "10",
            "--num-samples", "30",
            "--datasets", TEMP_SWEEP_DATASET,
            "--output-dir", str(out_dir),
        ]
        if dry_run:
            print(f"[DRY RUN] {' '.join(cmd)}")
        else:
            _run(cmd, f"Temperature sweep T={temp}")


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for the paper."
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=["n_sweep", "temp_sweep", "all"],
        default=["all"],
        help="Which ablations to run (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    args = parser.parse_args()

    run_all = "all" in args.ablations
    dry = args.dry_run

    if run_all or "n_sweep" in args.ablations:
        print("\n--- Ablation 1: N-samples sweep ---")
        run_n_sweep(dry_run=dry)

    if run_all or "temp_sweep" in args.ablations:
        print("\n--- Ablation 2: Temperature sweep ---")
        run_temp_sweep(dry_run=dry)

    print("\nDone. Run `python experiments/ablation_analysis.py` to generate figures.")


if __name__ == "__main__":
    main()
