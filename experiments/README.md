# Experiments

This directory contains the benchmark runner for comparing vanilla RAG and KG-RAG under a shared evaluation protocol.

The current pipeline does three things:
- builds or reuses a dataset-scoped Neo4j KG
- runs vanilla RAG and KG-RAG on the same seeded question subset
- scores answer quality and, in `full_metrics` mode, computes the current 15-metric uncertainty suite

## Quick Start

Use the source-checkout-safe form below. If your `.venv` has already picked up the console script, `ontograph evaluate ...` is equivalent.

```bash
source .venv/bin/activate

# Cheap smoke test on one dataset
python experiments/experiment.py \
  --datasets hotpotqa \
  --num-samples 30 \
  --subset-seed 42 \
  --rebuild-kg \
  --evaluation-mode accuracy_only

# Full uncertainty run on recommended datasets
python experiments/experiment.py \
  --datasets hotpotqa 2wikimultihopqa musique pubmedqa multihoprag \
  --num-samples 100 \
  --subset-seed 42 \
  --rebuild-kg \
  --evaluation-mode full_metrics
```

For BioASQ retrieval benchmarking, first build the shared PubMed abstract corpus:

```bash
python experiments/prepare_bioasq_corpus.py \
  --bioasq-path MIRAGE/rawdata/bioasq/Task10BGoldenEnriched/10B1_golden.json \
  --output MIRAGE/rawdata/bioasq/pubmed_abstracts.jsonl \
  --email you@example.com \
  --verbose
```

Then run:

```bash
python experiments/experiment.py \
  --datasets bioasq \
  --num-samples 100 \
  --subset-seed 42 \
  --rebuild-kg \
  --evaluation-mode full_metrics
```

## Current Flags

These are the live flags supported by [experiment.py](experiment.py).

| Flag | Default | What it does |
|---|---:|---|
| `--num-samples` | all | Number of questions per dataset |
| `--subset-seed` | `42` | Deterministic question-subset seed |
| `--entropy-samples` | `5` | Number of response samples used for uncertainty estimation |
| `--similarity-thresholds` | `0.1` | Retrieval threshold sweep |
| `--max-chunks-values` | `10` | Retrieved chunk-count sweep |
| `--llm-provider` | `openai` | Generation and KG extraction provider |
| `--llm-model` | `gpt-4o-mini` | Generation and KG extraction model |
| `--datasets` | `pubmedqa bioasq` | Datasets to run |
| `--rebuild-kg` | off | Rebuild the dataset KG instead of reusing an existing one |
| `--max-kg-contexts` | unset | Cap the number of passages sent into KG construction |
| `--dataset-kg-scope` | `evaluation_subset` | Build the KG from the selected subset or the full normalized dataset |
| `--allow-gold-evidence-contexts` | off | Allow oracle evidence contexts to be indexed directly; controlled-evidence only |
| `--no-llm-judge` | off | Disable LLM-as-judge and use heuristic matching only |
| `--judge-provider` | generation provider | Separate provider for the answer judge |
| `--judge-model` | generation model | Separate model for the answer judge |
| `--temperature` | `1.0` | Generation temperature |
| `--multi-temperature` | off | Also run T=0, 0.5, 1.0 sweeps for uncertainty analysis |
| `--evaluation-mode` | `full_metrics` | `accuracy_only` or `full_metrics` |
| `--output-dir` | `results` | CLI argument accepted; run artifacts are currently written under `results/runs/<run_id>/` |

Important removals:
- there is no `--skip-kg-build`
- evaluation is no longer described in terms of the old 6 LLM-scored quality metrics

## Supported Datasets

The loader currently supports these datasets via [dataset_adapters.py](dataset_adapters.py):

| Dataset | Status | Context role | Notes |
|---|---|---|---|
| `pubmedqa` | ready | `source_document` | Biomedical yes/no/maybe over source abstract segments |
| `realmedqa` | ready after local download | `no_context` + shared corpus | Uses the verified ideal subset and builds a shared corpus from NICE recommendations |
| `hotpotqa` | ready | `retrieval_bundle` | Multi-hop Wikipedia bundle per question |
| `2wikimultihopqa` | ready | `retrieval_bundle` | Multi-hop Wikipedia bundle per question |
| `musique` | ready | `retrieval_bundle` | Multi-hop paragraph bundle per question |
| `multihoprag` | ready | gold evidence + shared corpus | Uses `corpus.json` for fair retrieval |
| `bioasq` | ready after corpus prep | `gold_evidence` | Requires shared PubMed abstract corpus for fair retrieval benchmarking |
| `medhop` | not ready for fair retrieval yet | `gold_evidence` | Loader exists; needs a proper shared corpus to benchmark retrieval fairly |
| `medqa` | loader only | `no_context` | MCQ reasoning, not retrieval-native |
| `medmcqa` | loader only | `no_context` | MCQ reasoning, not retrieval-native |
| `mmlu` | loader only | `no_context` | MCQ reasoning, not retrieval-native |

Recommended end-to-end retrieval benchmarks right now:
- `hotpotqa`
- `2wikimultihopqa`
- `musique`
- `pubmedqa`
- `realmedqa`
- `multihoprag`
- `bioasq` after shared-corpus prep

## Corpus Safety

The experiment runner now distinguishes between three different kinds of per-question context:
- `source_document`: the question already comes with the source abstract/document
- `retrieval_bundle`: the benchmark provides a per-question passage bundle that mixes relevant and distractor text
- `gold_evidence`: the benchmark provides oracle support snippets or support abstracts

Why this matters:
- `gold_evidence` should not be silently indexed as if it were a fair retrieval corpus
- for datasets like `bioasq`, the runner now fails closed unless a shared corpus exists or you explicitly opt into `--allow-gold-evidence-contexts`

So:
- `bioasq` without `pubmed_abstracts.jsonl` is blocked by default
- old BioASQ runs built from gold snippets should not be used for vanilla-vs-KG retrieval claims

## Dataset Files

Expected raw-data layout:

```text
MIRAGE/rawdata/
├── pubmedqa/data/test_set.json
├── realmedqa/RealMedQA.json                # or .jsonl / .csv export from Hugging Face
├── bioasq/Task10BGoldenEnriched/10B1_golden.json
├── bioasq/pubmed_abstracts.jsonl              # generated by prepare_bioasq_corpus.py
├── hotpotqa/hotpot_dev_fullwiki_v1.json
├── 2wikimultihopqa/dev.json
├── musique/dev.jsonl
├── multihoprag/MultiHopRAG.json
├── multihoprag/corpus.json
└── medhop/dev.json
```

Notes:
- `musique` also supports `musique_ans_v1.0_dev.jsonl`
- `realmedqa` loads the paper's verified ideal subset by default (`Plausible=Completely` and `Answered=Completely`)
- `bioasq/pubmed_abstracts.jsonl` is optional for controlled-evidence QA, but required for fair retrieval benchmarking
- `medhop` currently has no built-in shared corpus file path, so it is not a fair retrieval benchmark yet

## What The Pipeline Measures

For each dataset/configuration pair, the runner logs:
- clean accuracy excluding generation failures as the headline result
- raw accuracy only as a debugging / audit field
- per-system answered counts and generation-failure counts
- official-style `answer_em` / `answer_f1` where supported
- per-question details files and W&B tables

In `full_metrics` mode it also computes the current 15 uncertainty metrics:

Output-side metrics:
- `semantic_entropy`
- `discrete_semantic_entropy`
- `sre_uq`
- `p_true`
- `selfcheckgpt`
- `vn_entropy`
- `sd_uq`

Structural metrics:
- `graph_path_support`
- `graph_path_disagreement`
- `competing_answer_alternatives`
- `evidence_vn_entropy`
- `subgraph_informativeness`
- `subgraph_perturbation_stability`

Grounding metrics:
- `support_entailment_uncertainty`
- `evidence_conflict_uncertainty`

The runner also computes `AUROC` / `AUREC` summaries from the saved per-question outputs when `full_metrics` is enabled.

## Evaluation Protocol

Current high-level flow:

1. Normalize raw data into `InferenceRecord` / `GoldRecord`
2. Resolve and persist a deterministic seeded subset of question IDs
3. Enforce corpus-safety policy for the dataset
4. Build or reuse a dataset-scoped KG under the same subset/corpus conditions
5. Run vanilla RAG and KG-RAG on the same selected questions
6. Score correctness plus optional official-style `EM/F1`
7. If enabled, collect extra samples and compute uncertainty metrics
8. Save run artifacts under `results/runs/<run_id>/`

## Outputs

Each run writes under:

```text
results/runs/<run_id>/
├── manifest.json
├── mirage_evaluation_summary.json
└── questions/
```

The summary groups results by dataset and configuration and includes:
- dataset track (`biomedical_grounding`, `biomedical_multihop_reasoning`, `multihop_reasoning`)
- seeded subset metadata
- per-config answer accuracy / EM / F1
- uncertainty summaries
- task-type breakdowns

Checkpoints live separately under:

```text
results/checkpoints/
```

Deterministic subset selections live under:

```text
results/selections/
```

## Script Reference

| File | Purpose |
|---|---|
| [experiment.py](experiment.py) | Main experiment runner |
| [uncertainty_metrics.py](uncertainty_metrics.py) | All 15 uncertainty metric implementations |
| [dataset_adapters.py](dataset_adapters.py) | Dataset normalization and corpus-role metadata |
| [visualize_results.py](visualize_results.py) | Plotting and figure utilities |
| [hop_stratified_analysis.py](hop_stratified_analysis.py) | Per-hop-count accuracy and uncertainty stratification |
| [generate_auroc_heatmap.py](generate_auroc_heatmap.py) | AUROC/AUREC heatmaps across metrics and datasets |
| [prepare_bioasq_corpus.py](prepare_bioasq_corpus.py) | Build shared PubMed abstract corpus for BioASQ |
| [subset_selection.py](subset_selection.py) | Deterministic seeded question subsets |
| [official_answer_metrics.py](official_answer_metrics.py) | Official-style answer EM/F1 |
| [answer_formatting.py](answer_formatting.py) | Answer normalisation and judge prompts |
| [summary_utils.py](summary_utils.py) | Run-summary aggregation helpers |
| [kg_reuse.py](kg_reuse.py) | KG reuse/cache logic across runs |
