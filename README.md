# MediGraphX: AI-Powered Knowledge Graph for Medical Decision Support

[![Version](https://img.shields.io/badge/version-1.0.0--rc1--324db44-blue.svg)](https://github.com/julka01/MediGraphRAG)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/neo4j-5.0+-brightgreen.svg)](https://neo4j.com/)

MediGraphX structures medical data into ontology-guided knowledge graphs for evidence-based reasoning and clinical decision support.

## Quick Start

1. **Clone & Install**
   ```bash
   git clone https://github.com/julka01/MediGraphRAG.git
   cd MediGraphRAG
   uv sync  # Create venv & install deps
   source .venv/bin/activate
   ```

2. **Start Neo4j**
   ```bash
   docker compose up -d neo4j
   ```

3. **Configure** (Create `.env`)
   ```bash
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-password
   OPENROUTER_API_KEY=your-key  # Recommended for free access
   ```

4. **Run**
   ```bash
   python start_server.py  # API at http://localhost:8004
   ```

## Table of Contents

- [Setup](#setup)
- [Experiments](#experiments)
- [API](#api)
- [Configuration](#configuration)
- [Architecture](#architecture)

---

## Setup

### Requirements

- Python 3.11+
- Neo4j 5.0+ (via Docker)
- 8GB RAM minimum

### Environment Variables

Create `.env`:
```bash
# Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# LLM Providers (at least one required)
OPENROUTER_API_KEY=your-key    # Recommended - free access
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# OLLAMA_HOST=http://localhost:11434
```

---

## Experiments

This section documents how to run KG RAG vs Vanilla RAG evaluation experiments.

### MIRAGE Evaluation (Recommended)

The main experiment script compares KG-RAG against Vanilla RAG on biomedical QA datasets.

#### Basic Usage

```bash
# Activate venv
source venv/bin/activate

# Run on PubMedQA with 20 samples
python experiments/run_mirage_evaluation.py \
  --num-samples 20 \
  --entropy-samples 4 \
  --llm-provider openrouter \
  --llm-model gpt-oss-20b \
  --datasets pubmedqa
```

#### All Available Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--num-samples` | int | all | Number of questions to evaluate per dataset |
| `--entropy-samples` | int | 3 | Number of response generations for semantic entropy (max: 8) |
| `--similarity-thresholds` | float[] | [0.1] | Similarity thresholds to test |
| `--max-chunks-values` | int[] | [10] | Max chunks to retrieve per query |
| `--llm-provider` | str | openrouter | LLM provider (openrouter, openai, anthropic, ollama) |
| `--llm-model` | str | openai/gpt-oss-120b:free | Model name for the provider |
| `--datasets` | str[] | pubmedqa bioasq | Datasets to evaluate |
| `--skip-kg-build` | flag | False | Skip KG rebuild, reuse existing Neo4j data |

#### Example Experiments

```bash
# Quick test (5 samples, 3 entropy samples)
python experiments/run_mirage_evaluation.py \
  --num-samples 5 \
  --entropy-samples 3 \
  --datasets pubmedqa

# Full evaluation on multiple datasets
python experiments/run_mirage_evaluation.py \
  --num-samples 50 \
  --entropy-samples 4 \
  --similarity-thresholds 0.1 0.15 0.2 \
  --max-chunks-values 5 10 15 \
  --datasets pubmedqa bioasq medqa medmcqa

# Reuse existing KG (faster, no rebuild)
python experiments/run_mirage_evaluation.py \
  --num-samples 20 \
  --datasets pubmedqa \
  --skip-kg-build

# Use different LLM
python experiments/run_mirage_evaluation.py \
  --num-samples 10 \
  --llm-provider openai \
  --llm-model gpt-4o \
  --datasets pubmedqa
```

### Available Datasets

- **pubmedqa** - Yes/no questions from PubMed abstracts (recommended for initial testing)
- **bioasq** - Biomedical fact-based QA
- **medqa** - USMLE-style medical exam questions
- **medmcqa** - Indian medical exam questions

### Other Experiment Scripts

#### KG RAG Full Experiment
```bash
python experiments/kg_rag_full_experiment.py \
  --num-samples 10 \
  --similarity-thresholds 0.1 0.15 \
  --max-chunks 5 10
```

#### RAG Comparison
```bash
python experiments/rag_comparison_experiment.py \
  --num-samples 5 \
  --dataset path/to/dataset.json
```

---

## API

### Core Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/create_ontology_guided_kg` | POST | Build knowledge graph |
| `/chat` | POST | Medical Q&A |
| `/bulk_process_csv` | POST | Batch processing |
| `/health/neo4j` | GET | Database status |

### Example Usage

```bash
# Create KG from document
curl -X POST "http://localhost:8004/create_ontology_guided_kg" \
  -F "file=@medical_doc.pdf"

# Query
curl -X POST "http://localhost:8004/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "Treatment for prostate cancer?"}'
```

---

## Configuration

### Environment Variables

```bash
# Processing
CHUNK_SIZE=1500
CHUNK_OVERLAP=200
MAX_CHUNKS=50

# Embeddings
EMBEDDING_MODEL=sentence_transformers
VECTOR_SIMILARITY_THRESHOLD=0.1

# System
LOG_LEVEL=INFO
MAX_WORKERS=4
```

---

## Architecture

```
Medical Docs → LLM Parser → Knowledge Graph (Neo4j)
                                           │
Query → Vector Search + Graph Traversal → Evidence + Citations
```

**Pipeline**:
1. Document ingestion & chunking
2. Entity/relationship extraction via LLM
3. Graph construction with embeddings
4. Query with hybrid retrieval (vector + graph)

**Key Specs**:
- Chunking: 1500 chars with 200 overlap
- Embeddings: all-MiniLM-L6-v2 (384-dim) or OpenAI
- Database: Neo4j 5.0+ with vector indexes

---

## License

MIT License. See [LICENSE](LICENSE) for details.

*For support: [GitHub Issues](https://github.com/julka01/MediGraphRAG/issues)*
