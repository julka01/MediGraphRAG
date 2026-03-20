# OntographRAG

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/julka01/OntographRAG)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/neo4j-5.0+-brightgreen.svg)](https://neo4j.com/)

**Turn unstructured documents into schema-consistent knowledge graphs. Query them with RAG. Measure how much to trust the answers.**

OntographRAG extracts entities and relationships from raw text guided by a custom ontology, stores the result in Neo4j, and answers natural-language questions using hybrid vector + graph retrieval. A built-in uncertainty pipeline flags answers the model isn't confident in before they reach users.

Works for any domain — research literature, legal documents, financial reports, technical manuals. Particularly powerful for **clinical and biomedical data**, where ontology-constrained extraction enables population-level evidence generation from patient records at scale.

---

## Use cases

### Clinical intelligence and population-level evidence
Supply a clinical ontology (SNOMED CT, ICD-10, HPO) and process patient notes, discharge summaries, or EHR exports in bulk. Because every patient's data is extracted into the *same schema*, the entire population becomes queryable as a single graph:

```cypher
-- Which comorbidities most frequently co-occur with hypertension in patients over 60?
MATCH (p:Patient)-[:HAS_DIAGNOSIS]->(d:Diagnosis {name: "Hypertension"})
      -[:CO_OCCURS_WITH]->(c:Diagnosis)
WHERE p.age > 60
RETURN c.name, count(*) AS frequency ORDER BY frequency DESC
```

Ask the same question in plain English via the RAG layer. The ontology is what makes this possible — without schema enforcement, "T2DM", "type 2 diabetes", and "DM2" land as different nodes and aggregation breaks.

### Research and knowledge synthesis
Process a corpus of papers, extract entities and relationships consistently across all documents, then ask cross-paper questions the individual documents couldn't answer alone.

### Any domain with structured knowledge requirements
Legal (case law entities), finance (company relationships), engineering (component hierarchies). Supply the domain ontology; OntographRAG handles the rest.

---

## What makes this different

Most GraphRAG tools (including [Microsoft's GraphRAG](https://github.com/microsoft/graphrag)) let an LLM freely decide what to extract — producing inconsistent entity types, duplicate concepts, and graphs that drift between documents. OntographRAG takes the opposite approach: **you define the schema, the system respects it.**

| | OntographRAG | Microsoft GraphRAG |
|---|---|---|
| **Schema control** | Bring your own OWL/RDF ontology — entities and relationships are constrained to your types | LLM decides freely; no schema enforcement |
| **Graph storage** | Neo4j — production graph DB with Cypher, vector indexes, persistent named KGs | Parquet files in a local directory |
| **Hallucination detection** | 8 uncertainty metrics including RS-UQ (novel) and semantic entropy | None |
| **LLM providers** | OpenRouter, OpenAI, Gemini, Anthropic, Ollama, DeepSeek, HuggingFace | OpenAI / Azure OpenAI only |
| **Retrieval** | Hybrid: vector similarity + graph traversal in one query | Community summarisation (global) or entity search (local) |
| **Interface** | Web UI + REST API + Python library | CLI + Python library |
| **Domain** | Any domain; ontology is user-supplied | General purpose but tuned for summarisation |

---

## Key features

### 1. Ontology-guided KG construction
Supply a `.owl` / `.rdf` / `.ttl` ontology file and every extracted entity and relationship is validated against your schema. The same document processed twice produces the same graph shape. Across a corpus of documents, every entity lands in the same type hierarchy — enabling aggregation, comparison, and population-level queries that are impossible with free-form extraction.

This is the core differentiator. Without schema enforcement, LLMs produce synonym explosion ("myocardial infarction", "heart attack", "MI", "AMI" as four separate nodes), type drift (the same concept classified differently across documents), and graphs that can't be meaningfully queried at scale. The ontology collapses all of this into a consistent, traversable structure.

Without an ontology, extraction still works — the LLM infers types — but schema-constrained extraction is what unlocks population-level reasoning.

### 2. Neo4j as the graph store
Graphs are persisted in Neo4j with:
- **Vector indexes** (384-dim `all-MiniLM-L6-v2` by default) for semantic search over chunks
- **Named KGs** — multiple independent graphs in one database, scoped by name tag
- **Full Cypher access** — query or extend the graph with any Cypher statement

### 3. Hybrid retrieval
Queries run vector similarity search over chunk embeddings *and* graph traversal over entity neighborhoods simultaneously. The combined context is richer than either alone — the vector index finds relevant passages, the graph finds related concepts those passages didn't mention.

### 4. Uncertainty quantification and hallucination detection
A dedicated pipeline computes 8 metrics per answer to flag low-confidence responses:

| Metric | What it measures |
|--------|-----------------|
| `semantic_entropy` | Shannon entropy over meaning-clusters of N sampled responses (Farquhar et al., *Nature* 2023) |
| `discrete_semantic_entropy` | Same with hard cluster boundaries |
| `token_entropy` | Surface-level diversity across responses |
| `p_true` | NLI-based probability responses are supported by context |
| `embedding_consistency` | Pairwise NLI contradiction rate between response pairs |
| `spuq` | Semantic entropy weighted by variance under probability perturbation |
| `rs_uq` ⭐ | **Novel.** Cosine dissimilarity between the LLM's last-layer hidden state for the prompt alone vs. prompt + response. A large shift signals the model's answer diverges from its internal encoding of the question — no multiple samples needed. |

**Hypothesis:** ontology-constrained, deduplicated graph context lowers semantic entropy compared to vanilla RAG because the model receives less contradictory evidence.

### 5. Provider-agnostic LLM support
Every endpoint accepts a `provider` + `model` pair. Supported providers: OpenRouter (free tier available), OpenAI, Google Gemini, Anthropic, Ollama (local), DeepSeek, HuggingFace. Switch model per request with no code changes.

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/julka01/OntographRAG.git
cd OntographRAG
uv sync          # creates .venv and installs all dependencies
source .venv/bin/activate

# 2. Start Neo4j
docker compose up -d neo4j

# 3. Configure
cp .env.example .env
# Edit .env — set NEO4J_PASSWORD and at least one LLM provider key

# 4. Start server
python start_server.py
# → Web UI at http://localhost:8004
# → API docs at http://localhost:8004/docs
```

---

## Table of contents

- [Setup](#setup)
- [Web UI](#web-ui)
- [API Reference](#api-reference)
- [Experiments](#experiments)
- [Architecture](#architecture)
- [Utility scripts](#utility-scripts)
- [Docker](#docker)
- [Configuration](#configuration)

---

## Setup

### Requirements

- Python 3.11+
- Neo4j 5.0+ (via Docker or local install)
- 8 GB RAM minimum (16 GB recommended for large documents)

### Installation

```bash
uv sync          # recommended
# or
pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
# ── Neo4j ───────────────────────────────────────────────────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

# ── LLM providers (at least one required) ───────────────────────────────────
OPENROUTER_API_KEY=your-key   # free-tier models available
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
HF_API_TOKEN=...
OLLAMA_HOST=http://localhost:11434

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_PROVIDER=huggingface   # huggingface | openai | vertexai

# ── Document processing ──────────────────────────────────────────────────────
CHUNK_SIZE=1500
CHUNK_OVERLAP=200
MAX_CHUNKS=50

# ── Retrieval ────────────────────────────────────────────────────────────────
VECTOR_SIMILARITY_THRESHOLD=0.1

# ── Security (production) ────────────────────────────────────────────────────
APP_API_KEY=               # set to enforce API key auth on all endpoints
ALLOWED_ORIGINS=*          # comma-separated origins for CORS

# ── Server ───────────────────────────────────────────────────────────────────
LOG_LEVEL=INFO
MAX_WORKERS=4
LLM_TIMEOUT_SECONDS=120
```

---

## Web UI

The web interface is served at `http://localhost:8004`.

### Knowledge graph panel

- **Build KG** — upload a document (PDF, TXT, CSV, JSON, XML ≤ 50 MB), choose provider/model, optionally attach an ontology file. Extraction progress streams to the UI in real time via SSE and the graph loads automatically on completion.
- **Graph visualisation** — interactive vis.js network. Node size scales with degree. Click a node to open its detail panel (type, properties, connected nodes).
- **Search** — dims non-matching nodes rather than hiding them; shows match count.
- **Filter** — per-type checkboxes with node/edge counts.
- **Named KG management** — create, list, and switch between multiple saved graphs.

### Chat panel

- Ask questions against the active knowledge graph; answers cite source chunks.
- Chat history persisted in `localStorage`.
- Highlighted nodes — entities used in the answer are highlighted in the graph.
- Thinking indicator while waiting for the LLM response.

---

## API Reference

Server runs on **port 8004**. Interactive docs at `http://localhost:8004/docs`.

> **Authentication**: set `APP_API_KEY` in `.env` to require `X-API-Key: <key>` on all requests. Unset = open (development mode).

### Knowledge graph — build & query

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/create_ontology_guided_kg` | Build an ontology-guided KG from a file upload |
| `POST` | `/extract_graph` | Extract a raw KG (no ontology) from a file |
| `POST` | `/load_kg_from_file` | Load a graph from file into Neo4j |
| `GET`  | `/kg_progress_stream` | SSE stream of KG build progress |

#### `POST /create_ontology_guided_kg`

Multipart form:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Document (PDF/TXT/CSV/JSON/XML, ≤ 50 MB) |
| `provider` | string | `openai` | LLM provider |
| `model` | string | `gpt-3.5-turbo` | Model name |
| `ontology_file` | file | optional | Custom ontology (.owl/.rdf/.ttl/.xml) |
| `max_chunks` | int | `50` | Max text chunks to process |
| `kg_name` | string | optional | Name tag for the resulting KG |

Response:
```json
{
  "kg_id": "uuid",
  "kg_name": "my-kg",
  "graph_data": { "nodes": [...], "relationships": [...] },
  "method": "ontology_guided"
}
```

#### `GET /kg_progress_stream`

Server-Sent Events. Connect with `EventSource`:
```
data: {"line": "✓ Extracted 42 entities from chunk 3/10"}
data: {"done": true}
```

---

### Named KG management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST`   | `/kg/create` | Create a named KG record |
| `GET`    | `/kg/list` | List all KGs with document counts |
| `GET`    | `/kg/{kg_name}` | Stats for a specific KG |
| `DELETE` | `/kg/{kg_name}` | Delete a KG |
| `GET`    | `/kg/{kg_name}/entities` | List entities in a KG |

---

### Neo4j management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/save_kg_to_neo4j` | Persist an in-memory KG to Neo4j |
| `POST` | `/load_kg_from_neo4j` | Load a KG from Neo4j by name |
| `POST` | `/clear_kg` | Delete all nodes and relationships |
| `GET`  | `/health/neo4j` | Connectivity check |

---

### Chat / RAG

#### `POST /chat`

Rate limited: 30 requests/minute per IP.

JSON body:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | required | Question to answer (max 4096 chars) |
| `provider_rag` | string | `openrouter` | LLM provider |
| `model_rag` | string | `openai/gpt-oss-120b:free` | Model name |
| `kg_name` | string | optional | Restrict retrieval to a specific KG |
| `document_names` | string[] | `[]` | Restrict to specific documents |
| `session_id` | string | `default_session` | Session identifier |

Response:
```json
{
  "session_id": "default_session",
  "message": "...",
  "info": {
    "sources": ["chunk_id_1"],
    "model": "openai/gpt-oss-120b:free",
    "confidence": 0.87,
    "chunk_count": 5,
    "entity_count": 12,
    "relationship_count": 8,
    "entities": { "used_entities": [...] }
  }
}
```

---

### CSV bulk processing

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/validate_csv` | Validate a CSV before bulk processing |
| `POST` | `/bulk_process_csv` | Build KGs from all rows of a CSV |

#### `POST /bulk_process_csv`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | CSV file |
| `provider` | string | `openai` | LLM provider |
| `model` | string | `gpt-3.5-turbo` | LLM model |
| `text_column` | string | `full_report_text` | Column containing the text to process |
| `id_column` | string | optional | Column to use as document ID |
| `start_row` | int | `0` | First row to process |
| `batch_size` | int | `50` | Rows per batch |

---

### Models

`GET /models/{provider}` — lists available models for a provider.

---

### cURL examples

```bash
# Build a KG with ontology
curl -X POST http://localhost:8004/create_ontology_guided_kg \
  -F "file=@document.pdf" \
  -F "provider=openrouter" \
  -F "model=openai/gpt-4o-mini" \
  -F "ontology_file=@schema.owl" \
  -F "kg_name=my-kg"

# Ask a question
curl -X POST http://localhost:8004/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main concepts?", "kg_name": "my-kg", "provider_rag": "openrouter", "model_rag": "openai/gpt-4o-mini"}'

# Stream build progress
curl -N http://localhost:8004/kg_progress_stream

# List KGs
curl http://localhost:8004/kg/list

# Health check
curl http://localhost:8004/health/neo4j
```

---

## Experiments

The `experiments/` directory evaluates KG-RAG vs Vanilla RAG on biomedical QA benchmarks, measuring answer quality and hallucination via semantic uncertainty. See [experiments/README.md](experiments/README.md) for full details.

```bash
# 5-question smoke test
python experiments/experiment.py --num-samples 5 --entropy-samples 3 --datasets pubmedqa

# Sweep thresholds and chunk sizes
python experiments/experiment.py \
  --num-samples 50 --entropy-samples 4 \
  --similarity-thresholds 0.1 0.15 0.2 \
  --max-chunks-values 5 10 15 \
  --datasets pubmedqa bioasq
```

### Supported datasets

| Dataset | Task | Download |
|---------|------|----------|
| `pubmedqa` | Yes/no QA from PubMed abstracts | [pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa) |
| `bioasq` | Biomedical factoid QA | [bioasq.org](http://bioasq.org/participate/challenges) (free registration) |

Place downloaded files under `MIRAGE/rawdata/` — see [experiments/README.md](experiments/README.md) for exact paths.

### Evaluation flags

| Flag | Default | Description |
|------|---------|-------------|
| `--num-samples` | all | Questions per dataset |
| `--entropy-samples` | `3` | Responses per question for uncertainty metrics |
| `--similarity-thresholds` | `[0.1]` | Cosine similarity cutoffs to sweep |
| `--max-chunks-values` | `[10]` | Retrieved chunk counts to sweep |
| `--llm-provider` | `openrouter` | LLM provider |
| `--llm-model` | `openai/gpt-oss-120b:free` | Model |
| `--datasets` | `pubmedqa bioasq` | Datasets to run |
| `--skip-kg-build` | `False` | Reuse existing Neo4j graph |

Results are saved to `results/<dataset>_<timestamp>.json` and optionally synced to Weights & Biases.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Web UI (index.html)                     │
│   Graph panel (vis.js)  │  Chat panel  │  KG progress SSE   │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP / SSE
┌────────────────────────▼────────────────────────────────────┐
│               FastAPI server  ·  port 8004                   │
│        CORS  ·  rate limiting  ·  optional API key auth      │
└──────┬──────────────────────────────────────┬───────────────┘
       │                                      │
┌──────▼──────────────┐            ┌──────────▼──────────────┐
│    KG builder        │            │      RAG system          │
│                      │            │                          │
│ OntologyGuidedKG     │            │ EnhancedRAGSystem        │
│ Creator              │            │  ├ Vector search         │
│  ├ Chunking          │            │  ├ Graph traversal       │
│  ├ LLM extraction    │            │  └ LLM synthesis         │
│  ├ Ontology filter   │            │                          │
│  └ Neo4j write       │            │ VanillaRAGSystem         │
└──────┬───────────────┘            └──────────┬──────────────┘
       │                                       │
┌──────▼───────────────────────────────────────▼─────────────┐
│                     Neo4j graph database                     │
│   Nodes: Entity · Document · Chunk                          │
│   Relationships: typed and constrained by ontology          │
│   Indexes: vector (384-dim) + full-text                     │
└──────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   LLM provider layer                         │
│  OpenRouter · OpenAI · Gemini · Ollama · HuggingFace        │
│  DeepSeek · Anthropic  (configured per request)             │
└─────────────────────────────────────────────────────────────┘
```

### KG build pipeline

1. **Ingest** — file uploaded; PDF text extracted via PyMuPDF, plaintext decoded
2. **Chunk** — overlapping text windows (`CHUNK_SIZE=1500`, `CHUNK_OVERLAP=200`)
3. **Ontology load** — custom `.owl`/`.ttl` parsed (owlready2), or free-form extraction if none supplied
4. **LLM extraction** — each chunk sent with an ontology-constrained prompt; relationships and entities returned as structured JSON (relationships-first ordering prevents truncation data loss)
5. **Cross-chunk extraction** — sliding window over adjacent chunk pairs; a second LLM call extracts relationships that span chunk boundaries
6. **Entity harmonization** — duplicate entities merged by normalized text; most specific ontology type wins; stable UUID assigned per entity
7. **Confidence filtering** — each relationship triple is scored by co-occurrence in source chunks; hallucinated triples (score < 0.25) are dropped
8. **Embed** — chunk and entity embeddings computed with `all-MiniLM-L6-v2` (384-dim, CPU) or OpenAI
9. **Write** — nodes, relationships, and embeddings stored in Neo4j; entities tagged with `kgName` for scoped loading; progress streamed via SSE

### RAG query pipeline

1. **Embed query** — same model as build time
2. **Vector search** — top-K chunks by cosine similarity
3. **Graph traversal** — neighbour entities fetched via Cypher for each retrieved chunk
4. **Assemble context** — chunk text + entity graph merged into LLM prompt
5. **Synthesise** — LLM generates a grounded answer with citations

### Module layout

```
ontographrag/
├── api/
│   ├── app.py                         # FastAPI application, all endpoints
│   └── static/
│       └── index.html                 # Single-page web UI
├── kg/
│   ├── builders/
│   │   ├── ontology_guided_kg_creator.py   # OntologyGuidedKGCreator — core extraction, harmonization, Neo4j write
│   │   └── enhanced_kg_creator.py          # UnifiedOntologyGuidedKGCreator — API-facing wrapper + CSV bulk ops
│   ├── loaders/
│   │   └── kg_loader.py               # KGLoader — reads KG from Neo4j by kgName
│   └── utils/
│       ├── common_functions.py        # Shared helpers (embedding, text normalization)
│       └── constants.py               # Default values and Neo4j label constants
├── rag/
│   └── systems/
│       ├── enhanced_rag_system.py     # KG-RAG: hybrid vector + graph retrieval
│       └── vanilla_rag_system.py      # Vanilla RAG: vector-only baseline
└── providers/
    └── model_providers.py             # LLM + embedding provider abstractions
```

### Key specs

| Component | Detail |
|-----------|--------|
| Embeddings | `all-MiniLM-L6-v2` (384-dim), runs locally on CPU |
| Vector similarity | Cosine, default threshold 0.1 |
| Chunk size | 1500 chars, 200 overlap |
| Graph database | Neo4j 5.0+ with vector indexes |
| Graph visualisation | vis.js 9.1.0 |
| File upload limit | 50 MB |
| Chat rate limit | 30 req/min per IP |
| KG build rate limit | 5 req/min per IP |

---

## Utility scripts

| Script | Purpose |
|--------|---------|
| `start_server.py` | Start the FastAPI server on port 8004 |
| `run_kg_generation.py` | Build a KG from the command line without the API |
| `populate_neo4j.py` | Seed Neo4j with sample data |
| `clear_neo4j.py` | Delete all nodes and relationships |
| `cleanup_and_rebuild_kg.py` | Clear Neo4j and rebuild a KG in one step |
| `graphDB_dataAccess.py` | Low-level Neo4j data access layer (named KG CRUD) |
| `csv_processor.py` | `MedicalReportCSVProcessor` for medical-report CSVs |
| `compare_extraction_methods.py` | Compare ontology-guided vs free-form LLM extraction |
| `test_named_kg.py` | Integration test for named KG creation and retrieval |
| `MIRAGE_adaptation.py` | Adapter for the MIRAGE biomedical benchmark |

---

## Docker

```bash
# Neo4j only (recommended for development)
docker compose up -d neo4j

# Full stack (Neo4j + API server)
docker compose up -d

# Logs
docker compose logs -f

# Stop
docker compose down

# Neo4j Browser → http://localhost:7474
# Connect to bolt://localhost:7687
```

---

## Configuration reference

### LLM providers

| Provider | Env var | Notes |
|----------|---------|-------|
| `openrouter` | `OPENROUTER_API_KEY` | Recommended; free-tier models available |
| `openai` | `OPENAI_API_KEY` | GPT-3.5, GPT-4, GPT-4o |
| `gemini` | `GEMINI_API_KEY` | Gemini Pro, Flash |
| `ollama` | — | Local models; set `OLLAMA_HOST` if non-default |
| `huggingface` | `HF_API_TOKEN` | HuggingFace Inference API |
| `deepseek` | `DEEPSEEK_API_KEY` | DeepSeek Chat/Coder |

### Embedding providers

| Provider | Env var | Model |
|----------|---------|-------|
| `huggingface` (default) | — | `all-MiniLM-L6-v2`, runs locally |
| `openai` | `OPENAI_API_KEY` | `text-embedding-ada-002` |
| `vertexai` | GCP credentials | `textembedding-gecko` |

### Processing tuning

| Variable | Default | Effect |
|----------|---------|--------|
| `CHUNK_SIZE` | `1500` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `MAX_CHUNKS` | `50` | Max chunks processed per document |
| `VECTOR_SIMILARITY_THRESHOLD` | `0.1` | Minimum cosine similarity for retrieval |
| `MAX_WORKERS` | `4` | Parallel workers for batch processing |
| `LLM_TIMEOUT_SECONDS` | `120` | Per-request LLM timeout |

---

## License

MIT. See [LICENSE](LICENSE) for details.

*Issues and feature requests: [GitHub Issues](https://github.com/julka01/OntographRAG/issues)*
