# MediGraphX: AI-Powered Knowledge Graph for Medical Decision Support

[![Version](https://img.shields.io/badge/version-1.0.0--rc1--324db44-blue.svg)](https://github.com/julka01/MediGraphRAG)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/neo4j-5.0+-brightgreen.svg)](https://neo4j.com/)



MediGraphX structures medical data into ontology-guided knowledge graphs for evidence-based reasoning and clinical decision support.

## Quick Start

1. **Clone & Install** (Recommended setup)
   ```bash
   git clone https://github.com/julka01/MediGraphRAG.git
   cd MediGraphRAG  # Project directory
   curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv
   uv sync  # Create venv & install deps
   source .venv/bin/activate  # Linux/Mac
   # OR: .venv\Scripts\activate  # Windows
   ```

2. **Start Neo4j** (with Docker)
   ```bash
   docker compose up -d neo4j  # Access browser at http://localhost:7474
   ```

3. **Configure** (Create `.env` with your API keys)
   ```bash
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-password
   OPENAI_API_KEY=sk-your-key  # Or other provider key
   ```

4. **Run & Test**
   ```bash
   python start_server.py  # Start app on http://localhost:8004
   curl "http://localhost:8004/health/neo4j"  # Verify connection
   ```

## Table of Contents

- [Requirements & Setup](#requirements--setup)
- [Getting Started](#getting-started)
- [Features](#features)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Requirements & Setup

**System Requirements**: Python 3.9+ (3.11+ recommended), 8GB RAM minimum (16GB+ for large datasets), 10GB storage minimum, 4 CPU cores minimum. Optional: NVIDIA GPU for acceleration.

### Dependencies

- **Neo4j** 5.0+ : Graph database
- **Docker** 20.10+ : Containerized deployment

### Required: LLM Providers

At least one: OPENAI_API_KEY | ANTHROPIC_API_KEY | GEMINI_API_KEY | OPENROUTER_API_KEY | OLLAMA_HOST

### Development Setup (Quick Reference)

See Quick Start above for the recommended approach. For alternatives:

- **Package Managers**: For pip instead of uv, see [pyproject.toml](pyproject.toml)
- **Neo4j Setup**: Native installation available at [neo4j.com](https://neo4j.com/download/)
- **Docker Alternatives**: Standalone containers or Kubernetes via `docker-compose.yml`

### Environment Configuration

Create `.env` with:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
OPENAI_API_KEY=sk-your-key  # Or other provider key
```

Options: OPENAI_API_KEY | ANTHROPIC_API_KEY | GEMINI_API_KEY | OPENROUTER_API_KEY | OLLAMA_HOST

**Security**: Never commit `.env`; obtain keys from provider dashboards.

### Production Deployment

**Primary Option: Docker Compose**
```bash
docker compose up -d  # Full stack via docker-compose.yml
```

**Alternatives**:
- **Kubernetes**: Use included `docker-compose.yml` as base, deploy via Helm or K8s manifests
- **Standalone**: `docker build -t medigraph . && docker run -p 8004:8004 --env-file .env medigraph`
- **CI/CD**: Build images in pipelines, deploy to cloud registries

**Production Tips**: Use strong passwords, enable monitoring (Sentry), configure TLS. See [docker-compose.yml](docker-compose.yml) for full config.

## Getting Started

Assuming setup complete (see Quick Start), try these core functions. All examples use `{{BASE_URL}} = http://localhost:8004`.

### Build Knowledge Graph
```bash
curl -X POST "{{BASE_URL}}/create_ontology_guided_kg" \
  -F "file=@clinical_guidelines.pdf" \
  -F "provider=openai" \
  -F "model=gpt-4"
```

### Query Medical Data
```bash
curl -X POST "{{BASE_URL}}/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "Treatment options for stage II prostate cancer?", "document_names": ["guidelines.pdf"], "provider_rag": "openai", "model_rag": "gpt-4"}'
```

### Batch Process Patient Data
```bash
curl -X POST "{{BASE_URL}}/bulk_process_csv" \
  -F "csv_file=@patient_cohort.csv" \
  -F "batch_size=50"
```

**Advanced**: Use population-level queries or antimicrobial stewardship questions similarly.

## Features

- **Ontology-Guided KGs**: Structure medical data using biomedical ontologies (OWL, UMLS)
- **Evidence-Based Reasoning**: Retrieval-augmented generation with source attribution
- **Multi-Modal Processing**: Supports PDFs, CSVs, research documents
- **Medical Q&A**: Natural language queries with uncertainty quantification
- **Scalable Architecture**: Neo4j graph + vector embeddings for semantic search
- **Batch Cohort Analysis**: Process population-level datasets

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Medical       │───▶│   LLM Parser     │───▶│   Knowledge     │
│   Documents     │    │   + Ontology     │    │   Graph Store   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Natural       │───▶│   Vector Search  │───▶│  Evidence App   │
│   Language      │    │   + Graph        │    │  Citations      │
│   Questions     │    │   Traversal      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

**System Flow**: Medical Documents → LLM + Ontology Parser → Knowledge Graph (Neo4j) → Vector Search + Graph Traversal → Evidence Citations.

**Pipeline**:
1. Document ingestion & metadata
2. Semantic chunking + ontology validation
3. Graph construction & embeddings
4. Query processing with confidence scoring

**Key Specs**:
- **Chunking**: 1000-4000 chars (semantic boundaries)
- **Embedding Options**: Sentence-BERT (384-dim), OpenAI embeddings, local models via Ollama
- **Similarity**: Cosine distance, configurable thresholds (0.08 default)
- **Database**: Neo4j 5.0+ with ChromaDB vector acceleration
- **Latency**: <5s for typical medical queries

## API Reference

### Core Endpoints

| Endpoint | Method | Purpose | Authentication |
|----------|--------|---------|----------------|
| `/create_ontology_guided_kg` | POST | Knowledge graph creation | None |
| `/chat` | POST | Medical Q&A | None |
| `/bulk_process_csv` | POST | Batch processing | None |
| `/health/neo4j` | GET | Database status | None |
| `/visualize_graph` | GET | Graph visualization | None |

### `/create_ontology_guided_kg`

Create knowledge graphs from documents.

```bash
curl -X POST "http://localhost:8004/create_ontology_guided_kg" \
  -F "file=@medical_doc.pdf" \
  -F "provider=openai" \
  -F "model=gpt-4" \
  -F "max_chunks=20"
```

**Parameters:**
- `file`: Document file (PDF, TXT, etc.)
- `provider`: LLM provider (openai, anthropic, gemini)
- `model`: Specific model name
- `max_chunks`: Processing limit

### `/chat`

Evidence-based question answering.

```bash
curl -X POST "http://localhost:8004/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Treatment options for diabetes?",
    "document_names": ["guidelines.pdf"],
    "provider_rag": "openai",
    "model_rag": "gpt-4"
  }'
```

**Response Format:**
```json
{
  "recommendation_summary": "Evidence-based approach",
  "node_traversal_path": ["Patient → Symptoms → Treatment"],
  "reasoning_path": ["Clinical finding 1", "Evidence 2"],
  "evidence_synthesis": "Combined analysis",
  "confidence_metrics": {
    "similarity_score": 0.85,
    "entity_coverage": 88.5
  },
  "source_citations": ["Author A (2023) pg. 45"]
}
```

### Authentication & Security

- All endpoints currently open (development mode)
- Production deployments should implement:
  - API key authentication
  - Rate limiting
  - Input sanitization
  - Audit logging

## Configuration

### Environment Variables

#### Required

```bash
# Database connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=secure-password

# LLM providers (at least one)
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GEMINI_API_KEY=...
```

#### Processing Configuration

```bash
# Document processing
CHUNK_SIZE=2000                    # Character chunks (1000-4000)
CHUNK_OVERLAP=300                  # Overlap between chunks
MAX_CHUNKS=50                      # Processing limit per document

# Vector embeddings & search
EMBEDDING_MODEL=sentence_transformers  # Options: sentence_transformers, openai, local
VECTOR_SIMILARITY_THRESHOLD=0.08   # Relevance threshold (0.0-1.0)
EMBEDDING_BATCH_SIZE=32            # Batch size for embedding generation

# LLM configurations (provider-specific)
OPENAI_API_KEY=sk-your-key         # Required for OpenAI models
ANTHROPIC_API_KEY=sk-ant-your-key  # Required for Anthropic models
OLLAMA_HOST=http://localhost:11434 # Required for local Ollama models
OLLAMA_MODEL=mistral:7b            # Local model name
```

#### System Configuration

```bash
# Logging and monitoring
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
LOG_FILE=logs/medigraph.log
SENTRY_DSN=https://your-sentry-dsn

# Performance tuning
MAX_WORKERS=4                     # Concurrent processing
CACHE_TTL=3600                    # Cache expiration (seconds)
MEMORY_LIMIT=8GB                  # Process memory limit

# Security
ALLOW_ORIGINS=http://localhost:3000,https://yourapp.com
CORS_ORIGINS=*                    # In production, specify explicitly
```

### Configuration Files

- `.env`: Environment variables
- `config.yaml`: Advanced system configuration
- `logging.conf`: Logging configuration

## Troubleshooting

### Common Issues

#### Database Connection Problems

**Symptom**: "Unable to connect to Neo4j database"

**Solutions**:
```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Verify credentials
curl http://localhost:7474 -u neo4j:your-password

# Reset database
docker compose down neo4j
docker volume rm medigraph_neo4j_data
docker compose up -d neo4j
```

#### LLM Provider Issues

**Symptom**: "API rate limit exceeded" or "Authentication failed"

**Solutions**:
```bash
# Test API key validity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Switch providers for failover
export ANTHROPIC_API_KEY=sk-ant-your-key

# Check usage quotas in provider dashboard
```

#### Memory Issues

**Symptom**: "Out of memory" during large document processing

**Solutions**:
```bash
# Reduce chunk size
export CHUNK_SIZE=1500

# Process in smaller batches
curl -F "file=@large_doc.pdf" \
     -F "max_chunks=10" \
     http://localhost:8004/create_ontology_guided_kg

# Monitor resource usage
docker stats
```

#### Performance Problems

**Symptom**: Slow response times

**Solutions**:
```bash
# Enable GPU acceleration (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Increase workers for parallel processing
export MAX_WORKERS=8

# Optimize Neo4j memory
# Add JVM options to neo4j.conf
dbms.memory.heap.initial_size=2G
dbms.memory.heap.max_size=4G
```

### Diagnostic Tools

```bash
# Health checks
curl "http://localhost:8004/health/neo4j"

# Log analysis
tail -f logs/medigraph.log

# Database queries
cypher-shell -u neo4j -p password
MATCH (n) RETURN count(n) as node_count;

# Performance monitoring
docker stats
htop  # or top
```

### Getting Help

**Steps**:
1. Review `logs/medigraph.log` for details
2. Run health checks (see Diagnostic Tools)
3. Verify environment config vs docs
4. Submit issues with full logs & system info at [GitHub Issues](https://github.com/julka01/MediGraphRAG/issues)

## Contributing

We welcome contributions from healthcare professionals, data scientists, and open source enthusiasts!

### Development Workflow

#### 1. Fork and Clone
```bash
git clone https://github.com/YOUR-USERNAME/MediGraphRAG.git
cd MediGraphRAG
```

#### 2. Set Up Development Environment
```bash
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt

# Set up pre-commit hooks
pre-commit install
```

#### 3. Run Tests
```bash
# Test suite
pytest tests/ -v --cov=src/

# Lint code
black src/ tests/
flake8 src/ tests/

# Type checking
mypy src/ --strict
```

#### 4. Make Changes

Create a feature branch:
```bash
git checkout -b feature/your-feature-name

# Make changes...
# Write tests...
# Update documentation...

# Commit changes
git add .
git commit -m "Add feature: your descriptive message"

# Push commits
git push origin feature/your-feature-name
```

#### 5. Submit Pull Request

- Create PR with comprehensive description
- Ensure all tests pass
- Update documentation if needed

### Code Standards

- **Python Version**: 3.11+ compatible
- **Style**: [Black](https://black.readthedocs.io/) for formatting
- **Linting**: [flake8](https://flake8.pycqa.org/) + [mypy](https://mypy.readthedocs.io/)
- **Testing**: [pytest](https://docs.pytest.org/) with >= 80% coverage


## License

MediGraph is released under the MIT License. See [LICENSE](LICENSE) for full details.


### Additional Terms for Healthcare Use

- **Regulatory Compliance**: Users are responsible for ensuring compliance with applicable healthcare regulations (HIPAA, GDPR, etc.)
- **Clinical Use**: This software is for research and educational purposes. Clinical decisions should be made by qualified healthcare professionals
- **Data Privacy**: Implement appropriate data protection measures when handling patient information

---

**MediGraph v1.0.0-rc1**: Advancing clinical decision-making through transparent, evidence-based AI systems.

*For support: [GitHub Issues](https://github.com/julka01/MediGraphRAG/issues)*
