# MediGraph: AI-Powered Knowledge Graph for Medical Decision Support

[![Version](https://img.shields.io/badge/version-1.0.0--rc1--324db44-blue.svg)](https://github.com/julka01/MediGraphRAG)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/neo4j-5.0+-brightgreen.svg)](https://neo4j.com/)

```
   _____     _____     _____         _____     _____     _____
 .'     '. .'     '. .'     '.     .'     '. .'     '. .'     '.
'.'''_'''.'  '''_'''  '''_'''.   .'  '''_'''  '''_'''  '''_'''.
|                                   |        |            |        |
|          Knowledge               |        |   Medical   |        |
|          Graph                   |        |   Data      |        |
|                                   |        |            |        |
 '._''.     '__'.''_''.     '__'.'     '__'.'     '__'.'     '__
        '._ __ '_.'__'.'     '_.'__'.'     '_.'__'.'     '_.'__'.
           '--  '--' '""'     '"""' '""'     '"""' '""'     '"""'

            Transparent          →          Reliable
            Reasoning                          Findings
```

MediGraph implements an ontology-guided knowledge graph creation system for structuring unstructured medical data, enabling transparent querying and evidence-based reasoning.

## Overview

MediGraph provides an ontology-guided approach to structuring unstructured medical data into knowledge graphs, enabling efficient querying for evidence-based reasoning. The system processes diverse biomedical document types—including patient records, clinical guidelines, and medical literature—using large language models guided by biomedical ontologies to extract entities and relationships. This structured representation supports transparent querying for clinical decision support, population health analysis, and medical research.

Key contributions include:
- **Ontology-driven data structuring**: Systematic representation of medical knowledge using established biomedical taxonomies (e.g., OWL, UMLS) to ensure semantic consistency
- **Evidence-based querying**: Retrieval-augmented generation with full source attribution enabling traceability of clinical recommendations to original documents
- **Multi-modal document processing**: Support for PDFs, CSV files, and research articles, with future extensions planned for medical imaging and other multimodal data.
- **Scalable architecture**: Neo4j-based graph database with vector embeddings for semantic similarity search, supporting both individual query resolution and large-scale cohort analysis

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Development Environment Setup](#development-environment-setup)
  - [Production Deployment](#production-deployment)
- [Getting Started](#getting-started)
- [Features](#features)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Dependencies](#dependencies)
- [License](#license)

## Prerequisites

### System Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **Python** | 3.11 | 3.11+ | Required for all functionality |
| **RAM** | 8GB | 16GB+ | Graph operations are memory-intensive |
| **Storage** | 10GB | 50GB+ | Depends on dataset and graph size |
| **CPU** | 4 cores | 8+ cores | Parallel processing support |
| **GPU** | Optional | NVIDIA 11.0+ | Accelerates embeddings/LM inference |

### Software Dependencies

- **Neo4j** 5.0+ : Graph database for knowledge persistence
- **Docker** 20.10+ : Containerized deployment

### External API Requirements

At least one language model provider:
- **OpenAI API** (recommended for production)
- **Anthropic Claude API** (alternative enterprise option)
- **Google Gemini API** (cost-effective option)
- **OpenRouter** (unified API access to multiple models)
- **Ollama** (local model hosting for privacy/compliance)

## Installation

MediGraph can be installed for local development or deployed in production environments. Choose the appropriate method based on your use case.

### Development Environment Setup

#### Step 1: Clone the Repository
```bash
git clone https://github.com/julka01/MediGraphRAG.git
cd medigraph
```

#### Step 2: Set Up Python Environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR for Windows: venv\Scripts\activate

# Upgrade pip (recommended)
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements-prod.txt
```

#### Step 3: Set Up Neo4j Database

**Option A: Docker (Recommended)**

First, ensure Docker is installed and running. Then:
```bash
# Start Neo4j container
docker compose up -d neo4j

# Wait for Neo4j to initialize (~30 seconds)
# Access browser at http://localhost:7474
# Default credentials: neo4j/neo4j (you'll be prompted to change password immediately)
```

To stop and remove containers later:
```bash
docker compose down
```

**Option B: Native Installation**

- Download Neo4j Community Edition from the official website
- Follow the installation guide for your OS
- Ensure it's running on port 7687 (bolt protocol)
- Configure authentication and plugins as needed

#### Step 4: Configure Environment Variables

Create a `.env` file in the project root with the following content (adjust values as needed):

```bash
# Required: Neo4j connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password  # Change this to your actual password

# Required: At least one LLM provider
OPENAI_API_KEY=sk-your-openai-api-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
# OR
GEMINI_API_KEY=your-gemini-api-key-here

# Optional: Advanced configuration
OLLAMA_HOST=http://localhost:11434
CHUNK_SIZE=2000
VECTOR_SIMILARITY_THRESHOLD=0.08
```

**Important Notes:**
- Never commit the `.env` file to version control
- Obtain API keys from your provider's dashboard
- Test API keys by making a small request (refer to provider docs)

#### Step 5: Verify Installation
```bash
# Test database connectivity
python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'your-password')); driver.verify_connectivity(); print('Neo4j connected!')"

# Start the application
python start_server.py

# Test health endpoint (in a new terminal)
curl "http://localhost:8004/health/neo4j"

# If using API endpoints, verify they respond with 200 status
```

**Troubleshooting Installation:**
- If virtual environment activation fails, ensure Python 3.11+ is installed
- For Docker issues, check `docker ps` to confirm Neo4j is running
- API key errors usually mean invalid keys or network issues—test with provider's API directly
- If port 7687 conflicts, modify Docker Compose to use a different port

### Production Deployment

#### Option 1: Docker Compose (Complete Stack)

Use the provided `docker-compose.yml`:

```yaml
version: '3.8'
services:
  neo4j:
    image: neo4j:5.18
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    volumes:
      - neo4j_data:/data
      - ./apoc:/var/lib/neo4j/plugins
    environment:
      NEO4J_AUTH: neo4j/your-production-password

  medigraph:
    build: .
    ports:
      - "8004:8004"
    depends_on:
      - neo4j
    env_file:
      - .env
    restart: unless-stopped
```

Deploy with:
```bash
docker compose up -d
```

#### Option 2: Kubernetes Deployment

For scalable production environments:

```bash
# If using Helm (when available)
helm install medigraph ./k8s/

# Or deploy individual manifests
kubectl apply -f k8s/neo4j-deployment.yaml
kubectl apply -f k8s/medigraph-deployment.yaml
```

Ensure persistent volumes and ConfigMaps are set up for environment variables.

#### Option 3: Standalone Container

Useful for cloud deployments or CI/CD:

```bash
# Build the image
docker build -t medigraph:latest .

# Run the container
docker run -p 8004:8004 \
  --env-file .env \
  -e NEO4J_URI=bolt://external-neo4j:7687 \
  medigraph:latest
```

#### Production Environment Variables

```bash
# Database (use strong passwords)
NEO4J_URI=bolt://neo4j-cluster-url:7687
NEO4J_USER=medigraph_prod
NEO4J_PASSWORD=strong-password-here

# LLM Providers (multiple for redundancy)
OPENAI_API_KEY=sk-production-key
ANTHROPIC_API_KEY=sk-ant-production-key

# Performance tuning
CHUNK_SIZE=2000
MAX_WORKERS=4
CACHE_TTL=3600

# Logging and monitoring
LOG_LEVEL=INFO
SENTRY_DSN=https://your-sentry-dsn
```

**Production Security Considerations:**
- Use strong, unique passwords for Neo4j
- Rotate API keys regularly
- Implement rate limiting and IP whitelisting
- Enable HTTPS/TLS
- Set up monitoring and alerting

## Getting Started

### First Knowledge Graph Creation

Upload your first medical document:

```bash
# Using cURL
curl -X POST "http://localhost:8004/create_ontology_guided_kg" \
  -F "file=@clinical_guidelines.pdf" \
  -F "provider=openai" \
  -F "model=gpt-4"
```

### Medical Question Answering

Ask evidence-based questions:

```bash
# Natural language query
curl -X POST "http://localhost:8004/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the treatment options for stage II prostate cancer?",
    "document_names": ["EAU-EANM-ESTRO-ESUR-ISUP-SIOG-Pocket-on-Prostate-Cancer-2025_updated.pdf"],
    "provider_rag": "openai",
    "model_rag": "gpt-4"
  }'
```

### Batch Processing Patient Data

Process multiple patient records:

```bash
curl -X POST "http://localhost:8004/bulk_process_csv" \
  -F "csv_file=@patient_cohort.csv" \
  -F "batch_size=50" \
  -F "max_chunks=25"
```

### Advanced Query Examples

#### Population-Level Analysis
```bash
# Diuretic therapy outcomes
curl -X POST "http://localhost:8004/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "In heart failure patients with hepatic congestion, what diuretic combinations are most effective?",
    "provider_rag": "openai",
    "model_rag": "gpt-4o"
  }'
```

#### Antimicrobial Stewardship
```bash
curl -X POST "http://localhost:8004/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "For ICU patients with hospital-acquired infections, which antibiotic regimens show best outcomes?",
    "provider_rag": "openai",
    "model_rag": "gpt-4o"
  }'
```

## Features

### Computational Capabilities

- **Multi-modal Document Processing**: Automated ingestion of diverse biomedical document formats (PDF, CSV, research articles, clinical notes)
- **Ontology-Guided Entity Extraction**: Deep learning-based identification and validation of medical concepts against established biomedical ontologies (e.g., UMLS, SNOMED CT)
- **Evidence-Based Reasoning Framework**: Retrieval-augmented generation with full source attribution and confidence metrics

### Advanced Medical Informatics Features

- **Natural Language Medical Q&A**: Complex clinical question answering with temporal reasoning and uncertainty quantification
- **Interactive Knowledge Exploration**: Dynamic visualization and traversal of clinical relationships and treatment pathways
- **Scalable Cohort Analysis**: Batch processing of patient datasets for population-level clinical insights and research workflows

## Architecture

### System Overview

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

### Processing Pipeline

1. **Document Ingestion**: Multi-format file processing with metadata extraction
2. **Intelligent Chunking**: Semantic boundary preservation (configurable sizes)
3. **Ontology Validation**: Entity and relationship validation against biomedical standards
4. **Graph Construction**: Neo4j-based persistence with optimized indexing
5. **Vector Embeddings**: Semantic representation for similarity search
6. **Query Processing**: Multi-stage retrieval with confidence scoring

### Technical Specifications

| Component | Specification | Notes |
|-----------|----------------|-------|
| **Document Chunking** | 1000-4000 chars, adaptive | Semantic boundary detection |
| **Embedding Model** | 384-dim, Sentence-BERT | All-MiniLM-L6-v2 |
| **Similarity Function** | Cosine distance | Configurable thresholds |
| **Graph Database** | Neo4j 5.0+ | Cypher query language |
| **Vector Database** | ChromaDB | In-memory acceleration |
| **Response Latency** | <5 seconds | Typical medical queries |

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

#### Optional Processing Parameters

```bash
# Document processing
CHUNK_SIZE=2000                    # Character chunks (1000-4000)
CHUNK_OVERLAP=300                  # Overlap between chunks
MAX_CHUNKS=50                      # Processing limit

# Similarity search
VECTOR_SIMILARITY_THRESHOLD=0.08   # Similarity cutoff
EMBEDDING_MODEL=sentence_transformers
EMBEDDING_BATCH_SIZE=32

# Model configurations
LLM_MODEL_CONFIG_gpt_4=gpt-4,openai-key
LLM_MODEL_CONFIG_claude=claude-3-sonnet-20240229,anthropic-key

# Local model hosting
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2:7b
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

1. **Review logs**: Check `logs/medigraph.log` for error details
2. **Run diagnostics**: Execute health check endpoints
3. **Check documentation**: Verify environment configuration
4. **Submit issues**: Include full error logs and system information

## Contributing

We welcome contributions from healthcare professionals, data scientists, and open source enthusiasts!

### Development Workflow

#### 1. Fork and Clone
```bash
git clone https://github.com/YOUR-USERNAME/MediGraphRAG.git
cd medigraph
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
