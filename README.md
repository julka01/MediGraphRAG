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

---

## Table of Contents

- [Welcome](#welcome)
- [What's MediGraph?](#whats-medigraph)
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
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
- [Development Roadmap](#development-roadmap)
- [Dependencies](#dependencies)
- [License](#license)
- [Citing Our Work](#citing-our-work)
- [Acknowledgments](#acknowledgments)

---

## Welcome

Welcome to MediGraph! This project provides a comprehensive, AI-powered knowledge graph system specifically designed for healthcare and medical decision support. Our goal is to bridge the gap between unstructured medical data and actionable clinical insights through transparent, evidence-based reasoning.

---

## What's MediGraph?

MediGraph is an advanced knowledge graph platform that transforms raw medical documents, research papers, clinical guidelines, and patient records into structured, queryable knowledge graphs. By leveraging large language models and biomedical ontologies, MediGraph enables healthcare professionals to ask complex questions about patient care, treatment options, and medical research while maintaining full transparency and traceability.

### Philosophy
We believe that AI-powered medical systems must be:
- **Transparent**: Every recommendation tracks back to source evidence
- **Explainable**: Complex reasoning is broken down into comprehensible steps
- **Auditable**: All processing steps are logged and verifiable
- **Evidence-Based**: Results are grounded in validated medical knowledge

---

## Introduction

### System Overview

MediGraph addresses the critical challenge of knowledge discovery in healthcare by implementing a multi-stage pipeline that:

1. **Processes diverse medical document formats** (PDFs, CSVs, research papers, clinical notes)
2. **Extracts entities and relationships** using advanced language models
3. **Validates information against biomedical ontologies** (OWL, UMLS, etc.)
4. **Organizes knowledge into graph structures** that can be queried efficiently
5. **Provides comprehensive reasoning** with full source attribution

### Key Capabilities

- **Multi-format Document Ingestion**: Support for PDFs, CSV files, research articles, and clinical guidelines
- **Ontology-Guided Extraction**: Ensures terminological consistency with biomedical standards
- **Evidence-Based Querying**: Natural language questions return reasoned answers with source citations
- **Graph Visualization**: Interactive exploration of knowledge relationships
- **Batch Processing**: Large-scale data processing for clinical research workflows

### Response Transparency

Unlike traditional RAG systems, MediGraph provides detailed reasoning traces:

```json
{
  "recommendation_summary": "Evidence-based treatment plan",
  "reasoning_path": [
    "Patient symptoms align with clinical presentation X",
    "Treatment Y shows effectiveness in similar cases",
    "Evidence supports Z as recommended approach"
  ],
  "confidence_metrics": {
    "similarity_score": 0.87,
    "entity_coverage": 90.2
  },
  "source_citations": [
    "Author A, Journal B (2023) Section C"
  ]
}
```

---

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
- **Ollama** (local model hosting for privacy/compliance)

---

## Installation

MediGraph can be installed for local development or deployed in production environments. Choose the appropriate method based on your use case.

### Development Environment Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/julka01/MediGraphRAG.git
cd medigraph
```

#### 2. Set Up Python Environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate   # Windows

# Install Python dependencies
pip install -r requirements-prod.txt
```

#### 3. Set Up Neo4j Database

**Option A: Docker (Recommended)**
```bash
# Start Neo4j container
docker compose up -d neo4j

# Wait for Neo4j to initialize (~30 seconds)
# Access browser at http://localhost:7474
# Default credentials: neo4j/neo4j
```

**Option B: Native Installation**
```bash
# Install Neo4j locally following official documentation
# Ensure it's running on port 7687 (bolt protocol)
```

#### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Required: Neo4j connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password

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

#### 5. Verify Installation
```bash
# Test database connectivity
python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password')); driver.verify_connectivity(); print('Neo4j connected!')"

# Start the application
python start_server.py

# Test endpoints
curl "http://localhost:8004/health/neo4j"
```

### Production Deployment

#### Option 1: Docker Compose (Complete Stack)

```yaml
# docker-compose.yml (already provided)
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

```bash
# Install Helm chart (when available)
helm install medigraph ./k8s/

# Or deploy individual manifests
kubectl apply -f k8s/neo4j-deployment.yaml
kubectl apply -f k8s/medigraph-deployment.yaml
```

#### Option 3: Standalone Container

```bash
# Build and run
docker build -t medigraph:latest .
docker run -p 8004:8004 \
  -e OPENAI_API_KEY=sk-your-key \
  -e NEO4J_URI=bolt://external-neo4j:7687 \
  medigraph:latest
```

#### Production Environment Variables

```bash
# Database
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

---

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
    "document_names": ["urology-guidelines-2023.pdf"],
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

---

## Features

### Core Features

- **📄 Multi-Format Document Processing**: PDF, CSV, text files, research papers
- **🧠 Ontology-Guided Entity Extraction**: Validates entities against biomedical ontologies
- **🔗 Relationship Discovery**: Identifies meaningful connections between medical concepts
- **💬 Natural Language Queries**: Ask complex medical questions in plain English
- **📊 Evidence-Based Responses**: Every answer includes source citations and reasoning
- **🎯 Confidence Scoring**: Quality metrics for all generations
- **🌐 Graph Visualization**: Interactive exploration of knowledge relationships
- **⚡ High Performance**: Optimized for clinical workflows with <5s response times

### Advanced Capabilities

- **Batch Processing**: Handle large cohorts of patient data efficiently
- **Multi-Modal Support**: Text, images, and structured data integration
- **Regulatory Compliance**: HIPAA-ready architecture with audit trails
- **API-First Design**: RESTful APIs for seamless integration
- **Extensible Architecture**: Custom modules for specialized medical domains

### Quality Assurance Features

- **Citation Transparency**: Every conclusion links to source documents
- **Version Control**: Track changes in medical knowledge over time
- **Validation Pipeline**: Automated quality checks at each processing stage
- **Error Handling**: Comprehensive error reporting and recovery mechanisms

---

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

---

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

---

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

---

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

---

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

### Contribution Areas

- **Healthcare Domain Expertise**: Medical knowledge integration, clinical workflows
- **Machine Learning**: Better models, evaluation metrics, bias detection
- **Graph Algorithms**: Query optimization, visualization, analytics
- **Infrastructure**: Deployment, scaling, monitoring
- **Documentation**: Guides, tutorials, API documentation
- **Testing**: Unit tests, integration tests, performance benchmarks

### Research Collaboration

Interested in academic partnerships? Contact us for:
- Joint research projects
- Dataset sharing agreements
- Benchmarking collaborations
- Clinical validation studies

---

## Development Roadmap

### Current Version: 1.0.0-rc1 (git: 324db44)

### Recently Implemented
- ✅ Ontology-guided knowledge graph creation
- ✅ Multi-LLM provider support
- ✅ Evidence-based reasoning with citations
- ✅ Batch processing for large datasets
- ✅ Docker containerization
- ✅ RESTful API architecture

### Upcoming Features (v1.1.0)



### Future Releases (v1.2.0+)


---

## Dependencies

### Core Libraries

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| `fastapi` | 0.115.9 | Web framework | MIT |
| `uvicorn` | 0.34.2 | ASGI server | Apache 2.0 |
| `neo4j` | 5.25.0+ | Graph database driver | Apache 2.0 |
| `pypdf` | 5.4.0 | PDF processing | BSD-3 |
| `transformers` | 4.51.3 | LLM framework | Apache 2.0 |
| `chromadb` | 1.0.6 | Vector database | Apache 2.0 |
| `sentence-transformers` | 4.1.0 | Embeddings | Apache 2.0 |
| `langchain` | 0.3.27 | LLM orchestration | MIT |
| `pandas` | 2.2.3 | Data processing | BSD-3 |
| `scikit-learn` | 1.5.0 | Scientific computing | BSD-3 |
| `owlready2` | 0.48 | Ontology processing | LGPL |

### System Compatibility

| Component | Version | Status | Notes |
|-----------|---------|--------|-------|
| **Python** | 3.11+ | Verified | Main development target |
| **Python** | 3.8-3.10 | Compatible | May have limitations |
| **Neo4j** | 5.18+ | Tested | Recommended |
| **Neo4j** | 5.0-5.17 | Compatible | Some features limited |
| **Docker** | 20.10+ | Verified | All deployments |
| **Docker** | <20.10 | May work | Untested |
| **CUDA** | 12.0+ | Supported | GPU acceleration |
| **CUDA** | 11.0-11.8 | Supported | Legacy GPUs |
| **macOS** | Monterey+ | Tested | Native and Docker |
| **Linux** | Ubuntu 20.04+ | Tested | Primary platform |
| **Windows** | 10/11 | ⚠️ Compatible | Docker recommended |

### Development Dependencies

```
pytest>=7.4.0          # Testing framework
black>=23.0.0          # Code formatting
flake8>=6.0.0          # Linting
mypy>=1.0.0            # Type checking
pre-commit>=3.0.0      # Git hooks
coverage>=7.0.0        # Test coverage
```

---

## License

MediGraph is released under the MIT License. See [LICENSE](LICENSE) for full details.


### Additional Terms for Healthcare Use

- **Regulatory Compliance**: Users are responsible for ensuring compliance with applicable healthcare regulations (HIPAA, GDPR, etc.)
- **Clinical Use**: This software is for research and educational purposes. Clinical decisions should be made by qualified healthcare professionals
- **Data Privacy**: Implement appropriate data protection measures when handling patient information

---

## Citing Our Work

```
TBD
```

---

## Acknowledgments

MediGraph builds upon foundational work in:

- **Knowledge Graph Research**: Neo4j, RDF, and Semantic Web Technologies
- **Natural Language Processing**: Transformers, BERT, and LLM architectures
- **Healthcare Informatics**: Biomedical ontologies (UMLS, SNOMED CT)
- **Graph Neural Networks**: Research in representation learning on graphs
- **Clinical Decision Support**: Evidence-based medicine frameworks

Special thanks to the open-source communities that made this work possible.

---

**MediGraph v1.0.0-rc1**: Advancing clinical decision-making through transparent, evidence-based AI systems.

*For support: [GitHub Issues](https://github.com/julka01/MediGraphRAG/issues)*
274 *Research collaboration inquiries welcome.*
