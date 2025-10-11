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

## CITE OUR WORK


```
TBD
```

## System Capabilities

### Evidence-Based Inference
MediGraph provides traceable reasoning by explicitly linking all responses to source document sections, enabling verification of information provenance.

### Healthcare Domain Optimization
Designed for healthcare informatics workflows with:
- Multi-format document ingestion (PDF, CSV, structured text)
- Ontology-driven entity extraction and relationship identification
- Natural language query processing with source attribution
- Interactive graph visualisation of decision logic
- Regulatory compliance through standardised biomedical terminology

### Attribution Transparency
Responses include detailed citation trails linking each conclusion to specific document sections, enabling independent verification of clinical assertions.

### Key Advantages
- **Explainability**: Source document citation trails for all generated responses
- **Terminological Consistency**: Ontology-driven entity extraction ensures standardised medical terminology
- **Population-level Analysis**: Support for cohort-level research queries and statistical analysis
- **Auditability**: Comprehensive logging and compliance mechanisms for healthcare regulatory requirements

## Quick Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF/CSV/Text  │───▶│   LLM Parser     │───▶│   Knowledge     │
│   Documents     │    │   + Ontology     │    │   Graph Store   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Natural       │───▶│   Vector Search  │───▶│  Evidence App   │
│   Language      │    │   + Graph        │    │  Citations       │
│   Questions     │    │   Traversal      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## System Components

### Core Modules

| Component | Technologies | Purpose |
|-----------|--------------|---------|
| **Document Processing** | PyMuPDF, Text + CSV | File ingestion and text extraction |
| **Entity Extraction** | OpenAI GPT, Anthropic Claude, Ollama | Medical concept identification |
| **Ontology Framework** | OWLready2, Biomedical OWL | Clinical terminology consistency |
| **Graph Storage** | Neo4j 5.0+ | Interconnected data persistence |
| **Semantic Search** | ChromaDB, Sentence Transformers | Vector similarity matching |
| **Query Processing** | LangChain, FastAPI | RAG with transparency |

### Response Structure Format
```json
{
  "recommendation_summary": "Treatment approach based on patient's profile",
  "node_traversal_path": [
    "Patient_Record(123) → HAS_SYMPTOM → Frequent_Urination(456)",
    "Frequent_Urination(456) → INDICATES → Prostate_Cancer(789)",
    "Prostate_Cancer(789) → TREATED_BY → Radical_Prostatectomy(ABC)"
  ],
  "reasoning_path": [
    "Finding 1: Risk factors indicate malignancy",
    "Finding 2: PSA levels confirm diagnosis",
    "Conclusion: Surgery recommended based on evidence"
  ],
  "evidence_synthesis": "Combined analysis supports surgical intervention",
  "confidence_metrics": {
    "similarity_score": 0.87,
    "entity_coverage": 90.2,
    "relationship_consistency": 85.1
  },
  "source_citations": [
    "Smith et al. Journal of Urology (2019) pg. 234-240",
    "Patient Record A12345: PSA 12.5 ng/mL",
    "EAU Guidelines 2023 Chapter 4"
  ]
}
```

## Implementation Details

### Processing Pipeline Architecture

1. **Document Ingestion**: Multi-format text extraction with content validation
2. **Intelligent Chunking**: Semantic boundary preservation (default: 2000 chars with 300-char overlap)
3. **Ontology-Guided Extraction**: Entity and relationship validation against biomedical ontologies
4. **Harmonization**: Duplicate entity resolution and consistency checking
5. **Graph Construction**: Cypher-based persistence with automatic indexing
6. **Vector Embedding**: 384-dimension semantic representation for similarity search
7. **Query Processing**: Multi-stage retrieval with confidence scoring

### Technical Specifications
- **Chunk Size**: Configurable (1000-4000 characters optimal)
- **Overlap Strategy**: Adaptive semantic boundary detection
- **Ontology Support**: OWL 2.0 format with automatic class and property mapping
- **Vector Dimensions**: 384 (Sentence Transformers all-MiniLM-L6-v2)
- **Similarity Function**: Cosine distance with configurable thresholds
- **Response Latency**: \< 5 seconds for typical medical queries

## Setup Procedures

### Development Environment Installation
```bash
# Python environment preparation
pip install -r requirements-prod.txt

# Database infrastructure setup
docker compose up -d neo4j

# Environment configuration
export OPENAI_API_KEY="sk-your-api-key-here"

# Application startup
python start_server.py
```

### Production Deployment Options
```bash
# Containerized deployment
docker compose -f docker-compose.yml up --build

# Standalone container with external Neo4j
docker run -p 8004:8004 -v ./.env:/app/.env medigraph:latest
```

## Configuration Management

Centralized configuration through environment variables:

```bash
# Core infrastructure
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_DATABASE=neo4j

# LLM provider configuration
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...

# Processing parameters
CHUNK_SIZE=2000
EMBEDDING_MODEL=sentence_transformers
LLM_MODEL_CONFIG_gpt_4=gpt-4,your-openai-key
VECTOR_SIMILARITY_THRESHOLD=0.08

# Optional: Local model hosting
OLLAMA_HOST=http://localhost:11434
```

## User Workflow

### 1. Data Ingestion
```bash
# Medical document processing
curl -X POST "http://localhost:8004/create_ontology_guided_kg" \
  -F "file=@clinical_guidelines.pdf" \
  -F "provider=openai" \
  -F "model=gpt-4" \
  -F "max_chunks=10"
```

### 2. Knowledge Graph Querying
```bash
# Evidence-based information retrieval
curl -X POST "http://localhost:8004/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What treatment options exist for patients with PSA > 10?",
    "document_names": ["urology-guidelines-2023.pdf"],
    "provider_rag": "anthropic",
    "model_rag": "claude-3-sonnet-20240229"
  }'
```

### 3. Batch Processing Operations
```bash
# Large-scale medical data processing
curl -X POST "http://localhost:8004/bulk_process_csv" \
  -F "csv_file=@patient_records.csv" \
  -F "batch_size=100" \
  -F "max_chunks=25"
```

### 4. Cohort-Level Evidence Query Examples

#### Diuretic Therapy in Heart Failure with Liver Disease
```bash
# Evaluate diuretic combinations in heart failure patients
curl -X POST "http://localhost:8004/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "In patients admitted with heart failure signs and hepatic congestion, what diuretic combinations (furosemide + spironolactone vs furosemide alone) were more effective at resolving symptoms? What kidney function and electrolyte monitoring approaches prevented renal injury?",
    "provider_rag": "openai",
    "model_rag": "gpt-4o"
  }'
```

#### Antibiotic Management in Hospital-Acquired Infections
```bash
# Analyze antibiotic treatment patterns in ICU patients
curl -X POST "http://localhost:8004/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Among intensive care patients developing hospital-acquired infections, what antibiotic choices and durations were associated with better outcomes? How did Clostridiodes difficile development rates vary by antibiotic class and concurrent PPI use?",
    "provider_rag": "openai",
    "model_rag": "gpt-4o"
  }'
```

#### Orthopedic Trauma Surgery and Complication Rates
```bash
# Evaluate surgical outcomes in cervical spine fracture patients
curl -X POST "http://localhost:8004/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "In patients admitted with cervical spine fractures from falls, what preoperative factors predicted successful conservative vs surgical management? What complication rates were observed for cervical decompression procedures in anticoagulated atrial fibrillation patients?",
    "provider_rag": "openai",
    "model_rag": "gpt-4o"
  }'
```

## Performance Validation Results

## Research Applications Index



## API Specifications

### Core Endpoint Catalog

| Endpoint | Method | Purpose | Request Format |
|----------|--------|---------|----------------|
| `/create_ontology_guided_kg` | POST | Knowledge graph generation | Multipart form with document |
| `/chat` | POST | Transparent Q&A system | JSON with query and context |
| `/bulk_process_csv` | POST | Batch medical data processing | CSV file with parameters |
| `/health/neo4j` | GET | Database connectivity verification | None |

### Authentication and Security
- API key-based authentication for LLM providers
- Database credential management through environment variables
- Input validation and sanitization for all endpoints
- Rate limiting and usage monitoring

## Troubleshooting and Diagnostics

### Common Resolution Patterns

```bash
# Database connectivity verification
curl http://localhost:7474 -u neo4j:password

# Neo4j service restart procedure
docker compose down neo4j && docker compose up -d neo4j

# LLM provider connectivity testing
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Test"}]}' \
  https://api.openai.com/v1/chat/completions

# Memory usage monitoring
docker stats $(docker ps -q --filter name=neo4j)
```

### Diagnostic Logging Levels
```python
# Enhanced debugging configuration
import logging
logging.basicConfig(level=logging.DEBUG)

# Specific module debugging
logger = logging.getLogger("ontology_guided_kg_creator")
logger.setLevel(logging.DEBUG)
```

## Development Roadmap

### Current Version: 1.0.0-rc1 (git: 324db44)

### Planned Improvements
- Enhanced multi-modal reasoning capabilities
- Temporal relationship modeling for clinical progression
- Uncertainty quantification for medical recommendations
- Federated learning support for privacy-preserving collaboration

## Dependencies and Compatibility

### Core Libraries (Computed Dependencies)
```txt
fastapi==0.115.9                    # Web framework
uvicorn==0.34.2                     # ASGI server
neo4j>=5.25.0,<6.0.0                # Graph database driver
pypdf==5.4.0                        # PDF processing
transformers==4.51.3                # LLM framework
chromadb==1.0.6                     # Vector database
sentence-transformers==4.1.0        # Embedding generation
langchain==0.3.27                   # LLM orchestration
pandas==2.2.3                       # Data processing
scikit-learn==1.5.0                 # Scientific computing
owlready2==0.48                     # Ontology processing
```

### System Compatibility Matrix

| Component | Version | Architecture | Status |
|-----------|---------|--------------|---------|
| Python | 3.11+ | x86_64, ARM64 | ✅ Verified |
| Neo4j | 5.0+ | x86_64 | ✅ Tested |
| Docker | 20.10+ | Multi-platform | ✅ Compatible |
| CUDA | 11.0+ | NVIDIA GPUs | ✅ Optional GPU |

## Contributing and Collaboration

### Development Workflow
```bash
# Repository clone
git clone https://github.com/julka01/MediGraphRAG.git
cd medigraph

# Development environment setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt  # Include dev dependencies

# Run test suite
pytest tests/ -v --cov=src/

# Submit enhancements via pull request
git checkout -b feature/enhancement-name
# ... implementation ...
git push origin feature/enhancement-name
```

### Code Quality Standards


## Acknowledgments and Research Context

This implementation draws from established work in:

```

```

---

**MediGraph v1.0.0-rc1**: Advancing clinical decision-making through transparent, evidence-based AI systems.
