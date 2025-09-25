# Knowledge Graph RAG System

This application creates knowledge graphs from PDF documents and provides a chat interface for querying the knowledge graph using various AI models.

## Features
- Support for multiple AI providers: OpenAI, Ollama (local), Gemini, Hugging Face, DeepSeek, Anthropic
- PDF processing to extract text and create knowledge graphs
- Interactive visualization of knowledge graphs
- RAG-based chat interface using the knowledge graph
- Biomedical ontology support for enhanced knowledge extraction
- Vector similarity search with ChromaDB
- Neo4j graph database integration

## Requirements

### System Requirements
- Python 3.11 or higher
- Docker and Docker Compose (for containerized deployment)
- Neo4j database (included in Docker setup)

### Python Dependencies
The application uses a comprehensive set of dependencies organized into categories:

- **Web Framework**: FastAPI, Uvicorn, Gunicorn
- **AI/ML Libraries**: LangChain ecosystem, OpenAI, Anthropic, Google AI
- **Machine Learning**: PyTorch, Transformers, Sentence Transformers
- **Document Processing**: PyPDF, Unstructured, BeautifulSoup
- **Graph Database**: Neo4j driver
- **Vector Database**: ChromaDB
- **Scientific Computing**: NumPy, Pandas, Scikit-learn
- **NLP**: spaCy, NLTK, Language Detection

All dependencies are specified in `requirements.txt` with pinned versions for reproducibility.

## Installation

### Recommended: Hybrid Development Setup

**Best for development and testing** - Local Python app with Dockerized Neo4j database.

#### Quick Start:
```bash
# 1. Install Python dependencies
pip install -r requirements-prod.txt

# 2. Start Neo4j database
docker compose up -d neo4j

# 3. Configure your .env file with API keys
# 4. Start the application
python start_server.py
```

#### Detailed Setup:

1. **Clone the repository**:
```bash
git clone <repository-url>
cd tool-2025-kg-rag
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements-prod.txt
```

4. **Configure your `.env` file** with your API keys:
```env
# AI Provider API Keys (configure at least one)
OPENAI_API_KEY=your-openai-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GEMINI_API_KEY=your-gemini-api-key
HF_API_TOKEN=your-huggingface-api-token
DEEPSEEK_API_KEY=your-deepseek-api-key

# Database Configuration (Neo4j with no auth)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=
NEO4J_DATABASE=neo4j

# Model Configuration
EMBEDDING_MODEL=sentence_transformers
LLM_MODEL_CONFIG_gpt-3.5-turbo=gpt-3.5-turbo,your-openai-key
LLM_MODEL_CONFIG_meta_llama_llama_4_maverick_free=meta-llama/llama-4-maverick:free,your-openrouter-key,https://openrouter.ai/api/v1

# Optional: Ollama Configuration
OLLAMA_HOST=http://localhost:11434
```

5. **Start Neo4j database using Docker**:
```bash
docker compose up -d neo4j
```
This starts Neo4j with no authentication required.

6. **For Ollama (local models)**, install and download models on your host machine:
```bash
# Install Ollama (visit https://ollama.ai for installation instructions)
ollama pull llama2
ollama pull deepseek-coder
ollama pull llama3.1:8b
```

7. **Start the application**:
```bash
python start_server.py
```

### Option 2: Docker Compose (Full Stack)

1. **Clone the repository**:
```bash
git clone <repository-url>
cd tool-2025-kg-rag
```

2. **Configure your `.env` file** with API keys as shown above.

3. **Start all services**:
```bash
docker compose up --build
```

This will start:
- The KG-RAG application on http://localhost:8004
- Neo4j database on http://localhost:7474 (browser interface)
- All necessary services with proper networking and persistence

**Note**: The application runs on port 8004, not 8000 as mentioned in some older documentation.

### Docker Limitations

While Docker deployment is possible, the current setup has some limitations:

- **Frontend Build Issues**: The React frontend requires Node.js/npm which isn't included in the current Dockerfile
- **Version Warnings**: Docker Compose shows deprecation warnings for the `version` field
- **Network Configuration**: Requires changing `NEO4J_URI` from `localhost` to `neo4j` for container networking
- **Build Time**: Full container builds take significantly longer than local development
- **Debugging**: Harder to debug issues inside containers vs local development

For these reasons, the hybrid setup (local Python + Docker Neo4j) is recommended for development and testing.

### Option 3: Standalone Docker Container

For a single container deployment (without Neo4j):

```bash
# Build the image
docker build -t kg-rag-app .

# Run the container (requires separate Neo4j instance)
docker run -d \
  --name kg-rag-container \
  -p 8004:8004 \
  --env-file .env \
  kg-rag-app
```

## Running the Application

### Local Development
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8004
```

### Production
```bash
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8004
```

### Access the Application
- **Web Interface**: http://localhost:8004
- **API Documentation**: http://localhost:8004/docs
- **Neo4j Browser**: http://localhost:7474 (when using Docker Compose)

## Usage

1. **Upload a PDF file** through the web interface
2. **Select a KG provider and model** from the available options
3. **Process the PDF** to create a knowledge graph
4. **Visualize the knowledge graph** in the interactive viewer
5. **Query the knowledge graph** using the chat interface

## Configuration

### Model Providers
The application supports multiple AI providers:

- **OpenAI**: GPT-3.5, GPT-4 models
- **Anthropic**: Claude models
- **Google**: Gemini models
- **Hugging Face**: Various open-source models
- **DeepSeek**: DeepSeek models
- **Ollama**: Local models (Llama, CodeLlama, etc.)

### Ontology Support
The system includes biomedical ontology support for enhanced knowledge extraction:
<!-- - 15+ specialized biomedical entity types
- 19+ medically relevant relationship types
- Clinical property enrichment
- Evidence level classification -->

## Troubleshooting

### Common Issues

1. **Model not found errors**:
   ```bash
   # For Ollama models
   ollama pull <model_name>
   ollama serve  # Ensure Ollama is running
   ```

2. **Neo4j connection issues**:
   - Verify Neo4j is running and accessible
   - Check credentials in `.env` file
   - Ensure proper network connectivity

3. **Memory issues**:
   - The application implements memory management with limits
   - Increase Docker memory allocation if needed
   - Monitor system resources during processing

4. **API key issues**:
   - Verify all API keys in `.env` are valid and active
   - Check API quotas and rate limits
   - Ensure proper environment variable loading

### Docker Issues

1. **Complete system reset**:
   ```bash
   # Remove containers, networks, volumes, and images
   docker-compose down --rmi all --volumes --remove-orphans
   ```

2. **Build failures**:
   ```bash
   # Clean build
   docker-compose down --volumes
   docker-compose build --no-cache
   docker-compose up
   ```

2. **Permission issues**:
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER .
   ```

3. **Port conflicts**:
   - Ensure ports 8004 (app) and 7474 (Neo4j) are available
   - Modify port mappings in `docker-compose.yml` if needed

### Performance Optimization

- Use Docker Compose for production deployments
- Configure Neo4j memory settings based on available resources
- Monitor application logs for performance insights
- Consider using GPU acceleration for local models

## Development

### Project Structure
```
├── app.py                          # Main FastAPI application
├── ontology_guided_kg_creator.py   # Ontology-guided KG creation with LLM
├── enhanced_kg_creator_prod.py     # Production KG creator
├── enhanced_rag_system.py          # RAG system for KG queries
├── model_providers.py              # AI model provider configurations
├── start_server.py                 # Application startup script
├── requirements-prod.txt           # Production Python dependencies
├── requirements.txt                # Development dependencies
├── Dockerfile                      # Container configuration
├── docker-compose.yml              # Multi-service deployment
├── biomedical_ontology.owl         # Biomedical ontology definitions
├── ProstateCancerOntology.owl      # Prostate cancer specific ontology
├── llm-graph-builder/              # LLM Graph Builder submodule
│   ├── backend/                    # Backend services
│   └── frontend/                   # React frontend
├── shared/                         # Shared utilities
├── static/                         # Static web files
└── test/                           # Test files
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License
This project is licensed under the terms specified in the LICENSE file.
