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

### Option 1: Local Development Setup

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
pip install -r requirements.txt
```

4. **Create environment configuration**:
```bash
cp .env.example .env  # Copy the example configuration file
```

5. **Configure your `.env` file**:
```env
# AI Provider API Keys
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GEMINI_API_KEY=your-gemini-api-key
HF_API_TOKEN=your-huggingface-api-token
DEEPSEEK_API_KEY=your-deepseek-api-key

# Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password

# Optional: Ollama Configuration
OLLAMA_HOST=http://localhost:11434
```

6. **Set up Neo4j database** (if running locally):
   - Install Neo4j Desktop or use Docker
   - Create a database with the credentials from your `.env` file

7. **For Ollama (local models)**, install and download models on your host machine:
```bash
# Install Ollama (visit https://ollama.ai for installation instructions)
ollama pull llama2
ollama pull deepseek-coder
ollama pull llama3.1:8b
```

### Option 2: Docker Compose (Recommended)

1. **Clone the repository**:
```bash
git clone <repository-url>
cd tool-2025-kg-rag
```

2. **Create environment configuration**:
```bash
cp .env.example .env  # Copy the example configuration file
```

3. **Configure your `.env` file**:
```env
# AI Provider API Keys
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GEMINI_API_KEY=your-gemini-api-key
HF_API_TOKEN=your-huggingface-api-token
DEEPSEEK_API_KEY=your-deepseek-api-key

# Neo4j Password (required for Docker setup)
NEO4J_PASSWORD=your-secure-password
```

4. **Start the application**:
```bash
docker-compose up --build
```

This will start:
- The KG-RAG application on http://localhost:8000
- Neo4j database on http://localhost:7474 (browser interface)
- All necessary services with proper networking and persistence

### Option 3: Standalone Docker Container

For a single container deployment (without Neo4j):

```bash
# Build the image
docker build -t kg-rag-app .

# Run the container
docker run -d \
  --name kg-rag-container \
  -p 8000:8000 \
  --env-file .env \
  kg-rag-app
```

## Running the Application

### Local Development
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Access the Application
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
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
- 15+ specialized biomedical entity types
- 19+ medically relevant relationship types
- Clinical property enrichment
- Evidence level classification

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
   - Ensure ports 8000 and 7474 are available
   - Modify port mappings in `docker-compose.yml` if needed

### Performance Optimization

- Use Docker Compose for production deployments
- Configure Neo4j memory settings based on available resources
- Monitor application logs for performance insights
- Consider using GPU acceleration for local models

## Development

### Project Structure
```
├── app.py                 # Main FastAPI application
├── improved_kg_creator.py # Enhanced KG creation logic
├── kg_loader.py          # Knowledge graph loading utilities
├── model_providers.py    # AI model provider configurations
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Multi-service deployment
├── biomedical_ontology.owl # Biomedical ontology definitions
└── static/              # Web interface files
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License
This project is licensed under the terms specified in the LICENSE file.
