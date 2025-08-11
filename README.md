# Knowledge Graph RAG System

This application creates knowledge graphs from PDF documents and provides a chat interface for querying the knowledge graph using various AI models.

## Features
- Support for multiple AI providers: OpenAI, Ollama (local), Gemini, Hugging Face, DeepSeek
- PDF processing to extract text and create knowledge graphs
- Interactive visualization of knowledge graphs
- RAG-based chat interface using the knowledge graph

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your-openai-api-key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password
GEMINI_API_KEY=your-gemini-api-key
HF_API_TOKEN=your-huggingface-api-token
DEEPSEEK_API_KEY=your-deepseek-api-key
```

3. For Ollama (local models), download the models:
```bash
ollama pull llama2
ollama pull deepseek-coder
ollama pull llama3.1:8b
```

4. Start Neo4j database (if not already running)

## Running the Application
```bash
uvicorn app:app --reload
```

Open http://localhost:8000 in your browser.

## Usage
1. Upload a PDF file
2. Select a KG provider and model
3. Process the PDF to create a knowledge graph
4. Query the knowledge graph using the chat interface

## Troubleshooting
- If you get "model not found" for Ollama, run `ollama pull <model_name>`
- For local Ollama models, make sure to run `ollama serve` in a separate terminal
- Ensure all API keys in `.env` are valid

## Author
**Sahib Julka**  
Email: sahib.julka@med.lmu.de  
GitLab: [@sahibjulka](https://gitlab.lrz.de/sahibjulka)
