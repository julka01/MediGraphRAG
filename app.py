import os
from dotenv import load_dotenv
load_dotenv()

import json
import re
import uuid
from io import BytesIO
from typing import Dict, Any, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import uvicorn
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings
from neo4j import GraphDatabase
from kg_loader import KGLoader

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory storage for knowledge graphs
knowledge_graphs: Dict[str, Dict] = {}
vector_stores: Dict[str, Any] = {}

# Create KGLoader instance
kg_loader = KGLoader()

# Model providers configuration
MODEL_PROVIDERS = {
    "openrouter": {
        "deepseek/deepseek-r1-0528:free": ChatOpenAI(
            model="deepseek/deepseek-r1-0528:free",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY")
        ),
        "openai/gpt-4": ChatOpenAI(
            model="gpt-4",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY")
        ),
        "anthropic/claude-3-opus": ChatOpenAI(
            model="anthropic/claude-3-opus",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY")
        )
    },
    "ollama": {
        "deepseek-r1-0528": Ollama(model="deepseek-r1-0528"),
        "llama3": Ollama(model="llama3"),
        "mistral": Ollama(model="mistral")
    },
    "openai": {
        "gpt-4": ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY")),
        "gpt-3.5-turbo": ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY")),
        "gpt-4-turbo": ChatOpenAI(model="gpt-4-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))
    },
    "anthropic": {
        "claude-3-opus": ChatOpenAI(
            model="claude-3-opus",
            openai_api_base="https://api.anthropic.com",
            openai_api_key=os.getenv("ANTHROPIC_API_KEY")
        ),
        "claude-3-sonnet": ChatOpenAI(
            model="claude-3-sonnet",
            openai_api_base="https://api.anthropic.com",
            openai_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    },
    "google": {
        "gemini-pro": ChatOpenAI(
            model="gemini-pro",
            openai_api_base="https://generativelanguage.googleapis.com/v1beta/models",
            openai_api_key=os.getenv("GOOGLE_API_KEY")
        )
    }
}

# Endpoint to get available models
@app.get("/models/{vendor}")
async def get_models(vendor: str):
    if vendor in MODEL_PROVIDERS:
        return {"models": list(MODEL_PROVIDERS[vendor].keys())}
    return {"models": []}

# Embeddings providers
EMBEDDINGS_PROVIDERS = {
    "ollama": OllamaEmbeddings(model="nomic-embed-text"),
    "openai": OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
    "openrouter": OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )
}

class Message(BaseModel):
    question: str
    provider_rag: str
    model_rag: str
    kg_id: Optional[str] = None

@app.post("/generate_kg")
async def generate_kg(file: bytes = File(...), provider_kg: str = Form("openrouter"), model_kg: str = Form("deepseek/deepseek-r1-0528:free")):
    try:
        if not file:
            raise ValueError("No file provided")
            
        if len(file) > 1024 * 1024 * 5:
            raise ValueError("File size exceeds 5MB limit")
            
        text = extract_text(file)
        kg = generate_knowledge_graph(text, provider_kg, model_kg)
        
        kg_id = f"kg_{str(uuid.uuid4())[:12]}"
        knowledge_graphs[kg_id] = {
            "graph": kg,
            "provider": provider_kg,
            "model": model_kg
        }
        
        create_vector_store(kg_id, text, provider_kg)
        
        return {
            "message": "Knowledge graph generated successfully",
            "kg_id": kg_id,
            "graph_data": kg
        }
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"KG generation error: {error_traceback}")
        raise HTTPException(
            status_code=400,
            detail=f"KG generation failed: {str(e)} - Full traceback: {error_traceback}"
        )

@app.post("/load_kg_from_file")
async def load_kg_from_file(
    file: UploadFile = File(...),
    provider: str = Form("openrouter"),
    model: str = Form("deepseek/deepseek-r1-0528:free")
):
    try:
        # Read file contents
        contents = await file.read()
        
        # Check file size
        if len(contents) > 1024 * 1024 * 5:
            raise ValueError("File size exceeds 5MB limit")
            
        # Extract text from file
        text = extract_text(contents)
        if not text:
            raise ValueError("Failed to extract text from file")
            
        # Generate the knowledge graph
        kg = generate_knowledge_graph(text, provider, model)
        
        # Create a unique ID for this KG
        kg_id = f"kg_{str(uuid.uuid4())[:12]}"
        knowledge_graphs[kg_id] = {
            "graph": kg,
            "provider": provider,
            "model": model
        }
        
        # Create vector store for RAG
        create_vector_store(kg_id, text, provider)
        
        return {
            "message": "Knowledge graph generated successfully",
            "kg_id": kg_id,
            "graph_data": kg
        }
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"KG creation error: {error_traceback}")
        raise HTTPException(
            status_code=400,
            detail=f"KG creation failed: {str(e)} - Full traceback: {error_traceback}"
        )

@app.post("/load_kg_from_neo4j")
async def load_kg_from_neo4j(
    uri: str = Form(...),
    user: str = Form(...),
    password: str = Form(...),
    query: str = Form("MATCH (n) RETURN n LIMIT 100")
):
    try:
        result = kg_loader.load_from_neo4j(uri, user, password, query)
        
        if result['status'] == 'error':
            raise ValueError(result['message'])
            
        kg_id = f"kg_{str(uuid.uuid4())[:12]}"
        knowledge_graphs[kg_id] = {
            "graph": result,
            "provider": "neo4j",
            "model": "database"
        }
        
        return {
            "message": "Knowledge graph loaded from Neo4j successfully",
            "kg_id": kg_id,
            "graph_data": result
        }
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Neo4j loading error: {error_traceback}")
        raise HTTPException(
            status_code=400,
            detail=f"Neo4j loading failed: {str(e)}"
        )

def extract_text(file_bytes: bytes) -> str:
    try:
        with BytesIO(file_bytes) as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception:
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return file_bytes.decode('latin-1')

def generate_knowledge_graph(text: str, provider: str, model: str) -> Dict[str, Any]:
    # Default to OpenRouter with deepseek model
    if provider == "openrouter":
        model_mapping = {
            "gpt-4": "openai/gpt-4",
            "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
            "claude-3-opus": "anthropic/claude-3-opus",
            "claude-3-sonnet": "anthropic/claude-3-sonnet",
            "claude-3-haiku": "anthropic/claude-3-haiku",
            "gemini-pro": "google/gemini-pro",
            "deepseek/deepseek-r1-0528:free": "deepseek/deepseek-r1-0528:free",
            "mistral-7b": "mistralai/mistral-7b-instruct",
            "llama-3-70b": "meta-llama/llama-3-70b-instruct"
        }
        model = model_mapping.get(model, model)
    elif provider == "ollama" and model == "deepseek-r1-0528":
        # Map ollama's deepseek model to openrouter equivalent
        provider = "openrouter"
        model = "deepseek/deepseek-r1-0528:free"
    
    if provider not in MODEL_PROVIDERS or model not in MODEL_PROVIDERS[provider]:
        raise ValueError(f"Model {model} not available for provider {provider}")
    
    llm = MODEL_PROVIDERS[provider][model]
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert knowledge graph extraction system. Analyze the following text and extract a detailed knowledge graph with entities and relationships. Follow these guidelines:

    1. Identify all important entities (people, organizations, concepts, locations, etc.)
    2. Extract relationships between entities
    3. For each entity, include:
        - id: unique numerical ID
        - label: entity type (e.g., Person, Organization, Concept)
        - properties: key-value pairs of attributes (min. 2 properties per entity)
    4. For each relationship, include:
        - from: source entity ID
        - to: target entity ID
        - type: relationship type (e.g., WORKS_FOR, LOCATED_IN)
        - properties: relationship attributes if available
    
    5. Use this JSON structure:
        {{
            "nodes": [
                {{"id": 1, "label": "Person", "properties": {{"name": "John Doe", "title": "CEO"}}}},
                {{"id": 2, "label": "Company", "properties": {{"name": "Acme Inc", "industry": "Technology"}}}}
            ],
            "relationships": [
                {{"from": 1, "to": 2, "type": "WORKS_FOR", "properties": {{"role": "CEO", "since": "2020"}}}}
            ]
        }}
    
    6. Ensure all IDs are unique and relationships reference existing node IDs.
    7. Include at least 5 nodes and 3 relationships unless the text is very short.
    8. Return ONLY valid JSON - no additional text or explanations.

    Text:
    {text}
    """)
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": text[:3000]})
    
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        json_match = re.search(r'\{[\s\S]*\}', result)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                print(f"Extracted JSON parse error: {str(e)}")
                print(f"Extracted content: {json_match.group()[:200]}")
        raise ValueError("Failed to parse JSON from model response")

def create_vector_store(kg_id: str, text: str, provider: str):
    if provider not in EMBEDDINGS_PROVIDERS:
        raise ValueError(f"Embeddings provider {provider} not available")
    
    embeddings = EMBEDDINGS_PROVIDERS[provider]
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    
    try:
        vector_store = Chroma.from_texts(
            chunks, 
            embedding=embeddings,
            collection_name=f"kg_{kg_id}"
        )
        vector_stores[kg_id] = vector_store
    except Exception as e:
        print(f"Chroma vector store creation failed: {str(e)}")
        vector_stores[kg_id] = {"chunks": chunks, "embeddings": embeddings}

def get_graph_context(kg_id: str) -> str:
    if kg_id not in knowledge_graphs:
        return ""
    
    graph = knowledge_graphs[kg_id]["graph"]
    context_lines = []
    
    # Add graph summary
    num_nodes = len(graph.get("nodes", []))
    num_relationships = len(graph.get("relationships", []))
    context_lines.append(f"Knowledge Graph Summary: {num_nodes} nodes, {num_relationships} relationships")
    
    # Add node information
    context_lines.append("\nNodes:")
    for node in graph.get("nodes", []):
        props = ", ".join([f"{k}: {v}" for k, v in node.get("properties", {}).items()])
        context_lines.append(f"- {node.get('label', 'Entity')} (ID: {node['id']}): {props}")
    
    # Add relationship information
    context_lines.append("\nRelationships:")
    for rel in graph.get("relationships", []):
        source_node = next((n for n in graph["nodes"] if n["id"] == rel["from"]), None)
        target_node = next((n for n in graph["nodes"] if n["id"] == rel["to"]), None)
        source_label = source_node.get("label", "Entity") if source_node else "Unknown"
        target_label = target_node.get("label", "Entity") if target_node else "Unknown"
        props = ", ".join([f"{k}: {v}" for k, v in rel.get("properties", {}).items()])
        context_lines.append(f"- {source_label} ({rel['from']}) --[{rel.get('type', 'related')}]-> {target_label} ({rel['to']}): {props}")
    
    return "\n".join(context_lines)

@app.post("/chat")
async def chat(message: Message):
    try:
        if message.provider_rag not in MODEL_PROVIDERS or message.model_rag not in MODEL_PROVIDERS[message.provider_rag]:
            raise ValueError(f"Model {message.model_rag} not available for provider {message.provider_rag}")
        
        llm = MODEL_PROVIDERS[message.provider_rag][message.model_rag]
        context = ""
        if message.kg_id:
            context = get_graph_context(message.kg_id)
            if not context:
                context = "No knowledge graph context available"
        
        if message.kg_id and context:
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant answering questions based on a knowledge graph. "
                "Use the following graph context to provide a concise, structured response.\n\n"
                "Graph Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Provide your answer in this structured format:\n"
                "1. Summary: [concise summary of answer]\n"
                "2. Key Points:\n"
                "   - [point 1]\n"
                "   - [point 2]\n"
                "   ...\n"
                "3. Source: [graph element IDs if applicable]"
            )
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({
                "context": context[:10000],
                "question": message.question
            })
        else:
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the question concisely.\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({"question": message.question})
        
        return {"response": response}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/graph/{kg_id}")
async def get_graph(kg_id: str):
    try:
        if kg_id in knowledge_graphs:
            return knowledge_graphs[kg_id]["graph"]
        else:
            raise HTTPException(
                status_code=404,
                detail="Knowledge graph not found"
            )
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
