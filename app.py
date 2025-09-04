import os
from dotenv import load_dotenv
load_dotenv()

import json
import re
import uuid
from io import BytesIO
from typing import Dict, Any, Optional, List
import owlready2  # Add owlready2 for OWL parsing
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import uvicorn
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
# Import from langchain-ollama package to fix deprecation warnings
try:
    from langchain_ollama import OllamaLLM, OllamaEmbeddings
except ImportError:
    # Fallback to deprecated imports if new package not available
    from langchain_community.llms import Ollama as OllamaLLM
    from langchain_community.embeddings import OllamaEmbeddings as LegacyOllamaEmbeddings
    OllamaEmbeddings = LegacyOllamaEmbeddings
from neo4j import GraphDatabase
from kg_loader import KGLoader
from improved_kg_creator import ImprovedKGCreator

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

# In-memory storage for knowledge graphs with size limits
knowledge_graphs: Dict[str, Dict] = {}
vector_stores: Dict[str, Any] = {}

# Memory management constants
MAX_STORED_KGS = 50  # Maximum number of KGs to keep in memory
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size
MAX_TEXT_LENGTH = 50000  # Maximum text length for processing

def cleanup_old_kgs():
    """Remove oldest KGs if we exceed the limit"""
    if len(knowledge_graphs) > MAX_STORED_KGS:
        # Sort by creation time (assuming kg_id contains timestamp info)
        sorted_kgs = sorted(knowledge_graphs.items())
        # Remove oldest entries
        for kg_id, _ in sorted_kgs[:len(knowledge_graphs) - MAX_STORED_KGS]:
            del knowledge_graphs[kg_id]
            if kg_id in vector_stores:
                del vector_stores[kg_id]
            print(f"Cleaned up old KG: {kg_id}")

def validate_input_text(text: str) -> str:
    """Validate and sanitize input text"""
    if not text or not text.strip():
        raise ValueError("Empty text provided")
    
    if len(text) > MAX_TEXT_LENGTH:
        print(f"Text truncated from {len(text)} to {MAX_TEXT_LENGTH} characters")
        text = text[:MAX_TEXT_LENGTH]
    
    # Basic sanitization - remove potential harmful content
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    
    return text.strip()

def validate_file_size(file_bytes: bytes) -> None:
    """Validate file size"""
    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError(f"File size {len(file_bytes)} exceeds maximum allowed size of {MAX_FILE_SIZE} bytes")

# Create KGLoader and ImprovedKGCreator instances
kg_loader = KGLoader()

# Default Neo4j credentials for UI
DEFAULT_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
improved_kg_creator = ImprovedKGCreator()

def get_model_providers():
    """Initialize model providers with dynamic environment variable loading"""
    providers = {
        "openrouter": {
            "deepseek/deepseek-r1-0528:free": ChatOpenAI(
                model="deepseek/deepseek-r1-0528:free",
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=os.getenv("OPENROUTER_API_KEY")
            ) if os.getenv("OPENROUTER_API_KEY") else None,
            "openai/gpt-4": ChatOpenAI(
                model="gpt-4",
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=os.getenv("OPENROUTER_API_KEY")
            ) if os.getenv("OPENROUTER_API_KEY") else None,
            "anthropic/claude-3-opus": ChatOpenAI(
                model="anthropic/claude-3-opus",
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=os.getenv("OPENROUTER_API_KEY")
            ) if os.getenv("OPENROUTER_API_KEY") else None,
            "meta-llama/llama-4-maverick:free": ChatOpenAI(
                model="meta-llama/llama-4-maverick:free",
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=os.getenv("OPENROUTER_API_KEY")
            ) if os.getenv("OPENROUTER_API_KEY") else None,
        },
        "lmu_lightllm": {
            "eta/llama-3.3-70b-instruct": ChatOpenAI(
                model="eta/llama-3.3-70b-instruct",
                openai_api_base="https://litellm.info.med.uni-muenchen.de",
                openai_api_key=os.getenv("LMU_LIGHTLLM_API_KEY")
            ) if os.getenv("LMU_LIGHTLLM_API_KEY") else None,
            "meta/llama3.3-instruct:70b": ChatOpenAI(
                model="meta/llama3.3-instruct:70b",
                openai_api_base="https://litellm.info.med.uni-muenchen.de",
                openai_api_key=os.getenv("LMU_LIGHTLLM_API_KEY")
            ) if os.getenv("LMU_LIGHTLLM_API_KEY") else None,
            "meta/llama-3.3-70b-instruct": ChatOpenAI(
                model="meta/llama-3.3-70b-instruct",
                openai_api_base="https://litellm.info.med.uni-muenchen.de",
                openai_api_key=os.getenv("LMU_LIGHTLLM_API_KEY")
            ) if os.getenv("LMU_LIGHTLLM_API_KEY") else None,
            "bge-m3:567m": ChatOpenAI(
                model="bge-m3:567m",
                openai_api_base="https://litellm.info.med.uni-muenchen.de",
                openai_api_key=os.getenv("LMU_LIGHTLLM_API_KEY")
            ) if os.getenv("LMU_LIGHTLLM_API_KEY") else None,
            "deepseek-r1:32b-qwen-distill-q8_0": ChatOpenAI(
                model="deepseek-r1:32b-qwen-distill-q8_0",
                openai_api_base="https://litellm.info.med.uni-muenchen.de",
                openai_api_key=os.getenv("LMU_LIGHTLLM_API_KEY")
            ) if os.getenv("LMU_LIGHTLLM_API_KEY") else None,
            "deepseek-r1:70b": ChatOpenAI(
                model="deepseek-r1:70b",
                openai_api_base="https://litellm.info.med.uni-muenchen.de",
                openai_api_key=os.getenv("LMU_LIGHTLLM_API_KEY")
            ) if os.getenv("LMU_LIGHTLLM_API_KEY") else None,
            "linux6200/bge-reranker-v2-m3:latest": ChatOpenAI(
                model="linux6200/bge-reranker-v2-m3:latest",
                openai_api_base="https://litellm.info.med.uni-muenchen.de",
                openai_api_key=os.getenv("LMU_LIGHTLLM_API_KEY")
            ) if os.getenv("LMU_LIGHTLLM_API_KEY") else None,
            "llama3.2-vision:90b-instruct-q4_K_M": ChatOpenAI(
                model="llama3.2-vision:90b-instruct-q4_K_M",
                openai_api_base="https://litellm.info.med.uni-muenchen.de",
                openai_api_key=os.getenv("LMU_LIGHTLLM_API_KEY")
            ) if os.getenv("LMU_LIGHTLLM_API_KEY") else None,
            "nomic-embed-text:latest": ChatOpenAI(
                model="nomic-embed-text:latest",
                openai_api_base="https://litellm.info.med.uni-muenchen.de",
                openai_api_key=os.getenv("LMU_LIGHTLLM_API_KEY")
            ) if os.getenv("LMU_LIGHTLLM_API_KEY") else None,
            "phi4-mini:3.8b": ChatOpenAI(
                model="phi4-mini:3.8b",
                openai_api_base="https://litellm.info.med.uni-muenchen.de",
                openai_api_key=os.getenv("LMU_LIGHTLLM_API_KEY")
            ) if os.getenv("LMU_LIGHTLLM_API_KEY") else None,
            "qwen3:32b": ChatOpenAI(
                model="qwen3:32b",
                openai_api_base="https://litellm.info.med.uni-muenchen.de",
                openai_api_key=os.getenv("LMU_LIGHTLLM_API_KEY")
            ) if os.getenv("LMU_LIGHTLLM_API_KEY") else None,
        },
        "ollama": {
            "deepseek-r1-0528": OllamaLLM(model="deepseek-r1-0528"),
            "llama3": OllamaLLM(model="llama3"),
            "mistral": OllamaLLM(model="mistral")
        },
        "openai": {
            "gpt-4": ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None,
            "gpt-3.5-turbo": ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None,
            "gpt-4-turbo": ChatOpenAI(model="gpt-4-turbo", openai_api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
        },
        "anthropic": {
            "claude-3-opus": ChatOpenAI(
                model="claude-3-opus",
                openai_api_base="https://api.anthropic.com",
                openai_api_key=os.getenv("ANTHROPIC_API_KEY")
            ) if os.getenv("ANTHROPIC_API_KEY") else None,
            "claude-3-sonnet": ChatOpenAI(
                model="claude-3-sonnet",
                openai_api_base="https://api.anthropic.com",
                openai_api_key=os.getenv("ANTHROPIC_API_KEY")
            ) if os.getenv("ANTHROPIC_API_KEY") else None
        },
        "google": {
            "gemini-pro": ChatOpenAI(
                model="gemini-pro",
                openai_api_base="https://generativelanguage.googleapis.com/v1beta/models",
                openai_api_key=os.getenv("GOOGLE_API_KEY")
            ) if os.getenv("GOOGLE_API_KEY") else None
        }
    }
    
    # Remove any providers that have no valid models
    for provider, models in list(providers.items()):
        providers[provider] = {k: v for k, v in models.items() if v is not None}
        if not providers[provider]:
            del providers[provider]
            
    print("Loaded MODEL_PROVIDERS:", json.dumps(list(providers.keys()), indent=2))
    print(f"OpenRouter API key present: {bool(os.getenv('OPENROUTER_API_KEY'))}")
    return providers

MODEL_PROVIDERS = get_model_providers()
# Endpoint to get available models
@app.get("/models/{vendor}")
async def get_models(vendor: str):
    if vendor == "ollama":
        # Use Ollama's list command to get local models
        try:
            import subprocess
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            print("Ollama list output:", result.stdout)  # Debug print
            
            local_models = []
            for line in result.stdout.split('\n')[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if parts:
                        model_name = parts[0]
                        local_models.append(model_name)
            
            print("Local models found:", local_models)  # Debug print
            print("Available Ollama models in MODEL_PROVIDERS:", list(MODEL_PROVIDERS.get("ollama", {}).keys()))  # Debug print
            
            # More flexible matching: check if any part of the local model name matches a provider model
            available_models = []
            for local_model in local_models:
                for provider_model in MODEL_PROVIDERS.get("ollama", {}):
                    # Case-insensitive partial match
                    if local_model.lower() in provider_model.lower() or provider_model.lower() in local_model.lower():
                        available_models.append(provider_model)
                        break
            
            print("Matched models:", available_models)  # Debug print
            
            # Fallback to all Ollama models if no matches found
            if not available_models:
                available_models = list(MODEL_PROVIDERS.get("ollama", {}).keys())
            
            return {"models": available_models}
        except Exception as e:
            import traceback
            print(f"Error fetching Ollama models: {e}")
            traceback.print_exc()
            return {"models": list(MODEL_PROVIDERS.get("ollama", {}).keys())}
    elif vendor in MODEL_PROVIDERS:
        # Return original model names without modification
        models = list(MODEL_PROVIDERS[vendor].keys())
        return {"models": models}
    return {"models": []}


# Embeddings providers
from langchain_openai import OpenAIEmbeddings

EMBEDDINGS_PROVIDERS = {
    "ollama": OllamaEmbeddings(model="nomic-embed-text"),
    "openai": OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None,
    "openrouter": OllamaEmbeddings(model="nomic-embed-text")
}

class Message(BaseModel):
    question: str
    provider_rag: str
    model_rag: Optional[str] = None
    kg_id: Optional[str] = None

class ExportToFileRequest(BaseModel):
    kg_id: str
    folder_path: str
    filename: str
    base_path: Optional[str] = None

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
    model: str = Form("deepseek/deepseek-r1-0528:free"),
    ontology: UploadFile = File(None)
):
    try:
        # Read file contents
        contents = await file.read()
        
        # Validate file size
        validate_file_size(contents)
            
        # Extract text from file
        text = extract_text(contents)
        if not text:
            raise ValueError("Failed to extract text from file")
            
        # Validate and sanitize text
        text = validate_input_text(text)
            
        # Read and parse ontology if provided
        ontology_content = None
        if ontology:
            ontology_bytes = await ontology.read()
            if ontology_bytes:
                filename = ontology.filename.lower()
                if filename.endswith('.json'):
                    try:
                        ontology_content = json.loads(ontology_bytes.decode('utf-8'))
                    except json.JSONDecodeError:
                        raise ValueError("Invalid JSON ontology file format.")
                elif filename.endswith('.owl'):
                    try:
                        ontology_content = parse_owl_ontology(ontology_bytes)
                    except Exception as e:
                        raise ValueError(f"Failed to parse OWL ontology: {str(e)}")
                else:
                    raise ValueError("Unsupported ontology file format. Only JSON and OWL are supported.")
        
        # Generate the knowledge graph
        kg = generate_knowledge_graph(text, provider, model, ontology=ontology_content)
        
        # Create a unique ID for this KG
        kg_id = f"kg_{str(uuid.uuid4())[:12]}"
        knowledge_graphs[kg_id] = {
            "graph": kg,
            "provider": provider,
            "model": model
        }
        
        # Create vector store for RAG
        create_vector_store(kg_id, text, provider)
        
        # Clean up old KGs if needed
        cleanup_old_kgs()
        
        # Print confirmation that KG was stored with kg_id
        print(f"KG stored successfully with ID: {kg_id}")
        
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
        
def parse_owl_ontology(owl_bytes: bytes) -> dict:
    """Enhanced OWL ontology parsing with improved error handling and extraction"""
    try:
        # Create a BytesIO object from the bytes
        file_obj = BytesIO(owl_bytes)
        
        # Create a new world to avoid conflicts with existing ontologies
        world = owlready2.World()
        
        # Load the ontology directly from bytes with enhanced error handling
        onto = world.get_ontology("")
        
        try:
            onto.load(fileobj=file_obj)
        except Exception as e:
            print(f"Error loading OWL ontology: {str(e)}")
            # Fallback to manual parsing if owlready2 fails
            return parse_owl_manually(owl_bytes)
        
        # Extract classes, object properties, and data properties
        node_labels = set()
        relationship_types = set()
        
        for cls in onto.classes():
            if cls.name:
                node_labels.add(cls.name)
            # Extract labels and comments for additional context
            for label in cls.label:
                node_labels.add(str(label))
            for comment in cls.comment:
                node_labels.add(str(comment))
        
        for prop in onto.object_properties():
            if prop.name:
                relationship_types.add(prop.name)
            for label in prop.label:
                relationship_types.add(str(label))
        
        for prop in onto.data_properties():
            if prop.name:
                relationship_types.add(prop.name)
            for label in prop.label:
                relationship_types.add(str(label))
        
        return {
            "node_labels": list(node_labels),
            "relationship_types": list(relationship_types)
        }
        
    except Exception as e:
        print(f"Error parsing OWL ontology: {str(e)}")
        # Fallback to manual parsing if owlready2 fails
        return parse_owl_manually(owl_bytes)

def parse_owl_manually(owl_bytes: bytes) -> dict:
    """Manually parse OWL file using XML parsing as fallback"""
    try:
        import xml.etree.ElementTree as ET
        
        # Parse the XML content
        root = ET.fromstring(owl_bytes.decode('utf-8'))
        
        # Define namespaces
        namespaces = {
            'owl': 'http://www.w3.org/2002/07/owl#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
        }
        
        # Extract classes
        node_labels = []
        classes = root.findall('.//owl:Class', namespaces)
        for cls in classes:
            # Try to get the label first
            label_elem = cls.find('.//rdfs:label', namespaces)
            if label_elem is not None and label_elem.text:
                node_labels.append(label_elem.text)
            else:
                # Fallback to extracting from rdf:about attribute
                about = cls.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
                if about:
                    # Extract the local name (part after # or /)
                    local_name = about.split('#')[-1] if '#' in about else about.split('/')[-1]
                    if local_name and local_name not in node_labels:
                        node_labels.append(local_name)
        
        # Extract object properties
        relationship_types = []
        properties = root.findall('.//owl:ObjectProperty', namespaces)
        for prop in properties:
            # Try to get the label first
            label_elem = prop.find('.//rdfs:label', namespaces)
            if label_elem is not None and label_elem.text:
                relationship_types.append(label_elem.text)
            else:
                # Fallback to extracting from rdf:about attribute
                about = prop.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
                if about:
                    # Extract the local name (part after # or /)
                    local_name = about.split('#')[-1] if '#' in about else about.split('/')[-1]
                    if local_name and local_name not in relationship_types:
                        relationship_types.append(local_name)
        
        # Remove duplicates and filter out empty strings
        node_labels = list(set([label for label in node_labels if label and label.strip()]))
        relationship_types = list(set([rel for rel in relationship_types if rel and rel.strip()]))
        
        print(f"Manually parsed OWL: {len(node_labels)} classes, {len(relationship_types)} properties")
        
        return {
            "node_labels": node_labels,
            "relationship_types": relationship_types
        }
        
    except Exception as e:
        print(f"Manual OWL parsing also failed: {str(e)}")
        # Return a minimal fallback structure
        return {
            "node_labels": ["Entity", "Concept", "Resource"],
            "relationship_types": ["RELATED_TO", "ASSOCIATED_WITH", "CONNECTED_TO"]
        }

@app.post("/test_owl_parser")
async def test_owl_parser(file: UploadFile = File(...)):
    """
    Test endpoint for OWL parser functionality
    Accepts an OWL file and returns the parsed node labels and relationship types
    """
    try:
        contents = await file.read()
        result = parse_owl_ontology(contents)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"OWL parsing failed: {str(e)}")

@app.post("/list_neo4j_labels")
async def list_neo4j_labels(
    uri: str = Form(...),
    user: str = Form(...),
    password: str = Form(...)
):
    try:
        result = kg_loader.list_kg_labels(uri, user, password)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
        
@app.post("/load_kg_from_neo4j")
async def load_kg_from_neo4j(
    uri: str = Form(...),
    user: str = Form(...),
    password: str = Form(...),
    kg_label: str = Form(None),
    query: str = Form(None),
    limit: int = Form(None),
    sample_mode: bool = Form(False),
    load_complete: bool = Form(False)
):
    try:
        # Determine loading parameters based on user selection
        if load_complete:
            # Load complete graph without limits
            result = kg_loader.load_from_neo4j(uri, user, password, kg_label, query, limit=None, sample_mode=False)
        elif sample_mode:
            # Load representative sample for large graphs
            result = kg_loader.load_from_neo4j(uri, user, password, kg_label, query, limit=limit, sample_mode=True)
        else:
            # Default behavior with optional limit
            if limit is None:
                limit = 1000  # Default reasonable limit
            result = kg_loader.load_from_neo4j(uri, user, password, kg_label, query, limit=limit, sample_mode=False)
        
        if result['status'] == 'error':
            raise ValueError(result['message'])
            
        kg_id = f"kg_{str(uuid.uuid4())[:12]}"
        knowledge_graphs[kg_id] = {
            "graph": result,
            "provider": "neo4j",
            "model": "database"
        }
        
        # Create vector store for RAG if we have meaningful content
        try:
            # Extract text content from nodes for vector store
            text_content = []
            for node in result.get('nodes', []):
                props = node.get('properties', {})
                for key, value in props.items():
                    if isinstance(value, str) and len(value) > 10:
                        text_content.append(f"{key}: {value}")
            
            combined_text = " ".join(text_content)
            if combined_text.strip():
                create_vector_store(kg_id, combined_text, "ollama")
        except Exception as e:
            print(f"Warning: Could not create vector store for Neo4j KG: {str(e)}")
        
        return {
            "message": f"Knowledge graph loaded from Neo4j successfully. Loaded {result.get('loaded_nodes', 0)} nodes and {result.get('loaded_relationships', 0)} relationships from {result.get('total_nodes_in_db', 0)} total nodes in database.",
            "kg_id": kg_id,
            "graph_data": result,
            "stats": {
                "loaded_nodes": result.get('loaded_nodes', 0),
                "loaded_relationships": result.get('loaded_relationships', 0),
                "total_nodes_in_db": result.get('total_nodes_in_db', 0),
                "total_relationships_in_db": result.get('total_relationships_in_db', 0),
                "complete_import": result.get('complete_import', False),
                "sample_mode": result.get('sample_mode', False)
            }
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

def generate_knowledge_graph(text: str, provider: str, model: str, ontology: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Generate knowledge graph using the improved KG creator with enhanced ontology integration
    """
    # Default to OpenRouter with deepseek model
    if provider == "openrouter":
        model_mapping = {
            "gpt-4": "openai/gpt-4",
            "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
            "claude-3-opus": "anthropic/claude-3-opus",
            "claude-3-sonnet": "anthropic/claude-3-sonnet",
            "claude-3-haiku": "anthropic/claude-3-haiku",
            "gemini-pro": "google/gemini-pro",
            "deepseek-deepseek-r1-0528-free": "deepseek/deepseek-r1-0528:free",
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
    
    # Use the improved KG creator with temperature fixed at 0 for deterministic RAG chat
    try:
        # If the llm supports temperature, set it to 0
        if hasattr(llm, "temperature"):
            llm.temperature = 0
        
        return improved_kg_creator.generate_knowledge_graph(
            text=text,
            llm=llm,
            ontology=ontology
        )
    except Exception as e:
        print(f"Improved KG creator failed: {e}")
        # Fallback to basic creation if improved creator fails
        return _fallback_kg_generation(text, llm, ontology)

def _fallback_kg_generation(text: str, llm, ontology: Optional[Dict] = None) -> Dict[str, Any]:
    """Fallback KG generation method"""
    ontology_instructions = ""
    if ontology:
        ontology_instructions = f"""
    Strictly use ONLY the node labels and relationship types from the ontology below. 
    If no suitable node label exists for an entity, assign it the label "Custom" and provide a descriptive name in properties["name"].
    If no suitable relationship type exists, omit that relationship.
    Always prefer the exact label and type string from the ontology — do not rephrase or invent.
    Ontology:
    {json.dumps(ontology, indent=2)}
    """
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert biomedical knowledge graph extraction system. Analyze the following clinical text and extract a detailed knowledge graph with medically relevant entities. 
    {ontology_instructions}
    Follow these guidelines:

    1. Identify all medically relevant entities from the text (diseases, stages, treatments, procedures, biomarkers, clinical guidelines, organisations, and other biomedical concepts)
    2. For every extracted entity:
        - Set "label" as the exact ontology label if matched; otherwise "Custom"
        - If matched, include "ontology_id" from the ontology in properties
    3. Extract relationships between entities
    4. For each entity, include:
        - id: unique numerical ID
        - properties: key-value pairs of attributes (min. 2 properties per entity). 
          If available in the text, include:
            * evidence_level
            * guideline_section
            * publication_year
    5. For each relationship, include:
        - from: source entity ID
        - to: target entity ID
        - type: relationship type (e.g., TREATS, DIAGNOSES, ASSOCIATED_WITH)
        - properties: relationship attributes if available
    
    6. Use this JSON structure:
        {{
            "nodes": [
                {{"id": 1, "label": "Disease", "properties": {{"name": "Prostate Cancer", "stage": "T2c", "evidence_level": "1a", "ontology_id": "ONT:0001"}}}},
                {{"id": 2, "label": "Treatment", "properties": {{"name": "Radical Prostatectomy", "guideline_section": "5.2.1"}}}}
            ],
            "relationships": [
                {{"from": 2, "to": 1, "type": "TREATS", "properties": {{"efficacy": "High"}}}}
            ]
        }}
    
    7. Ensure all IDs are unique and relationships reference existing node IDs.
    8. Include at least 5 nodes and 3 relationships unless the text is very short.
    9. Return ONLY valid JSON - no additional text or explanations.

    Text:
    {text}
    """)
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "text": text[:3000],
        "ontology_instructions": ontology_instructions
    })
    
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        # First clean trailing commas that might cause parsing issues
        cleaned = re.sub(r',(\s*[}\]])', r'\1', result)
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                print(f"Extracted JSON parse error: {str(e)}")
                print(f"Extracted content: {json_match.group()[:200]}")
        raise ValueError("Failed to parse JSON from model response")

def create_vector_store(kg_id: str, text: str, provider: str):
    try:
        print(f"Creating vector store with provider: {provider}")
        
        # Map provider or model string to embeddings provider key
        provider_key_map = {
            "openrouter": "ollama",
            "ollama": "ollama",
            "openai": "openai",
            # Add possible model strings mapping to provider keys
            "deepseek/deepseek-r1-0528:free": "openrouter",
            "gpt-4": "openai",
            "gpt-3.5-turbo": "openai",
            "meta-llama/llama-4-maverick:free": "openrouter",
            "nomic-embed-text": "ollama"
        }
        
        embeddings_provider_key = provider_key_map.get(provider)
        if not embeddings_provider_key:
            raise ValueError(f"Embeddings provider for '{provider}' not found in mapping")
        
        if embeddings_provider_key not in EMBEDDINGS_PROVIDERS:
            raise ValueError(f"Embeddings provider '{embeddings_provider_key}' not available")
        
        embeddings = EMBEDDINGS_PROVIDERS[embeddings_provider_key]
        print(f"Embeddings object type: {type(embeddings)}")
        
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        
        # Use persistent Chroma client
        chroma_db_path = os.path.join(os.getcwd(), "chroma_db")
        os.makedirs(chroma_db_path, exist_ok=True)
        
        vector_store = Chroma.from_texts(
            chunks, 
            embedding=embeddings,
            collection_name=f"kg_{kg_id}",
            persist_directory=chroma_db_path
        )
        vector_store.persist()
        
        vector_stores[kg_id] = vector_store
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Chroma vector store creation failed: {str(e)}\n{error_traceback}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create vector store: {str(e)} - Full traceback: {error_traceback}"
        )

def get_graph_context(kg_id: str) -> str:
    if kg_id not in knowledge_graphs:
        return ""
    
    graph = knowledge_graphs[kg_id]["graph"]
    context_lines = []
    
    # Add graph summary
    num_nodes = len(graph.get("nodes", []))
    num_relationships = len(graph.get("relationships", []))
    context_lines.append(f"Knowledge Graph Summary: {num_nodes} nodes, {num_relationships} relationships")
    
    # Add node information with enhanced clinical context
    context_lines.append("\nClinical Entities:")
    for node in graph.get("nodes", []):
        props = node.get("properties", {})
        label = node.get('label', 'Entity')
        name = props.get('name', 'Unknown')
        
        # Create structured entity description
        entity_desc = f"- {label}: {name} (ID: {node['id']})"
        
        # Add clinical significance if available
        if 'clinical_significance' in props:
            entity_desc += f" [Significance: {props['clinical_significance']}]"
        if 'evidence_level' in props:
            entity_desc += f" [Evidence: {props['evidence_level']}]"
        if 'stage' in props:
            entity_desc += f" [Stage: {props['stage']}]"
        if 'severity' in props:
            entity_desc += f" [Severity: {props['severity']}]"
        
        # Add other relevant properties
        other_props = {k: v for k, v in props.items() 
                      if k not in ['name', 'clinical_significance', 'evidence_level', 'stage', 'severity']}
        if other_props:
            props_str = ", ".join([f"{k}: {v}" for k, v in other_props.items()])
            entity_desc += f" | {props_str}"
        
        context_lines.append(entity_desc)
    
    # Add relationship information with clinical context
    context_lines.append("\nClinical Relationships:")
    for rel in graph.get("relationships", []):
        source_node = next((n for n in graph["nodes"] if n["id"] == rel["from"]), None)
        target_node = next((n for n in graph["nodes"] if n["id"] == rel["to"]), None)
        
        if source_node and target_node:
            source_name = source_node.get("properties", {}).get("name", "Unknown")
            target_name = target_node.get("properties", {}).get("name", "Unknown")
            source_label = source_node.get("label", "Entity")
            target_label = target_node.get("label", "Entity")
            source_id = source_node.get("id")
            target_id = target_node.get("id")
            rel_type = rel.get('type', 'RELATED_TO')
            
            rel_desc = f"- {source_label}({source_name})[ID:{source_id}] --[{rel_type}]-> {target_label}({target_name})[ID:{target_id}]"
            
            # Add relationship properties
            rel_props = rel.get("properties", {})
            if rel_props:
                props_str = ", ".join([f"{k}: {v}" for k, v in rel_props.items()])
                rel_desc += f" | {props_str}"
            
            context_lines.append(rel_desc)
    
    return "\n".join(context_lines)

def get_ontology_context(kg_id: str) -> Dict[str, Any]:
    """Get ontology information for the knowledge graph"""
    if kg_id not in knowledge_graphs:
        return {}
    
    graph = knowledge_graphs[kg_id]["graph"]
    
    # Extract unique labels and relationship types from the graph
    node_labels = list(set([node.get('label', 'Entity') for node in graph.get('nodes', [])]))
    relationship_types = list(set([rel.get('type', 'RELATED_TO') for rel in graph.get('relationships', [])]))
    
    return {
        "node_labels": node_labels,
        "relationship_types": relationship_types
    }

def create_enhanced_rag_prompt(context: str, ontology: Dict[str, Any]) -> str:
    """Create a RAG prompt with structured response and multi-hop reasoning"""
    
    ontology_info = ""
    if ontology:
        ontology_info = f"""
Available Entity Types: {', '.join(ontology.get('node_labels', []))}
Available Relations: {', '.join(ontology.get('relationship_types', []))}
"""
    
    return f"""You are a clinical knowledge assistant. You must provide a structured response with the main answer first, followed by optional detailed reasoning.

{ontology_info}

KNOWLEDGE GRAPH CONTEXT:
{context}

CRITICAL: When referencing nodes in your reasoning paths, you MUST use the EXACT node IDs from the knowledge graph context above. Do NOT use placeholder IDs like X, Y, Z. Look at the context and use the actual numeric IDs shown (e.g., ID:1, ID:2, ID:3, etc.).

CRITICAL: You MUST NOT invent or hallucinate any nodes or relationships that are not present in the knowledge graph context above. Your reasoning and responses must be strictly grounded in the provided knowledge graph.

IMPORTANT: Structure your response EXACTLY as follows:

## MAIN ANSWER

### Summary
[2-3 sentences directly answering the question - be concise and clear]

### Key Evidence
- **Evidence 1**: [Most critical fact with source reference including actual node ID]
- **Evidence 2**: [Second critical fact with source reference including actual node ID]  
- **Evidence 3**: [Third critical fact with source reference including actual node ID]

### Reasoning Path
```
[Start Entity (ID:actual_number)] --[relationship]--> [Middle Entity (ID:actual_number)] --[relationship]--> [Conclusion Entity (ID:actual_number)]
```
*Show the main reasoning path through the knowledge graph using the ACTUAL node IDs from the context above*

### Confidence Assessment
- **Level**: [High/Medium/Low]
- **Basis**: [One-line explanation]

---

<details>
<summary>📊 Detailed Multi-Hop Reasoning Analysis</summary>

#### Hop 1: Query Decomposition
- **Primary Entity**: [Main subject identified with actual node ID]
- **Secondary Entities**: [Related concepts with actual node IDs]
- **Query Intent**: [What user wants to know]

#### Hop 2: Context Retrieval & Scoring
- **Retrieved Contexts**: 
  - Context A (95% relevance): [Description]
  - Context B (87% relevance): [Description]
  - Context C (73% relevance): [Description]
- **Knowledge Graph Traversal**:
  - Starting Node: [Node name (ID:actual_number)]
  - Path 1: [Node A (ID:actual_number)] → [Node B (ID:actual_number)] → [Node C (ID:actual_number)]
  - Path 2: [Node A (ID:actual_number)] → [Node D (ID:actual_number)] → [Node E (ID:actual_number)]

#### Hop 3: Inference Generation
- **Hypothesis 1**: [Description with evidence and actual node IDs]
- **Hypothesis 2**: [Alternative interpretation with actual node IDs]
- **Domain Rules Applied**: [Specific rules used]

#### Hop 4: Cross-Validation
- **Consistency Checks**: 
  - ✓ Hypothesis 1 aligns with [Evidence from node ID:actual_number]
  - ✓ No contradictions in primary path
  - ⚠️ Minor discrepancy resolved
- **Alternative Paths**: [Other considered paths with actual node IDs]

#### Hop 5: Confidence Scoring
- **Evidence Strength**: [Assessment]
- **Reasoning Reliability**: [Assessment]
- **Final Calculation**: [How confidence was determined]

### Complete Knowledge Graph Traversal
```
[Full graph showing all explored paths with actual node IDs from the context]
```

### Limitations & Recommendations
- **Data Gaps**: [Missing information]
- **Next Steps**: [Recommended actions]
- **Related Queries**: [Other relevant questions]

</details>
"""

@app.post("/chat")
async def chat(message: Message):
    try:
        print(f"Chat request received - Question: '{message.question}', KG ID: '{message.kg_id}'")
        
        # Default to llama-4-maverick:free for OpenRouter if not specified
        if message.provider_rag == "openrouter" and not message.model_rag:
            message.model_rag = "meta-llama/llama-4-maverick:free"
        
        if message.provider_rag not in MODEL_PROVIDERS or message.model_rag not in MODEL_PROVIDERS[message.provider_rag]:
            raise ValueError(f"Model {message.model_rag} not available for provider {message.provider_rag}")
        
        llm = MODEL_PROVIDERS[message.provider_rag][message.model_rag]
        
        if message.kg_id:
            print(f"Looking up KG context for ID: {message.kg_id}")
            ontology = get_ontology_context(message.kg_id)
            
            # Use vector store to retrieve relevant context chunks
            if message.kg_id in vector_stores:
                vector_store = vector_stores[message.kg_id]
                # Perform similarity search with question
                relevant_docs = vector_store.similarity_search(message.question, k=5)
                # Combine retrieved chunks as context
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                print(f"Retrieved {len(relevant_docs)} relevant context chunks from vector store")
            else:
                print(f"No vector store found for KG ID: {message.kg_id}, falling back to full KG context")
                context = get_graph_context(message.kg_id)
            
            if not context:
                print(f"Warning: No KG context found for ID: {message.kg_id}")
                context = "No knowledge graph context available"
                ontology = {}
            else:
                print(f"Using KG context for ID: {message.kg_id}")
            
            # Create enhanced prompt with ontology constraints
            enhanced_prompt_template = create_enhanced_rag_prompt(context, ontology)
            
            prompt = ChatPromptTemplate.from_template(
                enhanced_prompt_template + "\n\nUSER QUESTION: {question}\n\nCLINICAL RESPONSE:"
            )
            
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({
                "question": message.question
            })
        else:
            # Fallback for questions without KG context
            prompt = ChatPromptTemplate.from_template(
                """You are a clinical knowledge assistant. Provide a structured, evidence-based response.

RESPONSE FORMAT:
🔍 **Summary**: [Brief summary]
📋 **Key Points**: [Bullet points]
⚠️ **Note**: This response is not based on a specific knowledge graph.

Question: {question}

Response:"""
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

class SaveToNeo4jRequest(BaseModel):
    kg_id: str
    uri: str
    user: str
    password: str

@app.post("/save_kg_to_neo4j")
async def save_kg_to_neo4j(request: SaveToNeo4jRequest):
    kg_id = request.kg_id
    # Use KGLoader's automatic environment detection instead of client-provided URI
    uri = kg_loader.neo4j_uri
    user = kg_loader.neo4j_user
    password = kg_loader.neo4j_password
    try:
        print(f"Received save request for KG ID: {kg_id}")
        print(f"Using Neo4j URI: {uri}, User: {user}")
        
        if kg_id not in knowledge_graphs:
            print(f"KG ID {kg_id} not found in knowledge_graphs")
            raise HTTPException(
                status_code=404,
                detail="Knowledge graph not found"
            )
            
        graph_data = knowledge_graphs[kg_id]["graph"]
        print(f"Graph data for {kg_id}: Nodes: {len(graph_data.get('nodes', []))}, Relationships: {len(graph_data.get('relationships', []))}")
        
        result = kg_loader.save_to_neo4j(uri, user, password, graph_data)
        
        # Return the result from kg_loader, which includes status, message, and details
        if result['status'] == 'success':
            print(f"Successfully saved KG {kg_id} to Neo4j")
            return result
        else:
            print(f"Failed to save KG {kg_id}: {result.get('message')}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to save knowledge graph to Neo4j: {result.get('message')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Unexpected error saving KG: {error_traceback}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
        
class DirectoryListRequest(BaseModel):
    base_path: str = ""

@app.post("/list_directories")
async def list_directories(request: DirectoryListRequest):
    try:
        base_path = request.base_path or os.getcwd()
        items = os.listdir(base_path)
        directories = [item for item in items if os.path.isdir(os.path.join(base_path, item))]
        return {
            "status": "success",
            "path": base_path,
            "directories": directories
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/export_kg_to_file")
async def export_kg_to_file(request: ExportToFileRequest):
    kg_id = request.kg_id
    folder_path = request.folder_path
    filename = request.filename

    try:
        print(f"Export request - KG ID: '{kg_id}', Folder: '{folder_path}', Filename: '{filename}'")
        print(f"Available KG IDs in memory: {list(knowledge_graphs.keys())}")
        
        # Validate filename has .json extension
        if not filename.lower().endswith('.json'):
            filename += '.json'

        # Use base_path if provided
        full_folder_path = request.base_path or folder_path

        # Ensure the directory exists
        os.makedirs(full_folder_path, exist_ok=True)
        
        # Create full path
        file_path = os.path.join(full_folder_path, filename)
        
        if kg_id not in knowledge_graphs:
            # Try to auto-load a KG if none are in memory
            if not knowledge_graphs:
                print("No KGs in memory, attempting to load from storage...")
                try:
                    kg_storage_dir = os.path.join(os.getcwd(), "kg_storage")
                    if os.path.exists(kg_storage_dir):
                        kg_files = [f for f in os.listdir(kg_storage_dir) if f.endswith('.json')]
                        if kg_files:
                            # Load the first available KG
                            first_kg_file = kg_files[0]
                            with open(os.path.join(kg_storage_dir, first_kg_file), 'r') as f:
                                kg_data = json.load(f)
                            
                            # Create new KG ID and load it
                            new_kg_id = f"kg_{str(uuid.uuid4())[:12]}"
                            knowledge_graphs[new_kg_id] = {
                                "graph": kg_data,
                                "provider": "auto_loaded",
                                "model": "from_storage",
                                "filename": first_kg_file
                            }
                            print(f"Auto-loaded KG from {first_kg_file} with ID: {new_kg_id}")
                            
                            # Update kg_id to the newly loaded one
                            kg_id = new_kg_id
                        else:
                            raise HTTPException(
                                status_code=404,
                                detail="No knowledge graphs available. Please load or create a KG first."
                            )
                    else:
                        raise HTTPException(
                            status_code=404,
                            detail="No kg_storage directory found. Please create a KG first."
                        )
                except Exception as e:
                    print(f"Failed to auto-load KG: {str(e)}")
                    raise HTTPException(
                        status_code=404,
                        detail=f"Knowledge graph with ID '{kg_id}' not found. Available IDs: {list(knowledge_graphs.keys())}. Auto-load failed: {str(e)}"
                    )
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Knowledge graph with ID '{kg_id}' not found. Available IDs: {list(knowledge_graphs.keys())}"
                )
            
        graph_data = knowledge_graphs[kg_id]["graph"]
        print(f"Found KG data - Nodes: {len(graph_data.get('nodes', []))}, Relationships: {len(graph_data.get('relationships', []))}")
        
        result = kg_loader.save_to_file(graph_data, file_path)
        print(f"Export result: {result}")
        
        if result['status'] == 'success':
            return result
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to save knowledge graph to file: {result.get('message')}"
            )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Export error: {error_traceback}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

class SaveKGWithNameRequest(BaseModel):
    kg_id: str
    filename: str

@app.post("/save_kg_with_name")
async def save_kg_with_name(request: SaveKGWithNameRequest):
    """Save KG with a custom user-provided name to the kg_storage directory"""
    kg_id = request.kg_id
    filename = request.filename.strip()
    
    try:
        print(f"Save KG request - KG ID: '{kg_id}', Filename: '{filename}'")
        print(f"Available KG IDs: {list(knowledge_graphs.keys())}")
        
        if not filename:
            raise HTTPException(
                status_code=400,
                detail="Filename cannot be empty"
            )
        
        # Sanitize filename - remove invalid characters
        import re
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Ensure .json extension
        if not filename.lower().endswith('.json'):
            filename += '.json'
        
        # kg_loader.save_to_file will handle saving to kg_storage directory
        # Just check if the file already exists in kg_storage
        kg_storage_dir = os.path.join(os.getcwd(), "kg_storage")
        os.makedirs(kg_storage_dir, exist_ok=True)
        file_path = os.path.join(kg_storage_dir, filename)
        
        if os.path.exists(file_path):
            raise HTTPException(
                status_code=400,
                detail=f"File '{filename}' already exists. Please choose a different name."
            )
        
        if kg_id not in knowledge_graphs:
            print(f"KG ID '{kg_id}' not found in knowledge_graphs")
            print(f"Available KG IDs: {list(knowledge_graphs.keys())}")
            # Try to load a default KG if none is found
            if not knowledge_graphs:
                # If no KGs are loaded, try to load one from kg_storage
                try:
                    kg_files = os.listdir(kg_storage_dir)
                    json_files = [f for f in kg_files if f.endswith('.json')]
                    if json_files:
                        # Load the first JSON file found
                        with open(os.path.join(kg_storage_dir, json_files[0]), 'r') as f:
                            default_kg = json.load(f)
                            # Create a new KG ID
                            kg_id = f"kg_{str(uuid.uuid4())[:12]}"
                            knowledge_graphs[kg_id] = {
                                "graph": default_kg,
                                "provider": "default",
                                "model": "loaded_from_storage"
                            }
                            print(f"Loaded default KG from {json_files[0]} with ID {kg_id}")
                except Exception as e:
                    print(f"Failed to load default KG: {str(e)}")
            
            # If still not found, raise exception
            if kg_id not in knowledge_graphs:
                raise HTTPException(
                    status_code=404,
                    detail=f"Knowledge graph with ID '{kg_id}' not found. Available IDs: {list(knowledge_graphs.keys())}"
                )
            
        graph_data = knowledge_graphs[kg_id]["graph"]
        print(f"Graph data found - Nodes: {len(graph_data.get('nodes', []))}, Relationships: {len(graph_data.get('relationships', []))}")
        
        # Just pass the filename - kg_loader will handle the path
        result = kg_loader.save_to_file(graph_data, filename)
        print(f"Save result: {result}")
        
        if result['status'] == 'success':
            return {
                "status": "success",
                "message": f"Knowledge graph saved as '{filename}' in kg_storage directory",
                "filename": filename,
                "path": file_path,
                "nodes": len(graph_data.get('nodes', [])),
                "relationships": len(graph_data.get('relationships', []))
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to save knowledge graph: {result.get('message')}"
            )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error saving KG with name: {error_traceback}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
        
@app.get("/list_stored_kgs")
async def list_stored_kgs():
    """List all available KG files in the kg_storage directory"""
    try:
        kg_storage_path = os.path.join(os.getcwd(), "kg_storage")
        if not os.path.exists(kg_storage_path):
            return {"kg_files": []}
        
        kg_files = []
        for filename in os.listdir(kg_storage_path):
            if filename.endswith('.json'):
                file_path = os.path.join(kg_storage_path, filename)
                try:
                    # Get file stats
                    stat = os.stat(file_path)
                    file_size = stat.st_size
                    modified_time = stat.st_mtime
                    
                    # Try to read the file to get node/relationship counts
                    with open(file_path, 'r') as f:
                        kg_data = json.load(f)
                        node_count = len(kg_data.get('nodes', []))
                        rel_count = len(kg_data.get('relationships', []))
                    
                    kg_files.append({
                        "filename": filename,
                        "file_path": file_path,
                        "size": file_size,
                        "modified": modified_time,
                        "nodes": node_count,
                        "relationships": rel_count
                    })
                except Exception as e:
                    print(f"Error reading KG file {filename}: {str(e)}")
                    continue
        
        return {"kg_files": kg_files}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing KG files: {str(e)}"
        )

@app.post("/load_stored_kg")
async def load_stored_kg(filename: str = Form(...)):
    """Load a KG file from the kg_storage directory"""
    try:
        kg_storage_path = os.path.join(os.getcwd(), "kg_storage")
        file_path = os.path.join(kg_storage_path, filename)
        
        # Set the last_import_dir in kg_loader to ensure exports go to the same directory
        kg_loader.last_import_dir = kg_storage_path
        
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"KG file {filename} not found"
            )
        
        # Load the KG data
        with open(file_path, 'r') as f:
            kg_data = json.load(f)
        
        # Create a unique ID for this KG
        kg_id = f"kg_{str(uuid.uuid4())[:12]}"
        knowledge_graphs[kg_id] = {
            "graph": kg_data,
            "provider": "local_storage",
            "model": "stored_file",
            "filename": filename
        }
        
        # Create a simple text representation for vector store
        # Extract text from node properties for RAG
        text_content = []
        for node in kg_data.get('nodes', []):
            props = node.get('properties', {})
            if 'name' in props:
                text_content.append(props['name'])
            if 'description' in props:
                text_content.append(props['description'])
            # Add other text properties
            for key, value in props.items():
                if isinstance(value, str) and len(value) > 10:
                    text_content.append(f"{key}: {value}")
        
        # Add relationship information
        for rel in kg_data.get('relationships', []):
            rel_props = rel.get('properties', {})
            for key, value in rel_props.items():
                if isinstance(value, str) and len(value) > 10:
                    text_content.append(f"{key}: {value}")
        
        combined_text = " ".join(text_content)
        
        # Create vector store for RAG (default to ollama embeddings)
        try:
            create_vector_store(kg_id, combined_text, "ollama")
        except Exception as e:
            print(f"Warning: Could not create vector store: {str(e)}")
            # Continue without vector store - KG context will still work
        
        print(f"Loaded stored KG {filename} with ID: {kg_id}")
        
        return {
            "message": f"Knowledge graph {filename} loaded successfully",
            "kg_id": kg_id,
            "graph_data": kg_data,
            "filename": filename
        }
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error loading stored KG: {error_traceback}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load KG: {str(e)}"
        )

@app.get("/debug/kg_status")
async def debug_kg_status():
    """Debug endpoint to check KG status"""
    try:
        kg_storage_dir = os.path.join(os.getcwd(), "kg_storage")
        kg_files = []
        if os.path.exists(kg_storage_dir):
            kg_files = [f for f in os.listdir(kg_storage_dir) if f.endswith('.json')]
        
        return {
            "memory_kgs": list(knowledge_graphs.keys()),
            "memory_kg_count": len(knowledge_graphs),
            "storage_directory": kg_storage_dir,
            "storage_files": kg_files,
            "storage_file_count": len(kg_files)
        }
    except Exception as e:
        return {"error": str(e)}

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

# Endpoint to get default Neo4j credentials for UI
@app.get("/neo4j/default_credentials")
async def get_default_neo4j_credentials():
    return {
        "uri": DEFAULT_NEO4J_URI,
        "user": DEFAULT_NEO4J_USER,
        "password": DEFAULT_NEO4J_PASSWORD
    }

# Endpoint to update default Neo4j credentials from UI
from fastapi import Body

@app.post("/neo4j/update_credentials")
async def update_neo4j_credentials(
    uri: str = Body(...),
    user: str = Body(...),
    password: str = Body(...)
):
    global DEFAULT_NEO4J_URI, DEFAULT_NEO4J_USER, DEFAULT_NEO4J_PASSWORD
    DEFAULT_NEO4J_URI = uri
    DEFAULT_NEO4J_USER = user
    DEFAULT_NEO4J_PASSWORD = password
    # Optionally update kg_loader instance as well
    kg_loader.neo4j_uri = uri
    kg_loader.neo4j_user = user
    kg_loader.neo4j_password = password
    return {"message": "Neo4j default credentials updated successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
