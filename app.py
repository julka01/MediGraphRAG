import os
from dotenv import load_dotenv
load_dotenv()

import json
import re
import uuid
from io import BytesIO
from typing import Dict, Any, Optional
import owlready2  # Add owlready2 for OWL parsing
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
            ) if os.getenv("OPENROUTER_API_KEY") else None
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
            "deepseek-r1-0528": Ollama(model="deepseek-r1-0528"),
            "llama3": Ollama(model="llama3"),
            "mistral": Ollama(model="mistral")
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
    if vendor in MODEL_PROVIDERS:
        # Return original model names without modification
        models = list(MODEL_PROVIDERS[vendor].keys())
        return {"models": models}
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
        
        # Check file size
        if len(contents) > 1024 * 1024 * 5:
            raise ValueError("File size exceeds 5MB limit")
            
        # Extract text from file
        text = extract_text(contents)
        if not text:
            raise ValueError("Failed to extract text from file")
            
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
    """Parse OWL ontology file into our format"""
    # Create a BytesIO object from the bytes
    file_obj = BytesIO(owl_bytes)
    
    # Load the ontology directly from bytes
    onto = owlready2.get_ontology("")
    onto.load(fileobj=file_obj)
    
    # Extract classes as node labels
    node_labels = [cls.name for cls in onto.classes()]
    
    # Extract object properties as relationship types
    relationship_types = [prop.name for prop in onto.object_properties()]
    
    return {
        "node_labels": node_labels,
        "relationship_types": relationship_types
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

def generate_knowledge_graph(text: str, provider: str, model: str, ontology: Optional[Dict] = None) -> Dict[str, Any]:
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
        print(f"Chat request received - Question: '{message.question}', KG ID: '{message.kg_id}'")
        
        if message.provider_rag not in MODEL_PROVIDERS or message.model_rag not in MODEL_PROVIDERS[message.provider_rag]:
            raise ValueError(f"Model {message.model_rag} not available for provider {message.provider_rag}")
        
        llm = MODEL_PROVIDERS[message.provider_rag][message.model_rag]
        context = ""
        if message.kg_id:
            print(f"Looking up KG context for ID: {message.kg_id}")
            context = get_graph_context(message.kg_id)
            if not context:
                print(f"Warning: No KG context found for ID: {message.kg_id}")
                context = "No knowledge graph context available"
            else:
                print(f"Using KG context for ID: {message.kg_id}")
        
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

class SaveToNeo4jRequest(BaseModel):
    kg_id: str
    uri: str
    user: str
    password: str

@app.post("/save_kg_to_neo4j")
async def save_kg_to_neo4j(request: SaveToNeo4jRequest):
    kg_id = request.kg_id
    uri = request.uri
    user = request.user
    password = request.password
    try:
        print(f"Received save request for KG ID: {kg_id}")
        print(f"Using Neo4j URI: {uri}, User: {user}")
        
        if kg_id not in knowledge_graphs:
            print(f"KG ID {kg_id} not found in knowledge_graphs")
            return {
                "status": "error",
                "message": "Knowledge graph not found",
                "details": None
            }, 404
            
        graph_data = knowledge_graphs[kg_id]["graph"]
        print(f"Graph data for {kg_id}: Nodes: {len(graph_data.get('nodes', []))}, Relationships: {len(graph_data.get('relationships', []))}")
        
        result = kg_loader.save_to_neo4j(uri, user, password, graph_data)
        
        # Return the result from kg_loader, which includes status, message, and details
        if result['status'] == 'success':
            print(f"Successfully saved KG {kg_id} to Neo4j")
            return result
        else:
            print(f"Failed to save KG {kg_id}: {result.get('message')}")
            return {
                "status": "error",
                "message": result.get('message', 'Failed to save knowledge graph to Neo4j'),
                "details": result.get('details', {})
            }, 400
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Unexpected error saving KG: {error_traceback}")
        return {
            "status": "error",
            "message": "Internal server error",
            "details": error_traceback
        }, 500
        
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
        # Validate filename has .json extension
        if not filename.lower().endswith('.json'):
            filename += '.json'
        
        # Use base_path if provided
        full_folder_path = request.base_path or folder_path
        
        # Create full path
        file_path = os.path.join(full_folder_path, filename)
        
        if kg_id not in knowledge_graphs:
            return {
                "status": "error",
                "message": "Knowledge graph not found",
                "details": None
            }, 404
            
        graph_data = knowledge_graphs[kg_id]["graph"]
        result = kg_loader.save_to_file(graph_data, file_path)
        
        if result['status'] == 'success':
            return result
        else:
            return {
                "status": "error",
                "message": result.get('message', 'Failed to save knowledge graph to file'),
                "details": result.get('details', {})
            }, 400
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        return {
            "status": "error",
            "message": "Internal server error",
            "details": error_traceback
        }, 500
        
@app.get("/list_stored_kgs")
async def list_stored_kgs():
    """List all available KG files in the kg_storage directory"""
    try:
        kg_storage_path = "kg_storage"
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
        kg_storage_path = "kg_storage"
        file_path = os.path.join(kg_storage_path, filename)
        
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
    uvicorn.run(app, host="0.0.0.0", port=8004)
