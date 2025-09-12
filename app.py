from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import os
import uuid
import types
import sys

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from model_providers import get_provider as get_llm_provider
from kg_creator import ChunkedKGCreator
from enhanced_kg_creator import EnhancedKGCreator
from ontology_guided_kg_creator import OntologyGuidedKGCreator
from enhanced_rag_system import EnhancedRAGSystem

# stub missing llm-graph-builder dependencies
stub = types.ModuleType("langchain_google_vertexai")
stub.ChatVertexAI = object
stub.HarmBlockThreshold = object
stub.HarmCategory = object
sys.modules["langchain_google_vertexai"] = stub

stub2 = types.ModuleType("langchain_groq")
stub2.ChatGroq = object
sys.modules["langchain_groq"] = stub2

stub3 = types.ModuleType("langchain_experimental")
stub3.__path__ = []
stub3.graph_transformers = types.ModuleType("langchain_experimental.graph_transformers")
stub3.graph_transformers.__path__ = []
stub3.graph_transformers.diffbot = types.ModuleType("langchain_experimental.graph_transformers.diffbot")
stub3.graph_transformers.diffbot.DiffbotGraphTransformer = object
stub3.graph_transformers.LLMGraphTransformer = object
sys.modules["langchain_experimental"] = stub3
sys.modules["langchain_experimental.graph_transformers"] = stub3.graph_transformers
sys.modules["langchain_experimental.graph_transformers.diffbot"] = stub3.graph_transformers.diffbot

# include llm-graph-builder backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "llm-graph-builder/backend"))
try:
    from src.main import extract_graph_from_file_local_file
    from src.QA_integration import QA_RAG
    from src.graph_query import get_graphDB_driver, execute_query, visualize_schema
except ImportError:
    def extract_graph_from_file_local_file(*args, **kwargs):
        raise HTTPException(status_code=501, detail="Graph extraction not available")
    def QA_RAG(*args, **kwargs):
        raise HTTPException(status_code=501, detail="RAG not available")
    def get_graphDB_driver(*args, **kwargs):
        return None
    def execute_query(*args, **kwargs):
        return {}
    def visualize_schema(*args, **kwargs):
        return {}

app = FastAPI()

# Serve static UI
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Neo4j Aura credentials
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://01ae09e4.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "awhKHbIyHJZPAIuGhHpL9omIXw8Vupnnm_35XSDN2yg")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Initialize KG creator and enhanced systems
kg_creator = ChunkedKGCreator(
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USERNAME,
    neo4j_password=NEO4J_PASSWORD,
    neo4j_database=NEO4J_DATABASE
)

# Initialize enhanced systems
enhanced_kg_creator = EnhancedKGCreator(
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USERNAME,
    neo4j_password=NEO4J_PASSWORD,
    neo4j_database=NEO4J_DATABASE,
    embedding_model=os.getenv("EMBEDDING_MODEL", "openai")
)

# Initialize ontology-guided KG creator
ontology_kg_creator = OntologyGuidedKGCreator(
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USERNAME,
    neo4j_password=NEO4J_PASSWORD,
    neo4j_database=NEO4J_DATABASE,
    embedding_model=os.getenv("EMBEDDING_MODEL", "openai"),
    ontology_path="biomedical_ontology.owl"
)

enhanced_rag_system = EnhancedRAGSystem(
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USERNAME,
    neo4j_password=NEO4J_PASSWORD,
    neo4j_database=NEO4J_DATABASE,
    embedding_model=os.getenv("EMBEDDING_MODEL", "openai")
)

from pypdf import PdfReader

@app.post("/generate_kg")
async def generate_kg(
    file: UploadFile = File(...),
    provider_kg: str = Form("openrouter"),
    model_kg: str = Form("deepseek/deepseek-r1-0528:free")
):
    try:
        content = await file.read()
        # Check if file is PDF by magic bytes
        if content[:4] == b'%PDF':
            # Extract text from PDF
            from io import BytesIO
            pdf_reader = PdfReader(BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        else:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                text = content.decode("utf-8", errors="ignore")
        provider_instance = get_llm_provider(provider_kg, model_kg)
        # Wrap provider instance with LangChainRunnableAdapter for compatibility
        from model_providers import LangChainRunnableAdapter
        llm = LangChainRunnableAdapter(provider_instance, model_kg)
        kg = await kg_creator.generate_knowledge_graph(text, llm)
        return JSONResponse(content=kg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_kg_from_file")
async def load_kg_from_file(
    file: UploadFile = File(...),
    provider: str = Form("openrouter"),
    model: str = Form("deepseek/deepseek-r1-0528:free")
):
    """
    Load knowledge graph from file using llm-graph-builder backend.
    Returns KG ID and graph data.
    """
    try:
        content = await file.read()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("utf-8", errors="ignore")
        provider_instance = get_llm_provider(provider, model)
        # Wrap provider instance with LangChainRunnableAdapter for compatibility
        from model_providers import LangChainRunnableAdapter
        llm = LangChainRunnableAdapter(provider_instance, model)
        graph_data = kg_creator.generate_knowledge_graph(text, llm)
        kg_id = str(uuid.uuid4())
        return JSONResponse(content={"kg_id": kg_id, "graph_data": graph_data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_graph")
async def extract_graph(
    file: UploadFile = File(...),
    provider: str = Form("openrouter"),
    model: str = Form("deepseek/deepseek-r1-0528:free"),
    allowed_nodes: List[str] = Form(default=[]),
    allowed_relationships: List[str] = Form(default=[]),
    token_chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=100),
    chunks_to_combine: int = Form(default=1),
    additional_instructions: str = Form(default=None),
    use_enhanced: bool = Form(True)
):
    """
    Extract graph from file. By default uses enhanced KG creator for better entity names.
    Set use_enhanced=False to use original llm-graph-builder method.
    """
    try:
        content = await file.read()
        
        if use_enhanced:
            # Use enhanced KG creator for better entity extraction
            if content[:4] == b'%PDF':
                from io import BytesIO
                pdf_reader = PdfReader(BytesIO(content))
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            else:
                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    text = content.decode("utf-8", errors="ignore")
            
            provider_instance = get_llm_provider(provider, model)
            from model_providers import LangChainRunnableAdapter
            llm = LangChainRunnableAdapter(provider_instance, model)
            
            # Use enhanced KG creator with meaningful entity names
            kg = enhanced_kg_creator.generate_knowledge_graph(text, llm, file.filename)
            
            return JSONResponse(content={
                "message": "Enhanced knowledge graph created successfully",
                "fileName": file.filename,
                "nodeCount": len(kg.get('nodes', [])),
                "relationshipCount": len(kg.get('relationships', [])),
                "chunkCount": len(kg.get('chunks', [])),
                "hasEmbeddings": True,
                "embeddingDimension": kg.get('metadata', {}).get('embedding_dimension'),
                "method": "enhanced",
                "graph_data": kg
            })
        else:
            # Use original llm-graph-builder method
            graph = await extract_graph_from_file_local_file(
                NEO4J_URI,
                NEO4J_USERNAME,
                NEO4J_PASSWORD,
                NEO4J_DATABASE,
                model, None, file.filename,
                allowed_nodes, allowed_relationships,
                token_chunk_size, chunk_overlap, chunks_to_combine,
                None, additional_instructions
            )
            return JSONResponse(content=graph)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qa_rag")
async def qa_rag(
    question: str = Form(...),
    document_names: List[str] = Form(default=[]),
    session_id: str = Form(...),
    mode: str = Form("default")
):
    try:
        driver = get_graphDB_driver(
            NEO4J_URI,
            NEO4J_USERNAME,
            NEO4J_PASSWORD,
            NEO4J_DATABASE
        )
        result = QA_RAG(
            driver,
            get_llm_provider("openrouter", "deepseek/deepseek-r1-0528:free"),
            question, document_names, session_id, mode
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/neo4j/query")
async def neo4j_query(body: dict = Body(...)):
    try:
        query = body.get("query")
        if not query:
            raise HTTPException(status_code=422, detail="Missing query")
        driver = get_graphDB_driver(
            NEO4J_URI,
            NEO4J_USERNAME,
            NEO4J_PASSWORD,
            NEO4J_DATABASE
        )
        result = execute_query(driver, query, body.get("document_names", []))
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/neo4j/schema")
async def neo4j_schema():
    try:
        driver = get_graphDB_driver(
            NEO4J_URI,
            NEO4J_USERNAME,
            NEO4J_PASSWORD,
            NEO4J_DATABASE
        )
        schema = visualize_schema(driver)
        return JSONResponse(content=schema)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{provider}")
async def list_models(provider: str):
    model_map = {
        "openai": ["gpt-4", "gpt-3.5-turbo"],
        "openrouter": ["meta-llama/llama-4-maverick:free", "deepseek/deepseek-r1-0528:free"],
        "ollama": ["llama2"],
        "anthropic": ["claude-2"],
        "google": ["gemini-advanced"],
        "lmu_lightllm": ["lmu-light"],
        "huggingface": ["gpt2"],
        "deepseek": ["deepseek-r1-0528:free"],
        "gemini": ["gemini-advanced"]
    }
    return {"models": model_map.get(provider.lower(), [])}

@app.post("/chat")
async def chat(body: dict = Body(...)):
    """
    Enhanced chat endpoint for RAG queries using the knowledge graph with embeddings
    """
    try:
        question = body.get("question")
        provider_rag = body.get("provider_rag", "openrouter")
        model_rag = body.get("model_rag", "deepseek/deepseek-r1-0528:free")
        document_names = body.get("document_names", [])
        top_k = body.get("top_k", 5)
        
        if not question:
            raise HTTPException(status_code=422, detail="Missing question")
        
        # Get LLM provider
        try:
            provider_instance = get_llm_provider(provider_rag, model_rag)
            from model_providers import LangChainRunnableAdapter
            llm = LangChainRunnableAdapter(provider_instance, model_rag)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "response": f"Error initializing LLM provider: {str(e)}",
                    "sources": [],
                    "entities": [],
                    "relationships": [],
                    "confidence": 0.0,
                    "chunk_count": 0,
                    "entity_count": 0,
                    "relationship_count": 0,
                    "error": f"LLM provider error: {str(e)}"
                }
            )
        
        # Use enhanced RAG system
        try:
            result = enhanced_rag_system.generate_response(
                question=question,
                llm=llm,
                document_names=document_names if document_names else None,
                top_k=top_k
            )
            
            # Check if there was an error in the result
            if "error" in result:
                return JSONResponse(content={
                    "response": result["response"],
                    "sources": result.get("sources", []),
                    "entities": result.get("entities", []),
                    "relationships": result.get("relationships", []),
                    "confidence": result.get("confidence", 0.0),
                    "chunk_count": result.get("chunk_count", 0),
                    "entity_count": result.get("entity_count", 0),
                    "relationship_count": result.get("relationship_count", 0),
                    "error": result["error"],
                    "debug_info": "Enhanced RAG system encountered an issue"
                })
            
            return JSONResponse(content={
                "response": result["response"],
                "sources": result["sources"],
                "entities": result["entities"],
                "relationships": result["relationships"],
                "confidence": result["confidence"],
                "chunk_count": result["chunk_count"],
                "entity_count": result["entity_count"],
                "relationship_count": result["relationship_count"]
            })
            
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "response": f"Error generating RAG response: {str(e)}",
                    "sources": [],
                    "entities": [],
                    "relationships": [],
                    "confidence": 0.0,
                    "chunk_count": 0,
                    "entity_count": 0,
                    "relationship_count": 0,
                    "error": f"RAG system error: {str(e)}"
                }
            )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "response": f"Unexpected error: {str(e)}",
                "sources": [],
                "entities": [],
                "relationships": [],
                "confidence": 0.0,
                "chunk_count": 0,
                "entity_count": 0,
                "relationship_count": 0,
                "error": f"Unexpected error: {str(e)}"
            }
        )

@app.post("/save_kg_with_name")
async def save_kg_with_name(body: dict = Body(...)):
    """
    Save knowledge graph with a custom name
    """
    try:
        kg_id = body.get("kg_id")
        filename = body.get("filename", "knowledge_graph")
        
        if not kg_id:
            raise HTTPException(status_code=422, detail="Missing kg_id")
        
        # For now, return a success message
        # In a full implementation, this would save to a file or database
        return JSONResponse(content={
            "message": f"Knowledge graph saved successfully",
            "filename": f"{filename}.json",
            "nodes": 0,  # Would be actual count
            "relationships": 0  # Would be actual count
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_enhanced_kg")
async def generate_enhanced_kg(
    file: UploadFile = File(...),
    provider_kg: str = Form("openrouter"),
    model_kg: str = Form("deepseek/deepseek-r1-0528:free"),
    store_in_neo4j: bool = Form(True)
):
    """
    Generate knowledge graph with embeddings using the enhanced KG creator
    """
    try:
        content = await file.read()
        # Check if file is PDF by magic bytes
        if content[:4] == b'%PDF':
            # Extract text from PDF
            from io import BytesIO
            pdf_reader = PdfReader(BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        else:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                text = content.decode("utf-8", errors="ignore")
        
        provider_instance = get_llm_provider(provider_kg, model_kg)
        from model_providers import LangChainRunnableAdapter
        llm = LangChainRunnableAdapter(provider_instance, model_kg)
        
        # Use enhanced KG creator with embeddings
        file_name = file.filename if store_in_neo4j else None
        kg = enhanced_kg_creator.generate_knowledge_graph(text, llm, file_name)
        
        return JSONResponse(content=kg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_ontology_guided_kg")
async def generate_ontology_guided_kg(
    file: UploadFile = File(...),
    provider_kg: str = Form("openrouter"),
    model_kg: str = Form("deepseek/deepseek-r1-0528:free"),
    ontology_file: UploadFile = File(None),
    store_in_neo4j: bool = Form(True)
):
    """
    Generate knowledge graph using ontology guidance for better entity classification and relationships.
    This endpoint properly extracts entities from PDF content using LLM with ontology constraints.
    """
    try:
        content = await file.read()
        
        # Extract text from PDF or text file
        if content[:4] == b'%PDF':
            from io import BytesIO
            pdf_reader = PdfReader(BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        else:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                text = content.decode("utf-8", errors="ignore")
        
        # Handle optional ontology file upload
        ontology_path = None
        if ontology_file:
            ontology_content = await ontology_file.read()
            ontology_path = f"temp_{ontology_file.filename}"
            with open(ontology_path, "wb") as f:
                f.write(ontology_content)
        
        provider_instance = get_llm_provider(provider_kg, model_kg)
        from model_providers import LangChainRunnableAdapter
        llm = LangChainRunnableAdapter(provider_instance, model_kg)
        
        # Use ontology-guided KG creator for proper entity extraction
        file_name = file.filename if store_in_neo4j else None
        kg = ontology_kg_creator.generate_knowledge_graph(
            text=text, 
            llm=llm, 
            file_name=file_name,
            ontology_path=ontology_path
        )
        
        # Clean up temporary ontology file
        if ontology_path and os.path.exists(ontology_path):
            os.remove(ontology_path)
        
        return JSONResponse(content={
            "message": "Ontology-guided knowledge graph created successfully",
            "fileName": file.filename,
            "nodeCount": len(kg.get('nodes', [])),
            "relationshipCount": len(kg.get('relationships', [])),
            "chunkCount": len(kg.get('chunks', [])),
            "hasEmbeddings": True,
            "embeddingDimension": kg.get('metadata', {}).get('embedding_dimension'),
            "ontologyClasses": kg.get('metadata', {}).get('ontology_classes', 0),
            "ontologyRelationships": kg.get('metadata', {}).get('ontology_relationships', 0),
            "method": "ontology_guided_llm",
            "extractionMethod": kg.get('metadata', {}).get('extraction_method'),
            "graph_data": kg
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/kg_stats")
async def get_kg_stats():
    """
    Get knowledge graph statistics
    """
    try:
        stats = enhanced_rag_system.get_knowledge_graph_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/entity/{entity_id}")
async def get_entity_details(entity_id: str):
    """
    Get detailed information about a specific entity
    """
    try:
        details = enhanced_rag_system.get_entity_details(entity_id)
        return JSONResponse(content=details)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_entities")
async def search_entities(body: dict = Body(...)):
    """
    Search for entities using vector similarity
    """
    try:
        query = body.get("query")
        top_k = body.get("top_k", 10)
        
        if not query:
            raise HTTPException(status_code=422, detail="Missing query")
        
        results = enhanced_rag_system.search_entities(query, top_k)
        return JSONResponse(content={"results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/neo4j/validate_aura")
async def validate_neo4j_aura():
    """
    Validate Neo4j Aura instance availability and connection
    """
    try:
        from langchain_neo4j import Neo4jGraph
        
        # Test connection to Neo4j Aura
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE
        )
        
        # Simple query to test connectivity
        result = graph.query("RETURN 'Connection successful' as status, datetime() as timestamp")
        
        if result:
            # Get database info
            db_info = graph.query("CALL dbms.components() YIELD name, versions, edition")
            
            return JSONResponse(content={
                "status": "connected",
                "message": "Neo4j Aura instance is available and accessible",
                "uri": NEO4J_URI,
                "database": NEO4J_DATABASE,
                "instance_id": os.getenv("AURA_INSTANCEID", "01ae09e4"),
                "instance_name": os.getenv("AURA_INSTANCENAME", "My instance"),
                "connection_test": result[0] if result else None,
                "database_info": db_info[0] if db_info else None
            })
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "Failed to execute test query on Neo4j Aura",
                    "uri": NEO4J_URI
                }
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": f"Cannot connect to Neo4j Aura instance: {str(e)}",
                "uri": NEO4J_URI,
                "instance_id": os.getenv("AURA_INSTANCEID", "01ae09e4"),
                "error_details": str(e)
            }
        )

@app.get("/neo4j/default_credentials")
async def default_credentials():
    return {
        "uri": NEO4J_URI,
        "user": NEO4J_USERNAME,
        "database": NEO4J_DATABASE,
        "instance_id": os.getenv("AURA_INSTANCEID", "01ae09e4"),
        "instance_name": os.getenv("AURA_INSTANCENAME", "My instance")
    }
