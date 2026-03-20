from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Body, Request, Depends
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Optional
import asyncio
import hashlib
import logging
import threading
import os, uuid, sys, tempfile, io, json
from collections import deque
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

from ontographrag.providers.model_providers import get_provider as get_llm_provider, LangChainRunnableAdapter
from ontographrag.kg.builders.enhanced_kg_creator import UnifiedOntologyGuidedKGCreator
from ontographrag.rag.systems.enhanced_rag_system import EnhancedRAGSystem

# Module-level singleton with lock to prevent race conditions at startup
_rag_system: EnhancedRAGSystem = None
_rag_system_lock = threading.Lock()

def get_rag_system() -> EnhancedRAGSystem:
    global _rag_system
    if _rag_system is None:
        with _rag_system_lock:
            if _rag_system is None:  # double-checked locking
                _rag_system = EnhancedRAGSystem()
    return _rag_system

from csv_processor import MedicalReportCSVProcessor

# Configuration constants for input validation
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_FILE_EXTENSIONS = {'.pdf', '.txt', '.csv', '.json', '.xml'}
ALLOWED_ONTOLOGY_EXTENSIONS = {'.owl', '.rdf', '.ttl', '.xml'}

def validate_file_upload(file: UploadFile, max_size_bytes: int = MAX_FILE_SIZE_BYTES, allowed_extensions: set = None) -> None:
    """
    Validate file upload for size and extension.
    Raises HTTPException if validation fails.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Check file extension
    if allowed_extensions:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            logger.warning(f"Invalid file extension: {file_ext}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed extensions: {', '.join(allowed_extensions)}"
            )
    
    # Check file size (peek at beginning)
    # Note: For full validation, we'd need to read the entire file which we do in the endpoint
    # This is a preliminary check that can be enhanced with streaming size validation
    logger.info(f"Validating file: {file.filename}")

# Import from local kg_utils
from ontographrag.kg.utils.extract_graph import extract_graph_from_file_local_file
from ontographrag.kg.utils.graph_query import get_graphDB_driver

# Retain actual langchain_experimental if available
# import importlib
# if "langchain_experimental" not in sys.modules:
#     importlib.import_module("langchain_experimental")

# Core graph imports moved inside endpoint functions

# ---------------------------------------------------------------------------
# Document text extraction — tiered OCR strategy (inspired by MOSAICX)
# Tier 1: PyMuPDF — fast, zero extra deps, works on digitally-created PDFs
# Tier 2: Surya   — layout-aware OCR, activates when PyMuPDF yield is poor
#                   (< MIN_CHARS_PER_PAGE chars/page on average), meaning the
#                   document is likely a scan. Surya is optional; if not
#                   installed the pipeline warns and accepts the thin output.
# ---------------------------------------------------------------------------
_MIN_CHARS_PER_PAGE = 80  # below this avg we treat the PDF as a scan

def _extract_text_from_bytes(data: bytes, filename: str) -> tuple[str, str]:
    """
    Extract text from raw file bytes.

    Returns (text_content, ocr_method) where ocr_method is one of:
      'pymupdf', 'surya', 'plaintext'
    Raises HTTPException on unrecoverable errors.
    """
    import fitz  # PyMuPDF — always available (in requirements.txt)

    ext = os.path.splitext(filename)[1].lower()

    if ext != '.pdf':
        try:
            return data.decode('utf-8'), 'plaintext'
        except UnicodeDecodeError:
            return data.decode('latin-1', errors='ignore'), 'plaintext'

    # --- Tier 1: PyMuPDF ---
    try:
        doc = fitz.open(stream=io.BytesIO(data), filetype="pdf")
        pages_text = [page.get_text() for page in doc]
        doc.close()
        text_pymupdf = "\n".join(pages_text)
        page_count = max(len(pages_text), 1)
        avg_chars = len(text_pymupdf.strip()) / page_count

        if avg_chars >= _MIN_CHARS_PER_PAGE:
            return text_pymupdf, 'pymupdf'

        logger.warning(
            "PyMuPDF extracted only %.0f chars/page — PDF looks like a scan; trying Surya OCR",
            avg_chars,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

    # --- Tier 2: Surya OCR (optional dependency) ---
    try:
        from surya.recognition import batch_recognition  # type: ignore
        from surya.detection import batch_text_detection  # type: ignore
        from surya.model.detection.model import load_model as load_det_model  # type: ignore
        from surya.model.recognition.model import load_model as load_rec_model  # type: ignore
        from surya.model.recognition.processor import load_processor  # type: ignore
        from PIL import Image  # type: ignore

        logger.info("Surya OCR available — running OCR on scanned PDF")
        surya_pages: list[str] = []
        pdf_doc = fitz.open(stream=io.BytesIO(data), filetype="pdf")
        images = []
        for page in pdf_doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        pdf_doc.close()

        det_model, det_processor = load_det_model(), load_det_model()  # det processor
        rec_model, rec_processor = load_rec_model(), load_processor()

        det_results = batch_text_detection(images, det_model, det_processor)
        rec_results = batch_recognition(images, det_results, rec_model, rec_processor, langs=[["en"]] * len(images))

        for page_result in rec_results:
            surya_pages.append(" ".join(line.text for line in page_result.text_lines))

        text_surya = "\n".join(surya_pages)
        if text_surya.strip():
            logger.info("Surya OCR produced %d chars across %d pages", len(text_surya), len(surya_pages))
            return text_surya, 'surya'
        logger.warning("Surya OCR returned empty text — falling back to PyMuPDF output")
    except ImportError:
        logger.warning("Surya not installed (pip install surya-ocr) — using PyMuPDF output for scan")
    except Exception as surya_err:
        logger.warning("Surya OCR failed (%s) — using PyMuPDF output", surya_err)

    # Fall back to whatever PyMuPDF gave us (may be sparse)
    if text_pymupdf.strip():
        return text_pymupdf, 'pymupdf_fallback'
    raise HTTPException(status_code=400, detail="PDF contains no extractable text and Surya OCR is not available")


# Rate limiter (keyed on client IP)
limiter = Limiter(key_func=get_remote_address)

# Configure FastAPI
app = FastAPI(
    title="OntographRAG",
    description=(
        "Turn unstructured documents into schema-consistent knowledge graphs. "
        "Query them with hybrid vector + graph RAG. "
        "Measure answer confidence with 8 uncertainty metrics including RS-UQ."
    ),
    version="1.0.0",
)

# CORS — tighten ALLOWED_ORIGINS via env var in production
_cors_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.mount("/static", StaticFiles(directory="ontographrag/api/static"), name="static")

# Optional API key authentication — only enforced when APP_API_KEY is set in the environment.
# In development (no APP_API_KEY) all requests pass through.
_APP_API_KEY = os.getenv("APP_API_KEY")

def require_api_key(request: Request) -> None:
    if _APP_API_KEY:
        key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if key != _APP_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

# Global storage for current graph data
current_graph_data = None

# KG build progress log (ring buffer of last 200 lines)
_kg_progress: deque = deque(maxlen=200)
_kg_building: bool = False

def _log_progress(line: str) -> None:
    """Append a line to the KG build progress log."""
    _kg_progress.append(line)


@app.get("/kg_progress_stream")
async def kg_progress_stream(request: Request):
    """SSE endpoint — streams KG build progress to the browser."""
    async def event_generator():
        last_idx = 0
        while True:
            if await request.is_disconnected():
                break
            lines = list(_kg_progress)
            if len(lines) > last_idx:
                for line in lines[last_idx:]:
                    yield f"data: {json.dumps({'line': line})}\n\n"
                last_idx = len(lines)
            if not _kg_building and last_idx >= len(lines) and last_idx > 0:
                yield f"data: {json.dumps({'done': True})}\n\n"
                break
            await asyncio.sleep(0.4)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Tracks whether the Neo4j connection was successfully verified at startup.
# Endpoints that require Neo4j call require_neo4j() which raises 503 when False.
neo4j_ready: bool = False


def require_neo4j() -> None:
    """Raise 503 if Neo4j was not available at startup."""
    if not neo4j_ready:
        raise HTTPException(
            status_code=503,
            detail="Neo4j database is unavailable. Check connection settings and restart the server.",
        )


@app.on_event("startup")
def check_neo4j_connection():
    global neo4j_ready
    try:
        driver = get_graphDB_driver(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD"),
            os.getenv("NEO4J_DATABASE"),
        )
        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            session.run("RETURN 1").single()
        neo4j_ready = True
        logger.info("Neo4j connection check passed")
    except Exception as e:
        logger.warning("Neo4j health check failed: %s — Neo4j-dependent endpoints will return 503", e)
        # Allow the app to start so health/static endpoints still work

# ========== Named KG Management Endpoints ==========

@app.post("/kg/create")
async def create_kg(
    kg_name: str = Form(...),
    description: str = Form(None),
    data_source: str = Form(None)
):
    """
    Create a new named Knowledge Graph.
    """
    require_neo4j()
    try:
        from graphDB_dataAccess import graphDBdataAccess
        from langchain_neo4j import Neo4jGraph
        
        # Create Neo4jGraph (langchain) instead of driver
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE")
        )
        
        db_access = graphDBdataAccess(graph)
        
        result = db_access.create_kg(
            kg_name=kg_name,
            description=description,
            data_source=data_source
        )
        
        return JSONResponse(content={
            "status": "success",
            "kg": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create KG: {str(e)}")


@app.get("/kg/list")
async def list_kgs():
    """
    List all named Knowledge Graphs by querying Document nodes with kgName property.
    """
    require_neo4j()
    try:
        from ontographrag.kg.utils.graph_query import get_graphDB_driver
        
        driver = get_graphDB_driver(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD"),
            os.getenv("NEO4J_DATABASE"),
        )
        
        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            # Query for distinct kgName values from Document nodes
            result = session.run("""
                MATCH (d:Document)
                WHERE d.kgName IS NOT NULL AND d.kgName <> ''
                RETURN DISTINCT d.kgName AS kgName, count(d) AS documentCount, max(d.updatedAt) AS lastUpdated
                ORDER BY d.kgName
            """)
            
            kgs = []
            for record in result:
                kgs.append({
                    "name": record["kgName"],
                    "kg_name": record["kgName"],
                    "document_count": record["documentCount"],
                    "last_updated": record["lastUpdated"].isoformat() if record["lastUpdated"] else None
                })
        
        return JSONResponse(content={
            "status": "success",
            "kgs": kgs
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list KGs: {str(e)}")


@app.get("/kg/{kg_name}")
async def get_kg(kg_name: str):
    """
    Get details of a specific Knowledge Graph.
    """
    try:
        from graphDB_dataAccess import graphDBdataAccess
        from langchain_neo4j import Neo4jGraph

        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE")
        )

        db_access = graphDBdataAccess(graph)
        kg = db_access.get_kg(kg_name)
        
        if not kg:
            raise HTTPException(status_code=404, detail=f"KG '{kg_name}' not found")
        
        # Also get stats
        stats = db_access.get_kg_stats(kg_name)
        
        return JSONResponse(content={
            "status": "success",
            "kg": kg,
            "stats": stats[0] if stats else {}
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get KG: {str(e)}")


@app.delete("/kg/{kg_name}")
async def delete_kg(kg_name: str, delete_entities: bool = Query(True)):
    """
    Delete a named Knowledge Graph.
    """
    try:
        from graphDB_dataAccess import graphDBdataAccess
        from langchain_neo4j import Neo4jGraph

        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE")
        )

        db_access = graphDBdataAccess(graph)
        deleted_count = db_access.delete_kg_by_name(kg_name, delete_entities)
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Deleted KG '{kg_name}' with {deleted_count} documents",
            "deleted_documents": deleted_count
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete KG: {str(e)}")


@app.get("/kg/{kg_name}/entities")
async def get_kg_entities(kg_name: str, limit: int = 100):
    """
    Get entities from a specific Knowledge Graph.
    """
    try:
        from graphDB_dataAccess import graphDBdataAccess
        from langchain_neo4j import Neo4jGraph

        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE")
        )

        db_access = graphDBdataAccess(graph)
        entities = db_access.get_kg_entities(kg_name, limit)
        
        return JSONResponse(content={
            "status": "success",
            "kg_name": kg_name,
            "entities": entities,
            "count": len(entities)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get KG entities: {str(e)}")


@app.get("/health/neo4j")
def neo4j_health():
    try:
        driver = get_graphDB_driver(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD"),
            os.getenv("NEO4J_DATABASE"),
        )
        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            record = session.run("RETURN count(*) AS c").single()
        return {"status": "ok", "nodeCount": record["c"]}
    except Exception as e:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Neo4j health check failed: {e}"
        )

@app.get("/")
async def root():
    return FileResponse("ontographrag/api/static/index.html")

@app.post("/load_kg_from_file")
async def load_kg_from_file(
    file: UploadFile = File(...),
    provider: str = Form("openai"),
    model: str = Form("gpt-3.5-turbo")
):
    """
    Return the full raw KG from llm-graph-builder (no ontology filtering).
    """
    file_path = None
    try:
        data = await file.read()
        safe_name = f"{uuid.uuid4()}_{os.path.basename(file.filename)}"
        file_path = os.path.join(tempfile.gettempdir(), safe_name)
        await asyncio.to_thread(lambda: open(file_path, "wb").write(data))
        latency, graph_data = await extract_graph_from_file_local_file(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD"),
            os.getenv("NEO4J_DATABASE"),
            model,
            file_path,
            file.filename,
            [], [], 1000, 100, 1,
            None, None
        )
        driver = get_graphDB_driver(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD"),
            os.getenv("NEO4J_DATABASE"),
        )
        nodes = []
        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            for record in session.run("MATCH (n) RETURN n"):
                node = record["n"]
                props = {}
                for k, v in dict(node).items():
                    if hasattr(v, "isoformat"):
                        props[k] = v.isoformat()
                    else:
                        props[k] = v
                nodes.append({"id": node.id, "labels": list(node.labels), "properties": props})
        relationships = []
        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            for record in session.run("MATCH ()-[r]->() RETURN r"):
                rel = record["r"]
                props = {}
                for k, v in dict(rel).items():
                    if hasattr(v, "isoformat"):
                        props[k] = v.isoformat()
                    else:
                        props[k] = v
                relationships.append({
                    "id": rel.id,
                    "type": rel.type,
                    "start": rel.start_node.id,
                    "end": rel.end_node.id,
                    "properties": props
                })
        return JSONResponse(content=jsonable_encoder({"kg_id": str(uuid.uuid4()), "graph_data": {"nodes": nodes, "relationships": relationships}}))
    except (ImportError, ModuleNotFoundError):
        return JSONResponse(content=jsonable_encoder({"kg_id": str(uuid.uuid4()), "graph_data": {"nodes": [], "relationships": []}}))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path:
            try:
                os.unlink(file_path)
            except OSError:
                pass

@app.post("/extract_graph")
async def extract_graph(
    file: UploadFile = File(...),
    provider: str = Form("openai"),
    model: str = Form("gpt-3.5-turbo")
):
    """
    Return raw graph JSON without ontology post-processing.
    """
    file_path = None
    try:
        data = await file.read()
        safe_name = f"{uuid.uuid4()}_{os.path.basename(file.filename)}"
        file_path = os.path.join(tempfile.gettempdir(), safe_name)
        await asyncio.to_thread(lambda: open(file_path, "wb").write(data))
        _, graph_data = await extract_graph_from_file_local_file(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD"),
            os.getenv("NEO4J_DATABASE"),
            model,
            file_path,
            file.filename,
            [], [], 1000, 100, 1,
            None, None
        )
        return JSONResponse(content=graph_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path:
            try:
                os.unlink(file_path)
            except OSError:
                pass

@app.post("/create_ontology_guided_kg")
@limiter.limit("5/minute")
async def create_ontology_guided_kg(  # noqa: C901
    request: Request,
    file: UploadFile = File(...),
    provider: str = Form("openai"),
    model: str = Form("gpt-3.5-turbo"),
    embedding_model: str = Form("sentence_transformers"),
    ontology_file: Optional[UploadFile] = File(None),
    max_chunks: int = Form(None),
    kg_name: str = Form(None),
    neo4j_uri: str = Form(None),
    neo4j_user: str = Form(None),
    neo4j_password: str = Form(None),
    neo4j_database: str = Form(None)
):
    """
    Create knowledge graph with optional ontology guidance.
    If ontology file is provided, uses it to ensure consistent entity types and relationships.
    If no ontology is provided, performs basic LLM-based entity extraction.
    """
    global _kg_building, _kg_progress
    if max_chunks is not None and (max_chunks < 1 or max_chunks > 500):
        raise HTTPException(status_code=422, detail="max_chunks must be between 1 and 500")
    require_neo4j()
    ontology_path = None  # declared here so the finally block can always reference it
    _kg_progress.clear()
    _kg_building = True
    try:
        # Read file content with proper encoding handling
        _log_progress(f"📄 Reading file: {file.filename}")
        data = await file.read()

        # SHA-256 content deduplication — skip re-extraction for identical documents
        doc_hash = hashlib.sha256(data).hexdigest()
        _log_progress("🔎 Checking for duplicate document…")
        try:
            _neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
            _neo4j_user = neo4j_user or os.getenv("NEO4J_USERNAME", "neo4j")
            _neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
            _neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")
            _dup_driver = get_graphDB_driver(_neo4j_uri, _neo4j_user, _neo4j_password, _neo4j_database)
            with _dup_driver.session() as _session:
                _dup_result = _session.run(
                    "MATCH (d:Document {contentHash: $hash, kgName: $kg_name}) RETURN d.kgName AS kgName, d.fileName AS fileName LIMIT 1",
                    {"hash": doc_hash, "kg_name": kg_name}
                ).single()
            _dup_driver.close()
            if _dup_result:
                _existing_kg_name = _dup_result["kgName"]
                _log_progress(f"♻️  Duplicate detected — reusing existing KG '{_existing_kg_name}'")
                logger.info("Duplicate document (SHA-256 %s) — returning existing KG %s", doc_hash[:12], _existing_kg_name)
                from ontographrag.kg.loaders.kg_loader import KGLoader
                _dup_loader = KGLoader()
                _dup_kg = _dup_loader.load_from_neo4j(
                    uri=_neo4j_uri,
                    user=_neo4j_user,
                    password=_neo4j_password,
                    kg_label=_existing_kg_name
                ) if _existing_kg_name else None
                _kg_building = False
                return JSONResponse(content={
                    "kg_id": str(uuid.uuid4()),
                    "kg_name": _existing_kg_name,
                    "graph_data": _dup_kg,
                    "method": "deduplicated",
                    "doc_hash": doc_hash,
                    "deduplicated": True,
                    "message": f"Document already ingested (SHA-256 match). Returning existing KG '{_existing_kg_name}'."
                })
        except Exception as _dup_err:
            # Non-fatal: if duplicate check fails, proceed with normal extraction
            logger.warning("Deduplication check failed (proceeding normally): %s", _dup_err)

        # Determine file type and extract text (tiered OCR strategy)
        text_content, ocr_method = _extract_text_from_bytes(data, file.filename)
        _log_progress(f"📝 Text extracted via {ocr_method} · {len(text_content)} chars")

        if len(text_content.strip()) == 0:
            raise HTTPException(status_code=400, detail="File contains no readable text content")

        logger.info("Creating KG with model: %s from provider: %s", model, provider)
        logger.info("File: %s, ocr=%s, size: %d bytes, text: %d chars", file.filename, ocr_method, len(data), len(text_content))
        _log_progress(f"🎯 Using {provider}/{model} · {len(text_content)} chars")

        # Get LLM provider (use defaults matching test if not specified)
        provider = provider or "openrouter"
        model = model or "openai/gpt-oss-120b:free"
        llm = get_llm_provider(provider, model)

        # Handle ontology file if provided
        ontology_path = None
        if ontology_file:
            logger.debug("Ontology file: %s (%s)", ontology_file.filename, getattr(ontology_file, 'content_type', 'unknown'))
            ontology_data = await ontology_file.read()
            ontology_filename = f"ontology_{uuid.uuid4()}{os.path.splitext(os.path.basename(ontology_file.filename))[1]}"
            ontology_path = os.path.join(tempfile.gettempdir(), ontology_filename)
            with open(ontology_path, "wb") as tmpf:
                tmpf.write(ontology_data)
            logger.info("Ontology saved to %s (%d bytes)", ontology_path, len(ontology_data))

            # Quick format sanity-check
            try:
                with open(ontology_path, 'r', encoding='utf-8') as f:
                    sample = f.read(500).lower()
                if 'owl:' not in sample and 'rdf:' not in sample:
                    logger.warning("Ontology file may not be valid OWL/RDF (no owl:/rdf: tags in first 500 chars)")
            except Exception as e:
                logger.warning("Could not inspect ontology file: %s", e)
        else:
            logger.debug("No ontology file provided")

        # Generate unique KG name if not provided
        if not kg_name:
            kg_name = f"kg_{str(uuid.uuid4())}"

        logger.info("Initializing OntologyGuidedKGCreator (ontology_path=%s)", ontology_path)

        # Initialize ontology-guided KG creator (with defaults matching test)
        # Use provided neo4j credentials or fall back to environment variables
        neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = neo4j_user or os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
        neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

        kg_creator = UnifiedOntologyGuidedKGCreator(
            chunk_size=2000,  # Larger chunks for better patient report context
            chunk_overlap=300,
            ontology_path=ontology_path,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
            embedding_model=embedding_model or "sentence_transformers"
        )

        logger.debug(
            "KG creator: ontology_path=%s, classes=%d, relationships=%d",
            kg_creator.ontology_path,
            len(kg_creator.ontology_classes),
            len(kg_creator.ontology_relationships),
        )

        # Generate KG with ontology guidance (or without if no ontology provided)
        # Run in a thread so the blocking LLM/Neo4j calls don't block the event loop.
        _log_progress("🔍 Extracting entities and relationships…")
        kg = await asyncio.to_thread(
            kg_creator.generate_knowledge_graph,
            text_content, llm, file.filename, model, max_chunks, kg_name, None, doc_hash,
        )

        # Log results like test script
        entities = kg.get('metadata', {}).get('total_entities', 0)
        relationships = kg.get('metadata', {}).get('total_relationships', 0)
        stored = kg.get('metadata', {}).get('stored_in_neo4j', False)

        logger.info("KG results: %d entities, %d relationships, stored=%s", entities, relationships, stored)
        _log_progress(f"📊 Extracted {entities} entities, {relationships} relationships")

        # Reload KG from Neo4j to ensure ontology labels are properly displayed
        loaded_kg = None
        if stored:
            logger.info("Reloading KG from Neo4j to apply ontology labels")
            from ontographrag.kg.loaders.kg_loader import KGLoader

            kg_loader = KGLoader()
            reload_success = False
            if kg_name:
                # Load by KG name (ontology label) - now includes Document nodes
                loaded_kg = kg_loader.load_from_neo4j(
                    uri=neo4j_uri,
                    user=neo4j_user,
                    password=neo4j_password,
                    kg_label=kg_name
                )
                if loaded_kg and loaded_kg.get('status') == 'success':
                    reload_success = True
            else:
                # Load all entities (excluding system nodes)
                loaded_kg = kg_loader.load_from_neo4j(
                    uri=neo4j_uri,
                    user=neo4j_user,
                    password=neo4j_password
                )
                if loaded_kg and loaded_kg.get('status') == 'success':
                    reload_success = True

            if reload_success:
                loaded_entities = loaded_kg.get('loaded_nodes', 0) if 'loaded_nodes' in loaded_kg else len(loaded_kg.get('nodes', []))
                loaded_relationships = loaded_kg.get('loaded_relationships', 0) if 'loaded_relationships' in loaded_kg else len(loaded_kg.get('relationships', []))
                logger.info("Reloaded %d nodes, %d relationships from Neo4j", loaded_entities, loaded_relationships)
            else:
                logger.warning("Failed to reload KG from Neo4j, using initial KG data")
                loaded_kg = None

        # Use reloaded KG data if available and valid, otherwise use initial KG
        final_kg_data = loaded_kg if loaded_kg and loaded_kg.get('status') == 'success' else kg

        method = "ontology_guided" if ontology_path else "basic_llm"
        determinism_improvements = [
            "fixed_chunk_size",
            "temperature=0_for_all_LLMs",
            "node_label_fix",
            "neo4j_reload" if loaded_kg else None
        ]
        determinism_improvements = [x for x in determinism_improvements if x is not None]
        if ontology_path:
            determinism_improvements.append("ontology_constraints_applied")

        return JSONResponse(content={
            "kg_id": str(uuid.uuid4()),
            "kg_name": kg_name,
            "graph_data": final_kg_data,
            "method": method,
            "ocr_method": ocr_method,
            "ontology_file": ontology_file.filename if ontology_file else None,
            "determinism_improvements": determinism_improvements
        })

    except HTTPException:
        raise
    except Exception as e:
        # Provide fallback: try to return the initial KG data even if Neo4j operations failed
        error_msg = f"Ontology-guided KG creation failed: {str(e)}"

        # If we have initial KG data (created before Neo4j failure), return it
        if 'kg' in locals() and kg:
            logger.warning("Neo4j storage/reload failed, returning locally generated KG data")
            return JSONResponse(content={
                "kg_id": str(uuid.uuid4()),
                "graph_data": kg,
                "method": "ontology_guided" if ontology_path else "basic_llm",
                "ontology_file": ontology_file.filename if ontology_file else None,
                "determinism_improvements": [
                    "fixed_chunk_size",
                    "temperature=0_for_all_LLMs",
                    "node_label_fix"
                ] + (["ontology_constraints_applied"] if ontology_path else []),
                "warning": "Neo4j connection/storage failed - using locally processed data only",
                "error_details": error_msg
            })

        # No fallback data available, raise the error
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        _kg_building = False
        # Always clean up the ontology temp file, regardless of success or failure.
        if ontology_path:
            try:
                os.unlink(ontology_path)
            except OSError:
                pass

@app.post("/chat")
@limiter.limit("30/minute")
async def chat(request: Request, body: dict = Body(..., max_length=65536)):
    """
    Enhanced KG-focused RAG chat that ensures responses come from KG alone.
    Supports optional kg_name parameter to filter retrieval to a specific named KG.
    """
    require_neo4j()
    try:
        question = body.get("question", "")
        if not question or not isinstance(question, str):
            raise HTTPException(status_code=422, detail="Missing question")
        if len(question) > 4096:
            raise HTTPException(status_code=422, detail="Question too long (max 4096 chars)")

        docs = body.get("document_names", [])
        session = body.get("session_id", "default_session")
        mode = body.get("mode", "default")
        provider = body.get("provider_rag", "openrouter")
        model = body.get("model_rag", "openai/gpt-oss-120b:free")
        kg_name = body.get("kg_name", None)

        # Validate kg_name exists before querying (avoids confusing empty-result errors)
        if kg_name:
            _driver = get_graphDB_driver(
                os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password"),
                os.getenv("NEO4J_DATABASE", "neo4j"),
            )
            with _driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as _s:
                _exists = _s.run(
                    "MATCH (d:Document {kgName: $kg_name}) RETURN count(d) AS c",
                    {"kg_name": kg_name}
                ).single()
            _driver.close()
            if (_exists or {}).get("c", 0) == 0:
                raise HTTPException(status_code=404, detail=f"Knowledge graph '{kg_name}' not found")

        rag_system = get_rag_system()

        # Get LLM provider
        llm = LangChainRunnableAdapter(get_llm_provider(provider, model), model)

        # Generate response; cap at 120 s to prevent thread pool exhaustion on hung LLMs.
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(rag_system.generate_response, question, llm, docs, kg_name=kg_name),
                timeout=120,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="LLM response timed out — try again or choose a faster model")

        # Format response to match expected structure
        if "error" in result:
            return JSONResponse(content={
                "session_id": session,
                "message": f"KG Error: {result['error']}",
                "info": {
                    "sources": [],
                    "model": model,
                    "nodedetails": [],
                    "total_tokens": 0,
                    "response_time": 0,
                    "mode": mode,
                    "entities": result.get("entities", []),
                    "metric_details": {},
                    "kg_only": True,
                    "kg_stats": result.get("context", {}).get("kg_stats", {})
                },
                "user": "chatbot"
            })

        # Convert enhanced response to standard format
        return JSONResponse(content={
            "session_id": session,
            "message": result["response"],
            "info": {
                "sources": result["sources"],
                "model": model,
                "nodedetails": {
                    "chunkdetails": result.get("context", {}).get("chunks", []),
                    "entitydetails": result.get("context", {}).get("entities", {}),
                    "communitydetails": []
                },
                "total_tokens": 0,  # TODO: Implement token counting
                "response_time": 0,
                "mode": mode,
                "entities": {
                    "entityids": result.get("entities", []),
                    "relationshipids": [r.get("key", "") for r in result.get("relationships", [])],
                    "used_entities": result.get("used_entities", []),  # Nodes highlighted in KG visualization
                    "reasoning_edges": result.get("reasoning_edges", [])  # Edges forming the reasoning path
                },
                "metric_details": {
                    "question": question,
                    "contexts": [chunk["text"] for chunk in result.get("context", {}).get("chunks", [])],
                    "answer": result["response"]
                },
                "kg_only": True,
                "chunk_count": result.get("chunk_count", 0),
                "entity_count": result.get("entity_count", 0),
                "relationship_count": result.get("relationship_count", 0),
                "confidence": result.get("confidence", 0.0)
            },
            "user": "chatbot"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KG-RAG Error: {str(e)}")

@app.get("/models/{provider}")
def list_models(provider: str):
    """
    Return available models for provider.
    """
    model_map = {
        "openai": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ],
        "openrouter": [
            "openai/gpt-oss-120b:free",
            "meta-llama/llama-3.3-8b-instruct:free",
            "deepseek/deepseek-chat-v3.1:free",
            "x-ai/grok-4-fast:free",
        ],
        "ollama": [],  # populated dynamically below
    }

    if provider.lower() == "openai" and not os.getenv("OPENAI_API_KEY"):
        return {"models": [], "warning": "OPENAI_API_KEY not set"}

    if provider.lower() == "ollama":
        try:
            import ollama
            tags = ollama.list()
            model_map["ollama"] = [m.model for m in tags.models]
        except Exception:
            pass

    return {"models": model_map.get(provider.lower(), [])}

@app.get("/neo4j/default_credentials")
def default_credentials():
    """Return non-sensitive Neo4j connection defaults for the frontend form."""
    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USERNAME", "neo4j"),
        "database": os.getenv("NEO4J_DATABASE", "neo4j"),
        # Password is intentionally omitted — the user must enter it manually.
    }

@app.post("/clear_kg")
async def clear_kg():
    """
    Clear the entire Neo4j knowledge graph by removing all nodes and relationships.
    """
    require_neo4j()
    try:
        # Use Neo4j driver from environment
        driver = get_graphDB_driver(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD"),
            os.getenv("NEO4J_DATABASE"),
        )

        logger.info("Clearing entire Neo4j knowledge graph")

        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            # Drop constraints — backtick-quote names to guard against special characters
            try:
                constraints = [record["name"] for record in session.run("SHOW CONSTRAINTS")]
                for name in constraints:
                    safe = name.replace("`", "")  # strip any embedded backticks
                    try:
                        session.run(f"DROP CONSTRAINT `{safe}`")
                        logger.debug("Dropped constraint: %s", safe)
                    except Exception as e:
                        logger.warning("Could not drop constraint %s: %s", safe, e)
            except Exception as e:
                logger.warning("Error listing constraints: %s", e)

            # Drop indexes
            try:
                indexes = [
                    record["name"] for record in session.run("SHOW INDEXES")
                    if record["type"] != "LOOKUP"
                ]
                for name in indexes:
                    safe = name.replace("`", "")
                    try:
                        session.run(f"DROP INDEX `{safe}`")
                        logger.debug("Dropped index: %s", safe)
                    except Exception as e:
                        logger.warning("Could not drop index %s: %s", safe, e)
            except Exception as e:
                logger.warning("Error listing indexes: %s", e)

            # Delete all relationships first, then nodes
            session.run("MATCH ()-[r]-() DELETE r")
            result = session.run("MATCH (n) DELETE n RETURN count(n) as deleted_count")
            record = result.single()
            deleted_count = record["deleted_count"] if record else 0
            logger.info("Cleared %d nodes and all relationships", deleted_count)

            try:
                session.run("CALL db.resample.index.all()")
            except Exception:
                pass  # APOC not available — non-fatal

        logger.info("Neo4j knowledge graph cleared successfully")

        return JSONResponse(content={
            "message": f"Knowledge graph cleared successfully! Deleted {deleted_count} nodes and all relationships.",
            "status": "cleared",
            "nodes_deleted": deleted_count
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error clearing KG")
        raise HTTPException(status_code=500, detail=f"Failed to clear knowledge graph: {str(e)}")

@app.post("/test_create_working_kg")
async def test_create_working_kg():
    """
    Create a working test KG with vector embeddings that the RAG can actually use with semantic similarity search.
    """
    try:
        # Use the get_graphDB_driver function for consistency
        driver = get_graphDB_driver(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD"),
            os.getenv("NEO4J_DATABASE"),
        )

        logger.debug('Connected to Neo4j, creating test KG with placeholder embeddings...')

        # Clear any existing test data and indexes
        with driver.session() as session:
            try:
                logger.debug('🔄 Clearing existing test data...')

                # First drop existing vector indexes
                try:
                    session.run('DROP INDEX vector IF EXISTS')
                    session.run('DROP INDEX entity_vector IF EXISTS')
                    logger.debug('✅ Dropped existing vector indexes')
                except Exception as e:
                    logger.debug(f'Index drop warning (may not exist): {e}')

                # Clear all existing data before creating test KG
                session.run('MATCH (n) DETACH DELETE n')

                logger.debug('✅ Cleared existing test data')
            except Exception as e:
                logger.debug(f'Cleanup warning: {e}')

            # Create sample data with UUIDs
            doc_id = str(uuid.uuid4())
            chunk_id = str(uuid.uuid4())

            test_text = """Prostate cancer is a disease that affects men. Common symptoms include frequent urination, difficulty urinating, blood in urine, and erectile dysfunction. Treatment options include surgery (radical prostatectomy), radiation therapy, hormone therapy, and active surveillance for low-risk cases."""

            # Create document
            session.run('MERGE (d:Document {fileName: "test_medical_data.txt"}) SET d.id = $doc_id', {'doc_id': doc_id})

            # Create chunk with simple embeddings (placeholder for now - we can upgrade to real embeddings later)
            chunk_embedding = [0.1] * 384  # Placeholder vector similar to all-MiniLM-L6-v2 dimension

            session.run('''
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text = $text,
                c.embedding = $embedding,
                c.position = 0
            ''', {
                'chunk_id': chunk_id,
                'text': test_text,
                'embedding': chunk_embedding
            })

            # Link chunk to document
            session.run('MATCH (c:Chunk {id: $chunk_id}), (d:Document {id: $doc_id}) MERGE (c)-[:PART_OF]->(d)', {
                'chunk_id': chunk_id,
                'doc_id': doc_id
            })

            # No test entities created - removed hardcoded test entities

        logger.debug('✅ Test KG with embeddings created successfully!')

        # Test the KG by querying it
        with driver.session() as session:
            stats = session.run('''
            MATCH (d:Document) WHERE d.fileName = "test_medical_data.txt"
            OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)
            OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:__Entity__)
            OPTIONAL MATCH (e)-[r]-()
            RETURN count(DISTINCT d) AS docs, count(DISTINCT c) AS chunks, count(DISTINCT e) AS entities, count(DISTINCT r) AS rels
            ''')

            result = stats.single()

        return JSONResponse(content={
            "message": "Test KG with vector embeddings created successfully",
            "status": "ready_with_embeddings",
            "embeddings": "enabled",
            "vector_indexes": "created",
            "test_queries": [
                "What are the symptoms of prostate cancer?",
                "What treatments are available for prostate cancer?",
                "How does prostate cancer affect men?"
            ],
            "data_stats": {
                "documents": result.get('docs', 1) if 'result' in locals() and result else 1,
                "chunks": result.get('chunks', 1) if 'result' in locals() and result else 1,
                "entities": 0,  # No test entities created
                "relationships": 0,  # No test relationships created
                "embedding_dimensions": 384,
                "similarity_function": "cosine"
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error creating test KG")
        raise HTTPException(status_code=500, detail=f"Test KG creation failed: {str(e)}")


@app.post("/save_kg_to_neo4j")
async def save_kg_to_neo4j(
    kg_id: str = Form(...),
    uri: str = Form(...),
    user: str = Form(...),
    password: str = Form(...)
):
    """
    Save knowledge graph data to Neo4j database.
    """
    require_neo4j()
    try:
        global current_graph_data

        # Check if we have graph data to save
        if current_graph_data is None:
            raise HTTPException(status_code=400, detail="No graph data available to save. Load a KG first.")

        driver = get_graphDB_driver(uri, user, password, os.getenv("NEO4J_DATABASE"))

        nodes_saved = 0
        relationships_saved = 0

        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            # Clear existing data first (optional - you might want to keep this or make it configurable)
            try:
                session.run("MATCH (n) DETACH DELETE n")
                logger.debug("Cleared existing Neo4j database")
            except Exception as e:
                logger.debug(f"Warning: Could not clear existing data: {e}")

            # Save nodes
            for node in current_graph_data.get('nodes', []):
                labels = node.get('labels', [])
                if not labels:
                    labels = ['Entity']  # Default label

                # Build Cypher MERGE query
                label_str = ':'.join(f'`{label}`' for label in labels)
                properties = {}

                # Copy node properties, excluding internal ones
                for k, v in node.get('properties', {}).items():
                    if k not in ['embedding', 'element_id'] and v is not None:
                        properties[k] = v

                # Add id if it exists
                if node.get('properties', {}).get('id'):
                    properties['id'] = node['properties']['id']
                elif node.get('id'):
                    properties['id'] = str(node['id'])

                # Build parameterized MERGE query
                prop_str = ', '.join(f'`{k}`: ${k}' for k, v in properties.items())
                if prop_str:
                    merge_query = f"MERGE (n:{label_str} {{ {prop_str} }})"
                else:
                    # Fallback for nodes with no properties - use id if available
                    node_id = properties.get('id', str(node.get('id', str(uuid.uuid4()))))
                    merge_query = f"MERGE (n:{label_str} {{ id: '{node_id}' }})"

                try:
                    param_dict = {k: v for k, v in properties.items()}
                    session.run(merge_query, param_dict)
                    nodes_saved += 1
                except Exception as e:
                    logger.debug(f"Error saving node {node.get('id', 'unknown')}: {e}")
                    continue

            # Save relationships
            for rel in current_graph_data.get('relationships', []):
                start_id = rel.get('start') or rel.get('from')
                end_id = rel.get('end') or rel.get('to')
                rel_type = rel.get('type', 'RELATED_TO')

                if not start_id or not end_id:
                    continue

                # Build relationship query
                properties = {}
                for k, v in rel.get('properties', {}).items():
                    if v is not None:
                        properties[k] = v

                prop_str = ', '.join(f'`{k}`: ${k}' for k, v in properties.items())
                rel_prop = f" {{{prop_str}}}" if prop_str else ""

                match_query = f"""
                MATCH (a), (b)
                WHERE id(a) = $start_id AND id(b) = $end_id
                MERGE (a)-[r:`{rel_type}`{rel_prop}]->(b)
                """

                try:
                    param_dict = {k: v for k, v in properties.items()}
                    param_dict["start_id"] = start_id
                    param_dict["end_id"] = end_id
                    session.run(match_query, param_dict)
                    relationships_saved += 1
                except Exception as e:
                    logger.debug(f"Error saving relationship {start_id}-{rel_type}->{end_id}: {e}")
                    continue

        return JSONResponse(content={
            "message": "Knowledge graph saved to Neo4j successfully",
            "kg_id": kg_id,
            "nodes_saved": nodes_saved,
            "relationships_saved": relationships_saved
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save KG to Neo4j: {str(e)}")

@app.post("/load_kg_from_neo4j")
async def load_kg_from_neo4j(
    uri: str = Form(...),
    user: str = Form(...),
    password: str = Form(...),
    limit: int = Form(1000),
    sample_mode: bool = Form(False),
    load_complete: bool = Form(False),
    kg_label: str = Form(None)
):
    """
    Load the entire KG from Neo4j with optional sampling and filtering.
    """
    require_neo4j()
    try:
        driver = get_graphDB_driver(uri, user, password, os.getenv("NEO4J_DATABASE"))
        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            total_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

            def _serialize_props(raw_props: dict) -> dict:
                props = {}
                for k, v in raw_props.items():
                    if hasattr(v, "isoformat"):
                        props[k] = v.isoformat()
                    else:
                        props[k] = v
                return props

            def _append_node(nodes_list: list, node_obj) -> None:
                if node_obj is None:
                    return
                nodes_list.append({
                    "id": node_obj.id,
                    "labels": list(node_obj.labels),
                    "properties": _serialize_props(dict(node_obj))
                })

            def _append_relationship(rels_list: list, rel_obj, start_node, end_node) -> None:
                if rel_obj is None or start_node is None or end_node is None:
                    return
                rels_list.append({
                    "id": rel_obj.id,
                    "type": rel_obj.type,
                    "start": start_node.id,
                    "end": end_node.id,
                    "properties": _serialize_props(dict(rel_obj))
                })

            # If kg_label is provided, check if it matches a KG name (Document.kgName)
            kg_name_match = 0
            if kg_label:
                kg_name_match = session.run(
                    "MATCH (d:Document {kgName: $kg_name}) RETURN count(d) AS c",
                    {"kg_name": kg_label}
                ).single()["c"]

            if kg_label and kg_name_match:
                # Load by KG name — only __Entity__ + Document nodes, entity-to-entity edges.
                # Chunk and Mention infrastructure nodes are intentionally excluded from the
                # visualization so the graph matches what the creation flow renders.
                nodes = []
                relationships = []

                # Document node for this KG
                doc_records = session.run(
                    "MATCH (d:Document {kgName: $kg_name}) RETURN d",
                    {"kg_name": kg_label}
                )
                for record in doc_records:
                    _append_node(nodes, record["d"])

                # Load entity nodes scoped to this KG using the kgName property on each entity.
                # Previously used a Document←Chunk→Entity path, which silently excluded any
                # entity that wasn't linked to a chunk (mention-linking miss) → wrong counts.
                entity_nodes = []
                entity_ids = []
                order_clause = "ORDER BY rand()" if sample_mode else ""
                entity_limit = f"LIMIT {limit}" if not load_complete else ""
                entity_records = session.run(
                    f"MATCH (e:__Entity__ {{kgName: $kg_name}})"
                    f" RETURN DISTINCT e {order_clause} {entity_limit}",
                    {"kg_name": kg_label}
                )
                for record in entity_records:
                    entity = record.get("e")
                    if entity is None:
                        continue
                    entity_nodes.append(entity)
                    entity_ids.append(entity.id)
                    _append_node(nodes, entity)

                # Warn if any loaded entity has a kgName that differs from the requested KG
                if entity_ids:
                    shared_check = session.run(
                        "MATCH (e:__Entity__) "
                        "WHERE id(e) IN $eids AND e.kgName <> $kg_name "
                        "RETURN count(DISTINCT e) AS shared_count",
                        {"eids": entity_ids, "kg_name": kg_label}
                    ).single()
                    shared_count = (shared_check or {}).get("shared_count", 0)
                    if shared_count:
                        logging.warning(
                            "%d entities in KG '%s' are also referenced by other KGs — visualization may include shared entities",
                            shared_count, kg_label
                        )

                # Entity-to-entity relationships only
                if entity_ids:
                    entity_rel_records = session.run(
                        "MATCH (a:__Entity__)-[r]->(b:__Entity__) "
                        "WHERE id(a) IN $entity_ids AND id(b) IN $entity_ids "
                        "RETURN r, a AS start, b AS end",
                        {"entity_ids": entity_ids}
                    )
                    for record in entity_rel_records:
                        _append_relationship(relationships, record["r"], record["start"], record["end"])
                    if sample_mode and total_rels > len(relationships):
                        logging.info(
                            "Sampled %d/%d entities — loaded %d/%d relationships (remainder involve unloaded nodes)",
                            len(entity_ids), total_nodes, len(relationships), total_rels
                        )

                stats = {
                    "total_nodes_in_db": total_nodes,
                    "total_relationships_in_db": total_rels,
                    "loaded_nodes": len(nodes),
                    "loaded_relationships": len(relationships),
                    "sample_mode": sample_mode,
                    "complete_import": load_complete
                }
                return JSONResponse(content={
                    "kg_id": str(uuid.uuid4()),
                    "kg_name": kg_label,
                    "graph_data": {"nodes": nodes, "relationships": relationships},
                    "stats": stats
                })

            # Fallback: no kg_label match — load all __Entity__ nodes and their
            # entity-to-entity relationships only (exclude Document/Chunk/Mention).
            order_clause = "ORDER BY rand()" if sample_mode else ""
            limit_clause = f"LIMIT {limit}" if not load_complete else ""
            node_query = f"MATCH (n:__Entity__) RETURN n {order_clause} {limit_clause}"

            nodes = []
            for record in session.run(node_query):
                _append_node(nodes, record["n"])

            loaded_node_ids = [node["id"] for node in nodes]  # application-level id strings, not Neo4j internal IDs

            relationships = []
            query = "MATCH (n:__Entity__)-[r]->(m:__Entity__) WHERE n.id IN $node_ids AND m.id IN $node_ids RETURN r, n AS start, m AS end"
            rel_params = {"node_ids": loaded_node_ids}

            for record in session.run(query, rel_params):
                _append_relationship(relationships, record["r"], record["start"], record["end"])

        stats = {
            "total_nodes_in_db": total_nodes,
            "total_relationships_in_db": total_rels,
            "loaded_nodes": len(nodes),
            "loaded_relationships": len(relationships),
            "sample_mode": sample_mode,
            "complete_import": load_complete
        }
        return JSONResponse(content={
            "kg_id": str(uuid.uuid4()),
            "kg_name": kg_label,
            "graph_data": {"nodes": nodes, "relationships": relationships},
            "stats": stats
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate_csv")
async def validate_csv(csv_file: UploadFile = File(...)):
    """
    Validate CSV file format and structure for medical reports.
    """
    csv_path = None
    try:
        logger.info("Validating CSV file: %s", csv_file.filename)

        # Save uploaded file temporarily
        data = await csv_file.read()
        tmp_dir = tempfile.gettempdir()
        csv_path = os.path.join(tmp_dir, f"validate_{uuid.uuid4()}.csv")

        await asyncio.to_thread(lambda: open(csv_path, "wb").write(data))

        # Initialize CSV processor
        processor = MedicalReportCSVProcessor(delimiter='|')

        # Validate format (blocking I/O — run in thread)
        validation_result = await asyncio.to_thread(processor.validate_csv_format, csv_path)

        return JSONResponse(content={
            "is_valid": validation_result.get("is_valid", False),
            "delimiter": validation_result.get("delimiter", "|"),
            "num_columns": validation_result.get("num_columns", 0),
            "num_rows": validation_result.get("num_rows", 0),
            "field_mappings_count": len(validation_result.get("field_mappings", {})),
            "columns": validation_result.get("columns", []),
            "field_mappings": validation_result.get("field_mappings", {}),
            "errors": validation_result.get("validation_errors", [])
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CSV validation error")
        raise HTTPException(status_code=500, detail=f"CSV validation failed: {str(e)}")
    finally:
        if csv_path:
            try:
                os.unlink(csv_path)
            except OSError:
                pass

@app.post("/bulk_process_csv")
async def bulk_process_csv(
    csv_file: UploadFile = File(...),
    text_column: str = Form("full_report_text", description="CSV column containing the document text"),
    id_column: str = Form(None, description="CSV column to use as document ID (defaults to row index)"),
    kg_name: str = Form(None, description="Name for the resulting knowledge graph (auto-generated if omitted)"),
    batch_size: int = Form(50, description="Number of documents to process per batch"),
    start_row: int = Form(0, description="Starting row number (0-based)"),
    max_chunks: int = Form(20, description="Maximum number of chunks to process per document (for testing)")
):
    """
    Process documents from any CSV in bulk batches, guided by the loaded ontology.

    The ontology defines what entities and relationships to extract.
    Only `text_column` needs to match a column in your CSV.
    """
    require_neo4j()
    csv_path = None
    try:
        logger.info("Starting bulk CSV processing: %s (batch=%d, start=%d)", csv_file.filename, batch_size, start_row)

        # Save uploaded file temporarily
        data = await csv_file.read()
        tmp_dir = tempfile.gettempdir()
        csv_path = os.path.join(tmp_dir, f"bulk_{uuid.uuid4()}.csv")

        await asyncio.to_thread(lambda: open(csv_path, "wb").write(data))

        # Initialize enhanced KG creator for bulk processing
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        if not neo4j_password:
            raise HTTPException(status_code=500, detail="NEO4J_PASSWORD environment variable is not set")

        llm = get_llm_provider("openrouter", "openai/gpt-oss-120b:free")

        kg_creator = UnifiedOntologyGuidedKGCreator(
            chunk_size=2000,
            chunk_overlap=300,
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=neo4j_password,
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            embedding_model="sentence_transformers",
            max_chunks=max_chunks
        )

        # Process CSV in bulk.
        # Run in a thread so the blocking LLM/Neo4j calls don't block the event loop.
        resolved_kg_name = kg_name or f"bulk_{str(uuid.uuid4())[:8]}"
        bulk_result = await asyncio.to_thread(
            kg_creator.bulk_process_documents,
            csv_path=csv_path,
            text_column=text_column,
            id_column=id_column or None,
            start_row=start_row,
            batch_size=batch_size,
            llm=llm,
            kg_name=resolved_kg_name,
        )

        metadata = bulk_result.get("metadata", {})
        kg_id = str(uuid.uuid4())

        return JSONResponse(content={
            "kg_id": kg_id,
            "kg_name": resolved_kg_name,
            "message": f"Successfully processed {metadata.get('total_documents_processed', 0)} documents from CSV",
            "total_documents_processed": metadata.get("total_documents_processed", 0),
            "total_kgs": metadata.get("total_knowledge_graphs", 0),
            "batch_size": batch_size,
            "start_row": start_row,
            "csv_validation": metadata.get("csv_validation", {}),
            "bulk_processing_info": metadata.get("bulk_processing_info", {}),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Bulk CSV processing error")
        raise HTTPException(status_code=500, detail=f"Bulk CSV processing failed: {str(e)}")
    finally:
        if csv_path:
            try:
                os.unlink(csv_path)
            except OSError:
                pass

@app.get("/static/medical_reports_template.csv")
async def serve_csv_template():
    """
    Serve the medical reports CSV template for download.
    """
    from starlette.responses import StreamingResponse
    template_path = None
    try:
        processor = MedicalReportCSVProcessor()

        tmp_dir = tempfile.gettempdir()
        template_path = os.path.join(tmp_dir, f"template_{uuid.uuid4()}.csv")

        await asyncio.to_thread(processor.create_csv_template, template_path, num_sample_rows=3)

        # Read content into memory, then delete the temp file immediately — no threading.Timer needed.
        content = await asyncio.to_thread(lambda: open(template_path, "rb").read())

        headers = {
            "Content-Disposition": 'attachment; filename="medical_reports_template.csv"',
            "Content-Type": "text/csv",
        }

        return StreamingResponse(io.BytesIO(content), headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error serving CSV template")
        raise HTTPException(status_code=500, detail=f"Could not generate CSV template: {str(e)}")
    finally:
        if template_path:
            try:
                os.unlink(template_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# /doctor — infrastructure health check (inspired by `mosaicx doctor`)
# Validates all runtime dependencies without mutating any state.
# ---------------------------------------------------------------------------

@app.get("/doctor")
async def doctor():
    """
    Health check for all OntographRAG infrastructure.

    Checks:
      - Neo4j connectivity and node count
      - Embedding model loads and can embed a test sentence
      - Key environment variables are set
      - PyMuPDF available (PDF parsing)
      - Surya OCR available (optional — scan fallback)
      - LLM provider keys present (OpenAI / Anthropic / OpenRouter)

    Returns a JSON report with pass/warn/fail status per check.
    """
    import importlib
    from datetime import datetime, timezone

    checks: list[dict] = []
    overall = "ok"

    def _add(name: str, status: str, detail: str = ""):
        nonlocal overall
        checks.append({"check": name, "status": status, "detail": detail})
        if status == "fail":
            overall = "fail"
        elif status == "warn" and overall == "ok":
            overall = "warn"

    # 1. Neo4j connectivity
    try:
        _neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        _neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        _neo4j_pw = os.getenv("NEO4J_PASSWORD", "")
        _neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j")
        driver = get_graphDB_driver(_neo4j_uri, _neo4j_user, _neo4j_pw, _neo4j_db)
        with driver.session() as s:
            rec = s.run("MATCH (n) RETURN count(n) AS c").single()
            node_count = rec["c"] if rec else 0
        driver.close()
        _add("neo4j", "ok", f"Connected to {_neo4j_uri} — {node_count} nodes")
    except Exception as e:
        _add("neo4j", "fail", str(e))

    # 2. Embedding model
    try:
        from ontographrag.kg.utils.common_functions import load_embedding_model
        emb_fn, emb_dim = await asyncio.to_thread(load_embedding_model, "sentence_transformers")
        test_vec = await asyncio.to_thread(emb_fn.embed_query, "ontograph health check")
        _add("embedding_model", "ok", f"sentence_transformers dim={emb_dim}, test vector len={len(test_vec)}")
    except Exception as e:
        _add("embedding_model", "fail", str(e))

    # 3. PyMuPDF (PDF parsing)
    try:
        import fitz
        _add("pymupdf", "ok", f"fitz version {fitz.version[0]}")
    except ImportError:
        _add("pymupdf", "fail", "PyMuPDF not installed — PDF ingestion will fail")

    # 4. Surya OCR (optional scan fallback)
    try:
        importlib.import_module("surya")
        _add("surya_ocr", "ok", "surya installed — scan fallback available")
    except ImportError:
        _add("surya_ocr", "warn", "surya not installed — scanned PDFs will use PyMuPDF (may yield sparse text)")

    # 5. LLM provider keys
    key_checks = [
        ("OPENAI_API_KEY", "OpenAI"),
        ("ANTHROPIC_API_KEY", "Anthropic"),
        ("OPENROUTER_API_KEY", "OpenRouter"),
    ]
    any_llm = False
    for env_var, label in key_checks:
        if os.getenv(env_var):
            _add(f"llm_{label.lower()}", "ok", f"{env_var} is set")
            any_llm = True
        else:
            _add(f"llm_{label.lower()}", "warn", f"{env_var} not set")
    if not any_llm:
        # Downgrade to fail if zero LLM keys available
        _add("llm_any", "fail", "No LLM provider key found — KG extraction will fail without a local Ollama model")

    # 6. APP_API_KEY (security)
    if os.getenv("APP_API_KEY"):
        _add("api_key_auth", "ok", "APP_API_KEY set — endpoint auth enabled")
    else:
        _add("api_key_auth", "warn", "APP_API_KEY not set — all endpoints are publicly accessible")

    # 7. ALLOWED_ORIGINS (CORS)
    origins = os.getenv("ALLOWED_ORIGINS", "*")
    if origins == "*":
        _add("cors", "warn", "ALLOWED_ORIGINS=* — CORS is open to all origins (fine for local dev)")
    else:
        _add("cors", "ok", f"ALLOWED_ORIGINS restricted to: {origins}")

    return JSONResponse(content={
        "status": overall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    })
