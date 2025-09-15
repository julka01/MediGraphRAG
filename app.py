from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
import os, uuid, sys, tempfile
from dotenv import load_dotenv

load_dotenv()

from model_providers import get_provider as get_llm_provider

# Add llm-graph-builder backend to path so `src` package resolves
backend_root = os.path.join(os.path.dirname(__file__), "llm-graph-builder", "backend")
sys.path.insert(0, backend_root)

# Retain actual langchain_experimental if available
import importlib
if "langchain_experimental" not in sys.modules:
    importlib.import_module("langchain_experimental")

# Core graph imports moved inside endpoint functions

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
def check_neo4j_connection():
    from src.graph_query import get_graphDB_driver
    try:
        try:
            driver = get_graphDB_driver(
                os.getenv("NEO4J_URI"),
                os.getenv("NEO4J_USERNAME"),
                os.getenv("NEO4J_PASSWORD"),
                os.getenv("NEO4J_DATABASE"),
            )
        except Exception:
            return JSONResponse(content={"kg_id": str(uuid.uuid4()), "graph_data": graph_data})
        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            session.run("RETURN 1").single()
    except Exception as e:
        import sys
        print(f"Neo4j health check failed: {e}", file=sys.stderr)
        raise RuntimeError("Neo4j connection failed; check NEO4J_URI, credentials, and that the server is running")

@app.get("/health/neo4j")
def neo4j_health():
    from src.graph_query import get_graphDB_driver
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
    return FileResponse("static/index.html")

@app.post("/load_kg_from_file")
async def load_kg_from_file(
    file: UploadFile = File(...),
    provider: str = Form("openai"),
    model: str = Form("gpt-3.5-turbo")
):
    """
    Return the full raw KG from llm-graph-builder (no ontology filtering).
    """
    try:
        from src.main import extract_graph_from_file_local_file
        from src.graph_query import get_graphDB_driver
        data = await file.read()
        tmp_dir = tempfile.gettempdir()
        file_path = os.path.join(tmp_dir, file.filename)
        with open(file_path, "wb") as tmpf:
            tmpf.write(data)
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
    except (ImportError, ModuleNotFoundError) as e:
        return JSONResponse(content=jsonable_encoder({"kg_id": str(uuid.uuid4()), "graph_data": {"nodes": [], "relationships": []}}))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_graph")
async def extract_graph(
    file: UploadFile = File(...),
    provider: str = Form("openai"),
    model: str = Form("gpt-3.5-turbo")
):
    """
    Return raw graph JSON without ontology post-processing.
    """
    from src.main import extract_graph_from_file_local_file
    try:
        data = await file.read()
        tmp_dir = tempfile.gettempdir()
        file_path = os.path.join(tmp_dir, file.filename)
        with open(file_path, "wb") as tmpf:
            tmpf.write(data)
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

@app.post("/chat")
async def chat(body: dict = Body(...)):
    """
    Delegate chat to llm-graph-builder QA_RAG (raw graph-based QA).
    """
    from src.graph_query import get_graphDB_driver
    from src.QA_integration import QA_RAG
    try:
        question = body.get("question")
        docs = body.get("document_names", [])
        session = body.get("session_id")
        mode = body.get("mode", "default")
        provider = body.get("provider_rag", "openrouter")
        model = body.get("model_rag", "meta-llama/llama-4-maverick:free")
        if not question:
            raise HTTPException(status_code=422, detail="Missing question")
        driver = get_graphDB_driver(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD"),
            os.getenv("NEO4J_DATABASE")
        )
        from model_providers import LangChainRunnableAdapter
        llm = LangChainRunnableAdapter(get_llm_provider(provider, model), model)
        result = QA_RAG(driver, llm, question, docs, session, mode)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{provider}")
def list_models(provider: str):
    """
    Return available models for provider.
    """
    model_map = {
        "openai": ["gpt-4", "gpt-3.5-turbo"],
        "openrouter": ["meta-llama/llama-4-maverick:free", "deepseek/deepseek-r1-0528:free"]
    }
    return {"models": model_map.get(provider.lower(), [])}

@app.get("/neo4j/default_credentials")
def default_credentials():
    """
    Return Neo4j credentials for frontend.
    """
    return {
        "uri": os.getenv("NEO4J_URI"),
        "user": os.getenv("NEO4J_USERNAME"),
        "database": os.getenv("NEO4J_DATABASE")
    }

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
    from src.graph_query import get_graphDB_driver
    try:
        driver = get_graphDB_driver(uri, user, password, os.getenv("NEO4J_DATABASE"))
        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            total_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            # Determine node query
            if load_complete:
                node_query = "MATCH (n) RETURN n"
            elif sample_mode:
                node_query = f"MATCH (n) RETURN n ORDER BY rand() LIMIT {limit}"
            else:
                node_query = f"MATCH (n) RETURN n LIMIT {limit}"
            # Apply label filter if provided
            if kg_label:
                node_query = node_query.replace("MATCH (n)", f"MATCH (n:{kg_label})")
            nodes = [
                {"id": record["n"].id, "labels": list(record["n"].labels), "properties": dict(record["n"])}
                for record in session.run(node_query)
            ]
            relationships = [
                {
                    "id": record["r"].id,
                    "type": record["r"].type,
                    "start": record["r"].start_node.id,
                    "end": record["r"].end_node.id,
                    "properties": dict(record["r"])
                }
                for record in session.run("MATCH (n)-[r]->(m) RETURN r")
            ]
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
            "graph_data": {"nodes": nodes, "relationships": relationships},
            "stats": stats
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
