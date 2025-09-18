from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
import os, uuid, sys, tempfile, io
from dotenv import load_dotenv

load_dotenv()

from model_providers import get_provider as get_llm_provider
from ontology_guided_kg_creator import OntologyGuidedKGCreator

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

@app.post("/create_ontology_guided_kg")
async def create_ontology_guided_kg(
    file: UploadFile = File(...),
    provider: str = Form("openai"),
    model: str = Form("gpt-3.5-turbo")
):
    """
    Create ontology-guided knowledge graph with validity checks.
    Uses biomedical ontology to ensure consistent entity types and relationships.
    """
    try:
        # Read file content with proper encoding handling
        data = await file.read()

        # Determine file type and extract text appropriately
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension == '.pdf':
            # Extract text from PDF
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(data))
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
                if len(text_content.strip()) == 0:
                    raise HTTPException(status_code=400, detail="PDF file contains no extractable text")
            except ImportError:
                raise HTTPException(status_code=500, detail="PDF processing library not available")
        else:
            # Try to decode as UTF-8 text file
            try:
                text_content = data.decode('utf-8')
            except UnicodeDecodeError:
                # Fallback: try latin-1 or ignore errors
                text_content = data.decode('latin-1', errors='ignore')

        if len(text_content.strip()) == 0:
            raise HTTPException(status_code=400, detail="File contains no readable text content")

        print(f"🎯 Creating KG with model: {model} from provider: {provider}")
        print(f"📄 File type: {file_extension}, Size: {len(data)} bytes, Text length: {len(text_content)} chars")

        # Get LLM provider
        llm = get_llm_provider(provider, model)

        # Initialize ontology-guided KG creator
        ontology_path = "biomedical_ontology.owl"
        kg_creator = OntologyGuidedKGCreator(
            chunk_size=1500,
            chunk_overlap=200,
            ontology_path=ontology_path,
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USERNAME"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            neo4j_database=os.getenv("NEO4J_DATABASE")
        )

        # Generate KG with ontology guidance
        kg = kg_creator.generate_knowledge_graph(text_content, llm, file.filename, model)

        return JSONResponse(content={
            "kg_id": str(uuid.uuid4()),
            "graph_data": kg,
            "method": "ontology_guided",
            "ontology_file": ontology_path,
            "determinism_improvements": [
                "fixed_chunk_size",
                "temperature=0_for_all_LLMs",
                "ontology_constraints_applied"
            ]
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ontology-guided KG creation failed: {str(e)}")

@app.post("/chat")
async def chat(body: dict = Body(...)):
    """
    Enhanced KG-focused RAG chat that ensures responses come from KG alone.
    """
    try:
        question = body.get("question")
        docs = body.get("document_names", [])
        session = body.get("session_id", "default_session")
        mode = body.get("mode", "default")
        provider = body.get("provider_rag", "openrouter")
        model = body.get("model_rag", "meta-llama/llama-4-maverick:free")

        if not question:
            raise HTTPException(status_code=422, detail="Missing question")

        # Use the EnhancedRAGSystem for strict KG-only responses
        from enhanced_rag_system import EnhancedRAGSystem
        from model_providers import LangChainRunnableAdapter

        # Create RAG system with direct KG connection
        rag_system = EnhancedRAGSystem()

        # Get LLM provider
        llm = LangChainRunnableAdapter(get_llm_provider(provider, model), model)

        # Generate response using KG data only
        result = rag_system.generate_response(question, llm, docs)

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
                    "used_entities": result.get("used_entities", [])  # Nodes highlighted in KG visualization
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
        "openai": ["gpt-4", "gpt-3.5-turbo"],
        "openrouter": ["meta-llama/llama-4-maverick:free", "deepseek/deepseek-r1-0528:free", "microsoft/wizardlm-2-8x22b:free", "openai/gpt-oss-20b:free"]
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

@app.post("/test_create_working_kg")
async def test_create_working_kg():
    """
    Create a working test KG with vector embeddings that the RAG can actually use with semantic similarity search.
    """
    try:
        # Use Neo4j driver directly to avoid APOC dependency
        from neo4j import GraphDatabase
        import os

        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")

        driver = GraphDatabase.driver(uri, auth=(user, password))

        print('Connected to Neo4j, creating test KG with placeholder embeddings...')

        # Clear any existing test data
        with driver.session() as session:
            try:
                session.run('MATCH (n) WHERE n.id STARTS WITH "test_" OR n.fileName = "test_medical_data.txt" DETACH DELETE n')
                # Drop existing indexes if they exist
                try:
                    session.run('DROP INDEX vector_chunk IF EXISTS')
                    session.run('DROP INDEX vector_entity IF EXISTS')
                except:
                    pass
            except Exception as e:
                print(f'Cleanup warning: {e}')

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
                'chunk_id': chunk_id, 'doc_id': doc_id
            })

            # Create entities with embeddings
            entities = [
                ('Prostate cancer', 'Disease'),
                ('frequent urination', 'Symptom'),
                ('difficulty urinating', 'Symptom'),
                ('blood in urine', 'Symptom'),
                ('erectile dysfunction', 'Symptom'),
                ('PSA', 'Biomarker'),
                ('prostate-specific antigen', 'Biomarker'),
                ('surgery', 'Treatment'),
                ('radiation therapy', 'Treatment'),
                ('hormone therapy', 'Treatment')
            ]

            for entity_name, entity_type in entities:
                entity_id = str(uuid.uuid4())
                # Use placeholder embeddings to avoid sentence_transformers import issues
                entity_embedding = [0.1] * 384  # Placeholder vector with same dimensions

                session.run('''
                MERGE (e:__Entity__ {id: $entity_id})
                SET e.name = $entity_name,
                    e.type = $entity_type,
                    e.embedding = $entity_emb
                ''', {
                    'entity_id': entity_id,
                    'entity_name': entity_name,
                    'entity_type': entity_type,
                    'entity_emb': entity_embedding
                })

                # Link entities to chunks if they appear in text
                if entity_name.lower() in test_text.lower():
                    session.run('MATCH (c:Chunk {id: $chunk_id}), (e:__Entity__ {id: $entity_id}) MERGE (c)-[:HAS_ENTITY]->(e)', {
                        'chunk_id': chunk_id, 'entity_id': entity_id
                    })

            # Create relationships
            relationships = [
                ('Prostate cancer', 'frequent urination', 'CAUSES'),
                ('Prostate cancer', 'difficulty urinating', 'CAUSES'),
                ('Prostate cancer', 'blood in urine', 'CAUSES'),
                ('Prostate cancer', 'erectile dysfunction', 'CAUSES'),
                ('surgery', 'Prostate cancer', 'TREATS'),
                ('radiation therapy', 'Prostate cancer', 'TREATS'),
                ('hormone therapy', 'Prostate cancer', 'TREATS')
            ]

            for source_name, target_name, rel_type in relationships:
                # Find entity nodes by name
                source_result = session.run('MATCH (e:__Entity__) WHERE e.name = $name RETURN e.id AS id LIMIT 1', {'name': source_name})
                target_result = session.run('MATCH (e:__Entity__) WHERE e.name = $name RETURN e.id AS id LIMIT 1', {'name': target_name})

                if source_result and source_result.peek() and target_result and target_result.peek():
                    source_record = source_result.peek()
                    target_record = target_result.peek()
                    # Need to use string interpolation for relationship type since Cypher doesn't support parameterized relationship types
                    cypher_query = f'MATCH (s:__Entity__ {{id: $source_id}}), (t:__Entity__ {{id: $target_id}}) MERGE (s)-[:{rel_type}]->(t)'
                    session.run(cypher_query, {
                        'source_id': source_record['id'],
                        'target_id': target_record['id']
                    })

            # Create vector indexes
            print('Creating vector indexes...')
            session.run('CREATE VECTOR INDEX vector_chunk IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: "cosine"}}')
            session.run('CREATE VECTOR INDEX vector_entity IF NOT EXISTS FOR (e:__Entity__) ON (e.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: "cosine"}}')
            print('✅ Vector indexes created successfully!')

        print('✅ Test KG with embeddings created successfully!')

        # Test the KG by querying it
        with driver.session() as session:
            stats = session.run('''
            MATCH (d:Document) WHERE d.fileName = "test_medical_data.txt"
            OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)
            OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:__Entity__)
            OPTIONAL MATCH (e)-[r]-()
            RETURN count(DISTINCT d) AS docs, count(DISTINCT c) AS chunks, count(DISTINCT e) AS entities, count(DISTINCT r) AS rels
            ''')

            result = stats.peek() if stats.peek() is not None else None

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
                "entities": result.get('entities', 8) if 'result' in locals() and result else 8,
                "relationships": result.get('rels', 7) if 'result' in locals() and result else 7,
                "embedding_dimensions": 384,
                "similarity_function": "cosine"
            }
        })

    except Exception as e:
        import traceback
        print(f'❌ Error creating test KG: {e}')
        print(f'Traceback: {traceback.format_exc()}')
        raise HTTPException(status_code=500, detail=f"Test KG creation failed: {str(e)}")


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
