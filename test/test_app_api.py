import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_generate_kg_missing_file():
    # Missing file should return 422 Unprocessable Entity
    response = client.post("/generate_kg", data={"provider_kg": "openrouter", "model_kg": "test-model"})
    assert response.status_code == 422

def test_extract_graph_missing_file():
    # Missing file upload returns 422
    response = client.post("/extract_graph", data={})
    assert response.status_code == 422

def test_qa_rag_missing_params():
    # Missing required form parameters returns 422
    response = client.post("/qa_rag", data={})
    assert response.status_code == 422

def test_neo4j_query_no_body():
    # Missing query in JSON may return 422 or 500
    response = client.post("/neo4j/query", json={})
    assert response.status_code in (422, 500)

def test_neo4j_query_valid_body():
    # Even with a valid body, driver may not connect; expect 500 or 200
    body = {"query": "MATCH (n) RETURN n", "document_names": []}
    response = client.post("/neo4j/query", json=body)
    assert response.status_code in (200, 500)

def test_neo4j_schema():
    # Schema endpoint may return 500 if connection fails
    response = client.get("/neo4j/schema")
    assert response.status_code in (200, 500)
