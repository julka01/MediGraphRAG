import sys, os
import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import app

client = TestClient(app)

def test_load_kg_from_file_missing():
    # Missing file should yield 422 Unprocessable Entity
    response = client.post("/load_kg_from_file", data={})
    assert response.status_code == 422

def test_load_kg_from_file_success_txt():
    # Use a small sample text file to create a KG
    sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_biomedical.txt"))
    assert os.path.exists(sample_path), f"Sample file not found at {sample_path}"
    with open(sample_path, "rb") as f:
        files = {"file": ("test_biomedical.txt", f, "text/plain")}
        data = {"provider": "openai", "model": "gpt-3.5-turbo"}
        response = client.post("/load_kg_from_file", files=files, data=data)
    # Expect OK and valid structure
    assert response.status_code == 200
    payload = response.json()
    assert "graph_data" in payload, "Response missing 'graph_data'"
    graph_data = payload["graph_data"]
    # Nodes and relationships should be lists (even if empty)
    assert isinstance(graph_data.get("nodes", []), list)
    assert isinstance(graph_data.get("relationships", []), list)
