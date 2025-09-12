import os
import pytest
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chunked_kg_creator import ChunkedKGCreator
from model_providers import get_llm_provider

def test_generate_kg_with_openrouter(monkeypatch):
    # Mock the LLM call to return valid JSON
    mock_response = json.dumps({
        "nodes": [
            {"id": 1, "label": "Disease", "properties": {"name": "Diabetes Mellitus"}},
            {"id": 2, "label": "Medication", "properties": {"name": "Metformin"}}
        ],
        "relationships": [
            {"from": 2, "to": 1, "type": "TREATS", "properties": {}}
        ]
    })
    
    # Instead of a class, create a simple callable that returns the mock response
    def mock_llm(input_dict):
        return mock_response
    
    # Prepare test input text
    test_text = (
        "Patient diagnosed with diabetes mellitus. "
        "Metformin is used to treat diabetes. "
        "Hypertension is a common comorbidity."
    )

    # Initialize KG creator
    kg_creator = ChunkedKGCreator()

    # Generate knowledge graph
    kg = kg_creator.generate_knowledge_graph(test_text, mock_llm)

    # Assertions
    assert kg is not None
    assert len(kg["nodes"]) == 2
    assert kg["nodes"][0]["properties"]["name"] == "Diabetes Mellitus"
    assert kg["nodes"][1]["properties"]["name"] == "Metformin"
    assert len(kg["relationships"]) == 1
    assert kg["relationships"][0]["type"] == "TREATS"
    assert len(kg["relationships"]) > 0
    
    # Print output for manual inspection
    print("Generated KG nodes:", kg["nodes"])
    print("Generated KG relationships:", kg["relationships"])
