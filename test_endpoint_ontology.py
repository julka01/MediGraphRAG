#!/usr/bin/env python3
"""
Test ontology loading via the API endpoint to debug the issue.
This will help identify where the ontology file is being lost or not processed correctly.
"""

import os
import sys
import json
import requests
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_ontology_endpoint():
    """Test the ontology-guided KG creation endpoint with an ontology file."""
    print("🔍 Testing ontology loading via API endpoint...")

    # Use the biomedical ontology if it exists
    ontology_files = ["biomedical_ontology.owl", "ProstateCancerOntology.owl"]
    ontology_path = None

    for filename in ontology_files:
        potential_path = project_root / filename
        if potential_path.exists():
            ontology_path = potential_path
            print(f"📚 Found ontology file: {ontology_path}")
            break

    if not ontology_path:
        print("❌ No ontology file found for testing")
        return False

    # Create a simple test text file
    test_text = """
    Prostate cancer is a malignant tumor that develops in the prostate gland.
    Common symptoms include frequent urination, difficulty urinating, and blood in urine.
    Treatment options include radical prostatectomy, radiation therapy, and hormone therapy.
    """

    # Save test text to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_text)
        text_file_path = f.name

    try:
        # Prepare the request
        url = "http://localhost:8000/create_ontology_guided_kg"

        with open(text_file_path, 'rb') as text_file, open(ontology_path, 'rb') as ont_file:
            files = {
                'file': ('test_document.txt', text_file, 'text/plain'),
                'ontology_file': ('ontology.owl', ont_file, 'application/owl+xml')
            }

            data = {
                'provider': 'openrouter',
                'model': 'openai/gpt-oss-20b:free',
                'embedding_model': 'sentence_transformers',
                'kg_name': 'test_ontology_debug'
            }

            print(f"📤 Sending request to {url}")
            print("🔍 Request parameters:")
            print(f"   - Text file: test_document.txt")
            print(f"   - Ontology file: {ontology_path.name}")
            print(f"   - Model: {data['model']}")
            print(f"   - KG name: {data['kg_name']}")

            response = requests.post(url, files=files, data=data)

            print(f"📥 Response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print("✅ Request successful!")
                print(f"   - Method returned: {result.get('method', 'unknown')}")
                print(f"   - Ontology file reported: {result.get('ontology_file', 'none')}")

                # Check if ontology_guided was actually used
                if result.get('method') == 'ontology_guided':
                    print("✅ Ontology-guided KG creation was used!")
                else:
                    print("❌ Fell back to basic LLM extraction despite ontology provided")

                # Show graph data summary
                graph_data = result.get('graph_data', {})
                metadata = graph_data.get('metadata', {})
                print(f"   - Entities: {metadata.get('total_entities', 0)}")
                print(f"   - Relationships: {metadata.get('total_relationships', 0)}")
                print(f"   - Ontology classes: {metadata.get('ontology_classes', 0)}")
                print(f"   - Ontology relationships: {metadata.get('ontology_relationships', 0)}")

                return True
            else:
                print(f"❌ Request failed: {response.text}")
                return False

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

    finally:
        # Clean up temporary file
        try:
            os.unlink(text_file_path)
        except:
            pass

if __name__ == "__main__":
    success = test_ontology_endpoint()
    if success:
        print("\n✅ Ontology endpoint test completed successfully")
    else:
        print("\n❌ Ontology endpoint test failed")
        sys.exit(1)
