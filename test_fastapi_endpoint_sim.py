#!/usr/bin/env python3

import tempfile
import uuid
import os
import sys

# Add current directory to path for imports
sys.path.append('.')

# Import what the FastAPI endpoint imports
from model_providers import get_provider as get_llm_provider
from ontology_guided_kg_creator import OntologyGuidedKGCreator

def simulate_fastapi_endpoint():
    """
    Simulate exactly what the FastAPI create_ontology_guided_kg endpoint does
    """
    print("🔬 Simulating FastAPI endpoint execution with ontology...")

    # Step 1: Simulate reading ontology file (like FastAPI does)
    ontology_file_path = "biomedical_ontology.owl"

    if not os.path.exists(ontology_file_path):
        print(f"❌ Ontology file not found: {ontology_file_path}")
        return

    with open(ontology_file_path, "rb") as f:
        ontology_data = f.read()

    print(f"📖 Read {len(ontology_data)} bytes from ontology file")

    # Step 2: Save ontology file temporarily (like FastAPI does)
    tmp_dir = tempfile.gettempdir()
    ontology_filename = f"ontology_{uuid.uuid4()}{os.path.splitext(ontology_file_path)[1]}"
    ontology_path = os.path.join(tmp_dir, ontology_filename)
    with open(ontology_path, "wb") as tmpf:
        tmpf.write(ontology_data)

    print(f"💾 Saved ontology to temp path: {ontology_path}")
    print(f"🧪 File size: {os.path.getsize(ontology_path)} bytes")
    print(f"🧪 File exists: {os.path.exists(ontology_path)}")

    # Step 3: Create OntologyGuidedKGCreator exactly like FastAPI does
    print(f"\n🏗️ Creating OntologyGuidedKGCreator with ontology_path: {ontology_path}")

    try:
        kg_creator = OntologyGuidedKGCreator(
            chunk_size=1500,
            chunk_overlap=200,
            ontology_path=ontology_path,
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            neo4j_database="neo4j",
            embedding_model="openai"
        )

        print(f"✅ OntologyGuidedKGCreator created successfully")
        print(f"🔍 Ontology classes loaded: {len(kg_creator.ontology_classes)}")
        print(f"🔍 Ontology relationships loaded: {len(kg_creator.ontology_relationships)}")

        # Step 4: Check ontology availability (like in generate_knowledge_graph)
        has_ontology = bool(kg_creator.ontology_classes) or bool(kg_creator.ontology_relationships)
        extraction_method = "ontology_guided_llm" if has_ontology else "natural_llm"

        print(f"🔍 Has ontology: {has_ontology}")
        print(f"🔍 Extraction method: {extraction_method}")

        if has_ontology:
            print("✅ Ontology should be available for entity extraction")
        else:
            print("❌ Ontology loading failed - will use basic LLM extraction")

        # Step 5: Test one chunk processing to see the log message
        print("\n🚀 Testing entity extraction...")
        # Get a mock LLM provider
        llm = get_llm_provider("openrouter", "openai/gpt-oss-20b:free")

        # Test text
        test_text = "Prostate cancer is a malignant disease affecting the prostate gland. Symptoms include frequent urination and erectile dysfunction. Treatment options include surgery, radiation therapy, and hormone therapy."

        try:
            # This should trigger the ontology guidance or fallback to basic LLM
            chunk_kg = kg_creator._extract_entities_and_relationships_with_llm(test_text, llm, "openai/gpt-oss-20b:free")

            print(f"✅ Entity extraction completed successfully!")
            print(f"   - Entities found: {len(chunk_kg['entities'])}")
            print(f"   - Relationships found: {len(chunk_kg['relationships'])}")

        except Exception as e:
            print(f"❌ Entity extraction failed: {e}")

    except Exception as e:
        print(f"❌ Failed to create OntologyGuidedKGCreator: {e}")
        import traceback
        traceback.print_exc()

    # Clean up
    if os.path.exists(ontology_path):
        os.remove(ontology_path)
        print("🧹 Cleaned up temp file")

if __name__ == "__main__":
    simulate_fastapi_endpoint()
