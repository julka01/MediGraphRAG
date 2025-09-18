#!/usr/bin/env python3
"""
Test KG generation using OpenRouter model to isolate the 502 error issue.
"""

import os
import sys
import json
from pathlib import Path

# Add the project root to Python path
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / 'llm-graph-builder/backend/src'))

from dotenv import load_dotenv
from ontology_guided_kg_creator import OntologyGuidedKGCreator
from model_providers import get_llm_provider

# Load environment variables
load_dotenv()

def test_openrouter_kg_generation():
    """Test KG generation using OpenRouter model"""

    print("🔧 Testing OpenRouter KG Generation...")
    print("=" * 50)

    # Check API key
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    if not openrouter_key:
        print("❌ ERROR: OPENROUTER_API_KEY not found in environment")
        return False

    print(f"✅ OpenRouter API Key found: {openrouter_key[:10]}...")

    # Get ontology path
    ontology_path = ROOT_DIR / "biomedical_ontology.owl"
    if not ontology_path.exists():
        print(f"❌ ERROR: Ontology file not found: {ontology_path}")
        return False

    print(f"✅ Ontology file found: {ontology_path}")

    # Get test text
    test_file_path = ROOT_DIR / "prostate_cancer_guidelines.txt"
    if not test_file_path.exists():
        print(f"❌ ERROR: Test file not found: {test_file_path}")
        return False

    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_text = f.read()

        print(f"✅ Test file loaded: {len(test_text)} characters")

        # Test LLM provider loading first
        print("\n🔍 Testing LLM provider setup...")
        try:
            # Manual configuration for OpenRouter
            model_config = "meta-llama/llama-4-maverick:free"
            llm_provider = get_llm_provider("openrouter", model_config)
            if llm_provider:
                print("✅ OpenRouter LLM provider loaded successfully")
            else:
                print("❌ ERROR: Failed to load OpenRouter LLM provider")
                return False

        except Exception as e:
            print(f"❌ ERROR: LLM provider setup failed: {e}")
            print("This could be due to OpenRouter service issues or configuration problems")
            return False

        print("\n🚀 Starting KG generation...")

        # Initialize KG creator
        kg_creator = OntologyGuidedKGCreator(
            chunk_size=1500,
            chunk_overlap=200,
            neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            neo4j_user=os.getenv('NEO4J_USERNAME', 'neo4j'),
            neo4j_password=os.getenv('NEO4J_PASSWORD', 'password'),
            neo4j_database=os.getenv('NEO4J_DATABASE', 'neo4j'),
            embedding_model="openai",
            ontology_path=str(ontology_path)
        )

        print(f"✅ KG Creator initialized with ontology: {len(kg_creator.ontology_classes)} classes")

        # Generate the KG
        try:
            kg_result = kg_creator.generate_knowledge_graph(
                text=test_text,
                llm=llm_provider,
                file_name="OpenRouter-Test-KG",
                model_name=model_config
            )

            print("✅ SUCCESS: KG generation completed!")
            print(f"   - Nodes: {kg_result.get('metadata', {}).get('total_entities', 0)}")
            print(f"   - Relationships: {kg_result.get('metadata', {}).get('total_relationships', 0)}")
            print(f"   - Chunks: {kg_result.get('metadata', {}).get('total_chunks', 0)}")

            return True

        except Exception as e:
            print(f"❌ ERROR during KG generation: {e}")
            print(f"   Error type: {type(e).__name__}")
            if "502" in str(e):
                print("   💡 This is a server connectivity issue with OpenRouter")
                print("   💡 Try switching to OpenAI model in the UI")
            elif "rate limit" in str(e).lower():
                print("   ⚡ This is a rate limiting issue")
                print("   💡 Wait a few minutes and try again")
            elif "authentication" in str(e).lower():
                print("   🔐 This is an API key authentication issue")
                print("   💡 Check OPENROUTER_API_KEY in .env file")
            return False

    except Exception as e:
        print(f"❌ ERROR during setup: {e}")
        return False

if __name__ == "__main__":
    print("🧪 OpenRouter KG Generation Test")
    print("=" * 50)

    success = test_openrouter_kg_generation()

    print("\n" + "=" * 50)
    if success:
        print("🎉 TEST PASSED: OpenRouter KG generation working!")
        print("✅ The issue has been resolved - proceed with UI testing")
    else:
        print("⚠️  TEST FAILED: OpenRouter KG generation issue detected")
        print("💡 Try switching to OpenAI model in the UI")
        print("      ./start_server.py && navigate to http://localhost:8000")
        print("      Select OpenAI from the KG provider dropdown")
        print("      Upload file and create KG")
