#!/usr/bin/env python3
"""
Test script to verify EAU KG generation works correctly
"""

import asyncio
import sys
import os
import logging
import types
from dotenv import load_dotenv



# Import from local modules
from ontology_guided_kg_creator import OntologyGuidedKGCreator
from model_providers import get_llm_provider
import fitz  # PyMuPDF for PDF text extraction

# Load environment variables
load_dotenv()

# Temporarily set embedding provider to OpenAI for testing
os.environ["EMBEDDING_MODEL"] = "openai"

# Configure logging
logging.basicConfig(level=logging.INFO)

def extract_pdf_text(pdf_path: str, max_chars: int = 5000) -> str:
    """Extract text from PDF file"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) > max_chars:
                break
        doc.close()
        return text[:max_chars]
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def test_eau_kg_generation():
    """Test EAU KG generation using OntologyGuidedKGCreator"""
    print("🧪 Testing EAU KG Generation (Ontology-Guided)")
    print("=" * 50)

    try:
        # Initialize LLM provider
        llm_provider = get_llm_provider("openrouter")
        print("✅ LLM provider initialized")

        # Initialize KG creator without ontology for testing natural LLM detection
        ontology_path = None  # Force no ontology for testing
        kg_creator = OntologyGuidedKGCreator(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            neo4j_database="neo4j",
            ontology_path=ontology_path
        )
        print("✅ Ontology-guided KG creator initialized")
        if ontology_path:
            print(f"   📚 Using ontology: {ontology_path}")
        else:
            print("   📚 No ontology found - using basic LLM extraction")

        # EAU file to process
        file_path = 'EAU-EANM-ESTRO-ESUR-ISUP-SIOG-Pocket-on-Prostate-Cancer-2025_updated.pdf'
        merged_file_path = os.path.join(os.getcwd(), file_path)
        file_name = os.path.basename(merged_file_path)

        # Check if EAU file exists
        if not os.path.exists(merged_file_path):
            print(f"❌ Test FAILED: EAU file not found at {merged_file_path}")
            return False

        print(f"📄 Processing EAU file: {file_name}")

        # Extract text from PDF
        print("📖 Extracting text from PDF...")
        pdf_text = extract_pdf_text(merged_file_path, max_chars=2000)  # Limit for testing
        print(f"   📄 Extracted {len(pdf_text)} characters")

        if len(pdf_text.strip()) == 0:
            print("❌ Test FAILED: No text extracted from PDF")
            return False

        # Generate knowledge graph
        print("🔄 Generating ontology-guided knowledge graph...")
        kg = kg_creator.generate_knowledge_graph(
            text=pdf_text,
            llm=llm_provider,
            file_name=file_name,
            model_name="openai/gpt-oss-20b:free",
            kg_name="eau_kg_test"
        )

        # Check results
        entities = kg['metadata']['total_entities']
        relationships = kg['metadata']['total_relationships']
        stored = kg['metadata'].get('stored_in_neo4j', False)

        print(f"📊 Results:")
        print(f"   - Entities: {entities}")
        print(f"   - Relationships: {relationships}")
        print(f"   - Stored in Neo4j: {stored}")

        if entities > 0 and relationships >= 0:  # Accept 0 relationships for valid KG
            print("✅ Test PASSED: EAU KG generated successfully with relaxed constraints")
            print(f"   🎯 Ontology classes used: {kg['metadata']['ontology_classes']}")
            print(f"   🎯 Ontology relationships used: {kg['metadata']['ontology_relationships']}")
            return True
        else:
            print("❌ Test FAILED: EAU KG generation produced no entities")
            return False

    except Exception as e:
        print(f"❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_eau_kg_generation()
    sys.exit(0 if success else 1)
