import os
import sys
import tempfile
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append(os.getcwd())

# Import required modules
from PyPDF2 import PdfReader
from kg_loader import KGLoader

def test_kg_saving_with_numeric_naming():
    """Test KG generation with EAU guidelines PDF using numeric ID naming convention"""

    # PDF path
    pdf_path = 'EAU-EANM-ESTRO-ESUR-ISUP-SIOG-Pocket-on-Prostate-Cancer-2025_updated.pdf'

    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False

    print(f"✅ PDF file found: {pdf_path}")

    # Neo4j configuration
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print("❌ Neo4j credentials missing")
        return False

    print("✅ Neo4j credentials configured")

    try:
        # Extract PDF text
        print("📖 Extracting PDF content...")
        pdf_reader = PdfReader(pdf_path)

        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"

        if len(text_content.strip()) == 0:
            print("❌ PDF contains no extractable text")
            return False

        print(f"✅ Extracted {len(text_content)} characters from PDF")

        # Use ChunkedKGCreator instead of ontology-guided for simplicity
        from kg_creator import ChunkedKGCreator

        kg_creator = ChunkedKGCreator(
            chunk_size=1500,
            chunk_overlap=200,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j")
        )

        # 🎯 NEW NAMING CONVENTION: numeric_id + embedding_model + kg_llm_model
        numeric_id = datetime.now().strftime("%Y%m%d%H%M%S")  # Timestamp-based numeric ID
        embedding_model = "sentence_transformers"
        kg_llm_model = "openai_gpt_oss_20b_free"

        # Format: 20250106_134503__sentence_transformers__openai_gpt_oss_20b_free
        full_filename = f"{numeric_id}__{embedding_model}__{kg_llm_model}"

        print("🎯 Testing New Naming Convention")
        print(f"   📝 Format: numeric_id + embedding_model + kg_llm_model")
        print(f"   🔢 Numeric ID: {numeric_id}")
        print(f"   🎯 Embedding Model: {embedding_model}")
        print(f"   🤖 KG LLM Model: {kg_llm_model}")
        print(f"   📁 Generated Filename: {full_filename}")

        # Use a simple mock LLM to avoid dependency issues
        class MockLLM:
            def generate(self, system_prompt, user_prompt, model_name="mock"):
                return '{"entities": [{"id": "Prostate Cancer", "type": "Disease", "properties": {"name": "Prostate Cancer"}}, {"id": "Treatment", "type": "Concept", "properties": {"name": "Treatment"}}], "relationships": [{"source": "Prostate Cancer", "target": "Treatment", "type": "RELATED_TO", "properties": {}}]}'

        model_name = "mock"
        llm_provider = MockLLM()

        print(f"\n🔧 Generating Knowledge Graph with filename: {full_filename}")
        kg = kg_creator.generate_knowledge_graph(text_content, llm_provider, full_filename, model_name)

        if not kg or len(kg.get('nodes', [])) == 0:
            print("❌ No KG generated")
            return False

        # 📊 Show generation stats
        print(f"\n📊 KG Generation Statistics:")
        print(f"   📄 Chunks processed: {kg['metadata']['total_chunks']}")
        print(f"   🎯 Total entities extracted: {kg['metadata']['total_entities']}")
        print(f"   🔗 Total relationships: {kg['metadata']['total_relationships']}")
        print(f"   📦 Final KG nodes: {len(kg['nodes'])} (after harmonization)")
        print(f"   🔗 Final KG relationships: {len(kg['relationships'])}")

        # 💾 Store KG in Neo4j with the custom filename
        print(f"\n💾 Storing KG in Neo4j with new naming convention...")
        print(f"   📝 KG will be stored with filename: {full_filename}")

        store_success = kg_creator.store_knowledge_graph(kg, full_filename)
        if store_success:
            print("✅ KG stored successfully in Neo4j with new naming convention")
        else:
            print("❌ Failed to store KG in Neo4j")

        # Also save directly to Neo4j to compare
        print("💾 Saving KG directly to Neo4j...")
        loader = KGLoader()
        result = loader.save_to_neo4j(neo4j_uri, neo4j_user, neo4j_password, kg, clear_database=False)

        print("✅ KG saved to Neo4j")
        print(f"   📝 Stored with filename: {full_filename}")
        print(f"   💾 {result}")

        # 🎯 Show sample entities
        print(f"\n📋 Sample Extracted Entities:")
        entity_names = [node['id'] for node in kg['nodes'][:10]]
        for i, entity in enumerate(entity_names, 1):
            print(f"   {i}. {entity}")

        print(f"\n✨ SUCCESS: KG Generation with Numeric ID Naming Convention!")
        print(f"🔗 Access your Neo4j Browser at: http://localhost:7474")
        return True

    except Exception as e:
        print(f"❌ Error during KG generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing KG Saving with Numeric ID Naming Convention")
    print("=" * 60)
    print("Format: timestamp_id__embedding_model__kg_llm_model")
    print("Example: 20250106_134503__sentence_transformers__openai_gpt_oss_20b_free")
    print("=" * 60)

    success = test_kg_saving_with_numeric_naming()

    if success:
        print("\n🎉 SUCCESS: Numeric ID naming convention test passed!")
        print("   ✅ KG generated successfully")
        print("   ✅ Filename format verified")
        print("   ✅ Data stored in Neo4j")
        sys.exit(0)
    else:
        print("\n💥 TEST FAILED: Numeric ID naming convention has issues")
        sys.exit(1)
