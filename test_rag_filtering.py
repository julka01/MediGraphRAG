#!/usr/bin/env python3
"""
Test script for RAG filtering with real KG data.
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path to import local modules
sys.path.append('.')

from enhanced_rag_system import EnhancedRAGSystem
from model_providers import LangChainRunnableAdapter, get_provider

def test_actual_rag_filtering():
    """Test RAG filtering with actual KG and LLM responses"""

    print("🔍 Testing RAG Node Filtering with Real KG")

    # Initialize RAG system
    rag_system = EnhancedRAGSystem()

    # Get LLM provider
    llm = LangChainRunnableAdapter(get_provider("openrouter", "openai/gpt-oss-20b:free"), "openai/gpt-oss-20b:free")

    # Test question
    question = "What are the treatment options for prostate cancer?"

    print(f"❓ Question: {question}")

    # Generate RAG response
    result = rag_system.generate_response(question, llm, top_k=5)

    print("\n📄 RAG Response:")
    print(result["response"][:500] + "...")

    print(f"\n📊 Context Stats:")
    print(f"- Chunks retrieved: {len(result['context']['chunks'])}")
    print(f"- Entities in context: {result['context']['entity_count']}")
    print(f"- Relationships in context: {result['context']['relationship_count']}")

    print(f"\n🎯 Filtered Results:")
    print(f"- Used entities: {len(result.get('used_entities', []))}")
    print(f"- Used chunks: {len(result.get('used_chunks', []))}")
    print(f"- Reasoning edges: {len(result.get('reasoning_edges', []))}")

    if result.get('used_entities'):
        print("\n✅ Used Entities:")
        for entity in result['used_entities']:
            print(f"  • {entity['id']} (ID:{entity['element_id']}) - {entity['type']}")

    if result.get('used_chunks'):
        print("\n✅ Used Chunks:")
        for chunk in result['used_chunks']:
            print(f"  • {chunk['id']} (ID:{chunk['element_id']})")

    if result.get('reasoning_edges'):
        print("\n✅ Reasoning Edges:")
        for edge in result['reasoning_edges']:
            print(f"  • {edge['from']} → {edge['relationship']} → {edge['to']}")

    return result

def test_extraction_directly():
    """Test the extraction function directly with a sample response"""

    print("\n🧪 Testing Extraction Function Directly")
    print("=" * 50)

    rag_system = EnhancedRAGSystem()
    llm = LangChainRunnableAdapter(get_provider("openrouter", "openai/gpt-oss-20b:free"), "openai/gpt-oss-20b:free")

    # Get some context first
    context = rag_system.get_rag_context("prostate cancer treatments", top_k=3)

    if not context.get('chunks'):
        print("❌ No context data available")
        return

    print(f"Context has {context['entity_count']} entities and {len(context['chunks'])} chunks")

    # Create a mock response that follows the required format
    mock_response = """Based on the knowledge graph, prostate cancer (ID:12345) can be treated with surgery (ID:67890) and radiation (ID:54321).

## RECOMMENDATION/SUMMARY ##
Prostate cancer treatment options include surgical intervention and radiation therapy, which show good outcomes for patients.

## COMBINED EVIDENCE ##
The knowledge graph contains information about prostate cancer treatments including surgery (ID:67890) and radiation therapy (ID:54321) as mentioned in Chunk 1 (ID:chunk_001).

## REASONING PATH ##
Prostate cancer (ID:12345) is treated by surgery (ID:67890) → Surgery involves removing the prostate gland through radical prostatectomy (ID:chunk_002)."""

    print(f"\nMock RAG Response:\n{mock_response[:300]}...")

    # Test extraction
    extracted = rag_system._extract_used_entities_and_chunks(mock_response, context)

    print(f"\n🎯 Extracted Results:")
    print(f"- Used entities: {extracted['total_filtered_entities']}")
    print(f"- Used chunks: {extracted['total_filtered_chunks']}")
    print(f"- Reasoning edges: {extracted['total_reasoning_edges']}")

    if extracted['used_entities']:
        print("\n✅ Used Entities:")
        for entity in extracted['used_entities']:
            print(f"  • {entity['id']} (ID:{entity['element_id']}) - {entity['type']}")

    if extracted['used_chunks']:
        print("\n✅ Used Chunks:")
        for chunk in extracted['used_chunks']:
            print(f"  • {chunk['id']} (ID:{chunk['element_id']})")

    return extracted

def test_frontend_format():
    """Test the exact format expected by the frontend"""

    print("\n🖥️ Testing Frontend Format")
    print("=" * 50)

    # Simulate what should be returned to frontend
    mock_result = {
        "response": "Prostate cancer (ID:12345) can be treated with surgery (ID:67890).",
        "context": {
            "entities": {
                "prostate_cancer": {"id": "prostate_cancer", "element_id": "12345", "type": "Disease"},
                "surgery": {"id": "surgery", "element_id": "67890", "type": "Treatment"}
            }
        },
        "used_entities": [
            {"id": "prostate_cancer", "element_id": "12345", "type": "Disease"},
            {"id": "surgery", "element_id": "67890", "type": "Treatment"}
        ],
        "entities": ["prostate_cancer", "surgery"]
    }

    print("✅ Mock response that should highlight nodes in frontend:")
    print(f"- response.info.entities.used_entities: {mock_result['used_entities']}")
    print(f"- response.info.entities.entityids: {mock_result['entities']}")

    return mock_result

if __name__ == "__main__":
    try:
        test_actual_rag_filtering()
        test_extraction_directly()
        test_frontend_format()
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
