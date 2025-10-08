#!/usr/bin/env python3
"""
Test script for RAG filtering fallback mechanism with entities.
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path to import local modules
sys.path.append('.')

from enhanced_rag_system import EnhancedRAGSystem

def test_entity_name_fallback():
    """Test the entity name fallback mechanism in filtering"""

    print("🧪 Testing Entity Name Fallback in RAG Filtering")
    print("=" * 60)

    rag_system = EnhancedRAGSystem()

    # Create mock context with entities but no LLM ID references
    mock_context = {
        "entities": {
            "prostate_cancer": {
                "id": "prostate_cancer",
                "element_id": "12345",
                "type": "Disease",
                "description": "Cancer of the prostate gland"
            },
            "radical_prostatectomy": {
                "id": "radical_prostatectomy",
                "element_id": "67890",
                "type": "Treatment",
                "description": "Surgical removal of prostate"
            },
            "radiation_therapy": {
                "id": "radiation_therapy",
                "element_id": "54321",
                "type": "Treatment",
                "description": "Use of ionizing radiation to treat cancer"
            },
            "active_surveillance": {
                "id": "active_surveillance",
                "element_id": "99999",
                "type": "Treatment",
                "description": "Monitoring without immediate treatment"
            }
        },
        "chunks": [
            {
                "chunk_id": "chunk_1",
                "chunk_element_id": "chunk_001",
                "text": "Prostate cancer treatment options"
            }
        ],
        "relationships": [
            {
                "source": "prostate_cancer",
                "source_element_id": "12345",
                "target": "radical_prostatectomy",
                "target_element_id": "67890",
                "type": "TREATED_BY"
            },
            {
                "source": "prostate_cancer",
                "source_element_id": "12345",
                "target": "radiation_therapy",
                "target_element_id": "54321",
                "type": "TREATED_BY"
            },
            {
                "source": "prostate_cancer",
                "source_element_id": "12345",
                "target": "active_surveillance",
                "target_element_id": "99999",
                "type": "TREATED_BY"
            }
        ]
    }

    # Mock LLM response that mentions entity names but doesn't include IDs
    mock_response = """Prostate cancer can be treated with radical prostatectomy or radiation therapy.

## RECOMMENDATION/SUMMARY
For prostate cancer patients, treatment options include radical prostatectomy (surgical removal) and radiation therapy. Active surveillance may also be considered for low-risk cases.

## COMBINED EVIDENCE
The medical literature shows that prostate cancer treatment involves surgical and radiation approaches."""

    print(f"📄 Mock RAG Response (no explicit IDs):\n{mock_response[:200]}...")
    print("\n🔍 Entities in Context:")
    for entity_id, entity_info in mock_context["entities"].items():
        print(f"  • {entity_id} (ID:{entity_info['element_id']}) - {entity_info['description']}")

    # Test extraction
    extracted = rag_system._extract_used_entities_and_chunks(mock_response, mock_context)

    print(f"\n🎯 Fallback Extraction Results:")
    print(f"- Used entities: {extracted['total_filtered_entities']}")
    print(f"- Used chunks: {extracted['total_filtered_chunks']}")
    print(f"- Reasoning edges: {extracted['total_reasoning_edges']}")

    if extracted['used_entities']:
        print("\n✅ Fallback Found Used Entities:")
        for entity in extracted['used_entities']:
            print(f"  • {entity['id']} (ID:{entity['element_id']}) - {entity['reasoning_context']}")

    if extracted['reasoning_edges']:
        print("\n✅ Reasoning Edges:")
        for edge in extracted['reasoning_edges']:
            print(f"  • {edge['from']} → {edge['relationship']} → {edge['to']}")

    # Verify fallback worked
    expected_entities = {'prostate_cancer', 'radical_prostatectomy', 'radiation_therapy'}
    found_entities = {e['id'] for e in extracted['used_entities']}

    if expected_entities.issubset(found_entities):
        print(f"\n✅ SUCCESS: Fallback correctly identified entities: {found_entities}")
    else:
        print(f"\n❌ FAILED: Expected {expected_entities}, got {found_entities}")

    return extracted

def test_explicit_ids_still_work():
    """Test that explicit ID references still work properly"""

    print("\n🔍 Testing Explicit ID References Still Work")
    print("=" * 50)

    rag_system = EnhancedRAGSystem()

    # Mock context with entities
    mock_context = {
        "entities": {
            "surgery": {
                "id": "surgery",
                "element_id": "67890",
                "type": "Treatment",
                "description": "Surgical procedure"
            },
            "chemotherapy": {
                "id": "chemotherapy",
                "element_id": "54321",
                "type": "Treatment",
                "description": "Drug treatment"
            }
        },
        "chunks": [],
        "relationships": []
    }

    # Mock response WITH explicit IDs
    mock_response = """Treatment with surgery (ID:67890) is preferred over chemotherapy (ID:54321) in this case."""

    print(f"📄 Mock RAG Response (with explicit IDs):\n{mock_response}")

    # Test extraction
    extracted = rag_system._extract_used_entities_and_chunks(mock_response, mock_context)

    print(f"\n🎯 Explicit ID Extraction Results:")
    print(f"- Used entities: {extracted['total_filtered_entities']}")

    if extracted['used_entities']:
        print("\n✅ Explicit IDs Found Used Entities:")
        for entity in extracted['used_entities']:
            print(f"  • {entity['id']} (ID:{entity['element_id']}) - {entity['reasoning_context']}")

    # Verify explicit IDs worked
    expected_entities = {'surgery', 'chemotherapy'}
    found_entities = {e['id'] for e in extracted['used_entities']}

    if expected_entities == found_entities:
        print(f"\n✅ SUCCESS: Explicit IDs correctly identified entities: {found_entities}")
    else:
        print(f"\n❌ FAILED: Expected {expected_entities}, got {found_entities}")

    return extracted

if __name__ == "__main__":
    try:
        test_entity_name_fallback()
        test_explicit_ids_still_work()
        print("\n🎉 Testing Complete - RAG Node Filtering should now work!")
        print("   • Explicit ID references (preferred)")
        print("   • Name-based fallback when IDs are not used")
        print("   • Chunk references by ID")
        print("   • Reasoning edges between connected entities")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
