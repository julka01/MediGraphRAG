#!/usr/bin/env python3
"""
Test script for node filtering during RAG.
Tests that only nodes and chunks referenced in the RAG answer are used in filtered visualization.
"""

import re
from typing import Dict, Any, List

def extract_used_entities_and_chunks(response: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract entities and chunks that are actually mentioned in the RAG answer,
    focusing only on nodes that are referenced in the response content
    """
    used_entities = []
    used_chunks = []
    reasoning_edges = []

    try:
        context_entities = context.get("entities", {})
        context_chunks = context.get("chunks", [])
        context_relationships = context.get("relationships", [])

        # Find all element IDs referenced in the response text (ID:...)
        entity_id_pattern = r'\(ID:([^)]+)\)'
        mentioned_element_ids = set(re.findall(entity_id_pattern, response))

        # Also extract chunk IDs if they appear in the same format
        chunk_id_pattern = r'Chunk\s*(\d+)\s*\(ID:\s*([^)]+)\)'
        chunk_matches = re.findall(chunk_id_pattern, response)

        # Collect chunk IDs from matches
        mentioned_chunk_ids = set()
        for match in chunk_matches:
            chunk_num, chunk_id = match
            mentioned_chunk_ids.add(chunk_id)

        # Find entities that are actually mentioned in the response
        for entity_id, entity_info in context_entities.items():
            element_id = entity_info.get('element_id', '')

            # IMPORTANT: Only include entities that are EXPLICITLY referenced by their element_id in the response
            # This prevents false positives from entities that happen to share names with words in the response
            if element_id in mentioned_element_ids:
                used_entities.append({
                    "id": entity_id,
                    "element_id": element_id,
                    "type": entity_info.get("type", "Unknown"),
                    "description": entity_info.get("description", ""),
                    "reasoning_context": "explicitly referenced by ID in RAG response"
                })

        # Find chunks that are actually mentioned in the response
        for chunk in context_chunks:
            chunk_id = chunk.get("chunk_id", "")
            chunk_element_id = chunk.get("chunk_element_id", "")

            # Check if this chunk was mentioned
            chunk_mentioned = False

            if chunk_id in mentioned_chunk_ids:
                chunk_mentioned = True
            elif chunk_element_id in mentioned_element_ids:
                chunk_mentioned = True

            if chunk_mentioned:
                used_chunks.append({
                    "id": chunk_id,
                    "element_id": chunk_element_id,
                    "text": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", ""),
                    "reasoning_context": "directly referenced chunk"
                })

        # Find relationships between the used entities (reasoning edges)
        used_entity_element_ids = {e['element_id'] for e in used_entities}

        for rel in context_relationships:
            source_id = rel.get("source_element_id", "")
            target_id = rel.get("target_element_id", "")

            # Only include edges where both nodes are in our filtered set
            if source_id in used_entity_element_ids and target_id in used_entity_element_ids:
                reasoning_edges.append({
                    "from": source_id,
                    "to": target_id,
                    "relationship": rel.get("type", "CONNECTED_TO"),
                    "reasoning_context": "connects referenced entities"
                })

        return {
            "used_entities": used_entities,
            "used_chunks": used_chunks,
            "reasoning_edges": reasoning_edges,
            "total_filtered_entities": len(used_entities),
            "total_filtered_chunks": len(used_chunks),
            "total_reasoning_edges": len(reasoning_edges)
        }

    except Exception as e:
        return {
            "used_entities": [],
            "used_chunks": [],
            "reasoning_edges": [],
            "error": str(e)
        }

def test_node_filtering():
    """Test the node filtering logic with mock data"""

    print("🧪 Testing RAG Node Filtering Logic")
    print("=" * 50)

    # Mock context data representing a knowledge graph
    mock_context = {
        'entities': {
            'prostate_cancer': {
                'id': 'prostate_cancer',
                'element_id': '12345',
                'type': 'Disease',
                'description': 'Cancer of the prostate gland'
            },
            'surgery': {
                'id': 'surgery',
                'element_id': '67890',
                'type': 'Treatment',
                'description': 'Surgical intervention'
            },
            'radiation': {
                'id': 'radiation',
                'element_id': '54321',
                'type': 'Treatment',
                'description': 'Radiation therapy'
            },
            'weather': {
                'id': 'weather',
                'element_id': '99999',
                'type': 'Irrelevant',
                'description': 'Weather conditions'
            }
        },
        'chunks': [
            {
                'chunk_id': 'chunk_001',
                'chunk_element_id': 'chunk_001',
                'text': 'Prostate cancer treatment options include surgery and radiation therapy.'
            },
            {
                'chunk_id': 'chunk_002',
                'chunk_element_id': 'chunk_002',
                'text': 'Surgery involves removing the prostate gland through radical prostatectomy.'
            },
            {
                'chunk_id': 'chunk_003',
                'chunk_element_id': 'chunk_003',
                'text': 'Weather today is sunny with clear skies.'
            }
        ],
        'relationships': [
            {
                'source': 'prostate_cancer',
                'target': 'surgery',
                'source_element_id': '12345',
                'target_element_id': '67890',
                'type': 'TREATED_BY',
                'element_id': 'rel1'
            },
            {
                'source': 'prostate_cancer',
                'target': 'radiation',
                'source_element_id': '12345',
                'target_element_id': '54321',
                'type': 'TREATED_BY',
                'element_id': 'rel2'
            },
            {
                'source': 'weather',
                'target': 'surgery',
                'source_element_id': '99999',
                'target_element_id': '67890',
                'type': 'UNRELATED_TO',
                'element_id': 'rel3'
            }
        ]
    }

    # Test Case 1: RAG response mentions specific medical entities
    rag_response_1 = """Based on the knowledge graph, prostate cancer (ID:12345) can be treated with surgery (ID:67890). This treatment approach shows good outcomes for patients."""

    print("Test Case 1: Medical treatment query")
    print(f"RAG Response: {rag_response_1}")
    print("-" * 50)

    result1 = extract_used_entities_and_chunks(rag_response_1, mock_context)

    print("✅ Filtered Entities (should only include prostate_cancer and surgery):")
    for entity in result1['used_entities']:
        print(f"   • {entity['id']} (ID:{entity['element_id']}) - {entity['type']}")
    print()

    print("✅ Filtered Chunks (should be empty - no direct chunk references):")
    for chunk in result1['used_chunks']:
        print(f"   • {chunk['id']} (ID:{chunk['element_id']})")
    print()

    print("✅ Reasoning Edges (should only include treatments for prostate cancer):")
    for edge in result1['reasoning_edges']:
        print(f"   • {edge['from']} -> {edge['relationship']} -> {edge['to']}")
    print()

    # Test Case 2: RAG response mentions chunks directly
    rag_response_2 = """The information in Chunk 1 (ID:chunk_001) discusses prostate cancer treatment options."""

    print("Test Case 2: Direct chunk reference")
    print(f"RAG Response: {rag_response_2}")
    print("-" * 50)

    result2 = extract_used_entities_and_chunks(rag_response_2, mock_context)

    print("✅ Filtered Entities (should be empty - no entity references):")
    for entity in result2['used_entities']:
        print(f"   • {entity['id']} (ID:{entity['element_id']}) - {entity['type']}")
    print()

    print("✅ Filtered Chunks (should include chunk_001):")
    for chunk in result2['used_chunks']:
        print(f"   • {chunk['id']} (ID:{chunk['element_id']}) - {chunk['text'][:50]}...")
    print()

    print("✅ Reasoning Edges (should be empty - no entity connections):")
    for edge in result2['reasoning_edges']:
        print(f"   • {edge['from']} -> {edge['relationship']} -> {edge['to']}")
    print()

    # Test Case 3: Mixed response with both entities and chunks
    rag_response_3 = """Treatment for prostate cancer (ID:12345) includes surgery (ID:67890) as mentioned in Chunk 2 (ID:chunk_002). The weather is irrelevant here."""

    print("Test Case 3: Mixed entities and chunks")
    print(f"RAG Response: {rag_response_3}")
    print("-" * 50)

    result3 = extract_used_entities_and_chunks(rag_response_3, mock_context)

    print("✅ Filtered Entities (should include prostate_cancer and surgery):")
    for entity in result3['used_entities']:
        print(f"   • {entity['id']} (ID:{entity['element_id']}) - {entity['type']}")
    print()

    print("✅ Filtered Chunks (should include chunk_002):")
    for chunk in result3['used_chunks']:
        print(f"   • {chunk['id']} (ID:{chunk['element_id']}) - {chunk['text'][:50]}...")
    print()

    print("✅ Reasoning Edges (should include connection between prostate_cancer and surgery):")
    for edge in result3['reasoning_edges']:
        print(f"   • {edge['from']} -> {edge['relationship']} -> {edge['to']}")
    print()

    # Verification
    print("🎯 VERIFICATION:")
    print("-" * 50)

    success = True

    # Check Test 1: Only prostate_cancer and surgery entities, with their relationship
    if len(result1['used_entities']) != 2:
        print("❌ Test 1 failed: Expected 2 entities, got", len(result1['used_entities']))
        success = False
    else:
        entity_ids = {e['id'] for e in result1['used_entities']}
        if entity_ids != {'prostate_cancer', 'surgery'}:
            print("❌ Test 1 failed: Expected entities prostate_cancer and surgery, got:", entity_ids)
            success = False

    if len(result1['reasoning_edges']) != 1:
        print("❌ Test 1 failed: Expected 1 reasoning edge, got", len(result1['reasoning_edges']))
        success = False

    # Check Test 2: Only chunk_001 referenced, no entities
    if len(result2['used_chunks']) != 1 or result2['used_chunks'][0]['id'] != 'chunk_001':
        print("❌ Test 2 failed: Expected 1 chunk (chunk_001), got", len(result2['used_chunks']))
        success = False

    if len(result2['used_entities']) != 0:
        print("❌ Test 2 failed: Expected 0 entities, got", len(result2['used_entities']))
        success = False

    # Check Test 3: Both entities and chunk, with relationship
    if len(result3['used_entities']) != 2:
        print("❌ Test 3 failed: Expected 2 entities, got", len(result3['used_entities']))
        success = False

    if len(result3['used_chunks']) != 1 or result3['used_chunks'][0]['id'] != 'chunk_002':
        print("❌ Test 3 failed: Expected 1 chunk (chunk_002), got", len(result3['used_chunks']))
        success = False

    if len(result3['reasoning_edges']) != 1:
        print("❌ Test 3 failed: Expected 1 reasoning edge, got", len(result3['reasoning_edges']))
        success = False

    if success:
        print("✅ ALL TESTS PASSED!")
        print("🎉 Node filtering correctly shows only nodes and chunks referenced in the RAG answer!")
        print("📊 Summary:")
        print(f"   • Medical entities only (prostate_cancer, surgery) - irrelevant entities filtered out")
        print(f"   • Referenced chunks only (chunk_002) - unreferenced chunks filtered out")
        print(f"   • Only connecting relationships between referenced entities")
    else:
        print("❌ Some tests failed - node filtering needs adjustment")

if __name__ == "__main__":
    test_node_filtering()
