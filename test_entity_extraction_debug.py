#!/usr/bin/env python3
"""
Test script to debug entity extraction from RAG responses.
"""

import sys
import os
sys.path.append('.')

# Create sample context and response similar to user's example
sample_context = {
    'entities': {
        'psa_testing': {'id': 'PSA testing', 'element_id': 'elem_001', 'type': 'Procedure', 'description': 'PSA testing'},
        'age': {'id': 'Age', 'element_id': 'elem_002', 'type': 'RiskFactor', 'description': 'Age risk factor'},
        'race': {'id': 'Race', 'element_id': 'elem_003', 'type': 'RiskFactor', 'description': 'Race risk factor'},
        'gleason_scoring': {'id': 'Gleason scoring', 'element_id': 'elem_004', 'type': 'Procedure', 'description': 'Gleason scoring'},
        'psa': {'id': 'PSA', 'element_id': 'elem_005', 'type': 'Biomarker', 'description': 'PSA biomarker'},
    },
    'chunks': [
        {'chunk_id': 'chunk_1', 'text': 'Sample chunk text', 'entities': []}
    ],
    'relationships': []
}

# Sample RAG response similar to user's
sample_response = '''- Chunk 1 provides the foundational clinical framework: PSA testing is standard, PSA > 4 ng/mL is abnormal, and PSA 20 ng/mL is far above this threshold.
- Entities for Risk Factors (Age & Race) identify populations with increased baseline risk; a PSA of 20 ng/mL in these groups heightens concern for clinically significant prostate cancer.
- Procedure – Gleason scoring links PSA level to pathology grading, supporting the inference that a PSA of 20 likely corresponds to a higher Gleason grade, which would influence treatment aggressiveness.

▼ REASONING PATH
1. **PSA 20** (high PSA level) → **Alert flag for prostate cancer risk**
2. **High PSA** → **Potential for higher Gleason score**
3. **Age and Race risk factors** → **Augmented probability of aggressive disease'''

def test_extraction():
    # Import the extraction method directly
    from enhanced_rag_system import EnhancedRAGSystem

    # Create a mock system to test extraction
    class MockRAGSystem:
        def _extract_used_entities_and_chunks(self, response, context):
            # Copy the actual extraction logic from EnhancedRAGSystem
            import re
            used_entities = []
            used_chunks = []
            reasoning_edges = []

            try:
                context_entities = context.get("entities", {})
                context_chunks = context.get("chunks", [])
                context_relationships = context.get("relationships", [])

                # Strategy 1: Find all element IDs explicitly referenced in multiple formats
                entity_id_patterns = [
                    r'\(ID:([^)]+)\)',           # (ID:12345)
                    r'<ID:([^>]+)>',             # <ID:12345>
                    r'ID:\s*([^\s,.:;]+)',       # ID: 12345
                    r'elementId\([\'"]?([^\'")\s]+)'  # elementId('12345-123')
                ]

                mentioned_element_ids = set()
                for pattern in entity_id_patterns:
                    matches = re.findall(pattern, response)
                    mentioned_element_ids.update(matches)

                # Strategy 2: Extract chunk references
                chunk_patterns = [
                    r'Chunk\s*(\d+)\s*\(ID:\s*([^)]+)\)',  # Chunk 1 (ID:...)
                    r'chunk\s*\d+',                           # chunk 1, chunk 2
                ]

                mentioned_chunk_ids = set()
                for pattern in chunk_patterns:
                    chunk_matches = re.findall(pattern, response, re.IGNORECASE)
                    if chunk_matches:
                        # Handle different capture groups
                        for match in chunk_matches:
                            if isinstance(match, tuple) and len(match) > 1:
                                # Pattern with ID capture: (chunk_num, chunk_id)
                                mentioned_chunk_ids.add(match[1])
                            else:
                                # Pattern without ID: just chunk number
                                pass  # Would need string matching below

                # Strategy 3: Fuzzy name matching for entities (fallback)
                response_lower = response.lower()
                mentioned_entity_names = set()

                # Extract potential entity mentions from context
                for entity_key, entity_info in context_entities.items():
                    entity_name = entity_info.get("id", "").lower()
                    entity_desc = entity_info.get("description", "").lower()

                    # Check if entity name appears in response
                    if entity_name and len(entity_name) > 3:  # Avoid short matches
                        if entity_name in response_lower:
                            mentioned_entity_names.add(entity_key)

                    # Check description too (helps with fuzzy matching)
                    desc_words = entity_desc.split() if entity_desc else []
                    for word in desc_words:
                        if len(word) > 4 and word in response_lower:
                            mentioned_entity_names.add(entity_key)
                            break

                print(f"ID-based matches: {len(mentioned_element_ids)} element IDs")
                print(f"Name-based matches: {len(mentioned_entity_names)} entities")
                print(f"Mentioned entity names: {mentioned_entity_names}")

                # Combine ID-based and name-based entity matching
                for entity_key, entity_info in context_entities.items():
                    if entity_key in mentioned_entity_names:  # entity_key is the dict key
                        used_entities.append({
                            "id": entity_info['id'],  # Use the actual id field
                            "element_id": entity_info.get('element_id', ''),
                            "type": entity_info.get("type", "Unknown"),
                            "description": entity_info.get("description", ""),
                            "reasoning_context": "mentioned by name"
                        })

                # Strategy 4: Include chunks that contain mentioned entities (semantic linking)
                relevant_chunk_ids = set()
                for entity_id in {e['id'] for e in used_entities}:
                    for chunk in context_chunks:
                        chunk_entities = chunk.get('entities', [])
                        if any(ce.get('id') == entity_id for ce in chunk_entities):
                            relevant_chunk_ids.add(chunk.get('chunk_id'))
                            relevant_chunk_ids.add(chunk.get('chunk_element_id'))

                # Include explicitly mentioned chunks
                for chunk in context_chunks:
                    chunk_id = chunk.get("chunk_id", "")
                    chunk_element_id = chunk.get("chunk_element_id", "")

                    chunk_mentioned = False
                    if chunk_id in mentioned_chunk_ids or chunk_element_id in mentioned_element_ids:
                        chunk_mentioned = True

                    # Also include chunks that contain our selected entities
                    if chunk_id in relevant_chunk_ids or chunk_element_id in relevant_chunk_ids:
                        chunk_mentioned = True

                    if chunk_mentioned:
                        used_chunks.append({
                            "id": chunk_id,
                            "element_id": chunk_element_id,
                            "text": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", ""),
                            "reasoning_context": "directly referenced chunk" if chunk_id in mentioned_chunk_ids else "contains relevant entities"
                        })

                # Strategy 5: Find relationships between selected entities
                if used_entities:  # Only look for relationships if we have filtered entities
                    used_entity_element_ids = {e['element_id'] for e in used_entities if e['element_id']}
                    used_entity_ids = {e['id'] for e in used_entities}

                    for rel in context_relationships:
                        source_id = rel.get("source", "")
                        target_id = rel.get("target", "")
                        source_element_id = rel.get("source_element_id", "")
                        target_element_id = rel.get("target_element_id", "")

                        # Include edge if both connected entities are in our filtered set
                        source_in_set = (source_id in used_entity_ids or
                                       source_element_id in used_entity_element_ids)
                        target_in_set = (target_id in used_entity_ids or
                                       target_element_id in used_entity_element_ids)

                        if source_in_set and target_in_set:
                            reasoning_edges.append({
                                "from": source_element_id or source_id,
                                "to": target_element_id or target_id,
                                "relationship": rel.get("type", "CONNECTED_TO"),
                                "reasoning_context": "connects relevant entities"
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
                print(f"Error in extraction: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "used_entities": [],
                    "used_chunks": [],
                    "reasoning_edges": [],
                    "error": str(e)
                }

    rag_system = MockRAGSystem()
    extracted = rag_system._extract_used_entities_and_chunks(sample_response, sample_context)

    print('Sample RAG Response:')
    print(sample_response[:300] + '...')

    print(f'\nFound entities: {len(extracted["used_entities"])}')
    for entity in extracted['used_entities']:
        print(f'  - {entity["id"]} ({entity["element_id"]}) - {entity["reasoning_context"]}')

    print(f'\nEntities in context: {[v["id"] for v in sample_context["entities"].values()]}')

    return extracted

if __name__ == "__main__":
    test_extraction()
