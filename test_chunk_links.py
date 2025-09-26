#!/usr/bin/env python3
"""Simple test script to verify chunk-entity linking works"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kg_creator import ChunkedKGCreator

def test_chunk_entity_linking():
    """Test that Chunk nodes are properly linked to Entity nodes via MENTIONS relationships"""
    kg_creator = ChunkedKGCreator()
    test_text = "Prostate cancer treatment involves radiotherapy and chemotherapy."
    kg = kg_creator.generate_knowledge_graph(test_text, llm=None)

    print("Generated KG with metadata:", kg['metadata'])
    print("Generated", len(kg['nodes']), "entity nodes")
    print("Generated", len(kg['chunks']), "chunks")

    # Check if chunk_entity_map exists
    if 'chunk_entity_map' in kg:
        print("Chunk-entity map found:")
        for chunk_idx, entities in kg['chunk_entity_map'].items():
            print(f"  Chunk {chunk_idx}: {entities}")
    else:
        print("ERROR: chunk_entity_map not found in KG structure")
        return False

    # Verify MENTIONS relationships exist through Neo4j driver test
    driver = kg_creator._create_neo4j_connection()
    try:
        with driver.session() as session:
            # Store the KG first
            success = kg_creator.store_knowledge_graph(kg, file_name="test_chunk_linking.txt")
            if not success:
                print("ERROR: Failed to store knowledge graph")
                return False

            print("KG stored successfully")

            # Test that chunks link to entities (use proper Cypher syntax for any node label)
            result = session.run("""
                MATCH (c:Chunk)-[:MENTIONS]->(e)
                RETURN COUNT(c) as mention_count, COUNT(DISTINCT e) as entity_count
            """)
            record = result.single()

            print(f"Found {record['mention_count']} total mentions and {record['entity_count']} distinct entities")

            # Verify we have mentions and multiple entities
            if record['mention_count'] <= 0:
                print("ERROR: No MENTIONS relationships found")
                return False

            if record['entity_count'] <= 0:
                print("ERROR: Too few entities found")
                return False

            # Test reverse lookup: entities mentioned in chunks
            entity_result = session.run("""
                MATCH (e)<-[:MENTIONS]-(c:Chunk)
                WHERE e.id CONTAINS 'Prostate'
                RETURN e.id as entity_id, COUNT(c) as chunk_count
            """)
            entity_records = list(entity_result)

            if not entity_records:
                print("ERROR: No entities containing 'Prostate' found")
                return False

            entity_record = entity_records[0]
            print(f"Entity '{entity_record['entity_id']}' has {entity_record['chunk_count']} chunk mentions")

            # Verify specific entity has mentions
            if entity_record['chunk_count'] <= 0:
                print("ERROR: Entity has no chunk mentions")
                return False

            print("SUCCESS: Chunk-entity linking is working correctly!")
            return True

    except Exception as e:
        print(f"ERROR during Neo4j operations: {e}")
        return False
    finally:
        if driver:
            driver.close()

if __name__ == "__main__":
    success = test_chunk_entity_linking()
    sys.exit(0 if success else 1)
