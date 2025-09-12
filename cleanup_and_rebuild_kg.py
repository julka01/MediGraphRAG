#!/usr/bin/env python3

"""
Script to clean up random nodes and rebuild knowledge graph with meaningful entity names
"""

import os
import sys
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_and_rebuild_kg():
    """Clean up random nodes and rebuild with meaningful entities"""
    
    logger.info("🧹 Cleaning up and rebuilding knowledge graph")
    logger.info("=" * 60)
    
    try:
        # Import required modules
        from langchain_neo4j import Neo4jGraph
        from enhanced_kg_creator import EnhancedKGCreator
        from model_providers import get_provider as get_llm_provider, LangChainRunnableAdapter
        
        # Neo4j Aura credentials
        NEO4J_URI = "neo4j+s://01ae09e4.databases.neo4j.io"
        NEO4J_USERNAME = "neo4j"
        NEO4J_PASSWORD = "awhKHbIyHJZPAIuGhHpL9omIXw8Vupnnm_35XSDN2yg"
        NEO4J_DATABASE = "neo4j"
        
        # Create connection
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE
        )
        
        logger.info("✅ Connected to Neo4j Aura")
        
        # Step 1: Check current state
        logger.info("1. Checking current knowledge graph state...")
        
        stats_query = """
        MATCH (d:Document)
        OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)
        OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:__Entity__)
        OPTIONAL MATCH (e)-[r]-(e2:__Entity__)
        RETURN 
            count(DISTINCT d) AS documents,
            count(DISTINCT c) AS chunks,
            count(DISTINCT e) AS entities,
            count(DISTINCT r) AS relationships,
            collect(DISTINCT d.fileName) AS document_names
        """
        
        stats = graph.query(stats_query)
        if stats:
            result = stats[0]
            logger.info(f"   Documents: {result['documents']}")
            logger.info(f"   Chunks: {result['chunks']}")
            logger.info(f"   Entities: {result['entities']}")
            logger.info(f"   Relationships: {result['relationships']}")
            logger.info(f"   Document names: {result['document_names']}")
            
            if result['entities'] > 100:
                logger.warning(f"⚠️  Found {result['entities']} entities - this seems excessive")
                
                # Show sample entity names to confirm they're random
                sample_entities = graph.query("""
                    MATCH (e:__Entity__)
                    RETURN e.id AS id, e.name AS name, e.type AS type
                    LIMIT 10
                """)
                
                logger.info("   Sample entity names:")
                for i, entity in enumerate(sample_entities, 1):
                    logger.info(f"     {i}. {entity['id']} (Type: {entity['type']})")
                
                # Check if these look like random names
                random_indicators = ['chunk', 'node', 'entity', 'id_', 'uuid', 'hash']
                random_count = sum(1 for entity in sample_entities 
                                 if any(indicator in str(entity['id']).lower() 
                                       for indicator in random_indicators))
                
                if random_count > 5:
                    logger.warning("⚠️  Detected random/meaningless entity names")
                    
                    # Ask user if they want to clean up
                    logger.info("2. Cleaning up random entities...")
                    
                    # Delete all entities and relationships (keep chunks and documents)
                    cleanup_query = """
                    MATCH (e:__Entity__)
                    DETACH DELETE e
                    """
                    graph.query(cleanup_query)
                    
                    logger.info("✅ Cleaned up random entities")
                    
                    # Verify cleanup
                    verify_stats = graph.query(stats_query)
                    if verify_stats:
                        verify_result = verify_stats[0]
                        logger.info(f"   After cleanup - Entities: {verify_result['entities']}")
                        logger.info(f"   After cleanup - Relationships: {verify_result['relationships']}")
        
        # Step 2: Check if we have document content to rebuild from
        logger.info("3. Checking for document content to rebuild from...")
        
        doc_content_query = """
        MATCH (d:Document)<-[:PART_OF]-(c:Chunk)
        WITH d, collect(c.text) AS chunk_texts
        RETURN d.fileName AS fileName, 
               size(chunk_texts) AS chunk_count,
               reduce(s = '', text IN chunk_texts[0..3] | s + text + '\n') AS sample_text
        LIMIT 1
        """
        
        doc_content = graph.query(doc_content_query)
        
        if not doc_content:
            logger.error("❌ No document content found to rebuild from")
            logger.info("   Please upload a new document using /generate_enhanced_kg")
            return False
        
        doc_info = doc_content[0]
        logger.info(f"   Found document: {doc_info['fileName']}")
        logger.info(f"   Chunks available: {doc_info['chunk_count']}")
        logger.info(f"   Sample text: {doc_info['sample_text'][:100]}...")
        
        # Step 3: Rebuild entities using enhanced extraction
        logger.info("4. Rebuilding entities with meaningful names...")
        
        # Get all chunk texts
        all_chunks_query = """
        MATCH (d:Document {fileName: $fileName})<-[:PART_OF]-(c:Chunk)
        RETURN c.text AS text, c.id AS chunk_id
        ORDER BY c.position
        """
        
        all_chunks = graph.query(all_chunks_query, {"fileName": doc_info['fileName']})
        
        if not all_chunks:
            logger.error("❌ No chunks found for document")
            return False
        
        # Combine all chunk texts
        full_text = "\n".join([chunk['text'] for chunk in all_chunks])
        logger.info(f"   Combined text length: {len(full_text)} characters")
        
        # Initialize enhanced KG creator
        kg_creator = EnhancedKGCreator(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USERNAME,
            neo4j_password=NEO4J_PASSWORD,
            neo4j_database=NEO4J_DATABASE,
            embedding_model="openai"
        )
        
        # Get LLM provider
        provider_instance = get_llm_provider("openrouter", "meta-llama/llama-4-maverick:free")
        llm = LangChainRunnableAdapter(provider_instance, "meta-llama/llama-4-maverick:free")
        
        # Extract entities from the full text
        logger.info("   Extracting meaningful entities...")
        
        # Process in smaller chunks to avoid overwhelming the LLM
        chunk_size = 2000
        all_entities = []
        all_relationships = []
        
        for i in range(0, len(full_text), chunk_size):
            chunk_text = full_text[i:i+chunk_size]
            logger.info(f"   Processing text chunk {i//chunk_size + 1}...")
            
            try:
                chunk_kg = kg_creator._extract_entities_and_relationships_with_llm(chunk_text, llm)
                all_entities.extend(chunk_kg.get('entities', []))
                all_relationships.extend(chunk_kg.get('relationships', []))
            except Exception as e:
                logger.warning(f"   LLM extraction failed for chunk {i//chunk_size + 1}: {e}")
                fallback_kg = kg_creator._extract_entities_and_relationships_fallback(chunk_text)
                all_entities.extend(fallback_kg.get('entities', []))
                all_relationships.extend(fallback_kg.get('relationships', []))
        
        # Harmonize entities
        harmonized_entities = kg_creator._harmonize_entities(all_entities)
        harmonized_relationships = kg_creator._harmonize_relationships(all_relationships, {})
        
        logger.info(f"   Extracted {len(harmonized_entities)} meaningful entities")
        logger.info(f"   Extracted {len(harmonized_relationships)} relationships")
        
        # Show sample entities
        if harmonized_entities:
            logger.info("   Sample meaningful entities:")
            for i, entity in enumerate(harmonized_entities[:10], 1):
                logger.info(f"     {i}. {entity['id']} (Type: {entity['type']})")
        
        # Step 4: Store the meaningful entities
        logger.info("5. Storing meaningful entities in Neo4j...")
        
        # Create entity nodes with embeddings
        for entity in harmonized_entities:
            entity_query = """
            MERGE (e:__Entity__ {id: $id})
            SET e.name = $name,
                e.type = $type,
                e.description = $description,
                e.embedding = $embedding
            """
            
            properties = entity.get('properties', {})
            graph.query(entity_query, {
                "id": entity['id'],
                "name": properties.get('name', entity['id']),
                "type": entity['type'],
                "description": properties.get('description', ''),
                "embedding": entity.get('embedding')
            })
        
        # Create relationships
        for rel in harmonized_relationships:
            rel_query = f"""
            MATCH (source:__Entity__ {{id: $source_id}})
            MATCH (target:__Entity__ {{id: $target_id}})
            MERGE (source)-[r:{rel['type']}]->(target)
            SET r += $properties
            """
            graph.query(rel_query, {
                "source_id": rel['source'],
                "target_id": rel['target'],
                "properties": rel.get('properties', {})
            })
        
        # Link entities to chunks
        logger.info("6. Linking entities to chunks...")
        
        for chunk in all_chunks:
            chunk_text_lower = chunk['text'].lower()
            
            for entity in harmonized_entities:
                entity_name_lower = entity['id'].lower()
                if entity_name_lower in chunk_text_lower:
                    link_query = """
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (e:__Entity__ {id: $entity_id})
                    MERGE (c)-[:HAS_ENTITY]->(e)
                    """
                    graph.query(link_query, {
                        "chunk_id": chunk['chunk_id'],
                        "entity_id": entity['id']
                    })
        
        # Create vector indexes
        logger.info("7. Creating vector indexes...")
        kg_creator._create_vector_indexes(graph)
        
        # Final verification
        logger.info("8. Verifying rebuilt knowledge graph...")
        final_stats = graph.query(stats_query)
        if final_stats:
            final_result = final_stats[0]
            logger.info(f"   Final entities: {final_result['entities']}")
            logger.info(f"   Final relationships: {final_result['relationships']}")
        
        logger.info("=" * 60)
        logger.info("🎉 Knowledge graph cleanup and rebuild completed successfully!")
        logger.info("   The knowledge graph now has meaningful entity names")
        logger.info("   RAG chat should work properly with actual entity references")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cleanup and rebuild failed: {e}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main execution"""
    logger.info("Starting Knowledge Graph Cleanup and Rebuild")
    
    success = cleanup_and_rebuild_kg()
    
    if success:
        logger.info("✅ Knowledge graph cleanup and rebuild completed successfully")
        logger.info("You can now test RAG chat with meaningful entity names")
        sys.exit(0)
    else:
        logger.error("❌ Knowledge graph cleanup and rebuild failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
