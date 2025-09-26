def test_chunk_entity_linking(self):
    """Test that Chunk nodes are properly linked to Entity nodes via MENTIONS relationships"""
    kg_creator = ChunkedKGCreator()
    test_text = "Prostate cancer treatment involves radiotherapy and chemotherapy."
    kg = kg_creator.generate_knowledge_graph(test_text, llm=None)

    # Verify MENTIONS relationships exist through driver test
    driver = kg_creator._create_neo4j_connection()
    try:
        with driver.session() as session:
            # Store the KG first
            success = kg_creator.store_knowledge_graph(kg, file_name="test_chunk_linking.txt")
            self.assertTrue(success)

            # Test that chunks link to entities
            result = session.run("""
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                RETURN COUNT(c) as mention_count, COUNT(DISTINCT e) as entity_count
            """)
            record = result.single()

            # Verify we have mentions and multiple entities
            self.assertGreater(record['mention_count'], 0)
            self.assertGreater(record['entity_count'], 1)

            # Test reverse lookup: entities mentioned in chunks
            entity_result = session.run("""
                MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)
                WHERE e.id CONTAINS 'Prostate'
                RETURN e.id as entity_id, COUNT(c) as chunk_count
            """)
            entity_record = entity_result.single()

            # Verify specific entity has mentions
            self.assertEqual(entity_record['entity_id'], 'Prostate')
            self.assertGreater(entity_record['chunk_count'], 0)

    finally:
        if driver:
            driver.close()
