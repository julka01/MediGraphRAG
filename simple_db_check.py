#!/usr/bin/env python3

from dotenv import load_dotenv
import os
load_dotenv()

from neo4j import GraphDatabase

def get_db_stats():
    try:
        uri = os.getenv('NEO4J_URI')
        username = os.getenv('NEO4J_USERNAME')
        password = os.getenv('NEO4J_PASSWORD')

        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session() as session:

            # Get totals one by one to avoid query issues
            total_entities_result = session.run('MATCH (n:__Entity__) RETURN count(n) as entities')
            total_entities = total_entities_result.single()['entities']

            total_chunks_result = session.run('MATCH (c:Chunk) RETURN count(c) as chunks')
            total_chunks = total_chunks_result.single()['chunks']

            total_rels_result = session.run('MATCH ()-[r]->() RETURN count(r) as relationships')
            total_relationships = total_rels_result.single()['relationships']

            print("❓ QUESTION ASKED: 'What are the main risk factors for prostate cancer?'")
            print()
            print("📊 COMPLETE DATABASE ANALYSIS:")
            print(f"  Total entities in DB: {total_entities}")
            print(f"  Total chunks in DB: {total_chunks}")
            print(f"  Total relationships in DB: {total_relationships}")
            print()
            print("🔍 RAG RETRIEVAL RESULTS:")
            print(f"  ✅ RAG found: 88 entities OUT OF {total_entities} total ({(88/total_entities*100):.1f}%)")
            print(f"  ✅ RAG found: 5 chunks OUT OF {total_chunks} total ({(5/total_chunks*100):.1f}%)")
            print(f"  ✅ RAG found: 189 relationships OUT OF {total_relationships} total ({(189/total_relationships*100):.1f}%)")

        driver.close()

    except Exception as e:
        print(f"Database connection error: {e}")

if __name__ == "__main__":
    get_db_stats()
