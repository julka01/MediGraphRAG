#!/usr/bin/env python3
import os
from neo4j import GraphDatabase

def fix_indexes():
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        auth=(os.getenv('NEO4J_USERNAME', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'password'))
    )

    try:
        with driver.session() as s:
            print("🔍 Checking current data and indexes...")

            # Check existing data
            chunks = s.run('MATCH (c:Chunk) RETURN count(c) as cnt').single()['cnt']
            entities = s.run('MATCH (e:__Entity__) RETURN count(e) as cnt').single()['cnt']
            print(f"📊 Found {chunks} chunks and {entities} entities")

            # Check existing vector indexes
            result = s.run("SHOW INDEXES WHERE type = 'VECTOR'")
            existing_vector_indexes = [record['name'] for record in result]
            print(f"📋 Existing vector indexes: {existing_vector_indexes}")

            # Drop incorrect indexes first
            old_indexes_to_drop = ['vector_chunk', 'vector_entity']
            for old_index in old_indexes_to_drop:
                if old_index in existing_vector_indexes:
                    print(f"🗑️ Dropping old index '{old_index}'...")
                    try:
                        s.run(f'DROP INDEX {old_index}')
                        print(f"✅ Dropped old index '{old_index}'")
                    except Exception as drop_e:
                        print(f"⚠️ Could not drop '{old_index}': {drop_e}")

            # Create correct indexes
            if chunks > 0:
                print("🔨 Creating 'vector' index for chunks...")
                s.run('CREATE VECTOR INDEX vector FOR (c:Chunk) ON (c.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: "cosine"}}')
                print("✅ Created vector index for chunks")

            if entities > 0:
                print("🔨 Creating 'entity_vector' index for entities...")
                s.run('CREATE VECTOR INDEX entity_vector FOR (e:__Entity__) ON (e.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: "cosine"}}')
                print("✅ Created entity_vector index for entities")

            # Final check
            result = s.run("SHOW INDEXES WHERE type = 'VECTOR'")
            final_indexes = [record['name'] for record in result]
            print(f"🏁 Final vector indexes: {final_indexes}")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    finally:
        driver.close()

    print("✨ Index creation complete!")

if __name__ == "__main__":
    fix_indexes()
