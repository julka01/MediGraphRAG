import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
driver = GraphDatabase.driver(uri)

try:
    with driver.session() as session:
        # Get node count and show some actual nodes
        result = session.run("MATCH (n) RETURN count(n) as node_count")
        record = result.single()
        node_count = record["node_count"]
        print(f"🎯 Total Nodes: {node_count}")

        # Show actual node properties
        if node_count > 0:
            result = session.run("MATCH (n) RETURN properties(n) as props, labels(n) as labels LIMIT 10")
            print("\n📋 Sample Nodes:")
            for i, record in enumerate(result, 1):
                print(f"  {i}. Labels: {record['labels']}, Properties: {record['props']}")

        # Get relationship count
        result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
        record = result.single()
        rel_count = record["rel_count"]
        print(f"\n🔗 Total Relationships: {rel_count}")

        # Show some relationship types
        if rel_count > 0:
            result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType LIMIT 10")
            relationship_types = [record['relationshipType'] for record in result]
            if relationship_types:
                print(f"\n📊 Relationship Types Found: {relationship_types}")

            # Show sample relationships
            result = session.run("MATCH (a)-[r]->(b) RETURN properties(a) as from_props, type(r) as rel_type, properties(b) as to_props LIMIT 5")
            print("\n🔗 Sample Relationships:")
            for i, record in enumerate(result, 1):
                print(f"  {i}. {record['rel_type']}: {record['from_props']} -> {record['to_props']}")

        # Show labels used
        result = session.run("CALL db.labels() YIELD label RETURN label LIMIT 10")
        labels = [record['label'] for record in result]
        if labels:
            print(f"\n🏷️  Node Labels Found: {labels}")

        print("\n✅ EAU Prostate Cancer Guidelines KG successfully loaded into Neo4j Browser!")
        print("🔗 Access your Neo4j Browser at: http://localhost:7474")

except Exception as e:
    print(f"❌ Error: {e}")
finally:
    driver.close()
