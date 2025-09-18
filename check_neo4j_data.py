import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

uri = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USER')
password = os.getenv('NEO4J_PASSWORD')

driver = GraphDatabase.driver(uri)

try:
    with driver.session() as session:
        # Query node count
        result = session.run("MATCH (n) RETURN count(n) as node_count")
        record = result.single()
        node_count = record["node_count"]
        print(f"Node count: {node_count}")

        # Query relationship count
        result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
        record = result.single()
        rel_count = record["rel_count"]
        print(f"Relationship count: {rel_count}")

        # If there are nodes, list a few
        if node_count > 0:
            result = session.run("MATCH (n) RETURN n.id as id, n.label as label LIMIT 5")
            print("\nSample nodes:")
            for record in result:
                print(f"id: {record['id']}, label: {record['label']}")

        # If there are relationships, list a few
        if rel_count > 0:
            result = session.run("MATCH (a)-[r]->(b) RETURN a.id as from, b.id as to, type(r) as type LIMIT 5")
            print("\nSample relationships:")
            for record in result:
                print(f"from: {record['from']}, to: {record['to']}, type: {record['type']}")

except Exception as e:
    print(f"Error: {e}")
finally:
    driver.close()
