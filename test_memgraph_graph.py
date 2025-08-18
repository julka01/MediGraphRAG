from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
username = "memgraph"
password = "password"

with GraphDatabase.driver(uri, auth=(username, password)) as driver:
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) AS node_count")
        record = result.single()
        print(f"Total nodes in Memgraph: {record['node_count']}")
        
        result = session.run("MATCH ()-[r]->() RETURN count(r) AS relationship_count")
        record = result.single()
        print(f"Total relationships in Memgraph: {record['relationship_count']}")
