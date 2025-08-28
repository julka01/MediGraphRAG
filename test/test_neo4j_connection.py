import os
from dotenv import load_dotenv
from kg_loader import KGLoader

# Load environment variables from .env file
load_dotenv()

# Get Neo4j credentials from environment variables
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

# Create a simple test graph
test_graph = {
    "nodes": [
        {"id": "test1", "label": "TestNode", "properties": {"name": "TestNode1"}}
    ],
    "relationships": []
}

# Test saving to Neo4j
loader = KGLoader()
result = loader.save_to_neo4j(uri, user, password, test_graph, clear_database=True)
print("Test result:", result)
