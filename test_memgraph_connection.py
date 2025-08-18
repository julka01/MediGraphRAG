import os
from dotenv import load_dotenv
from kg_loader import KGLoader

# Load environment variables from .env file
load_dotenv()

# Get Memgraph credentials from environment variables
uri = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
user = os.getenv("MEMGRAPH_USER", "memgraph")
password = os.getenv("MEMGRAPH_PASSWORD", "")

# Create a simple test graph
test_graph = {
    "nodes": [
        {"id": "test1", "label": "TestNode", "properties": {"name": "TestNode1"}}
    ],
    "relationships": []
}

# Test saving to Memgraph
loader = KGLoader()
result = loader.save_to_memgraph(uri, user, password, test_graph, clear_database=True)
print("Test result:", result)
