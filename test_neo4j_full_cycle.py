import os
from dotenv import load_dotenv
from kg_loader import KGLoader
import json

# Load environment variables
load_dotenv()

# Create KGLoader instance
loader = KGLoader()

def test_neo4j_full_cycle():
    print("\n=== Starting Neo4j Full Cycle Test ===")
    
    # Use KGLoader's automatically detected URI and credentials
    uri = loader.neo4j_uri
    user = loader.neo4j_user
    password = loader.neo4j_password
    
    print(f"Using Neo4j: {uri} with user: {user}")
    
    # Create test graph
    test_graph = {
        "nodes": [
            {"id": "test1", "label": "TestNode", "properties": {"name": "TestNode1"}},
            {"id": "test2", "label": "TestNode", "properties": {"name": "TestNode2"}}
        ],
        "relationships": [
            {"from": "test1", "to": "test2", "type": "CONNECTED_TO", "properties": {"strength": 0.8}}
        ]
    }
    
    print("Saving test graph to Neo4j...")
    save_result = loader.save_to_neo4j(uri, user, password, test_graph, clear_database=True)
    print("Save result:", save_result)
    
    if save_result['status'] != 'success':
        print("❌ Failed to save graph to Neo4j")
        return False
    
    print("Loading graph from Neo4j...")
    load_result = loader.load_from_neo4j(uri, user, password)
    print("Load result:", json.dumps(load_result, indent=2))
    
    if load_result['status'] != 'success':
        print("❌ Failed to load graph from Neo4j")
        return False
    
    # Verify loaded data
    loaded_nodes = load_result.get('nodes', [])
    loaded_relationships = load_result.get('relationships', [])
    
    if len(loaded_nodes) < 2:
        print(f"❌ Expected 2 nodes, got {len(loaded_nodes)}")
        return False
    
    if len(loaded_relationships) < 1:
        print(f"❌ Expected 1 relationship, got {len(loaded_relationships)}")
        return False
    
    # Check if our test nodes are present
    test_node_names = {"TestNode1", "TestNode2"}
    loaded_node_names = {n['properties'].get('name') for n in loaded_nodes}
    
    if not test_node_names.issubset(loaded_node_names):
        print(f"❌ Test nodes not found in loaded graph")
        return False
    
    print("✅ Neo4j full cycle test passed!")
    return True

if __name__ == "__main__":
    test_neo4j_full_cycle()
