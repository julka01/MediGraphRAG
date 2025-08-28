import os
import json
import requests
from neo4j import GraphDatabase
from PyPDF2 import PdfReader
from typing import Dict, List, Union, Tuple
from owlready2 import get_ontology

class KGLoader:
    def __init__(self):
        # Handle both Docker and local environments
        default_uri = "bolt://localhost:7687"
        configured_uri = os.getenv("NEO4J_URI", default_uri)
        
        # If running in Docker, use the configured URI, otherwise use localhost
        if os.getenv("DOCKER_ENV") == "true":
            self.neo4j_uri = configured_uri
        else:
            # Replace docker service name with localhost for local development
            self.neo4j_uri = configured_uri.replace("neo4j:7687", "localhost:7687")
        
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        print(f"KGLoader initialized with Neo4j URI: {self.neo4j_uri}")
        self.last_import_dir = None
        
    def _load_ontology(self, ontology_path: str) -> Dict:
        """Load ontology from JSON or OWL file"""
        if ontology_path.endswith('.json'):
            with open(ontology_path, 'r') as f:
                return json.load(f)
        elif ontology_path.endswith('.owl'):
            onto = get_ontology(ontology_path).load()
            return {
                "node_labels": [cls.name for cls in onto.classes()],
                "relationship_types": [prop.name for prop in onto.object_properties()]
            }
        else:
            raise ValueError(f"Unsupported ontology format: {ontology_path}")

    def load_from_pdf(self, file_path: str, ontology_path: str = None) -> Dict:
        """Extract text from PDF and structure as knowledge graph using optional ontology"""
        try:
            print(f"Loading PDF: {file_path}")
            self.last_import_dir = os.path.dirname(file_path)
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # Load ontology if provided
            ontology = {}
            if ontology_path:
                try:
                    ontology = self._load_ontology(ontology_path)
                    print(f"Loaded ontology: {ontology_path}")
                except Exception as e:
                    print(f"Error loading ontology: {str(e)}")
            
            # Create knowledge graph with ontology and supernodes
            nodes, supernodes, relationships = self._create_graph_with_ontology(text, ontology)
            
            return {
                "status": "success",
                "nodes": nodes,
                "supernodes": supernodes,
                "relationships": relationships,
                "source": "pdf",
                "filename": os.path.basename(file_path),
                "ontology": os.path.basename(ontology_path) if ontology_path else None
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def load_from_neo4j(self, uri: str, user: str, password: str, query: str = "MATCH (n) RETURN n LIMIT 100") -> Dict:
        """Fetch data from Neo4j database"""
        try:
            # Set last_import_dir to kg_storage directory for consistent export behavior
            kg_storage_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "kg_storage"))
            os.makedirs(kg_storage_dir, exist_ok=True)
            self.last_import_dir = kg_storage_dir
            
            driver = GraphDatabase.driver(
                uri, 
                auth=(user, password)
            )
            
            with driver.session() as session:
                result = session.run(query)
                nodes = []
                relationships = []
                
                for record in result:
                    node = record["n"]
                    # Convert properties to JSON-serializable format
                    properties = {}
                    for key, value in dict(node).items():
                        if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                            properties[key] = value
                        else:
                            properties[key] = str(value)
                    
                    nodes.append({
                        "id": node.id,
                        "labels": list(node.labels),
                        "label": list(node.labels)[0] if node.labels else "Node",
                        "properties": properties
                    })
                
                # Get relationships
                rel_result = session.run("MATCH ()-[r]->() RETURN r, startNode(r) as start, endNode(r) as end LIMIT 100")
                for record in rel_result:
                    rel = record["r"]
                    properties = {}
                    for key, value in dict(rel).items():
                        if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                            properties[key] = value
                        else:
                            properties[key] = str(value)
                    
                    relationships.append({
                        "id": rel.id,
                        "type": rel.type,
                        "from": record["start"].id,
                        "to": record["end"].id,
                        "properties": properties
                    })
                
                return {
                    "status": "success",
                    "nodes": nodes,
                    "relationships": relationships,
                    "source": "neo4j",
                    "query": query
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _create_graph_with_ontology(self, text: str, ontology: Dict) -> Tuple[List, List, List]:
        """Create knowledge graph with ontology harmonization and supernodes"""
        # Extract entities using ontology if available
        entities = []
        supernodes = []
        relationships = []
        
        # Create supernodes from ontology categories
        supernode_map = {}
        if "categories" in ontology:
            for i, category in enumerate(ontology["categories"]):
                supernode = {
                    "id": f"supernode_{i}",
                    "label": category["name"],
                    "properties": {
                        "description": category.get("description", ""),
                        "type": "supernode"
                    }
                }
                supernodes.append(supernode)
                supernode_map[category["name"]] = supernode["id"]
        
        # Extract entities and map to ontology
        words = set(text.split())
        entity_counter = 0
        for word in words:
            if len(word) < 4:  # Skip short words
                continue
                
            # Find matching ontology category
            supernode_id = None
            if "mappings" in ontology:
                for mapping in ontology["mappings"]:
                    if word.lower() in mapping.get("terms", []):
                        supernode_id = supernode_map.get(mapping["category"])
                        break
            
            # Create entity node
            entity = {
                "id": f"entity_{entity_counter}",
                "label": word.capitalize(),
                "properties": {
                    "text": word,
                    "frequency": text.count(word),
                    "type": "entity"
                }
            }
            entities.append(entity)
            
            # Link to supernode if found
            if supernode_id:
                relationships.append({
                    "id": f"rel_{entity_counter}",
                    "type": "MEMBER_OF",
                    "start": entity["id"],
                    "end": supernode_id,
                    "properties": {"source": "ontology"}
                })
            
            entity_counter += 1
        
        # Create relationships between entities
        for i in range(min(10, len(entities) - 1)):
            relationships.append({
                "id": f"rel_entity_{i}",
                "type": "RELATED_TO",
                "start": entities[i]["id"],
                "end": entities[i+1]["id"],
                "properties": {"strength": 0.5}
            })
        
        return entities, supernodes, relationships
        
    def save_to_neo4j(self, uri: str, user: str, password: str, graph_data: Dict, clear_database: bool = False) -> Dict:
        """Save knowledge graph to Neo4j database with supernode support
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            graph_data: Knowledge graph data to save
            clear_database: If True, clears the database before saving (default: False)
        """
        try:
            # Use password directly without escaping
            driver = GraphDatabase.driver(uri, auth=(user, password))
            
            with driver.session() as session:
                # Clear existing data only if requested
                if clear_database:
                    session.run("MATCH (n) DETACH DELETE n")
                    print("Cleared existing database data")
                
                node_map = {}
                
                # Create all nodes (entities and supernodes)
                all_nodes = graph_data.get("nodes", []) + graph_data.get("supernodes", [])
                for node in all_nodes:
                    # Sanitize labels: replace spaces with underscores
                    labels = node.get("label", "Node").replace(" ", "_")
                    properties = {k: v for k, v in node.get("properties", {}).items()}
                    result = session.run(
                        f"CREATE (n:{labels} $properties) RETURN elementId(n) as node_id",
                        properties=properties
                    )
                    node_id = result.single()["node_id"]
                    node_map[node["id"]] = node_id
                
                # Create relationships using the node_map to reference nodes
                for rel in graph_data.get("relationships", []):
                    # Use the original node IDs to look up Neo4j internal IDs
                    start_neo4j_id = node_map.get(rel["from"])
                    end_neo4j_id = node_map.get(rel["to"])
                    
                    if start_neo4j_id is not None and end_neo4j_id is not None:
                        session.run(
                            "MATCH (a), (b) WHERE elementId(a) = $start_neo4j_id AND elementId(b) = $end_neo4j_id "
                            "CREATE (a)-[r:%s $properties]->(b)" % rel['type'],
                            start_neo4j_id=start_neo4j_id,
                            end_neo4j_id=end_neo4j_id,
                            properties=rel.get("properties", {})
                        )
                    else:
                        print(f"Warning: Could not find nodes for relationship {rel['id']} "
                              f"({rel['from']} -> {rel['to']})")
                
                return {
                    "status": "success", 
                    "message": "Knowledge graph saved to Neo4j",
                    "clear_database": clear_database,
                    "nodes_created": len(all_nodes),
                    "relationships_created": len(graph_data.get("relationships", []))
                }
                
        except Exception as e:
            # Capture full error details for debugging
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "error_args": e.args
            }
            print(f"Error saving to Neo4j: {error_details}")
            return {"status": "error", "message": "Failed to save knowledge graph to Neo4j", "details": error_details}

    def save_to_file(self, graph_data: Dict, file_path: str) -> Dict:
        """Save knowledge graph to a JSON file
        
        Args:
            graph_data: Knowledge graph data to save
            file_path: Path or filename for the output JSON file
        """
        try:
            # If file_path is just a filename (no directory separators), save to kg_storage
            if os.path.dirname(file_path) == "":
                # Just a filename provided - save to kg_storage directory
                kg_storage_dir = os.path.join(os.getcwd(), "kg_storage")
                os.makedirs(kg_storage_dir, exist_ok=True)
                file_path = os.path.join(kg_storage_dir, file_path)
            else:
                # Full path provided - use it as is, but ensure directory exists
                dir_path = os.path.dirname(file_path)
                os.makedirs(dir_path, exist_ok=True)
            
            # Ensure .json extension
            if not file_path.endswith('.json'):
                file_path += '.json'
            
            print(f"Saving KG to: {os.path.abspath(file_path)}")  # Debug logging
            
            with open(file_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
            
            return {
                "status": "success", 
                "message": f"Knowledge graph saved to {file_path}",
                "file_path": file_path
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Example usage
if __name__ == "__main__":
    loader = KGLoader()
    print("Testing PDF loading:")
    print(loader.load_from_pdf("test_document.pdf"))
    
    print("\nTesting Neo4j loading:")
    print(loader.load_from_neo4j())
