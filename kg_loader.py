import os
import json
import requests
from neo4j import GraphDatabase
from PyPDF2 import PdfReader
from typing import Dict, List, Union

class KGLoader:
    def __init__(self):
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    def load_from_pdf(self, file_path: str) -> Dict:
        """Extract text from PDF and structure as knowledge graph"""
        try:
            print(f"Loading PDF: {file_path}")
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # Simple text processing to create nodes and relationships
            entities = self._extract_entities(text)
            relationships = self._create_relationships(entities)
            
            return {
                "status": "success",
                "nodes": entities,
                "relationships": relationships,
                "source": "pdf",
                "filename": os.path.basename(file_path)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def load_from_neo4j(self, query: str = "MATCH (n) RETURN n LIMIT 100") -> Dict:
        """Fetch data from Neo4j database"""
        try:
            driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_user, self.neo4j_password)
            )
            
            with driver.session() as session:
                result = session.run(query)
                nodes = []
                relationships = []
                
                for record in result:
                    node = record["n"]
                    nodes.append({
                        "id": node.id,
                        "labels": list(node.labels),
                        "properties": dict(node)
                    })
                
                # Get relationships
                rel_result = session.run("MATCH ()-[r]->() RETURN r LIMIT 100")
                for record in rel_result:
                    rel = record["r"]
                    relationships.append({
                        "id": rel.id,
                        "type": rel.type,
                        "start": rel.start_node.id,
                        "end": rel.end_node.id,
                        "properties": dict(rel)
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

    def _extract_entities(self, text: str) -> List[Dict]:
        """Simple entity extraction from text (placeholder implementation)"""
        # In a real implementation, this would use NLP/NER techniques
        entities = []
        words = set(text.split())
        for i, word in enumerate(words):
            if len(word) > 3:  # Filter out short words
                entities.append({
                    "id": i,
                    "label": word.capitalize(),
                    "properties": {"text": word, "frequency": text.count(word)}
                })
        return entities

    def _create_relationships(self, entities: List[Dict]) -> List[Dict]:
        """Create simple relationships between entities (placeholder)"""
        relationships = []
        for i in range(min(10, len(entities) - 1)):
            relationships.append({
                "id": i,
                "type": "RELATED_TO",
                "start": entities[i]["id"],
                "end": entities[i+1]["id"],
                "properties": {"strength": 0.5}
            })
        return relationships

# Example usage
if __name__ == "__main__":
    loader = KGLoader()
    print("Testing PDF loading:")
    print(loader.load_from_pdf("test_document.pdf"))
    
    print("\nTesting Neo4j loading:")
    print(loader.load_from_neo4j())
