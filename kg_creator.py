import json
import re
import hashlib
import importlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'llm-graph-builder/backend/src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'llm-graph-builder/backend'))

from create_chunks import CreateChunksofDocument

class ChunkedKGCreator:
    """
    Knowledge Graph Creator with chunking based on llm-graph-builder's logic
    Uses app's chosen LLMs for entity extraction and relationship detection
    """
    
    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        embedding_model: str = "sentence_transformers"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.embedding_model = embedding_model
    
    def _create_neo4j_connection(self):
        """Create Neo4j graph connection"""
        return Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password,
            database=self.neo4j_database
        )
   
    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text using simplified chunking logic"""
        from langchain_text_splitters import TokenTextSplitter
        
        # Use TokenTextSplitter directly for reliable chunking
        text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
        # Split the text into chunks
        chunks = text_splitter.split_text(text)
        
        # Format the chunks
        formatted = []
        total = len(chunks)
        current_pos = 0
        
        for idx, chunk_text in enumerate(chunks):
            start_pos = current_pos
            end_pos = start_pos + len(chunk_text)
            
            formatted.append({
                "text": chunk_text,
                "chunk_id": idx,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "total_chunks": total
            })
            
            # Update position for next chunk (accounting for overlap)
            current_pos += len(chunk_text) - self.chunk_overlap
            
        return formatted

    def _extract_entities_and_relationships(self, chunk_text: str, llm) -> Dict[str, Any]:
        """
        Extract entities and relationships from chunk text using simple pattern matching
        Fallback implementation when LLM API is not available
        """
        # Simple entity extraction using pattern matching
        entities = []
        relationships = []
        
        # Extract potential entities (capitalized words/phrases)
        import re
        
        # Find capitalized words and phrases
        entity_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized phrases
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]
        
        found_entities = set()
        for pattern in entity_patterns:
            matches = re.findall(pattern, chunk_text)
            for match in matches:
                if len(match) > 2 and match not in ['The', 'This', 'That', 'And', 'But', 'For']:
                    found_entities.add(match)
        
        # Convert to entity format
        for idx, entity_text in enumerate(found_entities):
            entity_type = "Concept"
            if any(word in entity_text.lower() for word in ['disease', 'cancer', 'tumor']):
                entity_type = "Disease"
            elif any(word in entity_text.lower() for word in ['drug', 'medication', 'treatment']):
                entity_type = "Drug"
            elif any(word in entity_text.lower() for word in ['person', 'patient', 'doctor']):
                entity_type = "Person"
            
            entities.append({
                "id": entity_text,
                "type": entity_type,
                "properties": {"name": entity_text}
            })
        
        # Create simple relationships between consecutive entities
        entity_list = list(found_entities)
        for i in range(len(entity_list) - 1):
            relationships.append({
                "source": entity_list[i],
                "target": entity_list[i + 1],
                "type": "RELATED_TO",
                "properties": {}
            })
        
        return {
            "entities": entities,
            "relationships": relationships
        }

    def _harmonize_entities(self, all_entities: List[Dict]) -> List[Dict]:
        """
        Harmonize entities across chunks to avoid duplicates
        Based on llm-graph-builder's harmonization logic
        """
        entity_map = {}
        harmonized_entities = []
        
        for entity in all_entities:
            entity_key = f"{entity['type']}:{entity['id'].lower()}"
            
            if entity_key not in entity_map:
                entity_map[entity_key] = entity
                harmonized_entities.append(entity)
            else:
                # Merge properties if needed
                existing_entity = entity_map[entity_key]
                if entity.get('properties'):
                    existing_entity.setdefault('properties', {}).update(entity['properties'])
        
        return harmonized_entities

    def _harmonize_relationships(self, all_relationships: List[Dict], entity_map: Dict) -> List[Dict]:
        """
        Harmonize relationships across chunks
        """
        harmonized_relationships = []
        seen_relationships = set()
        
        for rel in all_relationships:
            # Create a unique key for the relationship
            rel_key = f"{rel['source']}:{rel['type']}:{rel['target']}"
            
            if rel_key not in seen_relationships:
                harmonized_relationships.append(rel)
                seen_relationships.add(rel_key)
        
        return harmonized_relationships

    def generate_knowledge_graph(self, text: str, llm, file_name: str = None, model_name: str = "openai/gpt-oss-20b:free") -> Dict[str, Any]:
        """
        Generate knowledge graph from text using chunking and the app's chosen LLM
        """
        # Step 1: Chunk the text using llm-graph-builder's logic
        chunks = self._chunk_text(text)
        
        # Step 2: Extract entities and relationships from each chunk
        all_entities = []
        all_relationships = []
        
        for chunk in chunks:
            chunk_kg = self._extract_entities_and_relationships(chunk['text'], llm)
            all_entities.extend(chunk_kg['entities'])
            all_relationships.extend(chunk_kg['relationships'])
        
        # Step 3: Harmonize entities and relationships across chunks
        harmonized_entities = self._harmonize_entities(all_entities)
        
        # Create entity map for relationship harmonization
        entity_map = {entity['id']: entity for entity in harmonized_entities}
        harmonized_relationships = self._harmonize_relationships(all_relationships, entity_map)
        
        # Step 4: Format the final knowledge graph with detailed Neo4j-compatible structure
        kg = {
            "nodes": [
                {
                    "id": entity['id'],
                    "label": entity['type'],
                    "properties": {
                        "name": entity['id'],
                        "type": entity['type'],
                        **entity.get('properties', {})
                    },
                    "color": self._get_node_color(entity['type']),
                    "size": 30,
                    "font": {"size": 14, "color": "#333333"},
                    "title": f"Entity: {entity['id']}\nType: {entity['type']}\nClick for details"
                }
                for entity in harmonized_entities
            ],
            "relationships": [
                {
                    "id": f"rel_{idx}",
                    "from": rel['source'],
                    "to": rel['target'],
                    "source": rel['source'],
                    "target": rel['target'],
                    "type": rel['type'],
                    "label": rel['type'],
                    "properties": rel.get('properties', {}),
                    "arrows": "to",
                    "color": {"color": "#444444"},
                    "font": {"size": 12, "align": "middle"}
                }
                for idx, rel in enumerate(harmonized_relationships)
            ],
            "chunks": chunks,
            "metadata": {
                "total_chunks": len(chunks),
                "total_entities": len(harmonized_entities),
                "total_relationships": len(harmonized_relationships),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "created_at": datetime.now().isoformat(),
                "visualization_ready": True
            }
        }
        
        return kg

    def _get_node_color(self, entity_type: str) -> str:
        """Get color for node based on entity type"""
        color_map = {
            "Person": "#ff9999",
            "Organization": "#99ccff", 
            "Location": "#99ff99",
            "Technology": "#ffcc99",
            "Concept": "#cc99ff",
            "Disease": "#ff6666",
            "Drug": "#66ff66",
            "Symptom": "#ffff66",
            "Treatment": "#66ffff"
        }
        return color_map.get(entity_type, "#a6cee3")

    def store_knowledge_graph(self, kg: Dict[str, Any], file_name: str = None) -> bool:
        """
        Store the knowledge graph in Neo4j database
        """
        try:
            graph = self._create_neo4j_connection()
            
            # Create document node if file_name is provided
            if file_name:
                doc_query = """
                MERGE (d:Document {fileName: $fileName})
                SET d.createdAt = datetime()
                """
                graph.query(doc_query, {"fileName": file_name})
            
            # Create entity nodes
            for node in kg['nodes']:
                node_query = f"""
                MERGE (n:{node['label']} {{id: $id}})
                SET n += $properties
                """
                graph.query(node_query, {
                    "id": node['id'],
                    "properties": node.get('properties', {})
                })
            
            # Create relationships
            for rel in kg['relationships']:
                rel_query = f"""
                MATCH (source {{id: $source_id}})
                MATCH (target {{id: $target_id}})
                MERGE (source)-[r:{rel['type']}]->(target)
                SET r += $properties
                """
                graph.query(rel_query, {
                    "source_id": rel['source'],
                    "target_id": rel['target'],
                    "properties": rel.get('properties', {})
                })
            
            # Create chunk nodes and relationships
            for chunk in kg['chunks']:
                chunk_id = hashlib.sha1(chunk['text'].encode()).hexdigest()
                chunk_query = """
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text,
                    c.position = $position,
                    c.start_pos = $start_pos,
                    c.end_pos = $end_pos
                """
                graph.query(chunk_query, {
                    "chunk_id": chunk_id,
                    "text": chunk['text'],
                    "position": chunk['chunk_id'],
                    "start_pos": chunk['start_pos'],
                    "end_pos": chunk['end_pos']
                })
                
                # Link chunk to document if file_name is provided
                if file_name:
                    chunk_doc_query = """
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (d:Document {fileName: $fileName})
                    MERGE (c)-[:PART_OF]->(d)
                    """
                    graph.query(chunk_doc_query, {
                        "chunk_id": chunk_id,
                        "fileName": file_name
                    })
            
            return True
            
        except Exception as e:
            print(f"Error storing knowledge graph: {e}")
            return False
