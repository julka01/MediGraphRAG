import json
import re
import hashlib
import importlib
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from neo4j import GraphDatabase, basic_auth
from langchain_experimental.graph_transformers import LLMGraphTransformer
import os
import sys
import logging

# Import from local kg_utils
from medigraphrag_x.kg.utils.create_chunks import CreateChunksofDocument

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
        """Create Neo4j driver connection"""
        return GraphDatabase.driver(
            self.neo4j_uri,
            auth=basic_auth(self.neo4j_user, self.neo4j_password)
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
        Also prevents duplicate nodes when same text content is repeated
        """
        entity_map = {}
        harmonized_entities = []

        for entity in all_entities:
            # Create text hash for content-based deduplication
            entity_text = entity.get('id', '').strip()
            entity_text_hash = None
            if entity_text:
                entity_text_hash = hashlib.sha256(entity_text.lower().encode('utf-8')).hexdigest()

                # Check if we've seen this exact text content before (in-memory deduplication)
                if hasattr(self, '_seen_entity_text_hashes'):
                    if entity_text_hash in self._seen_entity_text_hashes:
                        # Skip this entity - it's identical text content already processed
                        continue

                # Record this text hash as seen
                if not hasattr(self, '_seen_entity_text_hashes'):
                    self._seen_entity_text_hashes = set()
                self._seen_entity_text_hashes.add(entity_text_hash)

            # Type-based entity harmonization (different from text hash deduplication)
            entity_key = f"{entity['type']}:{entity['id'].lower()}"

            if entity_key not in entity_map:
                entity_map[entity_key] = entity
                # Store text hash in properties for database-level deduplication
                if entity_text_hash:
                    entity.setdefault('properties', {})['text_hash'] = entity_text_hash
                    entity.setdefault('properties', {})['name'] = entity_text
                    entity.setdefault('properties', {})['description'] = f"{entity['type']}: {entity_text}"
                harmonized_entities.append(entity)
            else:
                # Merge properties if needed, but ensure text_hash is preserved
                existing_entity = entity_map[entity_key]
                if entity_text_hash and not existing_entity.get('properties', {}).get('text_hash'):
                    existing_entity.setdefault('properties', {})['text_hash'] = entity_text_hash
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

    def generate_knowledge_graph(self, text: str, llm, file_name: str = None, model_name: str = "openai/gpt-oss-20b:free", kg_name: str = None) -> Dict[str, Any]:
        """
        Generate knowledge graph from text using chunking and the app's chosen LLM
        Prevents duplicate nodes when same text appears multiple times
        """
        # Initialize hash cache for text-based deduplication
        self._seen_entity_text_hashes = set()

        # Step 1: Chunk the text using llm-graph-builder's logic
        chunks = self._chunk_text(text)

        # Step 2: Extract entities and relationships from each chunk, tracking chunk-entity associations
        all_entities = []
        all_relationships = []
        chunk_entity_map = {}  # Track which entities came from which chunks

        for chunk in chunks:
            chunk_kg = self._extract_entities_and_relationships(chunk['text'], llm)

            # Store chunk-entity associations for provenance
            chunk_entities = [entity['id'] for entity in chunk_kg['entities']]
            chunk_entity_map[chunk['chunk_id']] = chunk_entities

            all_entities.extend(chunk_kg['entities'])
            all_relationships.extend(chunk_kg['relationships'])

        # Step 3: Harmonize entities and relationships across chunks
        harmonized_entities = self._harmonize_entities(all_entities)

        # Create entity map for relationship harmonization
        entity_map = {entity['id']: entity for entity in harmonized_entities}
        harmonized_relationships = self._harmonize_relationships(all_relationships, entity_map)

        # Step 4: Format the final knowledge graph with detailed Neo4j-compatible structure
        # Make IDs unique per KG to avoid conflicts in visualization
        kg_prefix = f"{kg_name}_" if kg_name else ""

        kg = {
            "nodes": [
                {
                    "id": f"{kg_prefix}{entity['id']}",
                    "label": entity['type'],
                    "properties": {
                        "name": entity['id'],
                        "type": entity['type'],
                        "original_id": entity['id'],  # Keep original ID for reference
                        **entity.get('properties', {})
                    },
                    "color": self._get_node_color(entity['type']),
                    "size": 30,
                    "font": {"size": 14, "color": "#333333"},
                    "title": f"Entity: {entity['id']}\nType: {entity['type']}\nKG: {kg_name or 'default'}\nClick for details"
                }
                for entity in harmonized_entities
            ],
            "relationships": [
                {
                    "id": f"{kg_prefix}rel_{idx}",
                    "from": f"{kg_prefix}{rel['source']}",
                    "to": f"{kg_prefix}{rel['target']}",
                    "source": f"{kg_prefix}{rel['source']}",
                    "target": f"{kg_prefix}{rel['target']}",
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
            "chunk_entity_map": chunk_entity_map,  # Store provenance mapping for entity-chunk links
            "metadata": {
                "total_chunks": len(chunks),
                "total_entities": len(harmonized_entities),
                "total_relationships": len(harmonized_relationships),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "kg_name": kg_name,
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

    def store_knowledge_graph(self, kg: Dict[str, Any], file_name: str = None, version_name: str = None, version_description: str = None, kg_name: str = None) -> bool:
        """
        Store the knowledge graph in Neo4j database with optional version tracking
        """
        driver = None
        session = None
        try:
            driver = self._create_neo4j_connection()
            session = driver.session(database=self.neo4j_database)

            # Create KG version node if version_name is provided
            version_id = None
            if version_name:
                version_id = str(uuid.uuid4())
                version_query = """
                CREATE (v:KGVersion {
                    id: $version_id,
                    name: $version_name,
                    description: $version_description,
                    created: datetime()
                })
                """
                session.run(version_query, {
                    "version_id": version_id,
                    "version_name": version_name,
                    "version_description": version_description or f"KG version: {version_name}"
                })

            # Create document node if file_name is provided
            if file_name:
                doc_query = """
                MERGE (d:Document {fileName: $fileName})
                SET d.createdAt = datetime()
                """
                session.run(doc_query, {"fileName": file_name})

            # Create entity nodes with text hash deduplication
            for node in kg['nodes']:
                properties = node.get('properties', {})

                # Check if we have a text hash for deduplication
                text_hash = properties.get('text_hash')
                if text_hash:
                    # Try to find existing entity with same text hash
                    existing_entity_query = """
                    MATCH (n)
                    WHERE n.text_hash = $text_hash
                    RETURN n.id AS existing_id, n.name AS existing_name
                    LIMIT 1
                    """
                    existing_result = session.run(existing_entity_query, {"text_hash": text_hash})

                    existing_entity = existing_result.single() if existing_result.peek() else None

                    if existing_entity:
                        # Skip creating this node - link existing entity instead
                        logging.info(f"Skipping duplicate entity '{node['id']}' - using existing entity '{existing_entity['existing_name']}'")

                        # Update chunk mentions to point to existing entity
                        # We'll handle this in the chunk mention creation below
                        continue

                node_query = f"""
                MERGE (n:{node['label']} {{id: $id}})
                SET n += $properties
                """
                session.run(node_query, {
                    "id": node['id'],
                    "properties": properties
                })

                # Link entity to version if version_id is provided
                if version_id:
                    version_link_query = """
                    MATCH (n {id: $node_id})
                    MATCH (v:KGVersion {id: $version_id})
                    MERGE (v)-[:CONTAINS_ENTITY]->(n)
                    """
                    session.run(version_link_query, {
                        "node_id": node['id'],
                        "version_id": version_id
                    })

            # Create relationships
            for rel in kg['relationships']:
                # Sanitize relationship type to replace spaces with underscores for valid Cypher
                sanitized_rel_type = rel['type'].replace(' ', '_').replace('-', '_').upper()

                rel_query = f"""
                MATCH (source {{id: $source_id}})
                MATCH (target {{id: $target_id}})
                MERGE (source)-[r:{sanitized_rel_type}]->(target)
                SET r += $properties
                """
                session.run(rel_query, {
                    "source_id": rel['source'],
                    "target_id": rel['target'],
                    "properties": rel.get('properties', {})
                })

                # Note: Relationship versioning is complex in graph databases
                # For now, we only version entities. Relationships are shared across versions.

            # Create chunk nodes and relationships
            chunk_id_map = {}  # Map chunk_id to actual chunk hash for MENTIONS links

            for chunk in kg['chunks']:
                chunk_id = hashlib.sha1(chunk['text'].encode()).hexdigest()
                chunk_id_map[chunk['chunk_id']] = chunk_id  # Store mapping for entity linking

                chunk_query = """
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text,
                    c.position = $position,
                    c.start_pos = $start_pos,
                    c.end_pos = $end_pos
                """
                session.run(chunk_query, {
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
                    session.run(chunk_doc_query, {
                        "chunk_id": chunk_id,
                        "fileName": file_name
                    })

            # Create MENTIONS relationships between chunks and entities
            # Check if chunk_entity_map exists (for backwards compatibility)
            if 'chunk_entity_map' in kg:
                kg_prefix = f"{kg_name}_" if kg_name else ""

                for chunk_idx, entity_ids in kg['chunk_entity_map'].items():
                    if chunk_idx in chunk_id_map:
                        chunk_hash = chunk_id_map[chunk_idx]

                        for entity_id in entity_ids:
                            entity_full_id = f"{kg_prefix}{entity_id}"

                            mention_query = """
                            MATCH (c:Chunk {id: $chunk_id})
                            MATCH (e {id: $entity_id})
                            MERGE (c)-[:MENTIONS]->(e)
                            """
                            session.run(mention_query, {
                                "chunk_id": chunk_hash,
                                "entity_id": entity_full_id
                            })

            return True

        except Exception as e:
            print(f"Error storing knowledge graph: {e}")
            return False
        finally:
            if session:
                session.close()
            if driver:
                driver.close()
