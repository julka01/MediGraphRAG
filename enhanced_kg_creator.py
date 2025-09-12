import json
import re
import hashlib
import importlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.embeddings import OpenAIEmbeddings
import os
import sys
import logging

# Add llm-graph-builder paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'llm-graph-builder/backend/src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'llm-graph-builder/backend'))

from shared.common_fn import load_embedding_model

class EnhancedKGCreator:
    """
    Enhanced Knowledge Graph Creator with proper embedding integration for RAG
    """
    
    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        embedding_model: str = "openai"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        
        # Initialize embedding model
        self.embedding_function, self.embedding_dimension = load_embedding_model(embedding_model)
        logging.info(f"Initialized embedding model: {embedding_model}, dimension: {self.embedding_dimension}")
    
    def _create_neo4j_connection(self):
        """Create Neo4j graph connection"""
        return Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password,
            database=self.neo4j_database
        )
   
    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text using TokenTextSplitter"""
        from langchain_text_splitters import TokenTextSplitter
        
        text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
        chunks = text_splitter.split_text(text)
        
        formatted = []
        total = len(chunks)
        current_pos = 0
        
        for idx, chunk_text in enumerate(chunks):
            start_pos = current_pos
            end_pos = start_pos + len(chunk_text)
            
            # Generate embedding for the chunk
            try:
                chunk_embedding = self.embedding_function.embed_query(chunk_text)
            except Exception as e:
                logging.warning(f"Failed to generate embedding for chunk {idx}: {e}")
                chunk_embedding = None
            
            formatted.append({
                "text": chunk_text,
                "chunk_id": idx,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "total_chunks": total,
                "embedding": chunk_embedding
            })
            
            current_pos += len(chunk_text) - self.chunk_overlap
            
        return formatted

    def _extract_entities_and_relationships_with_llm(self, chunk_text: str, llm) -> Dict[str, Any]:
        """
        Extract entities and relationships using LLM with improved prompting
        """
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert knowledge graph extraction system. Extract entities and relationships from the given text.

IMPORTANT RULES:
1. Extract meaningful entities (people, places, organizations, concepts, diseases, treatments, etc.)
2. Create relationships that show how entities are connected
3. Use clear, descriptive relationship types (e.g., "TREATS", "CAUSES", "LOCATED_IN", "WORKS_FOR")
4. Avoid extracting dates, numbers, or measurements as separate entities - include them as properties
5. Focus on domain-specific entities relevant to the text content

Return your response as a JSON object with this exact structure:
{
  "entities": [
    {"id": "entity_name", "type": "EntityType", "properties": {"name": "entity_name", "description": "brief description"}}
  ],
  "relationships": [
    {"source": "source_entity", "target": "target_entity", "type": "RELATIONSHIP_TYPE", "properties": {"description": "relationship description"}}
  ]
}"""),
            ("human", f"Extract entities and relationships from this text:\n\n{chunk_text}")
        ])
        
        try:
            chain = extraction_prompt | llm | StrOutputParser()
            response = chain.invoke({})
            
            # Parse JSON response
            result = json.loads(response)
            return result
            
        except Exception as e:
            logging.warning(f"LLM extraction failed: {e}, falling back to pattern matching")
            return self._extract_entities_and_relationships_fallback(chunk_text)

    def _extract_entities_and_relationships_fallback(self, chunk_text: str) -> Dict[str, Any]:
        """
        Fallback entity extraction using pattern matching
        """
        entities = []
        relationships = []
        
        # Extract potential entities (capitalized words/phrases)
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
        
        # Convert to entity format with embeddings
        for idx, entity_text in enumerate(found_entities):
            entity_type = "Concept"
            if any(word in entity_text.lower() for word in ['disease', 'cancer', 'tumor', 'syndrome']):
                entity_type = "Disease"
            elif any(word in entity_text.lower() for word in ['drug', 'medication', 'treatment', 'therapy']):
                entity_type = "Treatment"
            elif any(word in entity_text.lower() for word in ['person', 'patient', 'doctor', 'physician']):
                entity_type = "Person"
            elif any(word in entity_text.lower() for word in ['hospital', 'clinic', 'university', 'company']):
                entity_type = "Organization"
            
            # Generate embedding for entity
            try:
                entity_embedding = self.embedding_function.embed_query(entity_text)
            except Exception as e:
                logging.warning(f"Failed to generate embedding for entity {entity_text}: {e}")
                entity_embedding = None
            
            entities.append({
                "id": entity_text,
                "type": entity_type,
                "properties": {
                    "name": entity_text,
                    "description": f"{entity_type}: {entity_text}"
                },
                "embedding": entity_embedding
            })
        
        # Create relationships between entities found in the same chunk
        entity_list = list(found_entities)
        for i in range(len(entity_list) - 1):
            relationships.append({
                "source": entity_list[i],
                "target": entity_list[i + 1],
                "type": "RELATED_TO",
                "properties": {"description": "Entities mentioned in the same context"}
            })
        
        return {
            "entities": entities,
            "relationships": relationships
        }

    def _harmonize_entities(self, all_entities: List[Dict]) -> List[Dict]:
        """
        Harmonize entities across chunks to avoid duplicates
        """
        entity_map = {}
        harmonized_entities = []
        
        for entity in all_entities:
            entity_key = f"{entity['type']}:{entity['id'].lower()}"
            
            if entity_key not in entity_map:
                entity_map[entity_key] = entity
                harmonized_entities.append(entity)
            else:
                # Merge properties and keep the best embedding
                existing_entity = entity_map[entity_key]
                if entity.get('properties'):
                    existing_entity.setdefault('properties', {}).update(entity['properties'])
                
                # Keep embedding if the existing one doesn't have it
                if not existing_entity.get('embedding') and entity.get('embedding'):
                    existing_entity['embedding'] = entity['embedding']
        
        return harmonized_entities

    def _harmonize_relationships(self, all_relationships: List[Dict], entity_map: Dict) -> List[Dict]:
        """
        Harmonize relationships across chunks
        """
        harmonized_relationships = []
        seen_relationships = set()
        
        for rel in all_relationships:
            rel_key = f"{rel['source']}:{rel['type']}:{rel['target']}"
            
            if rel_key not in seen_relationships:
                harmonized_relationships.append(rel)
                seen_relationships.add(rel_key)
        
        return harmonized_relationships

    def generate_knowledge_graph(self, text: str, llm, file_name: str = None) -> Dict[str, Any]:
        """
        Generate knowledge graph from text with proper embedding integration
        """
        logging.info("Starting knowledge graph generation with embeddings")
        
        # Step 1: Chunk the text
        chunks = self._chunk_text(text)
        logging.info(f"Created {len(chunks)} chunks")
        
        # Step 2: Extract entities and relationships from each chunk
        all_entities = []
        all_relationships = []
        
        for i, chunk in enumerate(chunks):
            logging.info(f"Processing chunk {i+1}/{len(chunks)}")
            try:
                chunk_kg = self._extract_entities_and_relationships_with_llm(chunk['text'], llm)
            except Exception as e:
                logging.warning(f"LLM extraction failed for chunk {i}: {e}")
                chunk_kg = self._extract_entities_and_relationships_fallback(chunk['text'])
            
            all_entities.extend(chunk_kg['entities'])
            all_relationships.extend(chunk_kg['relationships'])
        
        # Step 3: Harmonize entities and relationships
        harmonized_entities = self._harmonize_entities(all_entities)
        entity_map = {entity['id']: entity for entity in harmonized_entities}
        harmonized_relationships = self._harmonize_relationships(all_relationships, entity_map)
        
        logging.info(f"Harmonized to {len(harmonized_entities)} entities and {len(harmonized_relationships)} relationships")
        
        # Step 4: Format the final knowledge graph
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
                    "embedding": entity.get('embedding'),
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
                "embedding_model": type(self.embedding_function).__name__,
                "embedding_dimension": self.embedding_dimension,
                "created_at": datetime.now().isoformat(),
                "visualization_ready": True,
                "file_name": file_name
            }
        }
        
        # Step 5: Store in Neo4j if requested
        if file_name:
            success = self.store_knowledge_graph_with_embeddings(kg, file_name)
            kg['metadata']['stored_in_neo4j'] = success
        
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
            "Treatment": "#66ff66",
            "Symptom": "#ffff66",
            "Drug": "#66ffff"
        }
        return color_map.get(entity_type, "#a6cee3")

    def store_knowledge_graph_with_embeddings(self, kg: Dict[str, Any], file_name: str) -> bool:
        """
        Store the knowledge graph in Neo4j database with proper embedding support
        """
        try:
            graph = self._create_neo4j_connection()
            
            # Create document node
            doc_query = """
            MERGE (d:Document {fileName: $fileName})
            SET d.createdAt = datetime(),
                d.totalChunks = $totalChunks,
                d.totalEntities = $totalEntities,
                d.totalRelationships = $totalRelationships
            """
            graph.query(doc_query, {
                "fileName": file_name,
                "totalChunks": kg['metadata']['total_chunks'],
                "totalEntities": kg['metadata']['total_entities'],
                "totalRelationships": kg['metadata']['total_relationships']
            })
            
            # Create chunk nodes with embeddings
            for chunk in kg['chunks']:
                chunk_id = hashlib.sha1(chunk['text'].encode()).hexdigest()
                chunk_query = """
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text,
                    c.position = $position,
                    c.start_pos = $start_pos,
                    c.end_pos = $end_pos,
                    c.embedding = $embedding
                """
                graph.query(chunk_query, {
                    "chunk_id": chunk_id,
                    "text": chunk['text'],
                    "position": chunk['chunk_id'],
                    "start_pos": chunk['start_pos'],
                    "end_pos": chunk['end_pos'],
                    "embedding": chunk.get('embedding')
                })
                
                # Link chunk to document
                chunk_doc_query = """
                MATCH (c:Chunk {id: $chunk_id})
                MATCH (d:Document {fileName: $fileName})
                MERGE (c)-[:PART_OF]->(d)
                """
                graph.query(chunk_doc_query, {
                    "chunk_id": chunk_id,
                    "fileName": file_name
                })
            
            # Create entity nodes with embeddings
            for node in kg['nodes']:
                # Use __Entity__ label for compatibility with llm-graph-builder
                node_query = f"""
                MERGE (n:__Entity__ {{id: $id}})
                SET n.name = $name,
                    n.type = $type,
                    n.description = $description,
                    n.embedding = $embedding
                """
                
                properties = node.get('properties', {})
                graph.query(node_query, {
                    "id": node['id'],
                    "name": properties.get('name', node['id']),
                    "type": node['label'],
                    "description": properties.get('description', ''),
                    "embedding": node.get('embedding')
                })
            
            # Create relationships
            for rel in kg['relationships']:
                rel_query = f"""
                MATCH (source:__Entity__ {{id: $source_id}})
                MATCH (target:__Entity__ {{id: $target_id}})
                MERGE (source)-[r:{rel['type']}]->(target)
                SET r += $properties
                """
                graph.query(rel_query, {
                    "source_id": rel['source'],
                    "target_id": rel['target'],
                    "properties": rel.get('properties', {})
                })
            
            # Link entities to chunks (for RAG retrieval)
            for chunk in kg['chunks']:
                chunk_id = hashlib.sha1(chunk['text'].encode()).hexdigest()
                chunk_text = chunk['text'].lower()
                
                for node in kg['nodes']:
                    entity_name = node['id'].lower()
                    if entity_name in chunk_text:
                        entity_chunk_query = """
                        MATCH (c:Chunk {id: $chunk_id})
                        MATCH (e:__Entity__ {id: $entity_id})
                        MERGE (c)-[:HAS_ENTITY]->(e)
                        """
                        graph.query(entity_chunk_query, {
                            "chunk_id": chunk_id,
                            "entity_id": node['id']
                        })
            
            # Create vector indexes for RAG
            self._create_vector_indexes(graph)
            
            logging.info(f"Successfully stored knowledge graph for {file_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error storing knowledge graph: {e}")
            return False

    def _create_vector_indexes(self, graph):
        """
        Create vector indexes for RAG functionality
        """
        try:
            # Create vector index for chunks
            chunk_index_query = f"""
            CREATE VECTOR INDEX vector IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            graph.query(chunk_index_query)
            
            # Create vector index for entities
            entity_index_query = f"""
            CREATE VECTOR INDEX entity_vector IF NOT EXISTS
            FOR (e:__Entity__) ON (e.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            graph.query(entity_index_query)
            
            # Create keyword index for full-text search
            keyword_index_query = """
            CREATE FULLTEXT INDEX keyword IF NOT EXISTS
            FOR (c:Chunk) ON EACH [c.text]
            """
            graph.query(keyword_index_query)
            
            logging.info("Created vector and keyword indexes for RAG")
            
        except Exception as e:
            logging.warning(f"Error creating indexes (may already exist): {e}")

    def get_rag_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Get RAG context for a query using vector similarity search
        """
        try:
            graph = self._create_neo4j_connection()
            
            # Generate query embedding
            query_embedding = self.embedding_function.embed_query(query)
            
            # Vector search query
            search_query = """
            CALL db.index.vector.queryNodes('vector', $top_k, $query_vector)
            YIELD node AS chunk, score
            MATCH (chunk)-[:PART_OF]->(d:Document)
            OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
            WITH chunk, score, d, collect(e) AS entities
            RETURN 
                chunk.text AS text,
                chunk.id AS chunk_id,
                score,
                d.fileName AS document,
                [entity IN entities | {id: entity.id, type: entity.type, description: entity.description}] AS entities
            ORDER BY score DESC
            """
            
            results = graph.query(search_query, {
                "top_k": top_k,
                "query_vector": query_embedding
            })
            
            context = {
                "query": query,
                "chunks": [],
                "entities": set(),
                "documents": set()
            }
            
            for result in results:
                context["chunks"].append({
                    "text": result["text"],
                    "chunk_id": result["chunk_id"],
                    "score": result["score"],
                    "document": result["document"],
                    "entities": result["entities"]
                })
                
                context["documents"].add(result["document"])
                for entity in result["entities"]:
                    context["entities"].add(entity["id"])
            
            context["entities"] = list(context["entities"])
            context["documents"] = list(context["documents"])
            
            return context
            
        except Exception as e:
            logging.error(f"Error getting RAG context: {e}")
            return {"query": query, "chunks": [], "entities": [], "documents": []}
