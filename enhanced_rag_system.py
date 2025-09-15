import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph
from enhanced_kg_creator import EnhancedKGCreator

class EnhancedRAGSystem:
    """
    Enhanced RAG System that properly connects to the knowledge graph with embeddings
    """
    
    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        neo4j_database: str = "neo4j",
        embedding_model: str = "sentence_transformers"
    ):
        # Load Neo4j credentials from environment variables if not provided
        self.neo4j_uri = neo4j_uri if neo4j_uri is not None else os.getenv("NEO4J_URI")
        self.neo4j_user = neo4j_user if neo4j_user is not None else os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password if neo4j_password is not None else os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

        if not self.neo4j_uri or not self.neo4j_user or not self.neo4j_password:
            raise ValueError("Neo4j connection parameters not found. Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables.")
        
        # Initialize KG creator for embedding functionality
        self.kg_creator = EnhancedKGCreator(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
            embedding_model=embedding_model
        )
        
        # RAG prompt template with node ID instructions
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that answers questions based on a knowledge graph and document chunks.

CRITICAL INSTRUCTIONS FOR NODE IDs:
- When referencing entities from the knowledge graph, ALWAYS use their actual node IDs from the context
- NEVER use placeholder IDs like "ID:X", "ID:Y", "ID:Z", or "ID:actual_number"
- If you mention an entity, include its actual ID in parentheses like: "Entity Name (ID:actual_id_from_context)"
- Only reference entities that are explicitly provided in the context below

RESPONSE GUIDELINES:
1. Answer based ONLY on the provided context - do not use external knowledge
2. Be specific and cite relevant chunks and entities
3. If the context doesn't contain enough information, say so clearly
4. Use actual node IDs when referencing entities
5. Explain relationships between entities when relevant
6. Provide comprehensive answers that utilize the full context

Context Information:
{context}

Relevant Entities:
{entities}

Question: {question}"""),
            ("human", "{question}")
        ])

    def _create_neo4j_connection(self):
        """Create Neo4j graph connection"""
        return Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password,
            database=self.neo4j_database
        )

    def get_rag_context(self, query: str, top_k: int = 5, document_names: List[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive RAG context including chunks, entities, and relationships
        """
        try:
            graph = self._create_neo4j_connection()
            
            # First check if we have any data in the knowledge graph
            check_query = "MATCH (c:Chunk) RETURN count(c) as chunk_count LIMIT 1"
            check_result = graph.query(check_query)
            
            if not check_result or check_result[0]['chunk_count'] == 0:
                logging.warning("No chunks found in knowledge graph")
                return {
                    "query": query,
                    "chunks": [],
                    "entities": {},
                    "relationships": [],
                    "documents": [],
                    "total_score": 0,
                    "entity_count": 0,
                    "relationship_count": 0,
                    "error": "No data found in knowledge graph. Please upload and process a document first."
                }
            
            # Check if vector index exists
            try:
                index_check = graph.query("SHOW INDEXES YIELD name WHERE name = 'vector'")
                if not index_check:
                    logging.warning("Vector index 'vector' not found, falling back to text search")
                    return self._fallback_text_search(graph, query, top_k, document_names)
            except Exception as e:
                logging.warning(f"Could not check vector index: {e}, falling back to text search")
                return self._fallback_text_search(graph, query, top_k, document_names)
            
            # Generate query embedding
            try:
                query_embedding = self.kg_creator.embedding_function.embed_query(query)
            except Exception as e:
                logging.error(f"Failed to generate query embedding: {e}")
                return self._fallback_text_search(graph, query, top_k, document_names)
            
            # Try vector search
            try:
                if document_names:
                    search_query = """
                    CALL db.index.vector.queryNodes('vector', $top_k, $query_vector)
                    YIELD node AS chunk, score
                    MATCH (chunk)-[:PART_OF]->(d:Document)
                    WHERE d.fileName IN $document_names
                    OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
                    OPTIONAL MATCH (e)-[r]-(related:__Entity__)
                    WHERE (related)<-[:HAS_ENTITY]-()-[:PART_OF]->(d)
                    WITH chunk, score, d, 
                         collect(DISTINCT e) AS chunk_entities,
                         collect(DISTINCT {entity: e, relationship: r, related: related}) AS entity_relationships
                    RETURN 
                        chunk.text AS text,
                        chunk.id AS chunk_id,
                        elementId(chunk) AS chunk_element_id,
                        score,
                        d.fileName AS document,
                        [entity IN chunk_entities WHERE entity IS NOT NULL | {
                            id: entity.id, 
                            element_id: elementId(entity),
                            type: coalesce(entity.type, 'Unknown'), 
                            description: coalesce(entity.description, '')
                        }] AS entities,
                        [rel IN entity_relationships WHERE rel.relationship IS NOT NULL | {
                            source: rel.entity.id,
                            source_element_id: elementId(rel.entity),
                            target: rel.related.id,
                            target_element_id: elementId(rel.related),
                            relationship_type: type(rel.relationship),
                            relationship_element_id: elementId(rel.relationship)
                        }] AS relationships
                    ORDER BY score DESC
                    """
                    params = {
                        "top_k": top_k,
                        "query_vector": query_embedding,
                        "document_names": document_names
                    }
                else:
                    search_query = """
                    CALL db.index.vector.queryNodes('vector', $top_k, $query_vector)
                    YIELD node AS chunk, score
                    MATCH (chunk)-[:PART_OF]->(d:Document)
                    OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
                    OPTIONAL MATCH (e)-[r]-(related:__Entity__)
                    WITH chunk, score, d, 
                         collect(DISTINCT e) AS chunk_entities,
                         collect(DISTINCT {entity: e, relationship: r, related: related}) AS entity_relationships
                    RETURN 
                        chunk.text AS text,
                        chunk.id AS chunk_id,
                        elementId(chunk) AS chunk_element_id,
                        score,
                        d.fileName AS document,
                        [entity IN chunk_entities WHERE entity IS NOT NULL | {
                            id: entity.id, 
                            element_id: elementId(entity),
                            type: coalesce(entity.type, 'Unknown'), 
                            description: coalesce(entity.description, '')
                        }] AS entities,
                        [rel IN entity_relationships WHERE rel.relationship IS NOT NULL | {
                            source: rel.entity.id,
                            source_element_id: elementId(rel.entity),
                            target: rel.related.id,
                            target_element_id: elementId(rel.related),
                            relationship_type: type(rel.relationship),
                            relationship_element_id: elementId(rel.relationship)
                        }] AS relationships
                    ORDER BY score DESC
                    """
                    params = {
                        "top_k": top_k,
                        "query_vector": query_embedding
                    }
                
                results = graph.query(search_query, params)
            except Exception as e:
                logging.warning(f"Vector search failed: {e}, falling back to text search")
                return self._fallback_text_search(graph, query, top_k, document_names)
            
            context = {
                "query": query,
                "chunks": [],
                "entities": {},
                "relationships": [],
                "documents": set(),
                "total_score": 0
            }
            
            for result in results:
                chunk_info = {
                    "text": result["text"],
                    "chunk_id": result["chunk_id"],
                    "chunk_element_id": result["chunk_element_id"],
                    "score": result["score"],
                    "document": result["document"],
                    "entities": result["entities"]
                }
                context["chunks"].append(chunk_info)
                context["documents"].add(result["document"])
                context["total_score"] += result["score"]
                
                # Collect unique entities with their element IDs
                for entity in result["entities"]:
                    entity_id = entity["id"]
                    if entity_id not in context["entities"]:
                        context["entities"][entity_id] = {
                            "id": entity_id,
                            "element_id": entity["element_id"],
                            "type": entity["type"],
                            "description": entity["description"],
                            "mentioned_in_chunks": []
                        }
                    context["entities"][entity_id]["mentioned_in_chunks"].append(result["chunk_id"])
                
                # Collect relationships
                for rel in result["relationships"]:
                    rel_key = f"{rel['source']}-{rel['relationship_type']}-{rel['target']}"
                    if not any(r.get('key') == rel_key for r in context["relationships"]):
                        context["relationships"].append({
                            "key": rel_key,
                            "source": rel["source"],
                            "source_element_id": rel["source_element_id"],
                            "target": rel["target"],
                            "target_element_id": rel["target_element_id"],
                            "type": rel["relationship_type"],
                            "element_id": rel["relationship_element_id"]
                        })
            
            context["documents"] = list(context["documents"])
            context["entity_count"] = len(context["entities"])
            context["relationship_count"] = len(context["relationships"])
            
            return context
            
        except Exception as e:
            logging.error(f"Error getting RAG context: {e}")
            return {
                "query": query, 
                "chunks": [], 
                "entities": {}, 
                "relationships": [],
                "documents": [], 
                "error": str(e)
            }

    def format_context_for_llm(self, context: Dict[str, Any]) -> tuple[str, str]:
        """
        Format the context for the LLM prompt
        """
        # Format chunks
        chunk_texts = []
        for i, chunk in enumerate(context["chunks"], 1):
            chunk_text = f"Chunk {i} (ID: {chunk['chunk_id']}, Score: {chunk['score']:.3f}):\n{chunk['text']}\n"
            if chunk["entities"]:
                entities_in_chunk = [f"{e['id']} (ID:{e['element_id']})" for e in chunk["entities"]]
                chunk_text += f"Entities in this chunk: {', '.join(entities_in_chunk)}\n"
            chunk_texts.append(chunk_text)
        
        formatted_context = "\n".join(chunk_texts)
        
        # Format entities with their actual node IDs
        entity_texts = []
        for entity_id, entity_info in context["entities"].items():
            entity_text = f"- {entity_info['id']} (ID:{entity_info['element_id']}) - Type: {entity_info['type']}"
            if entity_info.get('description'):
                entity_text += f" - {entity_info['description']}"
            entity_texts.append(entity_text)
        
        # Add relationships
        if context["relationships"]:
            entity_texts.append("\nRelationships:")
            for rel in context["relationships"]:
                rel_text = f"- {rel['source']} (ID:{rel['source_element_id']}) --[{rel['type']}]--> {rel['target']} (ID:{rel['target_element_id']})"
                entity_texts.append(rel_text)
        
        formatted_entities = "\n".join(entity_texts)
        
        return formatted_context, formatted_entities

    def generate_response(self, question: str, llm, document_names: List[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Generate a RAG response using the knowledge graph
        """
        try:
            # Get context from knowledge graph
            context = self.get_rag_context(question, top_k=top_k, document_names=document_names)
            
            if not context["chunks"]:
                return {
                    "response": "I couldn't find any relevant information in the knowledge graph to answer your question.",
                    "context": context,
                    "sources": [],
                    "entities": [],
                    "confidence": 0.0
                }
            
            # Format context for LLM
            formatted_context, formatted_entities = self.format_context_for_llm(context)
            
            # Generate response using LLM
            chain = self.rag_prompt | llm | StrOutputParser()
            response = chain.invoke({
                "context": formatted_context,
                "entities": formatted_entities,
                "question": question
            })
            
            # Calculate confidence based on similarity scores
            avg_score = context["total_score"] / len(context["chunks"]) if context["chunks"] else 0
            confidence = min(avg_score * 2, 1.0)  # Scale to 0-1 range
            
            return {
                "response": response,
                "context": context,
                "sources": context["documents"],
                "entities": list(context["entities"].keys()),
                "relationships": context["relationships"],
                "confidence": confidence,
                "chunk_count": len(context["chunks"]),
                "entity_count": context["entity_count"],
                "relationship_count": context["relationship_count"]
            }
            
        except Exception as e:
            logging.error(f"Error generating RAG response: {e}")
            return {
                "response": f"An error occurred while generating the response: {str(e)}",
                "context": {},
                "sources": [],
                "entities": [],
                "confidence": 0.0,
                "error": str(e)
            }

    def get_entity_details(self, entity_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific entity
        """
        try:
            graph = self._create_neo4j_connection()
            
            query = """
            MATCH (e:__Entity__ {id: $entity_id})
            OPTIONAL MATCH (e)-[r]-(related:__Entity__)
            OPTIONAL MATCH (e)<-[:HAS_ENTITY]-(c:Chunk)-[:PART_OF]->(d:Document)
            RETURN 
                e.id AS id,
                e.type AS type,
                e.description AS description,
                elementId(e) AS element_id,
                collect(DISTINCT {
                    related_id: related.id,
                    related_type: related.type,
                    relationship_type: type(r),
                    relationship_element_id: elementId(r)
                }) AS relationships,
                collect(DISTINCT {
                    chunk_id: c.id,
                    document: d.fileName
                }) AS mentions
            """
            
            results = graph.query(query, {"entity_id": entity_id})
            
            if not results:
                return {"error": f"Entity {entity_id} not found"}
            
            result = results[0]
            return {
                "id": result["id"],
                "type": result["type"],
                "description": result["description"],
                "element_id": result["element_id"],
                "relationships": result["relationships"],
                "mentions": result["mentions"]
            }
            
        except Exception as e:
            logging.error(f"Error getting entity details: {e}")
            return {"error": str(e)}

    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph
        """
        try:
            graph = self._create_neo4j_connection()
            
            stats_query = """
            MATCH (d:Document)
            OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)
            OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:__Entity__)
            OPTIONAL MATCH (e)-[r]-(e2:__Entity__)
            RETURN 
                count(DISTINCT d) AS documents,
                count(DISTINCT c) AS chunks,
                count(DISTINCT e) AS entities,
                count(DISTINCT r) AS relationships,
                collect(DISTINCT d.fileName) AS document_names
            """
            
            results = graph.query(stats_query)
            
            if results:
                result = results[0]
                return {
                    "documents": result["documents"],
                    "chunks": result["chunks"],
                    "entities": result["entities"],
                    "relationships": result["relationships"],
                    "document_names": result["document_names"],
                    "has_embeddings": True,
                    "embedding_dimension": self.kg_creator.embedding_dimension
                }
            else:
                return {
                    "documents": 0,
                    "chunks": 0,
                    "entities": 0,
                    "relationships": 0,
                    "document_names": [],
                    "has_embeddings": False
                }
                
        except Exception as e:
            logging.error(f"Error getting KG stats: {e}")
            return {"error": str(e)}

    def search_entities(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for entities using vector similarity
        """
        try:
            graph = self._create_neo4j_connection()
            
            # Generate query embedding
            query_embedding = self.kg_creator.embedding_function.embed_query(query)
            
            search_query = """
            CALL db.index.vector.queryNodes('entity_vector', $top_k, $query_vector)
            YIELD node AS entity, score
            OPTIONAL MATCH (entity)<-[:HAS_ENTITY]-(c:Chunk)-[:PART_OF]->(d:Document)
            RETURN 
                entity.id AS id,
                entity.type AS type,
                entity.description AS description,
                elementId(entity) AS element_id,
                score,
                count(DISTINCT c) AS chunk_mentions,
                collect(DISTINCT d.fileName) AS documents
            ORDER BY score DESC
            """
            
            results = graph.query(search_query, {
                "top_k": top_k,
                "query_vector": query_embedding
            })
            
            return [
                {
                    "id": result["id"],
                    "element_id": result["element_id"],
                    "type": result["type"],
                    "description": result["description"],
                    "score": result["score"],
                    "chunk_mentions": result["chunk_mentions"],
                    "documents": result["documents"]
                }
                for result in results
            ]
            
        except Exception as e:
            logging.error(f"Error searching entities: {e}")
            return []

    def _fallback_text_search(self, graph, query: str, top_k: int = 5, document_names: List[str] = None) -> Dict[str, Any]:
        """
        Fallback text-based search when vector search is not available
        """
        try:
            logging.info("Using fallback text search")
            
            # Simple text-based search using CONTAINS
            if document_names:
                search_query = """
                MATCH (c:Chunk)-[:PART_OF]->(d:Document)
                WHERE d.fileName IN $document_names 
                AND (toLower(c.text) CONTAINS toLower($query))
                OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:__Entity__)
                WITH c, d, collect(DISTINCT e) AS chunk_entities,
                     size([word IN split(toLower($query), ' ') WHERE toLower(c.text) CONTAINS word]) AS relevance_score
                RETURN 
                    c.text AS text,
                    c.id AS chunk_id,
                    elementId(c) AS chunk_element_id,
                    toFloat(relevance_score) / size(split($query, ' ')) AS score,
                    d.fileName AS document,
                    [entity IN chunk_entities WHERE entity IS NOT NULL | {
                        id: entity.id, 
                        element_id: elementId(entity),
                        type: coalesce(entity.type, 'Unknown'), 
                        description: coalesce(entity.description, '')
                    }] AS entities,
                    [] AS relationships
                ORDER BY score DESC
                LIMIT $top_k
                """
                params = {
                    "query": query,
                    "top_k": top_k,
                    "document_names": document_names
                }
            else:
                search_query = """
                MATCH (c:Chunk)-[:PART_OF]->(d:Document)
                WHERE toLower(c.text) CONTAINS toLower($query)
                OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:__Entity__)
                WITH c, d, collect(DISTINCT e) AS chunk_entities,
                     size([word IN split(toLower($query), ' ') WHERE toLower(c.text) CONTAINS word]) AS relevance_score
                RETURN 
                    c.text AS text,
                    c.id AS chunk_id,
                    elementId(c) AS chunk_element_id,
                    toFloat(relevance_score) / size(split($query, ' ')) AS score,
                    d.fileName AS document,
                    [entity IN chunk_entities WHERE entity IS NOT NULL | {
                        id: entity.id, 
                        element_id: elementId(entity),
                        type: coalesce(entity.type, 'Unknown'), 
                        description: coalesce(entity.description, '')
                    }] AS entities,
                    [] AS relationships
                ORDER BY score DESC
                LIMIT $top_k
                """
                params = {
                    "query": query,
                    "top_k": top_k
                }
            
            results = graph.query(search_query, params)
            
            context = {
                "query": query,
                "chunks": [],
                "entities": {},
                "relationships": [],
                "documents": set(),
                "total_score": 0,
                "search_method": "text_fallback"
            }
            
            for result in results:
                chunk_info = {
                    "text": result["text"],
                    "chunk_id": result["chunk_id"],
                    "chunk_element_id": result["chunk_element_id"],
                    "score": result["score"],
                    "document": result["document"],
                    "entities": result["entities"]
                }
                context["chunks"].append(chunk_info)
                context["documents"].add(result["document"])
                context["total_score"] += result["score"]
                
                # Collect unique entities
                for entity in result["entities"]:
                    entity_id = entity["id"]
                    if entity_id not in context["entities"]:
                        context["entities"][entity_id] = {
                            "id": entity_id,
                            "element_id": entity["element_id"],
                            "type": entity["type"],
                            "description": entity["description"],
                            "mentioned_in_chunks": []
                        }
                    context["entities"][entity_id]["mentioned_in_chunks"].append(result["chunk_id"])
            
            context["documents"] = list(context["documents"])
            context["entity_count"] = len(context["entities"])
            context["relationship_count"] = len(context["relationships"])
            
            return context
            
        except Exception as e:
            logging.error(f"Error in fallback text search: {e}")
            return {
                "query": query,
                "chunks": [],
                "entities": {},
                "relationships": [],
                "documents": [],
                "total_score": 0,
                "entity_count": 0,
                "relationship_count": 0,
                "error": f"Text search failed: {str(e)}"
            }
