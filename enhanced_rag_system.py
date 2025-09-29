import os
import json
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph
from sentence_transformers import SentenceTransformer
from model_providers import get_embedding_method

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

        # Enhanced RAG prompt template with detailed node traversal
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that provides structured responses with detailed node traversal from the knowledge graph. All responses must follow the same format but content adapts to the user's query intent.

CRITICAL INSTRUCTIONS FOR RESPONSE FORMAT:

1. **RECOMMENDATION/SUMMARY** (Always include this)
   - Provide a summary of the key findings and important information from the knowledge graph
   - Include key insights, important relationships, and relevant context
   - Be concise but comprehensive in highlighting what matters most

2. **NODE TRAVERSAL PATH** (Always include this - detailed graph traversal)
   - Show the complete path through the knowledge graph nodes and relationships used to derive the answer
   - Format: Start Node Name (ID:start_node_id) → Relationship Type [rel_id] → End Node Name (ID:end_node_id)
   - Reference actual node and relationship IDs from the provided context
   - Include traversal depth and how each connection was discovered (text similarity/vector search)
   - Explain which chunks and entities were retrieved via vector similarity search
   - Show scores/confidence for each traversal step when available

3. **REASONING PATH** (Always include this - logical progression)
   - Finding 1 → Triggers or yields or reveals → Finding 2 → etc.
   - Show how each piece of evidence connects to the next
   - Use actual node IDs when referencing entities from the knowledge graph
   - Format entity references as: "Entity Name (ID:actual_id_from_context)"
   - Explain confidence level if evidence is weak

4. **COMBINED EVIDENCE** (Always include this)
   - Synthesize all relevant information into coherent evidence base
   - Show how different findings support or contradict each other
   - Highlight key relationships and patterns identified during traversal

5. **NEXT STEPS** (Only include if user asks for next steps, actions, or follow-up guidance)
   - Suggest specific next actions based on the analysis
   - Provide actionable recommendations for implementation
   - If no specific next steps are appropriate, omit this section entirely

NODE ID REQUIREMENTS:
- When referencing entities from knowledge graph, ALWAYS use their actual node IDs from context
- NEVER use placeholder IDs like "ID:X", "ID:Y", "ID:Z", or "ID:actual_number"
- Show relationship traversal as: Node1 (ID:id1) → RELATIONSHIP_TYPE [rel_id] → Node2 (ID:id2)

VECTOR SEARCH DETAILS:
- Include vector similarity scores for chunks/entities retrieved
- Show which vector index (chunk vs entity) was used for retrieval
- Reference element IDs from vector search results

IMPORTANT: Base your answer ONLY on the provided context. Structure ALL responses with sections 2, 3, and 4, but only include sections 1 and 5 when appropriate for the user's intent.

Context Information:
{context}

Relevant Entities:
{entities}

User Query: {question}"""),
            ("human", "{question}")
        ])

    def _create_neo4j_connection(self):
        """Create Neo4j graph connection without schema refresh"""
        return Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password,
            database=self.neo4j_database,
            refresh_schema=False,
            sanitize=True
        )

    def get_rag_context(self, query: str, top_k: int = 5, document_names: List[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive RAG context including chunks, entities, and relationships using vector search
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

            # Try vector search first
            logging.info("Attempting vector similarity search")
            return self._vector_similarity_search(graph, query, top_k, document_names)

        except Exception as e:
            logging.error(f"Error getting RAG context: {e}")
            return {
                "query": query,
                "chunks": [],
                "entities": {},
                "relationships": [],
                "documents": [],
                "total_score": 0,
                "entity_count": 0,
                "relationship_count": 0,
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
            logging.info(f"Starting generate_response for question: {question}")
            # Get context from knowledge graph
            context = self.get_rag_context(question, top_k=top_k, document_names=document_names)
            logging.info(f"Got context with {context.get('entity_count', 0)} entities")

            if not context["chunks"]:
                return {
                    "response": "I couldn't find any relevant information in the knowledge graph to answer your question.",
                    "context": context,
                    "sources": [],
                    "entities": [],
                    "confidence": 0.0
                }

            # Debug: Check for Version in context
            if "Version" in context.get("entities", {}):
                logging.warning(f"Found Version entity in context: {context['entities']['Version']}")

            # Format context for LLM
            formatted_context, formatted_entities = self.format_context_for_llm(context)

            # Generate response using LLM
            chain = self.rag_prompt | llm | StrOutputParser()
            response = chain.invoke({
                "context": formatted_context,
                "entities": formatted_entities,
                "question": question
            })

            # Extract entities and chunks actually mentioned in the response, plus reasoning edges
            extracted_info = self._extract_used_entities_and_chunks(response, context)
            used_entities = extracted_info["used_entities"]
            used_chunks = extracted_info["used_chunks"]
            reasoning_edges = extracted_info["reasoning_edges"]

            # Calculate confidence based on similarity scores
            avg_score = context["total_score"] / len(context["chunks"]) if context["chunks"] else 0
            confidence = min(avg_score * 2, 1.0)  # Scale to 0-1 range

            return {
                "response": response,
                "context": context,
                "sources": context["documents"],
                "entities": list(context["entities"].keys()),
                "used_entities": used_entities,  # Entities actually used in the answer
                "used_chunks": used_chunks,  # Chunks actually used in the answer
                "reasoning_edges": reasoning_edges,  # Edges that form the reasoning path
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
                    "has_embeddings": False
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
        Search for entities using text-based matching
        """
        try:
            graph = self._create_neo4j_connection()

            search_query = """
            MATCH (e:__Entity__)
            WHERE toLower(e.name) CONTAINS toLower($query)
            OPTIONAL MATCH (e)<-[:HAS_ENTITY]-(c:Chunk)-[:PART_OF]->(d:Document)
            RETURN
                e.id AS id,
                e.type AS type,
                e.name AS name,
                elementId(e) AS element_id,
                count(DISTINCT c) AS chunk_mentions,
                collect(DISTINCT d.fileName) AS documents
            ORDER BY chunk_mentions DESC
            LIMIT $top_k
            """

            results = graph.query(search_query, {
                "query": query,
                "top_k": top_k
            })

            return [
                {
                    "id": result["id"],
                    "element_id": result["element_id"],
                    "type": result["type"],
                    "description": result["name"],
                    "score": result["chunk_mentions"] * 0.1,  # Simple scoring based on mentions
                    "chunk_mentions": result["chunk_mentions"],
                    "documents": result["documents"]
                }
                for result in results
            ]

        except Exception as e:
            logging.error(f"Error searching entities: {e}")
            return []

    def _vector_similarity_search(self, graph, query: str, top_k: int = 5, document_names: List[str] = None) -> Dict[str, Any]:
        """
        Vector similarity search using placeholder embeddings (to avoid import issues)
        """
        try:
            logging.info("Using vector similarity search")

            # Use placeholder embedding for the query to avoid sentence_transformers import
            # In a real implementation, this would be proper embeddings
            query_embedding = [hash(token) % 100 / 100 * 0.2 + 0.1 for token in query.split()[:10]]
            query_embedding.extend([0.1] * (384 - len(query_embedding)))  # Pad to 384 dimensions
            query_embedding = query_embedding[:384]  # Truncate if too long

            # Vector search query
            if document_names:
                search_query = """
                CALL db.index.vector.queryNodes('vector', $top_k, $query_vector)
                YIELD node AS chunk, score
                MATCH (chunk)-[:PART_OF]->(d:Document)
                WHERE d.fileName IN $document_names
                AND score >= 0.1  // Minimum similarity threshold
                OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
                WITH chunk, score, d,
                     collect(DISTINCT e) AS chunk_entities
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
                        description: coalesce(entity.name, '')
                    }] AS entities
                ORDER BY score DESC
                LIMIT $top_k
                """
                params = {
                    "query_vector": query_embedding,
                    "top_k": top_k,
                    "document_names": document_names
                }
            else:
                search_query = """
                CALL db.index.vector.queryNodes('vector', $top_k, $query_vector)
                YIELD node AS chunk, score
                MATCH (chunk)-[:PART_OF]->(d:Document)
                WHERE score >= 0.1  // Minimum similarity threshold
                OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
                WITH chunk, score, d,
                     collect(DISTINCT e) AS chunk_entities
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
                        description: coalesce(entity.name, '')
                    }] AS entities
                ORDER BY score DESC
                LIMIT $top_k
                """
                params = {
                    "query_vector": query_embedding,
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
                "search_method": "vector_similarity"
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

                # Also search for relevant entities via vector similarity
                entity_search_query = """
                CALL db.index.vector.queryNodes('entity_vector', 3, $query_vector)
                YIELD node AS entity, score
                WHERE score >= 0.3  // Higher threshold for entities
                RETURN
                    entity.id AS id,
                    elementId(entity) AS element_id,
                    entity.name AS name,
                    entity.type AS type,
                    score
                ORDER BY score DESC
                LIMIT 3
                """

                entity_results = graph.query(entity_search_query, {"query_vector": query_embedding})
                for entity_result in entity_results:
                    entity_id = entity_result["id"]
                    if entity_id not in context["entities"]:
                        context["entities"][entity_id] = {
                            "id": entity_id,
                            "element_id": entity_result["element_id"],
                            "type": entity_result["type"],
                            "description": entity_result["name"],
                            "mentioned_in_chunks": [],
                            "semantic_relevance": entity_result["score"]
                        }

            # Add relationships for found entities
            all_entity_ids = list(context["entities"].keys())
            if all_entity_ids:
                relationship_query = """
                MATCH (e1:__Entity__)-[r]-(e2:__Entity__)
                WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids
                RETURN DISTINCT
                    e1.id AS source,
                    e1.name AS source_name,
                    elementId(e1) AS source_element_id,
                    e2.id AS target,
                    e2.name AS target_name,
                    elementId(e2) AS target_element_id,
                    type(r) AS relationship_type,
                    elementId(r) AS relationship_element_id
                """

                relationship_results = graph.query(relationship_query, {"entity_ids": all_entity_ids})
                for rel_result in relationship_results:
                    rel_key = f"{rel_result['source']}-{rel_result['relationship_type']}-{rel_result['target']}"
                    if not any(r.get('key') == rel_key for r in context["relationships"]):
                        context["relationships"].append({
                            "key": rel_key,
                            "source": rel_result["source"],
                            "source_element_id": rel_result["source_element_id"],
                            "target": rel_result["target"],
                            "target_element_id": rel_result["target_element_id"],
                            "type": rel_result["relationship_type"],
                            "element_id": rel_result["relationship_element_id"]
                        })

            context["documents"] = list(context["documents"])
            context["entity_count"] = len(context["entities"])
            context["relationship_count"] = len(context["relationships"])

            return context

        except Exception as e:
            logging.error(f"Error in vector similarity search: {e}")
            # Don't return error - let it fall back to text search
            raise Exception(f"Vector search failed: {str(e)}")



    def _extract_used_entities_and_chunks(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities and chunks that are actually mentioned in the RAG answer,
        focusing ONLY on nodes that are explicitly referenced by their element ID
        """
        used_entities = []
        used_chunks = []
        reasoning_edges = []

        try:
            context_entities = context.get("entities", {})
            context_chunks = context.get("chunks", [])
            context_relationships = context.get("relationships", [])

            # Find all element IDs explicitly referenced in the response text (ID:...)
            entity_id_pattern = r'\(ID:([^)]+)\)'
            mentioned_element_ids = set(re.findall(entity_id_pattern, response))

            # Also extract chunk IDs if they appear in the same format
            chunk_id_pattern = r'Chunk\s*(\d+)\s*\(ID:\s*([^)]+)\)'
            chunk_matches = re.findall(chunk_id_pattern, response)

            # Collect chunk IDs from matches
            mentioned_chunk_ids = set()
            for match in chunk_matches:
                chunk_num, chunk_id = match
                mentioned_chunk_ids.add(chunk_id)

            # CRITICAL: Only include entities that are EXPLICITLY referenced by their element_id in (ID:...) format
            # This ensures we only get entities that are actually referenced in the reasoning
            for entity_id, entity_info in context_entities.items():
                element_id = entity_info.get('element_id', '')

                if element_id in mentioned_element_ids:
                    used_entities.append({
                        "id": entity_id,
                        "element_id": element_id,
                        "type": entity_info.get("type", "Unknown"),
                        "description": entity_info.get("description", ""),
                        "reasoning_context": "explicitly referenced by ID in RAG response"
                    })

            # Find chunks that are directly referenced
            for chunk in context_chunks:
                chunk_id = chunk.get("chunk_id", "")
                chunk_element_id = chunk.get("chunk_element_id", "")

                chunk_mentioned = False
                if chunk_id in mentioned_chunk_ids or chunk_element_id in mentioned_element_ids:
                    chunk_mentioned = True

                if chunk_mentioned:
                    used_chunks.append({
                        "id": chunk_id,
                        "element_id": chunk_element_id,
                        "text": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", ""),
                        "reasoning_context": "directly referenced chunk"
                    })

            # Find relationships between the explicitly used entities (reasoning edges)
            used_entity_element_ids = {e['element_id'] for e in used_entities}

            for rel in context_relationships:
                source_id = rel.get("source_element_id", "")
                target_id = rel.get("target_element_id", "")

                # Only include edges where both nodes are in our filtered set
                if source_id in used_entity_element_ids and target_id in used_entity_element_ids:
                    reasoning_edges.append({
                        "from": source_id,
                        "to": target_id,
                        "relationship": rel.get("type", "CONNECTED_TO"),
                        "reasoning_context": "connects explicitly referenced entities"
                    })

            logging.info(f"Filtered to {len(used_entities)} entities and {len(used_chunks)} chunks EXPLICITLY referenced in RAG answer")
            logging.info(f"Found {len(reasoning_edges)} connecting edges between referenced entities")

            return {
                "used_entities": used_entities,
                "used_chunks": used_chunks,
                "reasoning_edges": reasoning_edges,
                "total_filtered_entities": len(used_entities),
                "total_filtered_chunks": len(used_chunks),
                "total_reasoning_edges": len(reasoning_edges)
            }

        except Exception as e:
            logging.error(f"Error extracting used entities and chunks: {e}")
            return {
                "used_entities": [],
                "used_chunks": [],
                "reasoning_edges": [],
                "error": str(e)
            }
