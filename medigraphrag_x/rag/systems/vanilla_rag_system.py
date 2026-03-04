import os
import json
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph

class VanillaRAGSystem:
    """
    Vanilla RAG system that retrieves chunks directly using vector similarity search
    without knowledge graph augmentation
    """

    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        neo4j_database: str = "neo4j",
        embedding_model: str = "sentence_transformers"  # Default: free, no API key needed
    ):
        # Load Neo4j credentials from environment variables if not provided
        self.neo4j_uri = neo4j_uri if neo4j_uri is not None else os.getenv("NEO4J_URI")
        self.neo4j_user = neo4j_user if neo4j_user is not None else os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password if neo4j_password is not None else os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

        if not self.neo4j_uri or not self.neo4j_user or not self.neo4j_password:
            raise ValueError("Neo4j connection parameters not found. Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables.")

        # Initialize embedding model - use OpenAI-compatible embeddings (1536 dimensions) for consistency with KG
        if embedding_model == "openai":
            from langchain_openai import OpenAIEmbeddings; self.embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
            self.embedding_dimension = 1536
        else:
            # Default to sentence transformers but use a 1536-dim model for consistency
            from sentence_transformers import SentenceTransformer
            # Use a 1536-dim model to match OpenAI for consistent vector search
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dimension = 768

        # Simple RAG prompt template (no KG traversal instructions)
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that provides accurate, factual answers based on the provided context. Answer the question using only the information available in the context provided.

Context Information:
{context}

Guidelines:
- Base your answer ONLY on the provided context
- If the context doesn't contain information to answer the question, say so directly
- Be concise but comprehensive
- Include specific facts and details from the context to support your answer
- If asked for explanations, provide them based on the context
- IMPORTANT: For yes/no questions (questions starting with: Is, Are, Does, Do, Can, Should, Was, Were, Has, Have), you MUST begin your response with either "Yes" or "No" as the very first word, followed by your explanation.

User Query: {question}"""),
            ("human", "{question}")
        ])

    def _create_neo4j_connection(self):
        """Create Neo4j graph connection"""
        return Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password,
            database=self.neo4j_database,
            refresh_schema=False,
            sanitize=True
        )

    def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the query using sentence transformer"""
        embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        return embedding.tolist()

    def get_vanilla_rag_context(self, query: str, document_names: List[str] = None, similarity_threshold: float = 0.1, max_chunks: int = 20) -> Dict[str, Any]:
        """
        Get vanilla RAG context using direct vector similarity search on chunks
        """
        try:
            graph = self._create_neo4j_connection()

            # First check if we have any data
            check_query = "MATCH (c:Chunk) RETURN count(c) as chunk_count LIMIT 1"
            check_result = graph.query(check_query)

            if not check_result or check_result[0]['chunk_count'] == 0:
                logging.warning("No chunks found in database")
                return {
                    "query": query,
                    "chunks": [],
                    "documents": [],
                    "total_score": 0,
                    "error": "No data found in database. Please upload and process a document first."
                }

            # Generate query embedding
            query_embedding = self._generate_query_embedding(query)

            # Vector search query - direct retrieval from chunks
            if document_names:
                search_query = f"""
                CALL db.index.vector.queryNodes('vector', {max_chunks}, $query_vector)
                YIELD node AS chunk, score
                MATCH (chunk)-[:PART_OF]->(d:Document)
                WHERE d.fileName IN $document_names
                AND score >= $similarity_threshold
                RETURN
                    chunk.text AS text,
                    chunk.id AS chunk_id,
                    elementId(chunk) AS chunk_element_id,
                    score,
                    d.fileName AS document
                ORDER BY score DESC
                """
                params = {
                    "query_vector": query_embedding,
                    "document_names": document_names,
                    "similarity_threshold": similarity_threshold
                }
            else:
                search_query = f"""
                CALL db.index.vector.queryNodes('vector', {max_chunks}, $query_vector)
                YIELD node AS chunk, score
                MATCH (chunk)-[:PART_OF]->(d:Document)
                WHERE score >= $similarity_threshold
                RETURN
                    chunk.text AS text,
                    chunk.id AS chunk_id,
                    elementId(chunk) AS chunk_element_id,
                    score,
                    d.fileName AS document
                ORDER BY score DESC
                """
                params = {
                    "query_vector": query_embedding,
                    "similarity_threshold": similarity_threshold
                }

            results = graph.query(search_query, params)

            context = {
                "query": query,
                "chunks": [],
                "documents": set(),
                "total_score": 0,
                "search_method": "vanilla_vector_similarity"
            }

            for result in results:
                chunk_info = {
                    "text": result["text"],
                    "chunk_id": result["chunk_id"],
                    "chunk_element_id": result["chunk_element_id"],
                    "score": result["score"],
                    "document": result["document"]
                }
                context["chunks"].append(chunk_info)
                context["documents"].add(result["document"])
                context["total_score"] += result["score"]

            context["documents"] = list(context["documents"])

            return context

        except Exception as e:
            logging.error(f"Error getting vanilla RAG context: {e}")
            return {
                "query": query,
                "chunks": [],
                "documents": [],
                "total_score": 0,
                "error": str(e)
            }

    def format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """
        Format the context for the LLM prompt (simplified, no entities)
        """
        # Format chunks with scores
        chunk_texts = []
        for i, chunk in enumerate(context["chunks"], 1):
            chunk_text = f"Chunk {i} (Similarity Score: {chunk['score']:.3f}):\n{chunk['text']}\n"
            chunk_texts.append(chunk_text)

        formatted_context = "\n".join(chunk_texts)
        return formatted_context

    def generate_response(self, question: str, llm, document_names: List[str] = None, similarity_threshold: float = 0.1, max_chunks: int = 20, extra_context_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a vanilla RAG response using direct vector retrieval.

        Args:
            extra_context_texts: Optional list of additional context strings to prepend
                to the retrieved chunks (e.g. ground-truth question contexts for MIRAGE eval).
        """
        try:
            logging.info(f"Starting vanilla RAG generate_response for question: {question}")

            # Get context using vanilla vector retrieval
            context = self.get_vanilla_rag_context(question, document_names=document_names, similarity_threshold=similarity_threshold, max_chunks=max_chunks)
            logging.info(f"Got context with {len(context.get('chunks', []))} chunks")

            # Format retrieved chunks
            formatted_context = self.format_context_for_llm(context)

            # Prepend extra (ground-truth) contexts if provided, so the LLM always has
            # the relevant source material even when vector retrieval misses it.
            if extra_context_texts:
                extra_block = "\n\n".join(
                    f"Provided Context {i+1}:\n{t}" for i, t in enumerate(extra_context_texts) if t.strip()
                )
                formatted_context = extra_block + "\n\n" + formatted_context if formatted_context else extra_block
                logging.info(f"Prepended {len(extra_context_texts)} extra context(s) to prompt")

            if not context["chunks"] and not extra_context_texts:
                return {
                    "response": "I couldn't find any relevant information to answer your question.",
                    "context": context,
                    "sources": [],
                    "confidence": 0.0
                }

            # Generate response using LLM
            chain = self.rag_prompt | llm | StrOutputParser()
            response = chain.invoke({
                "context": formatted_context,
                "question": question
            })

            # Calculate confidence based on similarity scores
            avg_score = context["total_score"] / len(context["chunks"]) if context["chunks"] else 0
            confidence = min(avg_score * 2, 1.0)  # Scale to 0-1 range

            return {
                "response": response,
                "context": context,
                "sources": context["documents"],
                "confidence": confidence,
                "chunk_count": len(context["chunks"]),
                "retrieval_params": {
                    "similarity_threshold": similarity_threshold,
                    "max_chunks": max_chunks
                }
            }

        except Exception as e:
            logging.error(f"Error generating vanilla RAG response: {e}")
            return {
                "response": f"An error occurred while generating the response: {str(e)}",
                "context": {},
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
