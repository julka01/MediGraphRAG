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
from ontographrag.rag.answer_guardrails import (
    RUNTIME_GUARDRAIL_ABSTENTION,
    evaluate_runtime_answer_guardrail,
)
from ontographrag.rag.retrieval_sampling import (
    compute_candidate_limit,
    select_ranked_subset,
)

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
        embedding_model: str = None  # Use environment variable if not provided
    ):
        # Load Neo4j credentials from environment variables if not provided
        self.neo4j_uri = neo4j_uri if neo4j_uri is not None else os.getenv("NEO4J_URI")
        self.neo4j_user = neo4j_user if neo4j_user is not None else os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password if neo4j_password is not None else os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

        if not self.neo4j_uri or not self.neo4j_user or not self.neo4j_password:
            raise ValueError("Neo4j connection parameters not found. Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables.")

        # Initialize embedding model - prefer EMBEDDING_PROVIDER for consistency
        embedding_provider = (
            embedding_model
            or os.getenv("EMBEDDING_PROVIDER")
            or os.getenv("EMBEDDING_MODEL", "sentence_transformers")
        )
        
        if embedding_provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
            self.embedding_dimension = 1536
            logging.info("Initialized OpenAI embedding model: text-embedding-ada-002 (1536 dimensions)")
        else:
            # Default to sentence transformers
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dimension = 384
            logging.info("Initialized SentenceTransformer embedding model: all-MiniLM-L6-v2 (384 dimensions)")
        self._vector_index_name: Optional[str] = None

        # Simple RAG prompt template (no KG traversal instructions)
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that provides accurate, factual answers based on the provided context.

Context Information:
{context}

Guidelines:
- Follow the task-specific answer instructions below when they are provided.
- If the task-specific instructions require an exact label-only answer, obey them exactly and do not add explanation.
- If the task-specific instructions require a short answer only, give only that short answer and do not add explanation.
- Read all context passages carefully — the answer is often present but may require connecting two passages.
- For multi-hop questions: explicitly chain your reasoning step by step (e.g. "The film starred X → X later held position Y").
- Base your answer on the provided context; do not invent facts.
- If the answer is not directly stated but can be inferred by connecting two pieces of evidence, make the inference explicitly and state your reasoning chain.
- Only say the context is insufficient if you genuinely cannot find any relevant evidence after carefully reading all passages.
- Be concise but comprehensive; include specific facts to support your answer.
- For source-document biomedical classification tasks, let the study conclusion in the text govern the final label.
- IMPORTANT: Unless task-specific instructions say otherwise, for yes/no questions
  (questions starting with: Is, Are, Does, Do, Can, Should, Was, Were, Has, Have),
  begin your response with either "Yes" or "No" as the very first word.

Task-Specific Answer Instructions:
{answer_instructions}

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
        """Generate embedding for the query across supported embedding backends."""
        if hasattr(self.embedding_model, "embed_query"):
            return self.embedding_model.embed_query(query)

        if hasattr(self.embedding_model, "encode"):
            embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

        raise ValueError("Unsupported embedding model interface. Expected embed_query or encode.")

    def _resolve_vector_index_name(self, graph) -> Optional[str]:
        """Prefer retrieval-span vectors when available, else fall back to Chunk vectors."""
        if self._vector_index_name is not None:
            return self._vector_index_name

        try:
            index_rows = graph.query(
                "SHOW INDEXES YIELD name, type, state WHERE type = 'VECTOR' RETURN name, state"
            ) or []
        except Exception as exc:
            logging.warning("Could not inspect Neo4j vector indexes: %s", exc)
            return None

        online_indexes = {
            str(row.get("name")): str(row.get("state") or "").upper()
            for row in index_rows
            if str(row.get("state") or "").upper() in {"", "ONLINE"}
        }
        probe_embedding = self._generate_query_embedding("test")
        for candidate in ("retrieval_vector", "vector"):
            if candidate not in online_indexes:
                continue
            try:
                graph.query(
                    f"""
                    CALL db.index.vector.queryNodes('{candidate}', 1, $query_vector)
                    YIELD node, score
                    RETURN count(node) AS n
                    """,
                    {"query_vector": probe_embedding},
                )
                self._vector_index_name = candidate
                return candidate
            except Exception as exc:
                logging.warning("Vector index %s probe failed: %s", candidate, exc)
        return None

    def get_vanilla_rag_context(
        self,
        query: str,
        document_names: List[str] = None,
        similarity_threshold: float = 0.1,
        max_chunks: int = 20,
        kg_name: str = None,
        question_id: str = None,
        retrieval_temperature: float = 0.0,
        retrieval_shortlist_factor: int = 4,
        retrieval_sample_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Get vanilla RAG context using direct vector similarity search on chunks.
        
        Args:
            query: The search query
            document_names: Optional list of document names to filter by
            similarity_threshold: Minimum similarity score for retrieval
            max_chunks: Maximum number of chunks to retrieve
            kg_name: Optional KG name to filter retrieval to a specific named KG
        """
        try:
            graph = self._create_neo4j_connection()

            # Check if we have data - optionally filter by kg_name
            if kg_name:
                check_query = """
                MATCH (c:Chunk)-[:PART_OF]->(d:Document {kgName: $kg_name})
                RETURN count(c) as chunk_count LIMIT 1
                """
                check_result = graph.query(check_query, {"kg_name": kg_name})
            else:
                check_query = "MATCH (c:Chunk) RETURN count(c) as chunk_count LIMIT 1"
                check_result = graph.query(check_query)

            if not check_result or check_result[0]['chunk_count'] == 0:
                logging.warning(f"No chunks found in database (kg_name: {kg_name})")
                return {
                    "query": query,
                    "chunks": [],
                    "documents": [],
                    "total_score": 0,
                    "error": f"No data found in database (kg: {kg_name}). Please upload and process a document first."
                }

            # Generate query embedding
            query_embedding = self._generate_query_embedding(query)
            vector_index_name = self._resolve_vector_index_name(graph)
            if not vector_index_name:
                logging.warning("No usable vector index found for vanilla retrieval")
                return {
                    "query": query,
                    "chunks": [],
                    "documents": [],
                    "total_score": 0,
                    "error": "No usable vector index found in Neo4j.",
                }

            # Vector search query - direct retrieval from chunks with optional filtering
            where_clauses = []
            params = {
                "query_vector": query_embedding,
                "similarity_threshold": similarity_threshold
            }
            
            if kg_name:
                where_clauses.append("d.kgName = $kg_name")
                params["kg_name"] = kg_name
                
            if document_names:
                where_clauses.append("d.fileName IN $document_names")
                params["document_names"] = document_names

            if question_id:
                where_clauses.append("chunk.questionId = $question_id")
                params["question_id"] = question_id
            
            # Build the complete WHERE clause
            if where_clauses:
                where_clause = "WHERE " + " AND ".join(where_clauses) + " AND score >= $similarity_threshold"
            else:
                where_clause = "WHERE score >= $similarity_threshold"
            
            candidate_limit = compute_candidate_limit(
                max_chunks,
                retrieval_temperature,
                retrieval_shortlist_factor,
                hard_cap=500,
            )
            retrieval_count = min(candidate_limit * 20, 500) if (kg_name or document_names) else candidate_limit

            if vector_index_name == "retrieval_vector":
                if question_id:
                    where_clause = where_clause.replace(
                        "chunk.questionId = $question_id",
                        "retrieval.questionId = $question_id",
                    )
                search_query = f"""
                CALL db.index.vector.queryNodes('{vector_index_name}', {retrieval_count}, $query_vector)
                YIELD node AS retrieval, score
                MATCH (retrieval:RetrievalChunk)-[:RETRIEVES_FROM]->(chunk:Chunk)-[:PART_OF]->(d:Document)
                {where_clause}
                RETURN
                    retrieval.text AS text,
                    retrieval.id AS chunk_id,
                    elementId(retrieval) AS chunk_element_id,
                    chunk.id AS parent_chunk_id,
                    elementId(chunk) AS parent_chunk_element_id,
                    chunk.position AS position,
                    chunk.source AS source,
                    retrieval.questionId AS question_id,
                    retrieval.passageIndex AS passage_index,
                    retrieval.chunkLocalIndex AS chunk_local_index,
                    retrieval.retrievalLocalIndex AS retrieval_local_index,
                    score,
                    d.fileName AS document,
                    d.kgName AS kg_name
                ORDER BY score DESC
                LIMIT $max_chunks
                """
            else:
                search_query = f"""
                CALL db.index.vector.queryNodes('{vector_index_name}', {retrieval_count}, $query_vector)
                YIELD node AS chunk, score
                MATCH (chunk)-[:PART_OF]->(d:Document)
                {where_clause}
                RETURN
                    chunk.text AS text,
                    chunk.id AS chunk_id,
                    elementId(chunk) AS chunk_element_id,
                    chunk.position AS position,
                    chunk.source AS source,
                    chunk.questionId AS question_id,
                    chunk.passageIndex AS passage_index,
                    chunk.chunkLocalIndex AS chunk_local_index,
                    score,
                    d.fileName AS document,
                    d.kgName AS kg_name
                ORDER BY score DESC
                LIMIT $max_chunks
                """
            params["max_chunks"] = candidate_limit

            results = graph.query(search_query, params)
            selected_results = select_ranked_subset(
                results,
                max_items=max_chunks,
                retrieval_temperature=retrieval_temperature,
                shortlist_factor=retrieval_shortlist_factor,
                sample_id=retrieval_sample_id,
                seed_parts=(
                    "vanilla",
                    kg_name,
                    tuple(document_names or []),
                    query,
                ),
                score_getter=lambda row: float(row.get("score", 0.0)),
            )

            context = {
                "query": query,
                "chunks": [],
                "documents": set(),
                "total_score": 0,
                "search_method": (
                    "vanilla_retrieval_span_similarity"
                    if vector_index_name == "retrieval_vector"
                    else "vanilla_vector_similarity"
                ),
                "retrieval_sampling": {
                    "temperature": float(retrieval_temperature or 0.0),
                    "shortlist_factor": int(retrieval_shortlist_factor or 1),
                    "sample_id": int(retrieval_sample_id or 0),
                    "candidate_limit": int(candidate_limit),
                },
            }

            seen_ids = set()
            for result in selected_results:
                chunk_info = {
                    "text": result["text"],
                    "chunk_id": result["chunk_id"],
                    "chunk_element_id": result["chunk_element_id"],
                    "position": result.get("position"),
                    "source": result.get("source"),
                    "question_id": result.get("question_id"),
                    "passage_index": result.get("passage_index"),
                    "chunk_local_index": result.get("chunk_local_index"),
                    "retrieval_local_index": result.get("retrieval_local_index"),
                    "parent_chunk_id": result.get("parent_chunk_id"),
                    "parent_chunk_element_id": result.get("parent_chunk_element_id"),
                    "score": result["score"],
                    "document": result["document"],
                    "kg_name": result.get("kg_name"),
                }
                context["chunks"].append(chunk_info)
                context["documents"].add(result["document"])
                context["total_score"] += result["score"]
                seen_ids.add(result["chunk_element_id"])

            # Adjacent chunk expansion: fetch position±1 neighbours of every
            # retrieved chunk to capture answers split across chunk boundaries.
            # When retrieval_vector is active, chunk_element_id is a RetrievalChunk
            # element id; use parent_chunk_element_id (the Chunk node) instead so
            # the adjacency query correctly matches Chunk nodes.
            if selected_results:
                seed_element_ids = [
                    r.get("parent_chunk_element_id") or r["chunk_element_id"]
                    for r in selected_results
                ]
                adj_params = {
                    "element_ids": seed_element_ids,
                    "kg_name": kg_name,
                    "max_adjacent": max_chunks,
                }
                kg_filter = "AND d.kgName = $kg_name" if kg_name else ""
                adj_query = f"""
                UNWIND $element_ids AS eid
                MATCH (seed:Chunk)-[:PART_OF]->(d:Document)
                WHERE elementId(seed) = eid
                MATCH (adj:Chunk)-[:PART_OF]->(d)
                WHERE ((
                    seed.questionId IS NOT NULL
                    AND adj.questionId = seed.questionId
                    AND coalesce(adj.passageIndex, -1) = coalesce(seed.passageIndex, -1)
                    AND abs(
                        coalesce(adj.chunkLocalIndex, adj.position)
                        - coalesce(seed.chunkLocalIndex, seed.position)
                    ) = 1
                ) OR (
                    seed.questionId IS NULL
                    AND adj.questionId IS NULL
                    AND abs(adj.position - seed.position) = 1
                ))
                  {kg_filter}
                RETURN DISTINCT
                    adj.text AS text,
                    adj.id AS chunk_id,
                    elementId(adj) AS chunk_element_id,
                    adj.position AS position,
                    adj.source AS source,
                    adj.questionId AS question_id,
                    adj.passageIndex AS passage_index,
                    adj.chunkLocalIndex AS chunk_local_index,
                    0.0 AS score,
                    d.fileName AS document,
                    d.kgName AS kg_name
                ORDER BY d.fileName ASC,
                         coalesce(adj.questionId, ''),
                         coalesce(adj.passageIndex, -1),
                         coalesce(adj.chunkLocalIndex, adj.position),
                         adj.position ASC
                LIMIT $max_adjacent
                """
                try:
                    adj_results = graph.query(adj_query, adj_params)
                    for result in adj_results:
                        if result["chunk_element_id"] not in seen_ids:
                            context["chunks"].append({
                                "text": result["text"],
                                "chunk_id": result["chunk_id"],
                                "chunk_element_id": result["chunk_element_id"],
                                "position": result.get("position"),
                                "source": result.get("source"),
                                "question_id": result.get("question_id"),
                                "passage_index": result.get("passage_index"),
                                "chunk_local_index": result.get("chunk_local_index"),
                                "score": 0.0,
                                "document": result["document"],
                                "kg_name": result.get("kg_name"),
                                "adjacent": True,
                            })
                            context["documents"].add(result["document"])
                            seen_ids.add(result["chunk_element_id"])
                except Exception as adj_err:
                    logging.debug("Adjacent chunk expansion failed (non-fatal): %s", adj_err)

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

    def _runtime_answer_guardrail_enabled(self, runtime_guardrail: Optional[bool]) -> bool:
        if runtime_guardrail is not None:
            return bool(runtime_guardrail)
        return str(os.getenv("ONTOGRAPHRAG_RUNTIME_ANSWER_GUARDRAIL", "0")).strip().lower() in {
            "1", "true", "yes", "on",
        }

    def _runtime_answer_guardrail_mode(self, runtime_guardrail_mode: Optional[str]) -> str:
        mode = str(
            runtime_guardrail_mode
            or os.getenv("ONTOGRAPHRAG_RUNTIME_ANSWER_GUARDRAIL_MODE", "retry_then_abstain")
        ).strip().lower()
        if mode not in {"retry_then_abstain", "abstain_only"}:
            return "retry_then_abstain"
        return mode

    def _invoke_answer_chain(
        self,
        *,
        question: str,
        llm,
        context: Dict[str, Any],
        answer_instructions: str = "",
        extra_context_texts: Optional[List[str]] = None,
    ) -> str:
        formatted_context = self.format_context_for_llm(context)

        if extra_context_texts:
            extra_block = "\n\n".join(
                f"Provided Context {i+1}:\n{t}" for i, t in enumerate(extra_context_texts) if t.strip()
            )
            formatted_context = extra_block + "\n\n" + formatted_context if formatted_context else extra_block
            logging.info(f"Prepended {len(extra_context_texts)} extra context(s) to prompt")

        chain = self.rag_prompt | llm | StrOutputParser()
        return chain.invoke({
            "context": formatted_context,
            "question": question,
            "answer_instructions": answer_instructions or "No additional formatting constraints.",
        })

    def _apply_runtime_answer_guardrail(
        self,
        *,
        question: str,
        llm,
        response: str,
        context: Dict[str, Any],
        runtime_guardrail: Optional[bool],
        runtime_guardrail_mode: Optional[str],
    ) -> tuple:
        if not self._runtime_answer_guardrail_enabled(runtime_guardrail):
            return response, {
                "enabled": False,
                "mode": self._runtime_answer_guardrail_mode(runtime_guardrail_mode),
            }

        mode = self._runtime_answer_guardrail_mode(runtime_guardrail_mode)
        verdict = evaluate_runtime_answer_guardrail(
            question=question,
            answer=response,
            chunks=context.get("chunks", []),
            llm=llm,
        )
        metadata: Dict[str, Any] = {
            "enabled": True,
            "mode": mode,
            "initial_verdict": verdict,
            "final_decision": verdict["decision"],
            "retried": False,
        }
        if verdict["decision"] == "keep":
            return response, metadata
        metadata["final_decision"] = "abstain"
        return RUNTIME_GUARDRAIL_ABSTENTION, metadata

    def generate_response(
        self,
        question: str,
        llm,
        document_names: List[str] = None,
        similarity_threshold: float = 0.1,
        max_chunks: int = 20,
        extra_context_texts: Optional[List[str]] = None,
        kg_name: str = None,
        answer_instructions: str = "",
        question_id: str = None,
        runtime_guardrail: Optional[bool] = None,
        runtime_guardrail_mode: Optional[str] = None,
        retrieval_temperature: float = 0.0,
        retrieval_shortlist_factor: int = 4,
        retrieval_sample_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Generate a vanilla RAG response using direct vector retrieval.

        Args:
            question: The question to answer
            llm: The LLM to use for response generation
            document_names: Optional list of document names to filter by
            similarity_threshold: Minimum similarity score for retrieval
            max_chunks: Maximum number of chunks to retrieve
            extra_context_texts: Optional list of additional context strings to prepend
                to the retrieved chunks (e.g. ground-truth question contexts for MIRAGE eval).
            kg_name: Optional KG name to filter retrieval to a specific named KG
        """
        try:
            logging.info(f"Starting vanilla RAG generate_response for question: {question}")

            # Get context using vanilla vector retrieval
            context = self.get_vanilla_rag_context(
                question,
                document_names=document_names,
                similarity_threshold=similarity_threshold,
                max_chunks=max_chunks,
                kg_name=kg_name,
                question_id=question_id,
                retrieval_temperature=retrieval_temperature,
                retrieval_shortlist_factor=retrieval_shortlist_factor,
                retrieval_sample_id=retrieval_sample_id,
            )
            logging.info(f"Got context with {len(context.get('chunks', []))} chunks")

            if not context["chunks"] and not extra_context_texts:
                return {
                    "response": "I couldn't find any relevant information to answer your question.",
                    "context": context,
                    "sources": [],
                    "confidence": 0.0
                }

            response = self._invoke_answer_chain(
                question=question,
                llm=llm,
                context=context,
                answer_instructions=answer_instructions,
                extra_context_texts=extra_context_texts,
            )
            response, guardrail = self._apply_runtime_answer_guardrail(
                question=question,
                llm=llm,
                response=response,
                context=context,
                runtime_guardrail=runtime_guardrail,
                runtime_guardrail_mode=runtime_guardrail_mode,
            )

            # Retrieval scores are already cosine similarities in [0, 1].
            avg_score = context["total_score"] / len(context["chunks"]) if context["chunks"] else 0
            confidence = avg_score
            if guardrail.get("enabled") and guardrail.get("final_decision") != "keep":
                confidence = 0.0

            return {
                "response": response,
                "context": context,
                "sources": context["documents"],
                "confidence": confidence,
                "guardrail": guardrail,
                "chunk_count": len(context["chunks"]),
                "retrieval_params": {
                    "similarity_threshold": similarity_threshold,
                    "max_chunks": max_chunks,
                    "retrieval_temperature": float(retrieval_temperature or 0.0),
                    "retrieval_shortlist_factor": int(retrieval_shortlist_factor or 1),
                    "retrieval_sample_id": int(retrieval_sample_id or 0),
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
