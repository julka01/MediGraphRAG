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
from ontographrag.providers.model_providers import get_embedding_method

# Configurable parameters for different question types
RAG_CONFIG = {
    "statistical": {
        "default_max_chunks": 100,  # More chunks for statistical analysis
        "threshold_floor": 0.05,
        "threshold_factor": 0.03
    },
    "semantic": {
        "default_max_chunks": 15,  # Fewer chunks for focused semantic questions
        "threshold_floor": 0.08,
        "threshold_ceiling": 0.15,
        "threshold_boost": 0.02
    },
    "generic": {
        "default_max_chunks": 20,  # Default chunk count
        "default_threshold": 0.08
    }
}

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
        embedding_model: str = None  # Use environment variable if not provided
    ):
        # Load Neo4j credentials from environment variables if not provided
        self.neo4j_uri = neo4j_uri if neo4j_uri is not None else os.getenv("NEO4J_URI")
        self.neo4j_user = neo4j_user if neo4j_user is not None else os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password if neo4j_password is not None else os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

        if not self.neo4j_uri or not self.neo4j_user or not self.neo4j_password:
            raise ValueError("Neo4j connection parameters not found. Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables.")

        self._vector_index_available: Optional[bool] = None  # cached after first check

        # Initialize embedding model with consistent precedence:
        # explicit arg > EMBEDDING_PROVIDER > EMBEDDING_MODEL > sentence_transformers
        embedding_provider = (
            embedding_model
            or os.getenv("EMBEDDING_PROVIDER")
            or os.getenv("EMBEDDING_MODEL", "sentence_transformers")
        )
        embedding_provider = str(embedding_provider).lower()
        
        if embedding_provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
            logging.info("Initialized OpenAI embedding model: text-embedding-ada-002 (1536 dimensions)")
        else:
            # Fallback to sentence transformers
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("Initialized SentenceTransformer embedding model: all-MiniLM-L6-v2 (384 dimensions)")

        # ========================================================================
        # ORIGINAL: Detailed structured prompt with mandatory sections (COMMENTED OUT)
        # ========================================================================
        # self.rag_prompt_original = ChatPromptTemplate.from_messages([
        #     ("system", """You are an AI assistant that provides structured responses with detailed node traversal from the knowledge graph. All responses must follow the same format but content adapts to the user's query intent.
        # 
        # CRITICAL INSTRUCTIONS FOR RESPONSE FORMAT:
        # 
        # 1. **RECOMMENDATION/SUMMARY** (Always include this)
        #    - Provide a summary of the key findings and important information from the knowledge graph
        #    - Include key insights, important relationships, and relevant context
        #    - Be concise but comprehensive in highlighting what matters most
        # 
        # 2. **NODE TRAVERSAL PATH** (Always include this - detailed graph traversal)
        #    - Show the complete path through the knowledge graph nodes and relationships used to derive the answer
        #    - Format: Start Node Name (ID:start_node_id) → Relationship Type [rel_id] → End Node Name (ID:end_node_id)
        #    - Reference actual node and relationship IDs from the provided context
        #    - Include traversal depth and how each connection was discovered (text similarity/vector search)
        #    - Explain which chunks and entities were retrieved via vector similarity search
        #    - Show scores/confidence for each traversal step when available
        # 
        # 3. **REASONING PATH** (Always include this - logical progression)
        #    - Finding 1 → Triggers or yields or reveals → Finding 2 → etc.
        #    - Show how each piece of evidence connects to the next
        #    - Use actual node IDs when referencing entities from the knowledge graph
        #    - Format entity references as: "Entity Name (ID:actual_id_from_context)"
        #    - Explain confidence level if evidence is weak
        # 
        # 4. **COMBINED EVIDENCE** (Always include this)
        #    - Synthesize all relevant information into coherent evidence base
        #    - Show how different findings support or contradict each other
        #    - Highlight key relationships and patterns identified during traversal
        # 
        # 5. **NEXT STEPS** (Only include if user asks for next steps, actions, or follow-up guidance)
        #    - Suggest specific next actions based on the analysis
        #    - Provide actionable recommendations for implementation
        #    - If no specific next steps are appropriate, omit this section entirely
        # 
        # NODE ID REQUIREMENTS:
        # - When referencing entities from knowledge graph, ALWAYS use their actual node IDs from context
        # - NEVER use placeholder IDs like "ID:X", "ID:Y", "ID:Z", or "ID:actual_number"
        # - Show relationship traversal as: Node1 (ID:id1) → RELATIONSHIP_TYPE [rel_id] → Node2 (ID:id2)
        # 
        # VECTOR SEARCH DETAILS:
        # - Include vector similarity scores for chunks/entities retrieved
        # - Show which vector index (chunk vs entity) was used for retrieval
        # - Reference element IDs from vector search results
        # 
        # IMPORTANT: Base your answer ONLY on the provided context. Structure ALL responses with sections 2, 3, and 4, but only include sections 1 and 5 when appropriate for the user's intent.
        # 
        # Context Information:
        # {context}
        # 
        # Relevant Entities:
        # {entities}
        # 
        # User Query: {question}"""),
        #     ("human", "{question}")
        # ])
        # ========================================================================
        
        # KG-RAG prompt: instructs the LLM to reason over graph traversal paths
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable AI assistant. Answer the question using only the provided knowledge graph context.

The context has three parts:

1. TEXT CHUNKS — document passages retrieved by semantic similarity.
2. GRAPH TRAVERSAL PATHS — multi-hop chains discovered by walking the knowledge graph from seed entities.
   Each path shows how concepts connect: Entity A --RELATIONSHIP--> Entity B --RELATIONSHIP--> Entity C
3. ENTITIES — individual concepts found in the graph.

HOW TO REASON:
- Start from the entities most relevant to the question.
- Follow the graph paths to discover indirect relationships and supporting evidence.
- Prefer evidence that appears in BOTH a text chunk AND a graph path — that is the strongest signal.
- When citing a relationship, name the entities and the relationship type (e.g. "X --TREATS--> Y").

IMPORTANT:
- For yes/no questions, begin with "Yes" or "No" followed by your explanation.
- Ground every claim in the provided chunks or paths; do not invent facts.
- When citing evidence, reference the entity names and relationships (e.g. "EntityA --TREATS--> EntityB"), not document filenames.
- If the context is insufficient, say so clearly.

Text Chunks:
{context}

Knowledge Graph Traversal Paths:
{graph_paths}

Entities:
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
            database=self.neo4j_database,
            refresh_schema=False,
            sanitize=True
        )

    def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate query embedding across supported embedding backends."""
        if hasattr(self.embedding_model, "embed_query"):
            return self.embedding_model.embed_query(query)

        if hasattr(self.embedding_model, "encode"):
            embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

        raise ValueError("Unsupported embedding model interface. Expected embed_query or encode.")

    def classify_question_type(self, query: str) -> str:
        """
        Classify the question type to determine retrieval strategy
        """
        query_lower = query.lower().strip()

        # Statistical indicators
        statistical_terms = [
            "statistic", "tendencies", "trend", "correlation", "rate", "incidence",
            "prevalence", "distribution", "frequency", "proportion", "percentage",
            "average", "mean", "median", "variance", "standard deviation",
            "regression", "p-value", "significance", "confidence interval",
            "sample size", "cohort", "meta-analysis", "epidemiology",
            "how many", "how much", "what percentage", "what proportion",
            "quantity", "quantity of", "number of", "count", "total", "sum"
        ]

        # Semantic indicators
        semantic_terms = [
            "explain", "describe", "what is", "how does", "define", "meaning",
            "concept", "principle", "theory", "framework", "model", "interpretation",
            "understanding", "overview", "context", "background", "history",
            "development", "evolution", "mechanism", "process", "function"
        ]

        # Check for statistical terms (highest priority)
        if any(term in query_lower for term in statistical_terms):
            return "statistical"

        # Check question starters for quantitative questions
        quantitative_starters = ["how many", "how much", "what percentage", "what proportion"]
        if any(query_lower.startswith(starter) for starter in quantitative_starters):
            return "statistical"

        # Check for semantic terms
        if any(term in query_lower for term in semantic_terms):
            return "semantic"

        # Check question starters for semantic questions
        semantic_starters = ["what is", "how does", "explain", "describe"]
        if any(query_lower.startswith(starter) for starter in semantic_starters):
            return "semantic"

        return "generic"

    def calculate_dynamic_threshold(self, query: str, entity_count: int = 0) -> float:
        """
        Calculate dynamic similarity threshold based on question type and context
        """
        question_type = self.classify_question_type(query)
        config = RAG_CONFIG[question_type]

        if question_type == "statistical":
            # Lower threshold for statistical queries to catch more data
            base_threshold = max(config["threshold_floor"], 0.08 - (entity_count * config["threshold_factor"]))
            return min(base_threshold, 0.15)

        elif question_type == "semantic":
            # Slightly higher threshold for focused semantic questions
            base_threshold = min(config["threshold_ceiling"], 0.08 + config["threshold_boost"])
            return max(base_threshold, 0.06)

        else:  # generic
            return config["default_threshold"]

    def get_adaptive_retrieval_params(self, query: str) -> Dict[str, Any]:
        """
        Get adaptive retrieval parameters based on question classification
        """
        question_type = self.classify_question_type(query)

        # For statistical questions, use reasonable chunk limit to avoid timeouts
        if question_type == "statistical":
            max_chunks = 200  # Aggressive limit to prevent timeouts - focus on quality over quantity
        else:
            max_chunks = RAG_CONFIG[question_type]["default_max_chunks"]

        params = {
            "question_type": question_type,
            "similarity_threshold": self.calculate_dynamic_threshold(query, entity_count=0),
            "max_chunks": max_chunks
        }

        logging.info(f"Question '{query[:50]}...' classified as '{question_type}': threshold={params['similarity_threshold']:.3f}, max_chunks={params['max_chunks']} (total available: all for stats)")
        return params

    def check_vector_index(self) -> bool:
        """
        Check if vector index exists and has correct dimensions. Result is cached after first call.
        """
        if self._vector_index_available is not None:
            return self._vector_index_available
        try:
            graph = self._create_neo4j_connection()
            result = graph.query("SHOW INDEXES WHERE type = 'VECTOR' AND name = 'vector'")
            if not result:
                logging.warning("No vector index found - falling back to text search")
                self._vector_index_available = False
                return False

            # Ensure index is online when state is available.
            index_info = result[0]
            state = index_info.get("state")
            if state and str(state).upper() != "ONLINE":
                logging.warning(f"Vector index exists but is not ONLINE (state={state})")
                self._vector_index_available = False
                return False

            # Probe queryNodes directly with current embedding backend to validate
            # interface compatibility and effective dimension match.
            probe_embedding = self._generate_query_embedding("test")
            graph.query(
                """
                CALL db.index.vector.queryNodes('vector', 1, $query_vector)
                YIELD node, score
                RETURN count(node) AS n
                """,
                {"query_vector": probe_embedding},
            )

            logging.info("Vector index probe succeeded — result cached")
            self._vector_index_available = True
            return True
        except Exception as e:
            logging.error(f"Error checking vector index: {e}")
            self._vector_index_available = False
            return False

    def _expand_entities_via_graph(
        self,
        graph,
        seed_entity_ids: List[str],
        kg_name: str = None,
        max_hops: int = 2,
        max_neighbors: int = 30,
    ) -> Dict[str, Any]:
        """
        Multi-hop graph traversal from seed entities.

        Starting from the entities found in the initially retrieved chunks,
        this method walks the knowledge graph up to ``max_hops`` relationship
        hops and returns:
          - ``neighbors``: newly discovered entities (not in the seed set)
          - ``paths``: human-readable traversal paths for the LLM prompt,
              e.g. "Metformin --TREATS--> Type2Diabetes --HAS_COMPLICATION--> Nephropathy"

        Neighbors are ranked by:
          1. How many distinct seed entities connect to them (higher = more central)
          2. Minimum hop distance (closer = higher priority)
        """
        if not seed_entity_ids:
            return {"neighbors": {}, "paths": []}

        try:
            params: Dict[str, Any] = {
                "seed_ids": seed_entity_ids,
                "max_neighbors": max_neighbors,
            }

            # Scope seeds and neighbors to the target KG when one is specified,
            # preventing cross-KG contamination when multiple KGs share a database.
            if kg_name:
                params["kg_name"] = kg_name
                seed_scope = """EXISTS {
                    MATCH (seed)<-[:MENTIONS|HAS_ENTITY]-(c:Chunk)-[:PART_OF]->(d:Document {kgName: $kg_name})
                }"""
                kg_filter = """
                AND EXISTS {
                    MATCH (neighbor)<-[:MENTIONS|HAS_ENTITY]-(c2:Chunk)-[:PART_OF]->(d2:Document {kgName: $kg_name})
                }"""
            else:
                seed_scope = "true"
                kg_filter = ""

            traversal_query = f"""
            MATCH (seed:__Entity__)
            WHERE seed.id IN $seed_ids
              AND {seed_scope}
            MATCH path = (seed)-[*1..{max_hops}]->(neighbor:__Entity__)
            WHERE NOT neighbor.id IN $seed_ids
              AND neighbor.id IS NOT NULL
              {kg_filter}
            WITH neighbor,
                 length(path) AS hops,
                 [n IN nodes(path) | coalesce(n.name, n.id)] AS node_names,
                 [r IN relationships(path) | type(r)] AS rel_types,
                 seed.id AS seed_id
            WITH neighbor,
                 min(hops) AS min_hops,
                 count(DISTINCT seed_id) AS seed_connections,
                 collect(DISTINCT {{nodes: node_names, rels: rel_types}})[0..3] AS sample_paths
            RETURN
                 neighbor.id AS id,
                 neighbor.name AS name,
                 coalesce(neighbor.type, 'Entity') AS type,
                 elementId(neighbor) AS element_id,
                 min_hops,
                 seed_connections,
                 sample_paths
            ORDER BY seed_connections DESC, min_hops ASC
            LIMIT $max_neighbors
            """

            results = graph.query(traversal_query, params)

            neighbors: Dict[str, Any] = {}
            seen_paths: set = set()
            paths: List[Dict[str, Any]] = []

            for row in results:
                neighbor_id = row["id"]
                if not neighbor_id:
                    continue

                neighbors[neighbor_id] = {
                    "id": neighbor_id,
                    "name": row["name"] or neighbor_id,
                    "type": row["type"],
                    "element_id": row["element_id"],
                    "min_hops": row["min_hops"],
                    "seed_connections": row["seed_connections"],
                    "source": "graph_traversal",
                }

                # Build human-readable path strings for the prompt
                for path_data in (row["sample_paths"] or []):
                    node_names = path_data.get("nodes", [])
                    rel_types = path_data.get("rels", [])
                    if node_names and rel_types and len(node_names) == len(rel_types) + 1:
                        parts = [node_names[0]]
                        for rel, node in zip(rel_types, node_names[1:]):
                            parts.extend([f"--{rel}-->", node])
                        path_str = " ".join(parts)
                        if path_str not in seen_paths:
                            seen_paths.add(path_str)
                            paths.append({"path": path_str, "hops": row["min_hops"]})

            logging.info(
                "Graph traversal: %d neighbors, %d paths from %d seeds (max_hops=%d)",
                len(neighbors), len(paths), len(seed_entity_ids), max_hops,
            )
            return {"neighbors": neighbors, "paths": paths}

        except Exception as e:
            logging.warning("Graph traversal failed: %s", e)
            return {"neighbors": {}, "paths": []}

    def get_rag_context(self, query: str, document_names: List[str] = None, similarity_threshold: float = 0.08, max_chunks: int = 20, kg_name: str = None) -> Dict[str, Any]:
        """
        Get comprehensive RAG context including chunks, entities, and relationships using vector search.
        
        Args:
            query: The search query
            document_names: Optional list of document names to filter by
            similarity_threshold: Minimum similarity score for retrieval
            max_chunks: Maximum number of chunks to retrieve
            kg_name: Optional KG name to filter retrieval to a specific named KG
        """
        try:
            graph = self._create_neo4j_connection()

            # First check if we have any data in the knowledge graph
            # If kg_name is specified, check only for that KG
            if kg_name:
                check_query = """
                MATCH (d:Document {kgName: $kg_name})<-[:PART_OF]-(c:Chunk)
                RETURN count(c) as chunk_count
                """
                check_result = graph.query(check_query, {"kg_name": kg_name})
                
                if not check_result or check_result[0]['chunk_count'] == 0:
                    logging.warning(f"No chunks found in KG '{kg_name}'")
                    return {
                        "query": query,
                        "chunks": [],
                        "entities": {},
                        "relationships": [],
                        "graph_neighbors": {},
                        "traversal_paths": [],
                        "documents": [],
                        "total_score": 0,
                        "entity_count": 0,
                        "relationship_count": 0,
                        "kg_name": kg_name,
                        "error": f"No data found in KG '{kg_name}'. Please process documents to this KG first."
                    }
            else:
                check_query = "MATCH (c:Chunk) RETURN count(c) as chunk_count LIMIT 1"
                check_result = graph.query(check_query)

                if not check_result or check_result[0]['chunk_count'] == 0:
                    logging.warning("No chunks found in knowledge graph")
                    return {
                        "query": query,
                        "chunks": [],
                        "entities": {},
                        "relationships": [],
                        "graph_neighbors": {},
                        "traversal_paths": [],
                        "documents": [],
                        "total_score": 0,
                        "entity_count": 0,
                        "relationship_count": 0,
                        "error": "No data found in knowledge graph. Please upload and process a document first."
                    }

            # Check vector index before attempting vector search
            has_vector_index = self.check_vector_index()
            
            if has_vector_index:
                # Try vector search with KG filtering
                logging.info(f"Attempting vector similarity search (kg_name: {kg_name})")
                try:
                    context = self._vector_similarity_search(graph, query, document_names, similarity_threshold, max_chunks, kg_name)
                    # If vector search returned no chunks, fall back to text search
                    if not context["chunks"]:
                        logging.warning("Vector search returned no results, falling back to text search")
                        return self._text_similarity_search(graph, query, document_names, max_chunks, kg_name)
                    return context
                except Exception as e:
                    logging.error(f"Vector search failed: {e}, falling back to text search")
                    return self._text_similarity_search(graph, query, document_names, max_chunks, kg_name)
            else:
                # Fall back to text search
                logging.info("No vector index available, using text similarity search")
                return self._text_similarity_search(graph, query, document_names, max_chunks, kg_name)

        except Exception as e:
            logging.error(f"Error getting RAG context: {e}")
            return {
                "query": query,
                "chunks": [],
                "entities": {},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
                "documents": [],
                "total_score": 0,
                "entity_count": 0,
                "relationship_count": 0,
                "error": str(e)
            }

    def _text_similarity_search(self, graph, query: str, document_names: List[str] = None, max_chunks: int = 20, kg_name: str = None) -> Dict[str, Any]:
        """
        Text-based similarity search as fallback when vector search fails
        """
        try:
            logging.info(f"Using text similarity search as fallback (kg_name: {kg_name})")

            # Build WHERE clause based on filters and text match
            where_conditions = ["ANY(term IN $search_terms WHERE toLower(c.text) CONTAINS term)"]
            params = {}
            
            if kg_name:
                where_conditions.append("d.kgName = $kg_name")
                params["kg_name"] = kg_name
                
            if document_names:
                where_conditions.append("d.fileName IN $document_names")
                params["document_names"] = document_names
            
            # Build a single WHERE clause to avoid duplicate WHERE errors
            where_clause = "WHERE " + " AND ".join(where_conditions)

            # Text similarity search: score by number of query terms matched in chunk text
            search_query = f"""
            MATCH (c:Chunk)-[:PART_OF]->(d:Document)
            {where_clause}
            WITH c, d,
                 size([term IN $search_terms WHERE toLower(c.text) CONTAINS term]) AS matched_terms
            OPTIONAL MATCH (c)-[:MENTIONS|HAS_ENTITY]-(e:__Entity__)
            WITH c, d, matched_terms, collect(DISTINCT e) AS chunk_entities
            RETURN
                c.text AS text,
                c.id AS chunk_id,
                elementId(c) AS chunk_element_id,
                toFloat(matched_terms) / size($search_terms) AS score,
                d.fileName AS document,
                d.kgName AS kg_name,
                [entity IN chunk_entities WHERE entity IS NOT NULL | {{
                    id: coalesce(entity.id, entity.name),
                    element_id: elementId(entity),
                    type: coalesce(entity.type, 'Entity'),
                    description: coalesce(entity.name, '')
                }}] AS entities
            ORDER BY score DESC
            LIMIT $max_chunks
            """

            # Tokenized fallback matching is more robust than a single 3-word phrase.
            raw_terms = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", query)]
            search_terms = [t for t in raw_terms if len(t) >= 3][:8]
            if not search_terms:
                search_terms = [query.lower().strip()]

            params.update({
                "search_terms": search_terms,
                "max_chunks": max_chunks
            })

            results = graph.query(search_query, params)

            context = {
                "query": query,
                "chunks": [],
                "entities": {},
                "relationships": [],
                "documents": set(),
                "total_score": 0,
                "search_method": "text_similarity_fallback"
            }

            for result in results:
                chunk_info = {
                    "text": result["text"],
                    "chunk_id": result["chunk_id"],
                    "chunk_element_id": result["chunk_element_id"],
                    "score": result["score"],
                    "document": result["document"],
                    "kg_name": result.get("kg_name"),
                    "entities": result.get("entities") or []
                }
                context["chunks"].append(chunk_info)
                context["documents"].add(result["document"])
                context["total_score"] += result["score"]

                # Collect unique entities
                for entity in (result.get("entities") or []):
                    entity_id = entity["id"]
                    if entity_id not in context["entities"]:
                        context["entities"][entity_id] = {
                            "id": entity_id,
                            "element_id": entity["element_id"],
                            "type": entity["type"],
                            "description": entity["description"],
                            "mentioned_in_chunks": []
                        }
                    if result.get("chunk_id"):
                        context["entities"][entity_id]["mentioned_in_chunks"].append(result["chunk_id"])

            # Multi-hop graph traversal from seed entities
            seed_ids = list(context["entities"].keys())
            expansion = self._expand_entities_via_graph(graph, seed_ids, kg_name=kg_name)
            context["graph_neighbors"] = expansion["neighbors"]
            context["traversal_paths"] = expansion["paths"]

            # Merge neighbor entities into the entity dict (tagged so they're
            # distinguishable from directly-retrieved seed entities)
            for nid, ninfo in expansion["neighbors"].items():
                if nid not in context["entities"]:
                    context["entities"][nid] = {
                        "id": nid,
                        "element_id": ninfo["element_id"],
                        "type": ninfo["type"],
                        "description": ninfo["name"],
                        "mentioned_in_chunks": [],
                        "source": "graph_traversal",
                        "min_hops": ninfo["min_hops"],
                    }

            # Relationships between ALL entities (seeds + neighbors)
            all_entity_ids = list(context["entities"].keys())
            if all_entity_ids:
                relationship_query = """
                MATCH (e1:__Entity__)-[r]->(e2:__Entity__)
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
                            "element_id": rel_result["relationship_element_id"],
                        })

            context["documents"] = list(context["documents"])
            context["entity_count"] = len(context["entities"])
            context["relationship_count"] = len(context["relationships"])

            return context

        except Exception as e:
            logging.error(f"Error in text similarity search: {e}")
            return {
                "query": query,
                "chunks": [],
                "entities": {},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
                "documents": [],
                "total_score": 0,
                "search_method": "text_similarity_fallback",
                "error": str(e)
            }

    def format_context_for_llm(self, context: Dict[str, Any]) -> tuple:
        """
        Format the context for the LLM prompt.

        Returns:
            (formatted_chunks, formatted_entities, formatted_paths)
        """
        # --- chunks ---
        chunk_texts = []
        for i, chunk in enumerate(context["chunks"], 1):
            doc = chunk.get('document', 'unknown')
            chunk_text = f"[Source: {doc}]\n{chunk['text']}\n"
            chunk_texts.append(chunk_text)
        formatted_context = "\n".join(chunk_texts)

        # --- traversal paths (most important new section) ---
        traversal_paths = context.get("traversal_paths", [])
        if traversal_paths:
            # Sort: 1-hop paths first, then multi-hop
            sorted_paths = sorted(traversal_paths, key=lambda p: p.get("hops", 99))
            path_lines = [f"  {p['path']}" for p in sorted_paths]
            formatted_paths = "\n".join(path_lines)
        else:
            formatted_paths = "(No graph paths discovered — only seed entities available)"

        # --- entities (seeds first, then graph-traversal neighbors) ---
        entity_texts = []
        seed_entities = {
            eid: info for eid, info in context["entities"].items()
            if info.get("source") != "graph_traversal"
        }
        neighbor_entities = {
            eid: info for eid, info in context["entities"].items()
            if info.get("source") == "graph_traversal"
        }

        if seed_entities:
            entity_texts.append("Seed entities (directly from retrieved chunks):")
            for entity_id, info in seed_entities.items():
                eid = info.get('id') or entity_id or 'Unknown'
                etype = info.get('type') or 'Unknown'
                line = f"  - {eid} | Type: {etype}"
                desc = info.get("description")
                if desc and desc != eid:
                    line += f" | {desc}"
                entity_texts.append(line)

        if neighbor_entities:
            entity_texts.append(f"\nGraph-traversal neighbors ({len(neighbor_entities)} discovered):")
            for entity_id, info in neighbor_entities.items():
                eid = info.get('id') or entity_id or 'Unknown'
                etype = info.get('type') or 'Unknown'
                hops = info.get("min_hops", "?")
                line = f"  - {eid} | Type: {etype} | {hops}-hop neighbor"
                desc = info.get("description")
                if desc and desc != eid:
                    line += f" | {desc}"
                entity_texts.append(line)

        formatted_entities = "\n".join(entity_texts) if entity_texts else "(No entities found)"

        return formatted_context, formatted_entities, formatted_paths

    def generate_response(self, question: str, llm, document_names: List[str] = None, similarity_threshold: float = None, max_chunks: int = None, timeout: float = None, extra_context_texts: Optional[List[str]] = None, kg_name: str = None) -> Dict[str, Any]:
        """
        Generate a RAG response using the knowledge graph with adaptive retrieval.

        Args:
            extra_context_texts: Optional list of additional context strings to prepend
                to the retrieved chunks (e.g. ground-truth question contexts for MIRAGE eval).
            kg_name: Optional KG name to filter retrieval to a specific named KG
        """
        try:
            logging.info(f"Starting generate_response for question: {question}")

            # Use adaptive retrieval parameters if not explicitly provided
            if similarity_threshold is None or max_chunks is None:
                retrieval_params = self.get_adaptive_retrieval_params(question)
                similarity_threshold = similarity_threshold or retrieval_params["similarity_threshold"]
                max_chunks = max_chunks or retrieval_params["max_chunks"]

            # Get context from knowledge graph (with optional KG filtering)
            context = self.get_rag_context(question, document_names=document_names, similarity_threshold=similarity_threshold, max_chunks=max_chunks, kg_name=kg_name)
            logging.info(f"Got context with {context.get('entity_count', 0)} entities")

            if not context["chunks"] and not extra_context_texts:
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
            formatted_context, formatted_entities, formatted_paths = self.format_context_for_llm(context)

            # Prepend extra (ground-truth) contexts if provided, so the LLM always has
            # the relevant source material even when KG retrieval misses it.
            if extra_context_texts:
                extra_block = "\n\n".join(
                    f"Provided Context {i+1}:\n{t}" for i, t in enumerate(extra_context_texts) if t.strip()
                )
                formatted_context = extra_block + "\n\n" + formatted_context if formatted_context else extra_block
                logging.info(f"Prepended {len(extra_context_texts)} extra context(s) to prompt")

            # Generate response using LLM
            chain = self.rag_prompt | llm | StrOutputParser()
            response = chain.invoke({
                "context": formatted_context,
                "graph_paths": formatted_paths,
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
            confidence = max(0.0, min((avg_score - 0.05) / 0.95, 1.0))  # Scale similarity score to 0-1
            # Penalize when multiple entities were retrieved but no graph relationships connected them
            # — this suggests the KG graph path wasn't useful, answer relies on text chunks only
            if context.get("relationship_count", 0) == 0 and context.get("entity_count", 0) > 1:
                confidence = min(confidence, 0.6)

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
                "relationship_count": context["relationship_count"],
                "retrieval_params": {
                    "question_type": self.classify_question_type(question),
                    "similarity_threshold": similarity_threshold,
                    "max_chunks": max_chunks,
                    "timeout": timeout
                }
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
            OPTIONAL MATCH (e)<-[:HAS_ENTITY|MENTIONS]-(c:Chunk)-[:PART_OF]->(d:Document)
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
                    "entities": result.get("entities") or [],
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

    def _vector_similarity_search(self, graph, query: str, document_names: List[str] = None, similarity_threshold: float = 0.08, max_chunks: int = 20, kg_name: str = None) -> Dict[str, Any]:
        """
        Vector similarity search using real sentence transformer embeddings.
        
        Args:
            graph: Neo4j graph connection
            query: The search query
            document_names: Optional list of document names to filter by
            similarity_threshold: Minimum similarity score for retrieval
            max_chunks: Maximum number of chunks to retrieve
            kg_name: Optional KG name to filter retrieval to a specific named KG
        """
        try:
            logging.info(f"Using vector similarity search with real embeddings (kg_name: {kg_name})")

            query_embedding = self._generate_query_embedding(query)
            logging.info(f"Generated query embedding with shape: {len(query_embedding)}")

            # Build the vector search query with optional filtering by kg_name and/or document_names
            # Relationship is: Chunk -[PART_OF]-> Document
            # We need to filter by either document_names or kg_name (or both)
            
            # Build one WHERE clause that always includes score threshold
            where_conditions = ["score >= $similarity_threshold"]
            params = {
                "query_vector": query_embedding,
                "similarity_threshold": similarity_threshold
            }
            
            if kg_name:
                where_conditions.append("d.kgName = $kg_name")
                params["kg_name"] = kg_name
                
            if document_names:
                where_conditions.append("d.fileName IN $document_names")
                params["document_names"] = document_names
            
            where_clause = "WHERE " + " AND ".join(where_conditions)

            # When filtering by kg_name or document_names the vector index is
            # queried globally first, then filtered.  Overfetch so that the
            # target KG's chunks are not accidentally excluded because they
            # didn't rank in the global top-N.
            # Overfetch so KG-filtered chunks aren't excluded by global top-N,
            # but cap at 500 to avoid timeouts on large statistical queries.
            retrieval_count = min(max_chunks * 20, 500) if (kg_name or document_names) else max_chunks

            search_query = f"""
            CALL db.index.vector.queryNodes('vector', {retrieval_count}, $query_vector)
            YIELD node AS chunk, score
            MATCH (chunk)-[:PART_OF]->(d:Document)
            {where_clause}
            OPTIONAL MATCH (chunk)-[:MENTIONS|HAS_ENTITY]-(e:__Entity__)
            WITH chunk, score, d,
                 collect(DISTINCT e) AS chunk_entities
            RETURN
                chunk.text AS text,
                chunk.id AS chunk_id,
                elementId(chunk) AS chunk_element_id,
                score,
                d.fileName AS document,
                d.kgName AS kg_name,
                [entity IN chunk_entities WHERE entity IS NOT NULL | {{
                    id: coalesce(entity.id, entity.name),
                    element_id: elementId(entity),
                    type: coalesce(entity.type, 'Entity'),
                    description: coalesce(entity.name, '')
                }}] AS entities
            ORDER BY score DESC
            LIMIT $max_chunks
            """
            params["max_chunks"] = max_chunks

            results = graph.query(search_query, params)

            context = {
                "query": query,
                "chunks": [],
                "entities": {},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
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
                    "kg_name": result.get("kg_name"),
                    "entities": result.get("entities") or []
                }
                context["chunks"].append(chunk_info)
                context["documents"].add(result["document"])
                context["total_score"] += result["score"]

                # Collect unique entities
                for entity in (result.get("entities") or []):
                    entity_id = entity["id"]
                    if entity_id not in context["entities"]:
                        context["entities"][entity_id] = {
                            "id": entity_id,
                            "element_id": entity["element_id"],
                            "type": entity["type"],
                            "description": entity["description"],
                            "mentioned_in_chunks": []
                        }
                    if result.get("chunk_id"):
                        context["entities"][entity_id]["mentioned_in_chunks"].append(result["chunk_id"])

            # Multi-hop graph traversal from seed entities found in chunks
            seed_ids = list(context["entities"].keys())
            expansion = self._expand_entities_via_graph(graph, seed_ids, kg_name=kg_name)
            context["graph_neighbors"] = expansion["neighbors"]
            context["traversal_paths"] = expansion["paths"]

            # Merge neighbor entities into the entity dict
            for nid, ninfo in expansion["neighbors"].items():
                if nid not in context["entities"]:
                    context["entities"][nid] = {
                        "id": nid,
                        "element_id": ninfo["element_id"],
                        "type": ninfo["type"],
                        "description": ninfo["name"],
                        "mentioned_in_chunks": [],
                        "source": "graph_traversal",
                        "min_hops": ninfo["min_hops"],
                    }

            # Relationships between ALL entities (seeds + neighbors)
            all_entity_ids = list(context["entities"].keys())
            if all_entity_ids:
                relationship_query = """
                MATCH (e1:__Entity__)-[r]->(e2:__Entity__)
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
                            "element_id": rel_result["relationship_element_id"],
                        })

            context["documents"] = list(context["documents"])
            context["entity_count"] = len(context["entities"])
            context["relationship_count"] = len(context["relationships"])

            return context

        except Exception as e:
            logging.error(f"Error in vector similarity search: {e}")
            # Don't return error — let caller fall back to text search
            raise Exception(f"Vector search failed: {str(e)}")

    def _extract_used_entities_and_chunks(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities and chunks that are actually mentioned in the RAG answer,
        using multiple fallback strategies to ensure filtering works
        """
        used_entities = []
        used_chunks = []
        reasoning_edges = []

        try:
            context_entities = context.get("entities", {})
            context_chunks = context.get("chunks", [])
            context_relationships = context.get("relationships", [])

            response_lower = response.lower()
            mentioned_entity_names = set()
            mentioned_chunk_ids = set()

            # Match entities by name: use description (= n.name, human-readable) as primary
            # key, falling back to id only if description is absent.
            # entity_info["id"] is a UUID-prefixed key, not a readable name — don't use it
            # for text matching against the LLM response.
            for entity_key, entity_info in context_entities.items():
                # description = coalesce(entity.name, '') from Cypher — the readable name
                entity_name = (entity_info.get("description") or entity_info.get("id") or "").lower().replace("_", " ").strip()

                if entity_name and len(entity_name) > 2:
                    if entity_name in response_lower:
                        mentioned_entity_names.add(entity_key)
                        continue
                    # Prefix match (handles abbreviations like "BRCA" matching "BRCA2")
                    if len(entity_name) > 4 and any(
                        word.startswith(entity_name[:4]) for word in response_lower.split()
                    ):
                        mentioned_entity_names.add(entity_key)
                        continue

                # Also match on individual significant words of the entity name
                for word in entity_name.split():
                    if len(word) > 4 and word in response_lower:
                        mentioned_entity_names.add(entity_key)
                        break

            # Map chunk ordinal references ("chunk 2") back to actual chunk IDs
            for chunk_num in re.findall(r'\bchunk\s*(\d+)\b', response, re.IGNORECASE):
                try:
                    chunk_index = int(chunk_num) - 1
                    if 0 <= chunk_index < len(context_chunks):
                        matched_chunk = context_chunks[chunk_index]
                        if matched_chunk.get("chunk_id"):
                            mentioned_chunk_ids.add(matched_chunk["chunk_id"])
                        if matched_chunk.get("chunk_element_id"):
                            mentioned_chunk_ids.add(matched_chunk["chunk_element_id"])
                except (ValueError, TypeError):
                    continue

            logging.info(f"Name-based entity matches: {len(mentioned_entity_names)}")

            for entity_id, entity_info in context_entities.items():
                if entity_id in mentioned_entity_names:
                    used_entities.append({
                        "id": entity_id,
                        "element_id": entity_info.get("element_id", ""),
                        "type": entity_info.get("type", "Unknown"),
                        "description": entity_info.get("description", ""),
                        "reasoning_context": "mentioned by name"
                    })

            # Strategy 4: Include chunks that contain mentioned entities (semantic linking)
            relevant_chunk_ids = set()
            for entity_id in {e['id'] for e in used_entities}:
                for chunk in context_chunks:
                    chunk_entities = chunk.get('entities', [])
                    if any(ce.get('id') == entity_id for ce in chunk_entities):
                        relevant_chunk_ids.add(chunk.get('chunk_id'))
                        relevant_chunk_ids.add(chunk.get('chunk_element_id'))

            # Include explicitly mentioned chunks
            for chunk in context_chunks:
                chunk_id = chunk.get("chunk_id", "")
                chunk_element_id = chunk.get("chunk_element_id", "")
                is_direct = chunk_id in mentioned_chunk_ids or chunk_element_id in mentioned_chunk_ids
                has_relevant_entity = chunk_id in relevant_chunk_ids or chunk_element_id in relevant_chunk_ids
                if is_direct or has_relevant_entity:
                    text = chunk.get("text", "")
                    used_chunks.append({
                        "id": chunk_id,
                        "element_id": chunk_element_id,
                        "text": text[:200] + "..." if len(text) > 200 else text,
                        "reasoning_context": "directly referenced" if is_direct else "contains relevant entities"
                    })

            # Strategy 5: Find relationships between selected entities
            if used_entities:  # Only look for relationships if we have filtered entities
                # Build lookup: element_id or id → human-readable name
                entity_name_lookup = {}
                for e in used_entities:
                    name = e.get("description") or e.get("id", "")
                    if e.get("element_id"):
                        entity_name_lookup[e["element_id"]] = name
                    if e.get("id"):
                        entity_name_lookup[e["id"]] = name

                used_entity_element_ids = {e['element_id'] for e in used_entities if e['element_id']}
                used_entity_ids = {e['id'] for e in used_entities}

                for rel in context_relationships:
                    source_id = rel.get("source", "")
                    target_id = rel.get("target", "")
                    source_element_id = rel.get("source_element_id", "")
                    target_element_id = rel.get("target_element_id", "")

                    # Include edge if both connected entities are in our filtered set
                    source_in_set = (source_id in used_entity_ids or
                                   source_element_id in used_entity_element_ids)
                    target_in_set = (target_id in used_entity_ids or
                                   target_element_id in used_entity_element_ids)

                    if source_in_set and target_in_set:
                        src_key = source_element_id or source_id
                        tgt_key = target_element_id or target_id
                        reasoning_edges.append({
                            "from": src_key,
                            "to": tgt_key,
                            "from_name": entity_name_lookup.get(src_key) or entity_name_lookup.get(source_id) or src_key,
                            "to_name": entity_name_lookup.get(tgt_key) or entity_name_lookup.get(target_id) or tgt_key,
                            "relationship": rel.get("type", "CONNECTED_TO"),
                            "reasoning_context": "connects relevant entities"
                        })

            logging.info(f"Filtered to {len(used_entities)} entities ({len([e for e in used_entities if 'mentioned by name' in e['reasoning_context']])} by name) and {len(used_chunks)} chunks")
            logging.info(f"Found {len(reasoning_edges)} reasoning edges")

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
