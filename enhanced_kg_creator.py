import json
import re
import hashlib
from datetime import datetime
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph
import os
import sys
import logging

# Import from local kg_utils
from kg_utils.common_functions import load_embedding_model

class EnhancedKGCreator:
    """
    Enhanced Knowledge Graph Creator with proper embedding integration for RAG
    """
    
    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        neo4j_uri: str = os.getenv("NEO4J_URI", "neo4j+s://01ae09e4.databases.neo4j.io"),
        neo4j_user: str = os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password: str = os.getenv("NEO4J_PASSWORD", "awhKHbIyHJZPAIuGhHpL9omIXw8Vupnnm_35XSDN2yg"),
        neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j"),
        embedding_model: str = os.getenv("EMBEDDING_PROVIDER", "openai")
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
        current_pos = 0
        
        for idx, chunk_text in enumerate(chunks):
            start_pos = current_pos
            end_pos = start_pos + len(chunk_text)
            
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
                "total_chunks": len(chunks),
                "embedding": chunk_embedding
            })
            
            current_pos += len(chunk_text) - self.chunk_overlap
            
        return formatted

    def _extract_entities_and_relationships_with_llm(self, chunk_text: str, llm) -> Dict[str, Any]:
        """
        Extract entities and relationships using LLM with proper escaped templates
        """
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert knowledge graph extraction system. Extract entities and relationships from medical text.

IMPORTANT RULES:
1. Focus on biomedical entities (diseases, treatments, symptoms, drugs)
2. Use ontology-compatible relationship types (TREATS, CAUSES, DIAGNOSES)
3. Include context-specific descriptions

Return response as JSON with this structure:
{{
  "entities": [
    {{"id": "entity_name", "type": "EntityType", "properties": {{"description": "domain-specific context"}}}}
  ],
  "relationships": [
    {{"source": "source_entity", "target": "target_entity", "type": "RELATIONSHIP_TYPE"}}
  ]
}}"""),
            ("human", "Extract knowledge from:\n{text}")
        ])
        
        try:
            chain = extraction_prompt | llm | StrOutputParser()
            response = chain.invoke({"text": chunk_text})
            
            # Handle JSON wrapped in markdown code blocks
            json_match = re.search(r'```(?:json)?\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
                
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing failed. Raw LLM response:\n{response}")
                return {"entities": [], "relationships": []}
            
        except Exception as e:
            logging.warning(f"LLM extraction failed: {str(e)}")
            return {"entities": [], "relationships": []}

    def generate_knowledge_graph(self, text: str, llm, file_name: str = None) -> Dict[str, Any]:
        """
        Generate knowledge graph from text with proper error handling
        """
        chunks = self._chunk_text(text)
        kg = {
            "nodes": [],
            "relationships": [],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "file_name": file_name
            }
        }
        
        for chunk in chunks:
            try:
                result = self._extract_entities_and_relationships_with_llm(chunk['text'], llm)
                kg["nodes"].extend(result.get("entities", []))
                kg["relationships"].extend(result.get("relationships", []))
            except Exception as e:
                logging.error(f"Error processing chunk {chunk['chunk_id']}: {e}")
        
        return kg
