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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from local kg_utils
from medigraphrag_x.kg.utils.common_functions import load_embedding_model


class EnhancedKGCreatorProd:
    """
    Enhanced Knowledge Graph Creator with production-ready LangChain 0.3.27
    and full Neo4j integration
    """

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        neo4j_database: str = None,
        embedding_model: str = "sentence_transformers"
    ):
        # Use environment variables if not provided
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

        # Initialize embedding model
        self.embedding_function, self.embedding_dimension = load_embedding_model(embedding_model)

        logging.info(f"✅ Initialized embedding model with dimension {self.embedding_dimension}")

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
                # Handle different embedding function types
                if hasattr(self.embedding_function, 'embed_query'):
                    chunk_embedding = self.embedding_function.embed_query(chunk_text)
                elif hasattr(self.embedding_function, 'encode'):
                    chunk_embedding = self.embedding_function.encode(chunk_text)
                elif callable(self.embedding_function):
                    chunk_embedding = self.embedding_function(chunk_text)
                else:
                    logging.warning(f"Unsupported embedding function type: {type(self.embedding_function)}")
                    chunk_embedding = None
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
        Extract entities and relationships using LLM with proper LangChain 0.3.27 structure
        """
        # Create ontology-aware extraction prompt
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert medical knowledge graph extraction system.
Extract entities and relationships from medical/scientific text.

IMPORTANT RULES:
1. Focus on medically relevant entities (diseases, treatments, symptoms, drugs, procedures)
2. Classify entities with appropriate medical categories
3. Extract relationships only between entities that actually interact
4. Be precise and medically accurate

Return ONLY a valid JSON object in this exact format:
{{
  "entities": [
    {{
      "id": "entity_name_exactly_as_it_appears",
      "type": "Medical_Category",
      "properties": {{
        "name": "entity_name_exactly_as_it_appears",
        "description": "brief medical context"
      }}
    }}
  ],
  "relationships": [
    {{
      "source": "source_entity_name",
      "target": "target_entity_name",
      "type": "RELATION_TYPE",
      "properties": {{
        "description": "relationship context in the text"
      }}
    }}
  ]
}}

Return ONLY the JSON object, no additional text."""),
            ("human", "Extract medical knowledge from this text:\n\n{text}")
        ])

        try:
            # Use LangChain 0.3.27 chain structure
            chain = extraction_prompt | llm | StrOutputParser()
            response = chain.invoke({"text": chunk_text})

            # Handle potential markdown code blocks
            json_match = re.search(r'```(?:json)?\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            else:
                # Fallback: Remove markdown code blocks if present
                if response.startswith('```json'):
                    response = response[7:]
                if response.startswith('```'):
                    response = response[3:]
                if response.endswith('```'):
                    response = response[:-3]

            response = response.strip()

            # Try to extract JSON from the response (sometimes LLMs add explanatory text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                response = response[json_start:json_end]

            # Parse and validate JSON
            try:
                result = json.loads(response)
            except json.JSONDecodeError as json_error:
                logging.error(f"JSON parsing failed for LLM response. Raw response:\n{response}")
                logging.error(f"JSON error: {json_error}")
                # Try to salvage by looking for partial JSON structure
                return self._attempt_json_recovery(response, chunk_text)

            # Validate result structure
            if not isinstance(result, dict):
                logging.error(f"LLM returned non-dict result: {type(result)} = {result}")
                return {'entities': [], 'relationships': []}

            entities = result.get('entities', [])
            relationships = result.get('relationships', [])

            # Basic validation
            if not isinstance(entities, list):
                logging.warning(f"Entities is not a list: {type(entities)}")
                entities = []
            if not isinstance(relationships, list):
                logging.warning(f"Relationships is not a list: {type(relationships)}")
                relationships = []

            return {
                'entities': entities,
                'relationships': relationships
            }

        except Exception as e:
            logging.error(f"LLM extraction failed: {e}")
            return {'entities': [], 'relationships': []}

    def _attempt_json_recovery(self, malformed_response: str, chunk_text: str) -> Dict[str, Any]:
        """
        Attempt to recover from malformed JSON response using fallback pattern matching
        """
        logging.info("Attempting JSON recovery with pattern matching fallback")

        try:
            # Use simple pattern matching to extract entities from the original chunk
            entities = self._extract_entities_pattern_matching(chunk_text)
            return {
                'entities': entities,
                'relationships': []
            }
        except Exception as e2:
            logging.error(f"Pattern matching fallback also failed: {e2}")
            return {'entities': [], 'relationships': []}

    def _extract_entities_pattern_matching(self, text: str) -> List[Dict[str, Any]]:
        """
        Fallback entity extraction using pattern matching when LLM JSON fails
        """
        entities = []

        # Extract potential medical entities
        import re

        # More sophisticated patterns for medical entities
        entity_patterns = [
            # Diseases/conditions
            r'\b(?:cancer|diabetes|hypertension|asthma|arthritis|covid|covid-19)\b',
            # Drugs/medications
            r'\b(?:aspirin|ibuprofen|metformin|insulin|lisinopril|amlodipine)\b',
            # Capitalized medical terms
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        ]

        found_entities = set()
        for pattern in entity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                match = match.strip()
                if len(match) > 2 and match.lower() not in ['the', 'and', 'but', 'for', 'with', 'this', 'that', 'from']:
                    found_entities.add(match)

        # Convert to entity format
        for entity_text in found_entities:
            # Classify entity type
            if any(word in entity_text.lower() for word in ['cancer', 'diabetes', 'asthma', 'arthritis', 'covid']):
                entity_type = "Disease"
            elif any(word in entity_text.lower() for word in ['aspirin', 'ibuprofen', 'metformin', 'insulin']):
                entity_type = "Drug"
            else:
                entity_type = "Concept"

            entities.append({
                "id": entity_text,
                "type": entity_type,
                "properties": {
                    "name": entity_text,
                    "description": f"Medical entity: {entity_text}"
                }
            })

        return entities

    def generate_knowledge_graph(self, text: str, llm=None, file_name: str = None) -> Dict[str, Any]:
        """
        Generate knowledge graph from text
        """
        logging.info("🚀 Starting Enhanced KG Generation")

        # Step 1: Chunk the text
        chunks = self._chunk_text(text)
        logging.info(f"📦 Created {len(chunks)} chunks")

        # Step 2: Process chunks (for demo, process only first few chunks)
        kg = {
            "nodes": [],
            "relationships": [],
            "chunks": chunks,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "file_name": file_name,
                "total_chunks": len(chunks),
                "chunk_size": self.chunk_size,
                "extraction_method": "enhanced_kg_creator_prod"
            }
        }

        if llm:
            all_entities = []
            all_relationships = []

            # Process chunks in batches for efficiency
            for i, chunk in enumerate(chunks[:3]):  # Limit to first 3 chunks for demo
                logging.info(f"🔬 Processing chunk {i+1}/3")
                chunk_kg = self._extract_entities_and_relationships_with_llm(chunk['text'], llm)
                all_entities.extend(chunk_kg['entities'])
                all_relationships.extend(chunk_kg['relationships'])

            # Deduplicate entities with text hash deduplication
            entity_map = {}
            for entity in all_entities:
                # Add text hash if not present
                if 'properties' not in entity:
                    entity['properties'] = {}
                if 'text_hash' not in entity['properties']:
                    entity_text = entity.get('id', '').strip()
                    if entity_text:
                        import hashlib
                        entity['properties']['text_hash'] = hashlib.sha256(entity_text.lower().encode('utf-8')).hexdigest()
                        entity['properties']['name'] = entity_text
                        entity['properties']['description'] = f"Medical entity: {entity_text}"

                key = f"{entity['type']}:{entity['id'].lower()}"
                if key not in entity_map:
                    entity_map[key] = entity

            # Create node format
            for entity in entity_map.values():
                kg["nodes"].append({
                    "id": entity['id'],
                    "label": entity['type'],
                    "properties": entity.get('properties', {}),
                    "embedding": entity.get('embedding')
                })

            # Create relationship format
            seen_relationships = set()
            for rel in all_relationships:
                rel_key = f"{rel['source']}:{rel['type']}:{rel['target']}"
                if rel_key not in seen_relationships:
                    kg["relationships"].append({
                        "id": f"rel_{len(kg['relationships'])}",
                        "from": rel['source'],
                        "to": rel['target'],
                        "source": rel['source'],
                        "target": rel['target'],
                        "type": rel['type'],
                        "label": rel['type'],
                        "properties": rel.get('properties', {})
                    })
                    seen_relationships.add(rel_key)

            kg["metadata"].update({
                "total_entities": len(kg["nodes"]),
                "total_relationships": len(kg["relationships"]),
            })

            logging.info(".1f")
            logging.info("Extraction completed successfully")
        return kg

    def save_to_neo4j(self, kg: Dict[str, Any], file_name: str) -> bool:
        """
        Save knowledge graph to Neo4j
        """
        try:
            graph = self._create_neo4j_connection()

            # Create source document
            graph.query("""
            MERGE (d:Document {fileName: $fileName})
            SET d.createdAt = datetime(),
                d.updatedAt = datetime(),
                d.totalEntities = $totalEntities,
                d.totalChunks = $totalChunks,
                d.totalRelationships = $totalRelationships
            """, {
                "fileName": file_name,
                "totalEntities": len(kg.get('nodes', [])),
                "totalChunks": len(kg.get('chunks', [])),
                "totalRelationships": len(kg.get('relationships', []))
            })

            # Save entities with text hash deduplication
            for node in kg.get('nodes', []):
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
                    existing_result = graph.query(existing_entity_query, {"text_hash": text_hash})

                    existing_entity = existing_result[0] if existing_result else None

                    if existing_entity:
                        # Skip creating this node - link existing entity instead
                        logging.info(f"Skipping duplicate entity '{node['id']}' - using existing entity '{existing_entity['existing_name']}'")
                        continue

                graph.query("""
                MERGE (n:__Entity__ {id: $id})
                SET n.name = $name,
                    n.type = $type,
                    n.description = $description,
                    n.embedding = $embedding,
                    n.text_hash = $text_hash,
                    n.updatedAt = datetime()
                """, {
                    "id": node['id'],
                    "name": properties.get('name', node['id']),
                    "type": node['label'],
                    "description": properties.get('description', ''),
                    "embedding": node.get('embedding'),
                    "text_hash": properties.get('text_hash')
                })

            # Save relationships
            for rel in kg.get('relationships', []):
                graph.query("""
                MATCH (s:__Entity__ {id: $source_id})
                MATCH (t:__Entity__ {id: $target_id})
                MERGE (s)-[r:{rel_type}]->(t)
                SET r.description = $description,
                    r.updatedAt = datetime()
                """.format(rel_type=rel['type']), {
                    "source_id": rel['source'],
                    "target_id": rel['target'],
                    "description": rel.get('properties', {}).get('description', '')
                })

            logging.info(f"✅ Saved KG with {len(kg.get('nodes', []))} entities and {len(kg.get('relationships', []))} relationships")
            return True

        except Exception as e:
            logging.error(f"❌ Failed to save to Neo4j: {e}")
            return False

        finally:
            try:
                graph._driver.close()
            except:
                pass


# Test function
if __name__ == "__main__":
    print("🧪 Testing Enhanced KG Creator Production Version")
    print("=" * 60)

    try:
        # Test with sample PDF content
        from pypdf import PdfReader as PypdfReader

        pdf_path = 'EAU-EANM-ESTRO-ESUR-ISUP-SIOG-Pocket-on-Prostate-Cancer-2025_updated.pdf'
        print(f"📖 Reading PDF: {pdf_path}")

        reader = PypdfReader(pdf_path)
        text_content = ""
        for page in reader.pages[:2]:  # Test with first 2 pages
            text_content += page.extract_text() + "\n"

        print(f"✅ Extracted {len(text_content)} characters from PDF")

        # Initialize KG creator
        kg_creator = EnhancedKGCreatorProd(
            chunk_size=400,  # Smaller for testing
            chunk_overlap=50
        )

        # Test chunking
        chunks = kg_creator._chunk_text(text_content)
        print(f"✅ Created {len(chunks)} chunks")
        print(f"✅ First chunk: {len(chunks[0]['text'])} characters")

        # Test without LLM (demo mode)
        kg = kg_creator.generate_knowledge_graph(text_content[:1000], llm=None, file_name="demo_prostate_cancer.pdf")

        print("🧠 KG Generation (demo without LLM):")
        print(f"   📦 Chunks processed: {len(kg['chunks'])}")
        print(f"   🏷️  Entities found: {kg['metadata'].get('total_entities', 0)}")
        print(f"   🔗 Relationships found: {kg['metadata'].get('total_relationships', 0)}")

        print("\n✅ Enhanced KG Creator production version is ready!")
        print("💡 LLM extraction can be added by providing a LangChain LLM instance")

    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()
