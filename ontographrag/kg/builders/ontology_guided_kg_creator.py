import json
import re
import hashlib
import difflib
import xml.etree.ElementTree as ET
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain_neo4j import Neo4jGraph
from langchain_text_splitters import TokenTextSplitter
from collections import defaultdict, Counter
import os
import sys
import logging
import time

# Import from local kg_utils
from ontographrag.kg.utils.common_functions import load_embedding_model

class OntologyGuidedKGCreator:
    """
    Ontology-Guided Knowledge Graph Creator that properly extracts entities from PDF content
    using LLM with ontology guidance for better entity classification and relationships
    """

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        embedding_model: str = "sentence_transformers",
        ontology_path: str = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.embedding_model = embedding_model
        self.ontology_path = ontology_path

        # Initialize embedding model
        self.embedding_function, self.embedding_dimension = load_embedding_model(embedding_model)
        logging.info(f"Initialized embedding model: {embedding_model}, dimension: {self.embedding_dimension}")

        # Load ontology if provided
        self.ontology_classes = []
        self.ontology_relationships = []
        if ontology_path and os.path.exists(ontology_path):
            logging.info(f"Ontology file exists at: {ontology_path} (size: {os.path.getsize(ontology_path)} bytes)")
            try:
                self._load_ontology(ontology_path)
                logging.info(f"✅ Successfully loaded ontology: {len(self.ontology_classes)} classes, {len(self.ontology_relationships)} relationships")

                # Validate ontology structure to prevent "string indices must be integers" errors
                self._validate_ontology_structure()

            except Exception as e:
                logging.error(f"❌ Failed to load ontology: {e}")
                # Continue with empty ontology - LLM extraction will still work
                self.ontology_classes = []
                self.ontology_relationships = []
        else:
            if ontology_path:
                logging.warning(f"Ontology file not found: {ontology_path}")
                logging.info(f"Available files in temp dir: {os.listdir(os.path.dirname(ontology_path)) if ontology_path else 'N/A'}")
            else:
                logging.info("No ontology provided - using basic LLM entity extraction")
            # No ontology - use empty lists (will fall back to pattern matching)

        # Pre-compute ontology class label embeddings for semantic classification
        self._ontology_class_embeddings: List[Tuple[str, Any]] = []  # [(class_id, embedding), ...]
        if self.ontology_classes and self.embedding_function:
            try:
                labels = [cls['label'] for cls in self.ontology_classes]
                embeddings = self.embedding_function.embed_documents(labels)
                self._ontology_class_embeddings = [
                    (cls['id'], emb)
                    for cls, emb in zip(self.ontology_classes, embeddings)
                ]
                logging.info(f"Pre-computed embeddings for {len(self._ontology_class_embeddings)} ontology classes")
            except Exception as e:
                logging.warning(f"Could not pre-compute ontology class embeddings: {e}. Falling back to keyword matching.")

    def _load_ontology(self, ontology_path: str):
        """
        Load ontology classes and relationships from OWL file
        """
        try:
            tree = ET.parse(ontology_path)
            root = tree.getroot()

            # Define namespaces
            ns = {
                'owl': 'http://www.w3.org/2002/07/owl#',
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'
            }

            # Extract classes
            for class_elem in root.findall('.//owl:Class', ns):
                class_id = class_elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about', '')
                if class_id:
                    # Extract local name from URI
                    class_name = class_id.split('#')[-1] if '#' in class_id else class_id.split('/')[-1]
                    if class_name:
                        self.ontology_classes.append({
                            'id': class_name,
                            'uri': class_id,
                            'label': class_name.replace('_', ' ').title()
                        })

            # Extract object properties (relationships)
            for prop_elem in root.findall('.//owl:ObjectProperty', ns):
                prop_id = prop_elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about', '')
                if prop_id:
                    prop_name = prop_id.split('#')[-1] if '#' in prop_id else prop_id.split('/')[-1]
                    if prop_name:
                        self.ontology_relationships.append({
                            'id': prop_name,
                            'uri': prop_id,
                            'label': prop_name.replace('_', ' ').title()
                        })

            logging.info(f"Successfully loaded {len(self.ontology_classes)} ontology classes and {len(self.ontology_relationships)} relationships")

        except Exception as e:
            logging.error(f"Error loading ontology: {e}")
            raise e

    def _validate_ontology_structure(self):
        """
        Validate and clean ontology class and relationship structures to prevent "string indices must be integers" errors
        """
        # Clean ontology classes
        valid_classes = []
        for cls in self.ontology_classes:
            if isinstance(cls, dict) and 'id' in cls and 'label' in cls:
                valid_classes.append(cls)
            else:
                logging.warning(f"Removing invalid ontology class entry: {cls}")

        self.ontology_classes = valid_classes
        logging.info(f"Validated ontology classes: {len(self.ontology_classes)} valid entries")

        # Clean ontology relationships
        valid_relationships = []
        for rel in self.ontology_relationships:
            if isinstance(rel, dict) and 'id' in rel and 'label' in rel:
                valid_relationships.append(rel)
            else:
                logging.warning(f"Removing invalid ontology relationship entry: {rel}")

        self.ontology_relationships = valid_relationships
        logging.info(f"Validated ontology relationships: {len(self.ontology_relationships)} valid entries")

    def _build_schema_card(self) -> dict:
        """
        Build a versioned snapshot of the ontology used for this KG build.
        Stored on the Document node so future queries can detect ontology drift.
        """
        classes = [c.get('id', '') for c in self.ontology_classes if isinstance(c, dict)]
        rels = [r.get('id', '') for r in self.ontology_relationships if isinstance(r, dict)]
        # Deterministic fingerprint: sort so order doesn't affect hash
        fingerprint_str = json.dumps({"classes": sorted(classes), "relationships": sorted(rels)}, sort_keys=True)
        schema_hash = hashlib.sha256(fingerprint_str.encode()).hexdigest()
        # Also hash the raw ontology file if available
        ontology_file_hash = None
        if self.ontology_path and os.path.exists(self.ontology_path):
            try:
                with open(self.ontology_path, 'rb') as f:
                    ontology_file_hash = hashlib.sha256(f.read()).hexdigest()
            except OSError:
                pass
        return {
            "schemaVersion": schema_hash[:16],  # short fingerprint for display
            "schemaHash": schema_hash,
            "ontologyFileHash": ontology_file_hash or "",
            "ontologyPath": os.path.basename(self.ontology_path) if self.ontology_path else "",
            "classes": sorted(classes),
            "relationships": sorted(rels),
            "classCount": len(classes),
            "relationshipCount": len(rels),
            "builtAt": datetime.now().isoformat(),
        }

    def _create_neo4j_connection(self):
        """Create Neo4j graph connection"""
        # Ensure password is not None - use environment variable as fallback
        password = self.neo4j_password
        if password is None or password == "":
            password = os.getenv("NEO4J_PASSWORD", "password")

        # Set environment variables to ensure LangChain Neo4jGraph can read them
        os.environ["NEO4J_URI"] = self.neo4j_uri
        os.environ["NEO4J_USERNAME"] = self.neo4j_user
        os.environ["NEO4J_PASSWORD"] = password
        os.environ["NEO4J_DATABASE"] = self.neo4j_database

        return Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=password,
            database=self.neo4j_database,
            refresh_schema=False,
            sanitize=True
        )

    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text using TokenTextSplitter"""
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

    def _extract_entities_and_relationships_with_llm(self, chunk_text: str, llm, model_name: str = "openai/gpt-oss-120b:free") -> Dict[str, Any]:
        """
        Extract entities and relationships using LLM with ontology guidance (if ontology available) or natural LLM detection
        """
        if llm is None:
            logging.warning("_extract_entities_and_relationships_with_llm called with llm=None; returning empty.")
            return {"entities": [], "relationships": []}

        # Check if ontology is available
        has_ontology = bool(self.ontology_classes) or bool(self.ontology_relationships)

        if has_ontology:
            # Ontology-guided extraction - with error handling
            try:
                # Use more ontology classes for better context
                ontology_classes_text = "\n".join([f"- {cls['label']} ({cls['id']})" for cls in self.ontology_classes[:100] if isinstance(cls, dict) and 'label' in cls and 'id' in cls])
                ontology_relationships_text = "\n".join([f"- {rel['label']} ({rel['id']})" for rel in self.ontology_relationships[:50] if isinstance(rel, dict) and 'label' in rel and 'id' in rel])
                logging.info(f"Ontology text generated: {len(self.ontology_classes)} classes (showing first 100), {len(self.ontology_relationships)} relationships (showing first 50)")
            except Exception as e:
                logging.warning(f"Error generating ontology text: {e}. Using basic extraction.")
                ontology_classes_text = "Basic ontology - see classification for details"
                ontology_relationships_text = "Basic relationships"
                has_ontology = False  # Fall back to natural LLM

            system_message = f"""
You are an expert medical knowledge graph extraction system.
Your task is to extract entities and relationships from medical/scientific text using the provided ontology.

ONTOLOGY CLASSES AVAILABLE:
{ontology_classes_text}

ONTOLOGY RELATIONSHIPS AVAILABLE:
{ontology_relationships_text}

INSTRUCTIONS:
1. Extract ONLY medically relevant entities from the text
2. Classify each entity using the ontology classes above
3. Create relationships ONLY between entities that actually interact in the text
4. Use ontology relationships when they match the context
5. Focus on diseases, treatments, symptoms, drugs, procedures, and medical concepts
6. Ignore generic words, dates, numbers, and administrative content

Return ONLY a valid JSON object in this exact format.
IMPORTANT: Output "relationships" FIRST, then "entities". This order is required.
{{
  "relationships": [
    {{
      "source": "source_entity_id",
      "target": "target_entity_id",
      "type": "ONTOLOGY_RELATIONSHIP",
      "properties": {{
        "description": "how they are related in the text"
      }}
    }}
  ],
  "entities": [
    {{
      "id": "exact_entity_name_from_text",
      "type": "OntologyClass",
      "properties": {{
        "name": "exact_entity_name_from_text",
        "description": "brief medical description"
      }}
    }}
  ]
}}

TEXT TO ANALYZE:
{chunk_text}

IMPORTANT: Return ONLY the JSON object, no additional text or explanation."""
        else:
            # No ontology - natural LLM detection with no filtering
            system_message = f"""
You are an expert knowledge graph extraction system.
Your task is to extract entities and relationships from text naturally and comprehensively.

INSTRUCTIONS:
1. Extract ALL significant entities and concepts from the text
2. No restrictions on entity types - extract anything that could be part of a knowledge graph
3. Create relationships between ANY entities that are meaningfully related in the text
4. Use descriptive relationship types that best capture how entities interact
5. Be comprehensive - detect as many nodes and relationships as naturally appear
6. Include both technical and non-technical concepts if contextually relevant

Return ONLY a valid JSON object in this exact format.
IMPORTANT: Output "relationships" FIRST, then "entities". This order is required.
{{
  "relationships": [
    {{
      "source": "source_entity_id",
      "target": "target_entity_id",
      "type": "RELATIONSHIP_TYPE",
      "properties": {{
        "description": "how they are related in the text"
      }}
    }}
  ],
  "entities": [
    {{
      "id": "exact_entity_name_from_text",
      "type": "EntityType",
      "properties": {{
        "name": "exact_entity_name_from_text",
        "description": "contextual description"
      }}
    }}
  ]
}}

TEXT TO ANALYZE:
{chunk_text}

IMPORTANT: Return ONLY the JSON object, no additional text or explanation."""

        try:

            # Get the response directly from LLM provider with timeout handling
            try:
                response = llm.generate(system_message, "", model_name)
            except Exception as timeout_error:
                if "timeout" in str(timeout_error).lower() or "read operation timed out" in str(timeout_error).lower():
                    logging.warning(f"LLM request timed out for chunk. Returning empty result to continue processing.")
                    return {'entities': [], 'relationships': []}
                else:
                    raise timeout_error

            # Debug: Log the raw response
            logging.info(f"Raw LLM response length: {len(response)}")
            logging.info(f"Raw LLM response preview: {response[:200]}...")

            # Clean the response to extract JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            # Robust JSON extraction with multiple fallback strategies
            json_content = ""

            # Strategy 1: Try to extract JSON from response with improved parsing
            json_start = response.find('{')
            if json_start == -1:
                logging.warning(f"No JSON start found in response: {response[:200]}... returning empty.")
                return {"entities": [], "relationships": []}

            # Try to find the complete JSON object by looking for balanced braces
            brace_count = 0
            json_end = json_start
            in_string = False
            escape_next = False

            for i, char in enumerate(response[json_start:], json_start):
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found the matching closing brace
                            json_end = i + 1
                            break

            if brace_count == 0 and json_end > json_start:
                json_content = response[json_start:json_end]
            else:
                logging.warning(f"Incomplete JSON structure (brace_count={brace_count}). Trying alternative extraction.")
                # Strategy 2: Try simple { } extraction
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = response[json_start:json_end]
                else:
                    logging.warning("Could not extract JSON with any strategy; returning empty.")
                    return {"entities": [], "relationships": []}

            # Parse the JSON content
            try:
                result = json.loads(json_content)
            except json.JSONDecodeError as e:
                logging.warning(f"JSON parsing error: {e}. Attempting partial extraction from: {json_content[:300]}...")
                result = self._partial_json_extract(json_content)

            # Validate the result has the expected structure
            if not isinstance(result, dict) or ('entities' not in result and 'relationships' not in result):
                logging.warning(f"Invalid response structure. Returning empty result.")
                return {'entities': [], 'relationships': []}
            result.setdefault('entities', [])
            result.setdefault('relationships', [])

            # Filter out non-medical entities - with better error handling
            medical_entities = []
            entities_raw = result.get('entities', [])

            # Check if entities is the wrong type (LLM returned malformed JSON)
            if not isinstance(entities_raw, list):
                logging.warning(f"LLM returned entities as {type(entities_raw)} instead of list: {entities_raw}. Skipping chunk.")
                return {'entities': [], 'relationships': []}

            for entity in entities_raw:
                # Handle both dict and string formats from LLM response
                if isinstance(entity, str):
                    # Convert string entity to dict format
                    try:
                        entity_type = self._classify_entity_with_ontology(entity)
                        entity = {
                            "id": entity,
                            "type": entity_type,
                            "properties": {
                                "name": entity,
                                "description": f"{entity_type}: {entity}"
                            }
                        }
                    except Exception as e:
                        logging.warning(f"Error processing string entity '{entity}': {e}. Skipping.")
                        continue
                elif isinstance(entity, dict):
                    # Ensure required fields exist
                    if 'id' not in entity:
                        logging.warning(f"Entity missing 'id' field: {entity}. Skipping.")
                        continue
                    # Ensure id is a string
                    if not isinstance(entity['id'], str):
                        logging.warning(f"Entity id is not a string: {type(entity['id'])} = {entity['id']}. Converting to string.")
                        entity['id'] = str(entity['id'])

                    try:
                        if 'type' not in entity:
                            entity['type'] = self._classify_entity_with_ontology(entity['id'])
                        if 'properties' not in entity:
                            entity['properties'] = {
                                "name": entity['id'],
                                "description": f"{entity['type']}: {entity['id']}"
                            }
                    except Exception as e:
                        logging.warning(f"Error processing dict entity {entity}: {e}. Skipping.")
                        continue
                else:
                    logging.warning(f"Entity is neither string nor dict: {type(entity)} = {entity}. Skipping.")
                    continue

                medical_entities.append(entity)

            # Filter relationships — source/target must reference known entities.
            # Use a case-insensitive lookup to tolerate LLMs that capitalise
            # inconsistently between the entity definition and the relationship field.
            medical_relationships = []
            # canonical lookup: lowercase → original-case entity id
            entity_id_lower = {e['id'].lower(): e['id'] for e in medical_entities}

            relationships_raw = result.get('relationships', [])

            # Check if relationships is the wrong type (LLM returned malformed JSON)
            if not isinstance(relationships_raw, list):
                logging.warning(f"LLM returned relationships as {type(relationships_raw)} instead of list: {relationships_raw}. Skipping relationships.")
                medical_relationships = []
            else:
                # Process relationships, handling both dict and string formats
                for rel in relationships_raw:
                    if isinstance(rel, str):
                        logging.warning(f"Relationship is string: {rel}. Skipping.")
                        continue
                    elif isinstance(rel, dict):
                        raw_src = rel.get('source', '')
                        raw_tgt = rel.get('target', '')
                        if not isinstance(raw_src, str) or not isinstance(raw_tgt, str):
                            logging.warning(f"Relationship source/target not strings: {rel}. Skipping.")
                            continue
                        # Resolve to canonical case (fallback: keep original)
                        canonical_src = entity_id_lower.get(raw_src.lower(), raw_src)
                        canonical_tgt = entity_id_lower.get(raw_tgt.lower(), raw_tgt)
                        if canonical_src.lower() in entity_id_lower and canonical_tgt.lower() in entity_id_lower:
                            rel_copy = dict(rel)
                            rel_copy['source'] = canonical_src
                            rel_copy['target'] = canonical_tgt
                            medical_relationships.append(rel_copy)
                        else:
                            logging.warning(f"Invalid relationship (source/target not in entities): {rel}. Skipping.")
                    else:
                        logging.warning(f"Relationship is neither string nor dict: {type(rel)} = {rel}. Skipping.")
                        continue

            return {
                'entities': medical_entities,
                'relationships': medical_relationships
            }

        except Exception as e:
            logging.error(f"LLM extraction failed: {e}")
            return {"entities": [], "relationships": []}






    def _entities_can_be_related(self, source_type: str, target_type: str) -> bool:
        """
        Check if two entity types can be meaningfully related
        """
        # Define which entity types can be related
        medical_relationships = {
            'Disease': ['Treatment', 'Symptom', 'Drug', 'MedicalProcedure', 'Patient'],
            'Treatment': ['Disease', 'Drug', 'MedicalProcedure', 'Patient', 'Physician'],
            'Drug': ['Disease', 'Treatment', 'MedicalProcedure', 'Patient'],
            'Symptom': ['Disease', 'Patient', 'MedicalProcedure'],
            'Patient': ['Disease', 'Treatment', 'Drug', 'Physician', 'Hospital', 'MedicalProcedure'],
            'Physician': ['Patient', 'Treatment', 'MedicalProcedure', 'Hospital'],
            'Hospital': ['Patient', 'Physician', 'MedicalProcedure'],
            'MedicalProcedure': ['Disease', 'Treatment', 'Patient', 'Physician', 'Hospital'],
            'MedicalDevice': ['MedicalProcedure', 'Treatment'],
            'Anatomy': ['Disease', 'MedicalProcedure']
        }

        return target_type in medical_relationships.get(source_type, [])

    def _partial_json_extract(self, text: str) -> Dict[str, Any]:
        """
        Best-effort recovery when json.loads() fails on an LLM response.

        Tries three strategies in order:
        1. Extract the 'entities' array via regex and parse it independently.
        2. Extract the 'relationships' array via regex and parse it independently.
        3. Return empty dict (caller will fall back to empty entities/relationships).

        This prevents silently discarding an entire chunk just because a trailing
        comma or stray character made the top-level JSON invalid.
        """
        def _extract_array(key: str, src: str):
            """Regex-extract the JSON array for a given key from a potentially broken JSON string."""
            pattern = rf'"{key}"\s*:\s*(\[.*?\])'
            match = re.search(pattern, src, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                # Try to clean common issues: trailing commas before ]
                cleaned = re.sub(r',\s*]', ']', match.group(1))
                cleaned = re.sub(r',\s*}', '}', cleaned)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    return None

        entities = _extract_array('entities', text)
        relationships = _extract_array('relationships', text)

        if entities is not None or relationships is not None:
            recovered = {
                'entities': entities if isinstance(entities, list) else [],
                'relationships': relationships if isinstance(relationships, list) else [],
            }
            logging.info(
                f"Partial JSON recovery succeeded: {len(recovered['entities'])} entities, "
                f"{len(recovered['relationships'])} relationships"
            )
            return recovered

        logging.warning("Partial JSON recovery failed; returning empty extraction for this chunk.")
        return {'entities': [], 'relationships': []}

    def _classify_entity_with_ontology(self, entity_text: str) -> str:
        """
        Classify entity using ontology guidance.

        Strategy (in priority order):
        1. Embedding-based cosine similarity against pre-computed ontology class label embeddings
           (threshold ≥ 0.50 required to accept the match).
        2. Exact substring match against ontology class id/label (legacy fallback).
        3. Keyword heuristics (final fallback).
        """
        entity_lower = entity_text.lower()

        # Strategy 1: embedding similarity (preferred — robust to abbreviations / paraphrases)
        if self._ontology_class_embeddings and self.embedding_function:
            try:
                entity_emb = self.embedding_function.embed_query(entity_text)
                best_class, best_score = None, 0.0
                for cls_id, cls_emb in self._ontology_class_embeddings:
                    # Cosine similarity (both vectors are L2-normalised by the embedding model)
                    score = float(
                        sum(a * b for a, b in zip(entity_emb, cls_emb))
                        / (
                            (sum(a * a for a in entity_emb) ** 0.5 + 1e-9)
                            * (sum(b * b for b in cls_emb) ** 0.5 + 1e-9)
                        )
                    )
                    if score > best_score:
                        best_score, best_class = score, cls_id
                if best_class and best_score >= 0.50:
                    return best_class
            except Exception as e:
                logging.debug(f"Embedding classification failed for '{entity_text}': {e}")

        # Strategy 2: exact substring match against ontology class labels
        for cls in self.ontology_classes:
            if cls['id'].lower() in entity_lower or cls['label'].lower() in entity_lower:
                return cls['id']

        # Strategy 3: keyword heuristics
        if any(word in entity_lower for word in ['disease', 'cancer', 'tumor', 'syndrome', 'disorder', 'carcinoma', 'malignancy']):
            return 'Disease'
        elif any(word in entity_lower for word in ['drug', 'medication', 'treatment', 'therapy', 'surgery', 'chemotherapy', 'radiotherapy']):
            return 'Treatment'
        elif any(word in entity_lower for word in ['patient', 'person', 'individual', 'male', 'female']):
            return 'Patient'
        elif any(word in entity_lower for word in ['doctor', 'physician', 'surgeon', 'specialist', 'oncologist', 'urologist']):
            return 'Physician'
        elif any(word in entity_lower for word in ['hospital', 'clinic', 'center', 'institute', 'department']):
            return 'Hospital'
        elif any(word in entity_lower for word in ['symptom', 'sign', 'manifestation', 'pain', 'fever']):
            return 'Symptom'
        elif any(word in entity_lower for word in ['gene', 'mutation', 'protein', 'biomarker', 'receptor', 'marker']):
            return 'Biomarker'
        elif any(word in entity_lower for word in ['score', 'grade', 'stage', 'classification', 'risk']):
            return 'ClinicalFinding'
        else:
            return 'Concept'

    def _classify_relationship_with_ontology(self, source: str, target: str) -> str:
        """
        Classify relationship using ontology guidance
        """
        source_lower = source.lower()
        target_lower = target.lower()

        # Check for treatment relationships
        if any(word in source_lower for word in ['treatment', 'therapy', 'drug']) or \
           any(word in target_lower for word in ['treatment', 'therapy', 'drug']):
            return 'treats'

        # Check for disease-symptom relationships
        if any(word in source_lower for word in ['disease', 'cancer']) and \
           any(word in target_lower for word in ['symptom', 'sign']):
            return 'hasSymptom'

        # Check for physician-patient relationships
        if any(word in source_lower for word in ['physician', 'doctor']) and \
           any(word in target_lower for word in ['patient', 'person']):
            return 'diagnoses'

        # Default relationship
        return 'RELATED_TO'

    def _canonicalize_relationship_type(self, raw_type: str) -> str:
        """
        Map a raw LLM-generated relationship type to the closest ontology relationship.

        Steps:
        1. Exact label match (case/space-insensitive) against ontology relationships.
        2. Fuzzy token-based match using difflib.SequenceMatcher with threshold ≥ 0.72.
           This catches paraphrases like "is_treated_by" → "treatedBy".
        3. Safe fallback: ASSOCIATED_WITH (semantically neutral, always valid).

        The result is uppercased with spaces/hyphens replaced by underscores so it
        is always a valid Neo4j relationship type.
        """
        if not raw_type:
            return 'ASSOCIATED_WITH'

        normalized_raw = raw_type.lower().replace(' ', '_').replace('-', '_')

        if self.ontology_relationships:
            # Step 1: exact match
            for ont_rel in self.ontology_relationships:
                candidate = ont_rel['label'].lower().replace(' ', '_')
                if candidate == normalized_raw or ont_rel['id'].lower() == normalized_raw:
                    return ont_rel['id'].replace(' ', '_').replace('-', '_').upper()

            # Step 2: fuzzy match
            best_match, best_score = None, 0.0
            for ont_rel in self.ontology_relationships:
                candidate = ont_rel['label'].lower().replace(' ', '_')
                score = difflib.SequenceMatcher(None, normalized_raw, candidate).ratio()
                if score > best_score:
                    best_score, best_match = score, ont_rel
                # Also compare against ont_rel id
                id_score = difflib.SequenceMatcher(None, normalized_raw, ont_rel['id'].lower()).ratio()
                if id_score > best_score:
                    best_score, best_match = id_score, ont_rel

            if best_match and best_score >= 0.72:
                logging.debug(
                    f"Fuzzy rel canonicalization: '{raw_type}' → '{best_match['id']}' (score={best_score:.2f})"
                )
                return best_match['id'].replace(' ', '_').replace('-', '_').upper()

        # Step 3: sanitize raw type and use as-is if it looks reasonable, else fallback
        sanitized = raw_type.strip().replace(' ', '_').replace('-', '_').upper()
        # If it's a suspiciously long or weird string, use safe fallback
        if len(sanitized) > 50 or not re.match(r'^[A-Z][A-Z0-9_]*$', sanitized):
            logging.debug(f"Rel type '{raw_type}' failed sanitization; using ASSOCIATED_WITH")
            return 'ASSOCIATED_WITH'
        return sanitized

    def _extract_relationships_only(self, combined_text: str, known_entities: List[Dict], llm, model_name: str) -> List[Dict]:
        """
        Given a combined text (two adjacent chunks) and entities already known from those chunks,
        ask the LLM only for relationships between those entities — no new entity extraction.

        Returns a list of raw relationship dicts {source, target, type, properties}.
        """
        if llm is None or len(known_entities) < 2:
            return []

        entity_list = "\n".join(
            f"- {e['id']} (type: {e.get('type', 'Unknown')})"
            for e in known_entities[:60]  # cap to avoid over-long prompts
        )

        prompt = f"""You are a medical knowledge graph expert.
Given the following text and list of known entities, identify ONLY relationships between these entities that are explicitly supported by the text.

KNOWN ENTITIES:
{entity_list}

TEXT:
{combined_text}

Return ONLY a JSON object with a "relationships" array (no "entities" key needed):
{{
  "relationships": [
    {{
      "source": "source_entity_id_from_list",
      "target": "target_entity_id_from_list",
      "type": "RELATIONSHIP_TYPE",
      "properties": {{"description": "how they relate in the text"}}
    }}
  ]
}}

Rules:
- source and target MUST be entity ids from the KNOWN ENTITIES list above
- Only include relationships explicitly supported by the text
- Prefer specific relationship types (TREATS, CAUSES, INDICATES, etc.) over generic ones
- Return ONLY the JSON object, no other text"""

        try:
            response = llm.generate(prompt, "", model_name)
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start < 0 or json_end <= json_start:
                return []
            result = json.loads(response[json_start:json_end])
            return result.get('relationships', [])
        except Exception as e:
            logging.warning(f"Cross-chunk relationship extraction failed: {e}")
            return []

    def _generate_entity_id(self, entity: Dict[str, Any]) -> str:
        """
        Generate a UUID-based entity ID to prevent duplicates
        """
        # Seed on normalized text only (not type) so the same entity text always maps
        # to the same UUID regardless of which ontology type the LLM assigned per chunk.
        unique_seed = self._normalize_entity_text(entity['id'])
        # Generate UUID5 (name-based) for deterministic but unique IDs
        return str(uuid.uuid5(uuid.NAMESPACE_OID, unique_seed))

    def _normalize_entity_text(self, text: str) -> str:
        """
        Normalize entity text for duplicate detection.

        Applies generic, domain-agnostic normalization:
        - lowercase + collapse whitespace
        - strip leading articles / conjunctions
        - remove punctuation that doesn't affect identity
        - collapse runs of underscores/spaces

        Domain-specific aliases are intentionally NOT hardcoded here.
        The LLM is expected to produce consistent names; UUID5-based
        deduplication in _harmonize_entities handles surface variants.
        """
        normalized = re.sub(r'\s+', ' ', text.lower().strip())

        # Strip leading articles and conjunctions
        normalized = re.sub(r'^(the |an |a |and |or )', '', normalized)

        # Remove punctuation that doesn't carry semantic weight
        normalized = re.sub(r'[,()\[\];:.]', '', normalized)

        # Collapse hyphens/slashes to underscores for stable keys
        normalized = re.sub(r'[\-/]', '_', normalized)

        # Final cleanup
        normalized = re.sub(r'\s+', '_', normalized.strip())
        normalized = re.sub(r'_+', '_', normalized)
        normalized = normalized.strip('_')

        return normalized

    def _verify_triple_confidence(
        self,
        source_name: str,
        target_name: str,
        rel_type: str,
        chunks: List[Dict],
    ) -> float:
        """
        Evidence-grounded verification for an extracted triple (inspired by MOSAICX verify).

        Searches all chunks for co-occurrence of the source and target entity names.
        Returns a confidence score in [0.0, 1.0]:
          - 1.0  both names found in the same sentence
          - 0.7  both names found in the same chunk (not same sentence)
          - 0.4  only one name found in any chunk
          - 0.1  neither name found (LLM may have hallucinated this triple)

        This is intentionally cheap (string matching only) so it doesn't add
        meaningful latency. An LLM-based re-verification pass can be added later
        for low-confidence triples as an opt-in upgrade.
        """
        if not chunks or not source_name or not target_name:
            return 0.5  # neutral when no evidence to check

        src_lower = source_name.strip().lower()
        tgt_lower = target_name.strip().lower()

        # Build boundary-aware patterns.
        # Use (?<!\w)/(?!\w) lookarounds instead of \b so that entity names that
        # start or end with non-word characters (parentheses, hyphens, dots) are
        # still matched correctly.  e.g. "gleason score (gs)" ends with ')' which
        # is not a word char — \b would fail there, but (?!\w) does not.
        def _boundary_pattern(name: str) -> re.Pattern:
            prefix = r'(?<!\w)' if not name[:1].isalnum() and name[:1] != '_' else r'\b'
            suffix = r'(?!\w)' if not name[-1:].isalnum() and name[-1:] != '_' else r'\b'
            return re.compile(prefix + re.escape(name) + suffix)

        src_pat = _boundary_pattern(src_lower)
        tgt_pat = _boundary_pattern(tgt_lower)

        found_src, found_tgt, same_sentence = False, False, False

        for chunk in chunks:
            text = chunk.get('text', '').lower()
            has_src = bool(src_pat.search(text))
            has_tgt = bool(tgt_pat.search(text))

            if has_src:
                found_src = True
            if has_tgt:
                found_tgt = True

            if has_src and has_tgt:
                # Check sentence-level co-occurrence (use same pattern objects)
                sentences = re.split(r'(?<=[.!?])\s+', text)
                if any(src_pat.search(s) and tgt_pat.search(s) for s in sentences):
                    same_sentence = True
                    break  # best possible score — stop scanning

        if same_sentence:
            return 1.0
        if found_src and found_tgt:
            return 0.7
        if found_src or found_tgt:
            return 0.4
        return 0.1

    def _harmonize_entities(self, all_entities: List[Dict], return_id_map: bool = False):
        """
        Harmonize entities across chunks to avoid duplicates using improved normalization
        """
        logging.info(f"Starting harmonization of {len(all_entities)} raw entities")

        # Step 1: Build initial grouping by normalized text only (NOT by type).
        # The original kg_creator deduped by entity text hash, ignoring type.
        # Using type:text as the key causes the same entity to become multiple nodes
        # when the LLM assigns different ontology classes across chunks — e.g.
        # "Prostate Cancer" as Disease in chunk 1 and Concept in chunk 2 → 2 nodes.
        # Grouping by text only and picking the most specific type restores that behaviour.
        entity_groups = defaultdict(list)

        for entity in all_entities:
            normalized_text = self._normalize_entity_text(entity['id'])
            entity_groups[normalized_text].append(entity)

        # Step 2: Log entity distribution before harmonization
        type_distribution = Counter()
        for entities in entity_groups.values():
            for entity in entities:
                type_distribution[entity['type']] += 1

        logging.info(f"Entity type distribution before harmonization: {dict(type_distribution)}")

        # Step 3: Create harmonized entities
        harmonized_entities = []
        entity_map = {}  # For mapping original IDs to harmonized versions
        total_duplicates_removed = 0

        for normalized_key, entities in entity_groups.items():
            if not entities:
                continue

            # --- Deterministic deduplication rules ---
            # Rule 1: prefer specific ontology types over generic fallbacks (Concept, Entity, Unknown)
            # Rule 2: among equally specific types, prefer the longest description
            # Rule 3: if still tied, prefer the longest entity name (most fully-qualified)
            _generic_types = {'Concept', 'Entity', 'Unknown', 'Other'}

            def _type_specificity(e):
                return 0 if e.get('type', 'Concept') in _generic_types else 1

            def _desc_len(e):
                return len(e.get('properties', {}).get('description') or '')

            def _name_len(e):
                return len(e.get('id') or '')

            representative_entity = max(
                entities, key=lambda e: (_type_specificity(e), _desc_len(e), _name_len(e))
            ).copy()

            # Merge information from all occurrences
            all_names = set()
            all_descriptions = set()

            for entity in entities:
                # Rule 3: accumulate all surface forms as synonyms
                all_names.add(entity['id'])
                desc = entity.get('properties', {}).get('description')
                if desc:
                    all_descriptions.add(desc)

                # Keep the best embedding (prefer any available embedding)
                if not representative_entity.get('embedding') and entity.get('embedding'):
                    representative_entity['embedding'] = entity['embedding']

            # Merge all non-description properties from variants into representative
            for entity in entities:
                if entity.get('properties'):
                    for k, v in entity['properties'].items():
                        if k not in ('description',):  # description already resolved above
                            representative_entity.setdefault('properties', {}).setdefault(k, v)

            # Update representative entity with merged information
            if len(all_names) > 1 or len(all_descriptions) > 1:
                representative_entity['properties']['all_names'] = sorted(all_names)
                representative_entity['properties']['all_descriptions'] = sorted(all_descriptions, key=len, reverse=True)
                total_duplicates_removed += len(entities) - 1

            # Generate deterministic UUID
            entity_uuid = self._generate_entity_id(representative_entity)
            representative_entity['uuid'] = entity_uuid

            harmonized_entities.append(representative_entity)

            # Map all original name variations to the harmonized entity
            for entity in entities:
                entity_map[entity['id']] = representative_entity

        logging.info(f"Harmonization complete: {len(harmonized_entities)} entities (removed {total_duplicates_removed} duplicates)")

        # Log final distribution
        final_type_distribution = Counter(e['type'] for e in harmonized_entities)
        logging.info(f"Entity type distribution after harmonization: {dict(final_type_distribution)}")

        if return_id_map:
            return harmonized_entities, entity_map  # entity_map: original_id → representative
        return harmonized_entities

    def _harmonize_relationships(self, all_relationships: List[Dict], entity_map: Dict) -> List[Dict]:
        """
        Harmonize relationships across chunks and map to UUID-based entity IDs.

        entity_map may be either:
          - original_id → representative entity  (from _harmonize_entities with return_id_map=True)
          - uuid → entity  (legacy call sites)
        In both cases we build original_to_uuid by walking .values().
        """
        harmonized_relationships = []
        seen_relationships = set()

        # Build variant_name → uuid from entity_map.
        # entity_map keys are ALL original variant IDs (every surface form seen during
        # extraction), values are the representative entities with uuid set.
        # Iterating .items() — not .values() — ensures every variant name is covered,
        # so relationships that used a non-canonical spelling are not silently dropped.
        original_to_uuid = {}
        for variant_name, representative in entity_map.items():
            if 'uuid' not in representative:
                logging.warning(f"Entity '{representative.get('id', '?')}' missing uuid in entity_map — skipping in relationship mapping")
                continue
            original_to_uuid[variant_name] = representative['uuid']
            # Also index by lowercase to catch case mismatches between extraction calls
            original_to_uuid[variant_name.lower()] = representative['uuid']

        for rel in all_relationships:
            src = rel.get('source')
            tgt = rel.get('target')
            # Map source and target to UUIDs — try exact match first, then lowercase fallback.
            # Guard isinstance before .lower() to handle any non-string values from malformed input.
            source_uuid = (original_to_uuid.get(src)
                           or (original_to_uuid.get(src.lower()) if isinstance(src, str) else None))
            target_uuid = (original_to_uuid.get(tgt)
                           or (original_to_uuid.get(tgt.lower()) if isinstance(tgt, str) else None))

            if not source_uuid or not target_uuid:
                logging.warning(
                    "Dropping relationship — entity not found in map: '%s' -[%s]-> '%s'",
                    src, rel.get('type', '?'), tgt,
                )
                continue

            if source_uuid and target_uuid:
                # Create new relationship with UUID-based IDs
                uuid_rel = rel.copy()
                uuid_rel['source'] = source_uuid
                uuid_rel['target'] = target_uuid

                rel_key = f"{source_uuid}:{rel['type']}:{target_uuid}"

                if rel_key not in seen_relationships:
                    harmonized_relationships.append(uuid_rel)
                    seen_relationships.add(rel_key)

        return harmonized_relationships

    def generate_knowledge_graph(self, text: str, llm, file_name: str = None, model_name: str = "openai/gpt-oss-120b:free", max_chunks: int = None, kg_name: str = None, doc_metadata: dict = None, doc_hash: str = None) -> Dict[str, Any]:
        """
        Generate knowledge graph from text with ontology-guided entity extraction

        Args:
            text: Input text to process
            llm: LLM provider instance
            file_name: Optional filename for storage
            model_name: LLM model name
            max_chunks: Maximum number of chunks to process (for large documents)
        """
        logging.info("Starting ontology-guided knowledge graph generation")

        # Determine if ontology is available
        has_ontology = bool(self.ontology_classes) or bool(self.ontology_relationships)
        extraction_method = "ontology_guided_llm" if has_ontology else "natural_llm"
        logging.info(f"Extraction method: {extraction_method}")

        # Step 1: Chunk the text
        chunks = self._chunk_text(text)
        logging.info(f"Created {len(chunks)} chunks")

        # Limit chunks if specified (for very large documents)
        if max_chunks and len(chunks) > max_chunks:
            logging.warning(f"Limiting processing to {max_chunks} chunks out of {len(chunks)} total")
            chunks = chunks[:max_chunks]

        # Step 2: Extract entities and relationships from each chunk
        all_entities = []
        all_relationships = []
        processed_chunks = 0
        failed_chunks = 0
        entities_per_chunk: Dict[int, List[Dict]] = {}  # chunk index → entities (for cross-chunk pass)

        for i, chunk in enumerate(chunks):
            try:
                logging.info(f"Processing chunk {i+1}/{len(chunks)}")
                chunk_kg = self._extract_entities_and_relationships_with_llm(chunk['text'], llm, model_name)

                # Ensure chunk_kg is a dictionary with expected keys
                try:
                    if isinstance(chunk_kg, dict) and (chunk_kg.get('entities', []) or chunk_kg.get('relationships', [])):
                        chunk_entities = chunk_kg.get('entities', [])
                        all_entities.extend(chunk_entities)
                        all_relationships.extend(chunk_kg.get('relationships', []))
                        entities_per_chunk[i] = chunk_entities
                        processed_chunks += 1
                        logging.info(f"✓ Chunk {i+1} processed: {len(chunk_entities)} entities, {len(chunk_kg.get('relationships', []))} relationships")
                    else:
                        logging.warning(f"⚠ Chunk {i+1} returned invalid or empty format: {type(chunk_kg)}")
                        failed_chunks += 1
                except Exception as inner_e:
                    logging.error(f"❌ Error processing chunk result: {inner_e}")
                    failed_chunks += 1

            except Exception as e:
                logging.error(f"❌ Failed to process chunk {i+1}: {e}")
                import traceback
                logging.error(f"Traceback: {traceback.format_exc()}")
                failed_chunks += 1
                continue

            # Add small delay between chunks only when an external LLM is in use
            if llm is not None and i < len(chunks) - 1:  # Don't delay after the last chunk
                time.sleep(1.0)  # 1 second delay between API calls

        logging.info(f"Processing complete: {processed_chunks} successful, {failed_chunks} failed")

        # Step 2b: Cross-chunk relationship extraction (sliding window of 2 adjacent chunks).
        # The per-chunk LLM pass can only see entities within one chunk at a time, so relationships
        # that span a chunk boundary are never extracted.  This pass combines adjacent chunk pairs
        # and asks the LLM only for relationships between already-known entities — no new entity
        # extraction, so it is cheap (no dedup work) and adds N-1 LLM calls total.
        if llm is not None and len(chunks) > 1:
            cross_chunk_rels = 0
            for i in range(len(chunks) - 1):
                left_entities = entities_per_chunk.get(i, [])
                right_entities = entities_per_chunk.get(i + 1, [])
                combined_text = chunks[i]['text'] + "\n\n" + chunks[i + 1]['text']
                combined_text_lower = combined_text.lower()
                # Pre-filter: only pass entities whose names actually appear in the combined text.
                # Avoids sending irrelevant entities that the LLM can't ground, and removes the
                # hard 60-entity cap from _extract_relationships_only becoming a problem.
                combined_entities = [
                    e for e in (left_entities + right_entities)
                    if isinstance(e.get('id'), str) and e['id'].lower() in combined_text_lower
                ]
                if len(combined_entities) < 2:
                    continue
                new_rels = self._extract_relationships_only(combined_text, combined_entities, llm, model_name)
                if new_rels:
                    all_relationships.extend(new_rels)
                    cross_chunk_rels += len(new_rels)
                    logging.info(f"Cross-chunk pass ({i+1}↔{i+2}): {len(new_rels)} relationships found")
                time.sleep(0.5)  # shorter delay for relationship-only calls
            logging.info(f"Cross-chunk pass complete: {cross_chunk_rels} additional relationships extracted")

        if processed_chunks == 0 and failed_chunks > 0:
            raise RuntimeError(
                f"KG extraction failed: all {failed_chunks} chunk(s) returned errors. "
                "Check LLM connectivity and rate limits."
            )

        # Step 3: Harmonize entities and relationships.
        # _harmonize_entities returns the representative entities AND builds the
        # full original-ID → representative mapping internally (stored on the
        # entities as entity['_all_ids'] is NOT available, so we rebuild it here).
        # We need every original variant ID to map to its representative so that
        # relationships whose source/target used a non-canonical name aren't dropped.
        harmonized_entities, id_to_representative = self._harmonize_entities(all_entities, return_id_map=True)
        harmonized_relationships = self._harmonize_relationships(all_relationships, id_to_representative)

        logging.info(f"Harmonized to {len(harmonized_entities)} entities and {len(harmonized_relationships)} relationships")

        # Step 4: Format the final knowledge graph
        # Use UUID-based IDs to prevent duplicates
        kg_prefix = f"{kg_name}_" if kg_name else ""

        kg = {
            "nodes": [
                {
                    "id": f"{kg_prefix}{entity['uuid']}",
                    "label": entity['type'],
                    "properties": {
                        "name": entity['id'],
                        "type": entity['type'],
                        "original_id": entity['id'],  # Keep original ID for reference
                        **entity.get('properties', {})
                    },
                    "embedding": entity.get('embedding'),
                    "color": self._get_node_color(entity['type']),
                    "size": 30,
                    "font": {"size": 14, "color": "#333333"},
                    "title": f"Entity: {entity['id']}\nType: {entity['type']}\nKG: {kg_name or 'default'}\nClick for details"
                }
                for entity in harmonized_entities
            ],
            "relationships": [
                {
                    "id": f"{kg_prefix}rel_{rel['source']}_{rel['type']}_{rel['target']}_{idx}",
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
            "metadata": {
                "total_chunks": len(chunks),
                "total_entities": len(harmonized_entities),
                "total_relationships": len(harmonized_relationships),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": type(self.embedding_function).__name__,
                "embedding_dimension": self.embedding_dimension,
                "ontology_classes": len(self.ontology_classes),
                "ontology_relationships": len(self.ontology_relationships),
                "extraction_method": "ontology_guided_llm" if has_ontology else "natural_llm",
                "kg_name": kg_name,
                "created_at": datetime.now().isoformat(),
                "visualization_ready": True,
                "file_name": file_name,
                "doc_hash": doc_hash,
                "schema_card": self._build_schema_card()
            }
        }

        # Attach any document-level metadata from the source (e.g. CSV columns)
        if doc_metadata:
            kg['metadata']['doc_metadata'] = doc_metadata

        # Step 5: Store in Neo4j if requested
        if file_name:
            success = self.store_knowledge_graph_with_embeddings(
                kg, file_name, doc_metadata=doc_metadata, doc_hash=doc_hash
            )
            kg['metadata']['stored_in_neo4j'] = success

        return kg

    def _get_node_color(self, entity_type: str) -> str:
        """Get color for node based on entity type"""
        color_map = {
            "Disease": "#ff6666",
            "Treatment": "#66ff66",
            "Drug": "#66ffff",
            "Symptom": "#ffff66",
            "Patient": "#ff9999",
            "Physician": "#ffcc99",
            "Hospital": "#99ccff",
            "MedicalProcedure": "#cc99ff",
            "MedicalDevice": "#ff99cc",
            "Anatomy": "#99ff99",
            "Concept": "#a6cee3"
        }
        return color_map.get(entity_type, "#a6cee3")

    def store_knowledge_graph_with_embeddings(self, kg: Dict[str, Any], file_name: str, doc_metadata: dict = None, doc_hash: str = None) -> bool:
        """
        Store the knowledge graph in Neo4j database with proper embedding support
        """
        try:
            # Try to create Neo4j connection - handle APOC issues gracefully
            try:
                graph = self._create_neo4j_connection()
            except Exception as conn_error:
                if "APOC" in str(conn_error) or "apoc" in str(conn_error):
                    logging.warning(f"APOC not available, skipping advanced KG storage: {conn_error}")
                    return False
                else:
                    raise conn_error

            # Create document node with versioning
            import uuid
            kg_version = str(uuid.uuid4())
            # Get kgName from metadata (set during KG generation)
            kg_name_value = kg['metadata'].get('kg_name') or file_name or "default"
            schema_card = self._build_schema_card()
            schema_card_json = json.dumps(schema_card, ensure_ascii=False)
            doc_query = """
            MERGE (d:Document {fileName: $fileName, kgName: $kgName})
            SET d.kgVersion = $kgVersion,
                d.kgName = $kgName,
                d.createdAt = datetime(),
                d.updatedAt = datetime(),
                d.totalChunks = $totalChunks,
                d.totalEntities = $totalEntities,
                d.totalRelationships = $totalRelationships,
                d.ontologyClasses = $ontologyClasses,
                d.ontologyRelationships = $ontologyRelationships,
                d.contentHash = $contentHash,
                d.schemaCard = $schemaCard,
                d.schemaVersion = $schemaVersion,
                d.schemaHash = $schemaHash
            """
            graph.query(doc_query, {
                "kgVersion": kg_version,
                "kgName": kg_name_value,
                "fileName": file_name,
                "totalChunks": kg['metadata']['total_chunks'],
                "totalEntities": kg['metadata']['total_entities'],
                "totalRelationships": kg['metadata']['total_relationships'],
                "ontologyClasses": kg['metadata']['ontology_classes'],
                "ontologyRelationships": kg['metadata']['ontology_relationships'],
                "contentHash": doc_hash or "",
                "schemaCard": schema_card_json,
                "schemaVersion": schema_card["schemaVersion"],
                "schemaHash": schema_card["schemaHash"],
            })

            # Store document-level metadata from source (e.g. CSV columns like SUBJECT_ID, HADM_ID)
            if doc_metadata:
                # Sanitise: Neo4j only accepts str/int/float/bool — drop NaN and convert the rest
                import math
                safe_meta = {
                    k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                    for k, v in doc_metadata.items()
                    if v is not None and not (isinstance(v, float) and math.isnan(v))
                }
                if safe_meta:
                    graph.query(
                        "MATCH (d:Document {fileName: $fileName, kgName: $kgName}) SET d += $meta",
                        {"fileName": file_name, "kgName": kg_name_value, "meta": safe_meta},
                    )

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
                MATCH (d:Document {fileName: $fileName, kgName: $kgName})
                MERGE (c)-[:PART_OF]->(d)
                """
                graph.query(chunk_doc_query, {
                    "chunk_id": chunk_id,
                    "fileName": file_name,
                    "kgName": kg_name_value,
                })

            # Create entity nodes with embeddings and ontology-based labels.
            # MERGE on node id (kg-prefixed UUID) so each KG's entities are independent.
            # content_hash is stored as a property for reference but is NOT the merge key,
            # which previously caused relationship storage to silently fail: a second KG
            # build would find the existing node by content_hash but leave n.id pointing at
            # the first KG's prefix, so MATCH (source {id: "kg2_uuid"}) never matched.
            for node in kg['nodes']:
                # Generate content-based deduplication hash (stored as property, not merge key)
                original_id = node.get('properties', {}).get('original_id', node['id'])
                normalized_content = self._normalize_entity_text(original_id)
                content_hash = hashlib.md5(f"{node['label']}:{normalized_content}".encode()).hexdigest()
                node['content_hash'] = content_hash

            for node in kg['nodes']:
                properties = node.get('properties', {})
                entity_type = node['label']  # This is the ontology class (Disease, Treatment, etc.)
                # Sanitize entity type for Cypher label compatibility and validate against whitelist
                cypher_safe_entity_type = re.sub(r'[^A-Za-z0-9_]', '_', entity_type).strip('_') or 'Concept'
                # Validate: label must start with a letter, max 64 chars, no injection risk
                if not re.match(r'^[A-Za-z][A-Za-z0-9_]{0,63}$', cypher_safe_entity_type):
                    logging.warning(f"Unsafe entity type '{entity_type}' → falling back to 'Concept'")
                    cypher_safe_entity_type = 'Concept'

                # Generate embedding for entity if it doesn't have one
                entity_embedding = node.get('embedding')
                if entity_embedding is None:
                    entity_text = properties.get('name', node['id'])
                    if properties.get('description'):
                        entity_text += " " + properties.get('description', '')
                    try:
                        entity_embedding = self.embedding_function.embed_query(entity_text)
                    except Exception as e:
                        logging.warning(f"Failed to generate embedding for entity {node['id']}: {e}")
                        entity_embedding = None

                # MERGE on id (kg-scoped) so relationship MATCH always finds this exact node.
                # kgName is stored as a property so entities can be loaded without traversing
                # chunk paths (which fail when mention-linking doesn't find the entity text).
                node_query = f"""
                MERGE (n:{cypher_safe_entity_type}:__Entity__ {{id: $id}})
                ON CREATE SET
                    n.name = $name,
                    n.type = $type,
                    n.description = $description,
                    n.embedding = $embedding,
                    n.ontology_class = $entity_type,
                    n.content_hash = $content_hash,
                    n.kgName = $kg_name,
                    n.all_names = $all_names,
                    n.original_ids = $original_ids,
                    n.created_at = datetime()
                ON MATCH SET
                    n.last_accessed = datetime(),
                    n.kgName = $kg_name,
                    n.all_names = coalesce(n.all_names, []) + $all_names,
                    n.original_ids = coalesce(n.original_ids, []) + $original_ids
                """
                graph.query(node_query, {
                    "id": node['id'],
                    "content_hash": node['content_hash'],
                    "kg_name": kg_name_value,
                    "name": properties.get('name', node['id']),
                    "type": node['label'],
                    "description": properties.get('description', ''),
                    "embedding": entity_embedding,
                    "entity_type": entity_type,
                    "all_names": list(set(properties.get('all_names', [node['id']]))),
                    "original_ids": list(set(properties.get('original_ids', [node['id']])))
                })

            # Build prefixed-UUID → human-readable name lookup for confidence verification.
            # kg['nodes'][i]['id'] is the prefixed UUID (e.g. "kg_abc_<uuid>")
            # kg['nodes'][i]['properties']['name'] is the actual entity name text.
            # rel['from'] / rel['to'] use the same prefixed-UUID format.
            _uuid_to_name = {
                _n['id']: (_n.get('properties', {}).get('name') or _n['id'])
                for _n in kg.get('nodes', [])
                if _n.get('id')
            }

            # Create relationships with improved error handling
            relationships_stored = 0
            for idx, rel in enumerate(kg['relationships']):
                try:
                    # Filter out 'id' property to avoid duplicate key issues in database
                    properties_filtered = {k: v for k, v in rel.get('properties', {}).items() if k != 'id'}

                    # Canonicalize relationship type against ontology using fuzzy matching
                    sanitized_rel_type = self._canonicalize_relationship_type(rel['type'])

                    # Resolve UUIDs to entity names for evidence-grounded confidence check.
                    # Prefer explicit source_name/target_name properties; fall back to name lookup.
                    _src_id = rel.get('from') or rel.get('source', '')
                    _tgt_id = rel.get('to') or rel.get('target', '')
                    source_name = (rel.get('properties', {}).get('source_name')
                                   or _uuid_to_name.get(_src_id)
                                   or _src_id)
                    target_name = (rel.get('properties', {}).get('target_name')
                                   or _uuid_to_name.get(_tgt_id)
                                   or _tgt_id)
                    triple_confidence = self._verify_triple_confidence(
                        source_name, target_name, sanitized_rel_type, kg.get('chunks', [])
                    )

                    # Reject only clear hallucinations: neither entity found anywhere in the document.
                    # Score 0.1 = neither entity present; 0.4+ = at least one entity grounded in text.
                    # Threshold just above 0.1 avoids discarding relationships where one entity
                    # is confirmed (score 0.4) or entity names have minor surface-form mismatches.
                    if triple_confidence < 0.15:
                        logging.info(
                            "Skipping hallucinated relationship (confidence=%.2f): %s -[%s]-> %s",
                            triple_confidence, rel.get('from'), sanitized_rel_type, rel.get('to'),
                        )
                        continue

                    properties_with_confidence = {**properties_filtered, "confidence": triple_confidence}

                    rel_query = f"""
                    MATCH (source:__Entity__ {{id: $source_id}})
                    MATCH (target:__Entity__ {{id: $target_id}})
                    MERGE (source)-[r:{sanitized_rel_type}]->(target)
                    SET r += $properties
                    """

                    logging.info(
                        "Creating relationship %d/%d: %s -[%s]-> %s (confidence=%.2f)",
                        idx + 1, len(kg['relationships']),
                        rel.get('from'), sanitized_rel_type, rel.get('to'), triple_confidence,
                    )

                    graph.query(rel_query, {
                        "source_id": rel.get('from'),  # Use 'from' field which has the prefixed UUID
                        "target_id": rel.get('to'),    # Use 'to' field which has the prefixed UUID
                        "properties": properties_with_confidence
                    })

                    relationships_stored += 1

                except Exception as rel_error:
                    logging.error(f"Failed to store relationship {idx+1}: {rel} - Error: {rel_error}")
                    continue

            logging.info(f"Successfully stored {relationships_stored} out of {len(kg['relationships'])} relationships")

            # Link entities to chunks via per-fact provenance Mention nodes
            # Pattern: (Entity)-[:MENTIONED_IN]->(Mention {quote, ...})-[:FROM_CHUNK]->(Chunk)
            def _mention_boundary(name: str) -> re.Pattern:
                """Adaptive boundary pattern — handles names ending in non-word chars like '(' or '-'."""
                prefix = r'(?<!\w)' if not name[:1].isalnum() and name[:1] != '_' else r'\b'
                suffix = r'(?!\w)' if not name[-1:].isalnum() and name[-1:] != '_' else r'\b'
                return re.compile(prefix + re.escape(name) + suffix)

            for chunk in kg['chunks']:
                chunk_id = hashlib.sha1(chunk['text'].encode()).hexdigest()
                chunk_text_lower = chunk['text'].lower()

                for node in kg['nodes']:
                    properties = node.get('properties', {})
                    candidate_names = []
                    candidate_names.extend(properties.get('all_names', []) if isinstance(properties.get('all_names', []), list) else [])
                    candidate_names.append(properties.get('name', ''))
                    candidate_names.append(properties.get('original_id', ''))

                    # Keep meaningful normalized names only
                    normalized_names = [n.strip().lower() for n in candidate_names if isinstance(n, str) and len(n.strip()) > 2]
                    matched_name = next(
                        (n for n in normalized_names if _mention_boundary(n).search(chunk_text_lower)),
                        None
                    )
                    if matched_name:
                        # Extract a short quote: the sentence containing the matched name
                        sentences = re.split(r'(?<=[.!?])\s+', chunk['text'])
                        quote = next(
                            (s.strip() for s in sentences if matched_name in s.lower()),
                            chunk['text'][:200]
                        )[:500]  # cap at 500 chars
                        # Seed mention_id from node['id'] (the unique key), not content_hash.
                        # content_hash is no longer unique across KGs (Fix 8 changed MERGE to id).
                        mention_id = hashlib.sha256(
                            f"{node['id']}::{chunk_id}".encode()
                        ).hexdigest()
                        mention_query = """
                        MATCH (c:Chunk {id: $chunk_id})
                        MATCH (e:__Entity__ {id: $entity_id})
                        MERGE (m:Mention {id: $mention_id})
                        SET m.quote = $quote,
                            m.chunkIndex = $chunk_index,
                            m.chunkStart = $chunk_start,
                            m.chunkEnd = $chunk_end,
                            m.entityName = $entity_name,
                            m.createdAt = datetime()
                        MERGE (e)-[:MENTIONED_IN]->(m)
                        MERGE (m)-[:FROM_CHUNK]->(c)
                        MERGE (c)-[:HAS_ENTITY]->(e)
                        """
                        graph.query(mention_query, {
                            "chunk_id": chunk_id,
                            "entity_id": node['id'],
                            "mention_id": mention_id,
                            "quote": quote,
                            "chunk_index": chunk.get('chunk_id', 0),
                            "chunk_start": chunk.get('start_pos', 0),
                            "chunk_end": chunk.get('end_pos', 0),
                            "entity_name": properties.get('name', ''),
                        })

            # Create vector indexes for RAG
            self._create_vector_indexes(graph)

            logging.info(f"Successfully stored ontology-guided knowledge graph for {file_name}")
            return True

        except Exception as e:
            logging.error(f"Error storing knowledge graph: {e}")
            return False

    def _create_vector_indexes(self, graph):
        """
        Create vector indexes and unique constraints for RAG functionality
        """
        try:
            # Create unique constraint for entity IDs to prevent duplicates
            entity_constraint_query = """
            CREATE CONSTRAINT unique_entity_id IF NOT EXISTS
            FOR (e:__Entity__) REQUIRE e.id IS UNIQUE
            """
            graph.query(entity_constraint_query)

            # Create unique constraint for chunk IDs
            chunk_constraint_query = """
            CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS
            FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """
            graph.query(chunk_constraint_query)

            # Create composite uniqueness for dataset-scoped documents
            # (same fileName may exist in different kgName datasets)
            doc_constraint_query = """
            CREATE CONSTRAINT unique_document_filename_kgname IF NOT EXISTS
            FOR (d:Document) REQUIRE (d.fileName, d.kgName) IS UNIQUE
            """
            graph.query(doc_constraint_query)

            # Unique constraint for Mention nodes (entity × chunk pair)
            mention_constraint_query = """
            CREATE CONSTRAINT unique_mention_id IF NOT EXISTS
            FOR (m:Mention) REQUIRE m.id IS UNIQUE
            """
            graph.query(mention_constraint_query)

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

            # Index on entity name for fast text-matching and multi-hop traversal
            # lookups (EnhancedRAGSystem._expand_entities_via_graph seeds from e.id)
            entity_id_index_query = """
            CREATE INDEX entity_id_index IF NOT EXISTS
            FOR (e:__Entity__) ON (e.id)
            """
            graph.query(entity_id_index_query)

            entity_name_index_query = """
            CREATE INDEX entity_name_index IF NOT EXISTS
            FOR (e:__Entity__) ON (e.name)
            """
            graph.query(entity_name_index_query)

            # Composite index for chunk→document lookup used in kg_name filtering
            chunk_kg_index_query = """
            CREATE INDEX chunk_document_index IF NOT EXISTS
            FOR ()-[r:PART_OF]-() ON (r)
            """
            try:
                graph.query(chunk_kg_index_query)
            except Exception:
                pass  # Relationship indexes not supported on all Neo4j versions

            logging.info("Created constraints, vector, keyword, and entity lookup indexes for RAG")

        except Exception as e:
            logging.warning(f"Error creating constraints/indexes (may already exist): {e}")

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
