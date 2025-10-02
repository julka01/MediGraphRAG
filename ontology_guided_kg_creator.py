import json
import re
import hashlib
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
from kg_utils.common_functions import load_embedding_model

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
            except Exception as e:
                logging.error(f"❌ Failed to load ontology: {e}")
                # Continue with empty ontology - LLM extraction will still work
        else:
            if ontology_path:
                logging.warning(f"Ontology file not found: {ontology_path}")
                logging.info(f"Available files in temp dir: {os.listdir(os.path.dirname(ontology_path)) if ontology_path else 'N/A'}")
            else:
                logging.info("No ontology provided - using basic LLM entity extraction")
            # No ontology - use empty lists (will fall back to pattern matching)

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
            database=self.neo4j_database
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

    def _extract_entities_and_relationships_with_llm(self, chunk_text: str, llm, model_name: str = "openai/gpt-oss-20b:free") -> Dict[str, Any]:
        """
        Extract entities and relationships using LLM with ontology guidance (if ontology available) or natural LLM detection
        """
        # Check if ontology is available
        has_ontology = bool(self.ontology_classes) or bool(self.ontology_relationships)

        if has_ontology:
            # Ontology-guided extraction
            ontology_classes_text = "\n".join([f"- {cls['label']} ({cls['id']})" for cls in self.ontology_classes[:50]])
            ontology_relationships_text = "\n".join([f"- {rel['label']} ({rel['id']})" for rel in self.ontology_relationships[:30]])

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

Return ONLY a valid JSON object in this exact format:
{{
  "entities": [
    {{
      "id": "exact_entity_name_from_text",
      "type": "OntologyClass",
      "properties": {{
        "name": "exact_entity_name_from_text",
        "description": "brief medical description"
      }}
    }}
  ],
  "relationships": [
    {{
      "source": "source_entity_id",
      "target": "target_entity_id",
      "type": "ONTOLOGY_RELATIONSHIP",
      "properties": {{
        "description": "how they are related in the text"
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

Return ONLY a valid JSON object in this exact format:
{{
  "entities": [
    {{
      "id": "exact_entity_name_from_text",
      "type": "EntityType",
      "properties": {{
        "name": "exact_entity_name_from_text",
        "description": "contextual description"
      }}
    }}
  ],
  "relationships": [
    {{
      "source": "source_entity_id",
      "target": "target_entity_id",
      "type": "RELATIONSHIP_TYPE",
      "properties": {{
        "description": "how they are related in the text"
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

            # Try to extract JSON from response - handle cases where LLM returns malformed JSON
            json_start = response.find('{')
            if json_start == -1:
                logging.warning(f"No JSON start found in response: {response[:200]}... Returning empty result.")
                return {'entities': [], 'relationships': []}

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
                try:
                    result = json.loads(json_content)
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON parsing error: {e}. Content: {json_content[:300]}... Returning empty result.")
                    return {'entities': [], 'relationships': []}
            else:
                logging.warning(f"Incomplete JSON in response (brace_count={brace_count}): {response[json_start:json_start+300]}... Returning empty result.")
                return {'entities': [], 'relationships': []}

            # Validate the result has the expected structure
            if not isinstance(result, dict) or 'entities' not in result or 'relationships' not in result:
                logging.warning(f"Invalid response structure. Returning empty result.")
                return {'entities': [], 'relationships': []}

            # Filter out non-medical entities
            medical_entities = []
            for entity in result.get('entities', []):
                # Handle both dict and string formats from LLM response
                if isinstance(entity, str):
                    # Convert string entity to dict format
                    entity = {
                        "id": entity,
                        "type": self._classify_entity_with_ontology(entity),
                        "properties": {
                            "name": entity,
                            "description": f"{self._classify_entity_with_ontology(entity)}: {entity}"
                        }
                    }
                if self._is_medical_entity(entity):
                    medical_entities.append(entity)

            # Filter relationships to only include medical ones
            medical_relationships = []
            entity_ids = {e['id'] for e in medical_entities}

            # Process relationships, handling both dict and string formats
            for rel in result.get('relationships', []):
                if isinstance(rel, str):
                    # Skip string relationships for now (LLM returned malformed data)
                    continue
                elif isinstance(rel, dict):
                    # Verify both source and target exist
                    if (isinstance(rel.get('source'), str) and
                        isinstance(rel.get('target'), str) and
                        rel.get('source', '') in entity_ids and
                        rel.get('target', '') in entity_ids):
                        medical_relationships.append(rel)
                    else:
                        # Try to create a valid relationship from incomplete data
                        source = rel.get('source', '')
                        target = rel.get('target', '')
                        rel_type = rel.get('type', 'RELATED_TO')

                        # Only add if we have valid source and target
                        if source and target and source in entity_ids and target in entity_ids:
                            medical_relationships.append({
                                'source': source,
                                'target': target,
                                'type': rel_type,
                                'properties': rel.get('properties', {})
                            })

            return {
                'entities': medical_entities,
                'relationships': medical_relationships
            }

        except Exception as e:
            logging.error(f"LLM extraction failed: {e}")
            # Return empty result instead of failing completely
            return {'entities': [], 'relationships': []}

    def _is_medical_entity(self, entity: Dict[str, Any]) -> bool:
        """
        Check if an entity is medically relevant - relaxed constraints
        """
        # Relaxed: Allow all entities extracted by LLM, trusting the ontology guidance and prompt instructions
        return True





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

    def _classify_entity_with_ontology(self, entity_text: str) -> str:
        """
        Classify entity using ontology guidance
        """
        entity_lower = entity_text.lower()

        # Check for matches with ontology classes
        for cls in self.ontology_classes:
            if cls['id'].lower() in entity_lower or cls['label'].lower() in entity_lower:
                return cls['id']

        # Fallback classification based on keywords
        if any(word in entity_lower for word in ['disease', 'cancer', 'tumor', 'syndrome', 'disorder']):
            return 'Disease'
        elif any(word in entity_lower for word in ['drug', 'medication', 'treatment', 'therapy', 'surgery']):
            return 'Treatment'
        elif any(word in entity_lower for word in ['patient', 'person', 'individual']):
            return 'Patient'
        elif any(word in entity_lower for word in ['doctor', 'physician', 'surgeon', 'specialist']):
            return 'Physician'
        elif any(word in entity_lower for word in ['hospital', 'clinic', 'center', 'institute']):
            return 'Hospital'
        elif any(word in entity_lower for word in ['symptom', 'sign', 'manifestation']):
            return 'Symptom'
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

    def _generate_entity_id(self, entity: Dict[str, Any]) -> str:
        """
        Generate a UUID-based entity ID to prevent duplicates
        """
        # Create a unique seed based on entity type and normalized name
        unique_seed = f"{entity['type']}_{entity['id'].lower().strip()}"
        # Generate UUID5 (name-based) for deterministic but unique IDs
        return str(uuid.uuid5(uuid.NAMESPACE_OID, unique_seed))

    def _normalize_entity_text(self, text: str) -> str:
        """
        Normalize entity text for better duplicate detection
        """
        # Convert to lowercase, remove extra whitespace, normalize common variations
        normalized = re.sub(r'\s+', ' ', text.lower().strip())

        # Common medical abbreviations and normalizations
        normalizations = {
            'prostate cancer': 'prostate_cancer',
            'breast cancer': 'breast_cancer',
            'lung cancer': 'lung_cancer',
            r'\bpc\b': 'prostate_cancer',  # Prostate cancer abbreviation
            r'\bbc\b': 'breast_cancer',    # Breast cancer abbreviation
            r'\blc\b': 'lung_cancer',      # Lung cancer abbreviation
            'radical prostatectomy': 'radical_prostatectomy',
            'brachytherapy': 'brachytherapy',
            'hormone therapy': 'hormone_therapy',
            'radiation therapy': 'radiation_therapy',
        }

        # Apply normalizations
        for full, normalized_form in normalizations.items():
            if isinstance(full, str):
                normalized = re.sub(rf'\b{re.escape(full)}\b', normalized_form, normalized)
            else:  # regex pattern
                normalized = re.sub(full, normalized_form, normalized)

        return normalized

    def _harmonize_entities(self, all_entities: List[Dict]) -> List[Dict]:
        """
        Harmonize entities across chunks to avoid duplicates using improved normalization
        """
        logging.info(f"Starting harmonization of {len(all_entities)} raw entities")

        # Step 1: Build initial grouping by normalized form
        entity_groups = defaultdict(list)

        for entity in all_entities:
            normalized_text = self._normalize_entity_text(entity['id'])
            normalized_key = f"{entity['type']}:{normalized_text}"
            entity_groups[normalized_key].append(entity)

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

            # Take the first entity as the representative
            representative_entity = entities[0].copy()

            # Merge information from all occurrences
            all_names = set()
            all_descriptions = set()

            for entity in entities:
                all_names.add(entity['id'])
                if entity.get('properties', {}).get('description'):
                    all_descriptions.add(entity['properties']['description'])

                # Keep the best embedding (prefer any available embedding)
                if not representative_entity.get('embedding') and entity.get('embedding'):
                    representative_entity['embedding'] = entity['embedding']

                # Merge additional properties
                if entity.get('properties'):
                    representative_entity.setdefault('properties', {}).update(entity['properties'])

            # Update representative entity with merged information
            if len(all_names) > 1 or len(all_descriptions) > 1:
                # Create a canonical name if multiple variations exist
                representative_entity['properties']['all_names'] = list(all_names)
                representative_entity['properties']['all_descriptions'] = list(all_descriptions)
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

        return harmonized_entities

    def _harmonize_relationships(self, all_relationships: List[Dict], entity_map: Dict) -> List[Dict]:
        """
        Harmonize relationships across chunks and map to UUID-based entity IDs
        """
        harmonized_relationships = []
        seen_relationships = set()

        # Create reverse mapping from original ID to UUID
        original_to_uuid = {}
        for entity in entity_map.values():
            original_to_uuid[entity['id']] = entity['uuid']

        for rel in all_relationships:
            # Map source and target to UUIDs
            source_uuid = original_to_uuid.get(rel['source'])
            target_uuid = original_to_uuid.get(rel['target'])

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

    def generate_knowledge_graph(self, text: str, llm, file_name: str = None, model_name: str = "openai/gpt-oss-20b:free", max_chunks: int = None, kg_name: str = None) -> Dict[str, Any]:
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

        for i, chunk in enumerate(chunks):
            try:
                logging.info(f"Processing chunk {i+1}/{len(chunks)}")
                chunk_kg = self._extract_entities_and_relationships_with_llm(chunk['text'], llm, model_name)

                if chunk_kg['entities'] or chunk_kg['relationships']:
                    all_entities.extend(chunk_kg['entities'])
                    all_relationships.extend(chunk_kg['relationships'])
                    processed_chunks += 1
                    logging.info(f"✓ Chunk {i+1} processed: {len(chunk_kg['entities'])} entities, {len(chunk_kg['relationships'])} relationships")
                else:
                    logging.warning(f"⚠ Chunk {i+1} returned no entities/relationships")
                    failed_chunks += 1


            except Exception as e:
                logging.error(f"❌ Failed to process chunk {i+1}: {e}")
                failed_chunks += 1
                continue

            # Add small delay between chunks to avoid rate limiting
            if i < len(chunks) - 1:  # Don't delay after the last chunk
                time.sleep(1.0)  # 1 second delay between API calls

        logging.info(f"Processing complete: {processed_chunks} successful, {failed_chunks} failed")

        # Step 3: Harmonize entities and relationships
        harmonized_entities = self._harmonize_entities(all_entities)
        # Create entity map using UUIDs for relationships
        entity_map = {entity['uuid']: entity for entity in harmonized_entities}
        harmonized_relationships = self._harmonize_relationships(all_relationships, entity_map)

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

    def store_knowledge_graph_with_embeddings(self, kg: Dict[str, Any], file_name: str) -> bool:
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
            doc_query = """
            MERGE (d:Document {fileName: $fileName})
            SET d.kgVersion = $kgVersion,
                d.createdAt = datetime(),
                d.totalChunks = $totalChunks,
                d.totalEntities = $totalEntities,
                d.totalRelationships = $totalRelationships,
                d.ontologyClasses = $ontologyClasses,
                d.ontologyRelationships = $ontologyRelationships
            """
            graph.query(doc_query, {
                "kgVersion": kg_version,
                "fileName": file_name,
                "totalChunks": kg['metadata']['total_chunks'],
                "totalEntities": kg['metadata']['total_entities'],
                "totalRelationships": kg['metadata']['total_relationships'],
                "ontologyClasses": kg['metadata']['ontology_classes'],
                "ontologyRelationships": kg['metadata']['ontology_relationships']
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

            # Create entity nodes with embeddings and ontology-based labels
            for node in kg['nodes']:
                properties = node.get('properties', {})
                entity_type = node['label']  # This is the ontology class (Disease, Treatment, etc.)
                # Sanitize entity type for Cypher compatibility (remove spaces and special chars)
                cypher_safe_entity_type = entity_type.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace(',', '')

                # Generate embedding for entity if it doesn't have one
                entity_embedding = node.get('embedding')
                if entity_embedding is None:
                    # Generate embedding from entity name and description
                    entity_text = properties.get('name', node['id'])
                    if properties.get('description'):
                        entity_text += " " + properties.get('description', '')
                    try:
                        entity_embedding = self.embedding_function.embed_query(entity_text)
                    except Exception as e:
                        logging.warning(f"Failed to generate embedding for entity {node['id']}: {e}")
                        entity_embedding = None

                # Use specific ontology class label first, then __Entity__ to ensure ontology class is the primary label
                node_query = f"""
                MERGE (n:{cypher_safe_entity_type}:__Entity__ {{id: $id}})
                SET n.name = $name,
                    n.type = $type,
                    n.description = $description,
                    n.embedding = $embedding,
                    n.ontology_class = $entity_type
                """
                graph.query(node_query, {
                    "id": node['id'],
                    "name": properties.get('name', node['id']),
                    "type": node['label'],
                    "description": properties.get('description', ''),
                    "embedding": entity_embedding,
                    "entity_type": entity_type
                })

            # Create relationships
            for rel in kg['relationships']:
                # Filter out 'id' property to avoid duplicate key issues in database
                properties_filtered = {k: v for k, v in rel.get('properties', {}).items() if k != 'id'}

                # Use ontology relationship type if available, otherwise sanitize
                sanitized_rel_type = rel['type']
                # Check if it's one of the defined ontology relationships
                for ont_rel in self.ontology_relationships:
                    if ont_rel['label'].lower().replace(' ', '_') == rel['type'].lower().replace(' ', '_'):
                        sanitized_rel_type = ont_rel['id']
                        break
                # Otherwise sanitize manually
                sanitized_rel_type = sanitized_rel_type.replace(' ', '_').replace('-', '_').upper()

                rel_query = f"""
                MATCH (source:__Entity__ {{id: $source_id}})
                MATCH (target:__Entity__ {{id: $target_id}})
                MERGE (source)-[r:{sanitized_rel_type}]->(target)
                SET r += $properties
                """
                graph.query(rel_query, {
                    "source_id": rel['source'],
                    "target_id": rel['target'],
                    "properties": properties_filtered
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

            # Create unique constraint for document filenames
            doc_constraint_query = """
            CREATE CONSTRAINT unique_document_filename IF NOT EXISTS
            FOR (d:Document) REQUIRE d.fileName IS UNIQUE
            """
            graph.query(doc_constraint_query)

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

            logging.info("Created constraints, vector and keyword indexes for RAG")

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
