import json
import re
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain_neo4j import Neo4jGraph
from langchain_text_splitters import TokenTextSplitter
import os
import sys
import logging

# Add llm-graph-builder paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'llm-graph-builder/backend/src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'llm-graph-builder/backend'))

from shared import common_fn

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
        embedding_model: str = "openai",
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
        self.embedding_function, self.embedding_dimension = common_fn.load_embedding_model(embedding_model)
        logging.info(f"Initialized embedding model: {embedding_model}, dimension: {self.embedding_dimension}")

        # Load ontology if provided
        self.ontology_classes = []
        self.ontology_relationships = []
        if ontology_path and os.path.exists(ontology_path):
            self._load_ontology(ontology_path)
            logging.info(f"Loaded ontology: {len(self.ontology_classes)} classes, {len(self.ontology_relationships)} relationships")
        else:
            logging.warning(f"Ontology file not found: {ontology_path}")

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
            # Fallback to basic medical ontology
            self._create_basic_medical_ontology()

    def _create_basic_medical_ontology(self):
        """
        Create a basic medical ontology as fallback
        """
        self.ontology_classes = [
            {'id': 'Disease', 'uri': 'medical:Disease', 'label': 'Disease'},
            {'id': 'Treatment', 'uri': 'medical:Treatment', 'label': 'Treatment'},
            {'id': 'Drug', 'uri': 'medical:Drug', 'label': 'Drug'},
            {'id': 'Symptom', 'uri': 'medical:Symptom', 'label': 'Symptom'},
            {'id': 'Patient', 'uri': 'medical:Patient', 'label': 'Patient'},
            {'id': 'Physician', 'uri': 'medical:Physician', 'label': 'Physician'},
            {'id': 'Hospital', 'uri': 'medical:Hospital', 'label': 'Hospital'},
            {'id': 'MedicalProcedure', 'uri': 'medical:MedicalProcedure', 'label': 'Medical Procedure'},
            {'id': 'MedicalDevice', 'uri': 'medical:MedicalDevice', 'label': 'Medical Device'},
            {'id': 'Anatomy', 'uri': 'medical:Anatomy', 'label': 'Anatomy'}
        ]

        self.ontology_relationships = [
            {'id': 'treats', 'uri': 'medical:treats', 'label': 'Treats'},
            {'id': 'causes', 'uri': 'medical:causes', 'label': 'Causes'},
            {'id': 'hasSymptom', 'uri': 'medical:hasSymptom', 'label': 'Has Symptom'},
            {'id': 'prescribes', 'uri': 'medical:prescribes', 'label': 'Prescribes'},
            {'id': 'diagnoses', 'uri': 'medical:diagnoses', 'label': 'Diagnoses'},
            {'id': 'locatedIn', 'uri': 'medical:locatedIn', 'label': 'Located In'},
            {'id': 'partOf', 'uri': 'medical:partOf', 'label': 'Part Of'},
            {'id': 'affects', 'uri': 'medical:affects', 'label': 'Affects'}
        ]

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
        Extract entities and relationships using LLM with ontology guidance
        """
        # Create ontology context for the prompt
        ontology_classes_text = "\n".join([f"- {cls['label']} ({cls['id']})" for cls in self.ontology_classes[:15]])
        ontology_relationships_text = "\n".join([f"- {rel['label']} ({rel['id']})" for rel in self.ontology_relationships[:10]])

        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an expert medical knowledge graph extraction system.
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

IMPORTANT: Return ONLY the JSON object, no additional text or explanation."""),
            ("human", f"Extract medical entities and relationships from this text:\n\n{chunk_text}")
        ])

        try:
            chain = extraction_prompt | llm | StrOutputParser()
            response = chain.invoke({})

            # Clean the response to extract JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            # Parse JSON response
            result = json.loads(response)

            # Validate the result has the expected structure
            if not isinstance(result, dict) or 'entities' not in result or 'relationships' not in result:
                raise ValueError("Invalid response structure")

            # Filter out non-medical entities
            medical_entities = []
            for entity in result.get('entities', []):
                if self._is_medical_entity(entity):
                    medical_entities.append(entity)

            # Filter relationships to only include medical ones
            medical_relationships = []
            entity_ids = {e['id'] for e in medical_entities}
            for rel in result.get('relationships', []):
                if rel['source'] in entity_ids and rel['target'] in entity_ids:
                    medical_relationships.append(rel)

            return {
                'entities': medical_entities,
                'relationships': medical_relationships
            }

        except Exception as e:
            logging.warning(f"LLM extraction failed: {e}, falling back to pattern matching")
            return self._extract_entities_and_relationships_fallback(chunk_text)

    def _is_medical_entity(self, entity: Dict[str, Any]) -> bool:
        """
        Check if an entity is medically relevant
        """
        entity_text = entity.get('id', '').lower()
        entity_type = entity.get('type', '').lower()

        # Check if entity type is from our ontology
        for cls in self.ontology_classes:
            if cls['id'].lower() == entity_type:
                return True

        # Check for medical keywords in the entity text
        medical_keywords = [
            'cancer', 'disease', 'treatment', 'therapy', 'drug', 'medication',
            'symptom', 'diagnosis', 'patient', 'physician', 'doctor', 'hospital',
            'clinic', 'surgery', 'procedure', 'tumor', 'disorder', 'syndrome',
            'clinical', 'medical', 'health', 'care', 'therapy', 'medicine'
        ]

        return any(keyword in entity_text for keyword in medical_keywords)

    def _extract_entities_and_relationships_fallback(self, chunk_text: str) -> Dict[str, Any]:
        """
        Fallback entity extraction using pattern matching with ontology guidance
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
                if len(match) > 2 and match not in ['The', 'This', 'That', 'And', 'But', 'For', 'With', 'From', 'Into', 'Upon']:
                    found_entities.add(match)

        # Filter for medical relevance and convert to entity format
        medical_entities = []
        for entity_text in found_entities:
            # Check if entity is medically relevant
            if self._is_medical_entity_fallback(entity_text):
                entity_type = self._classify_entity_with_ontology(entity_text)

                # Generate embedding for entity
                try:
                    entity_embedding = self.embedding_function.embed_query(entity_text)
                except Exception as e:
                    logging.warning(f"Failed to generate embedding for entity {entity_text}: {e}")
                    entity_embedding = None

                medical_entities.append({
                    "id": entity_text,
                    "type": entity_type,
                    "properties": {
                        "name": entity_text,
                        "description": f"{entity_type}: {entity_text}"
                    },
                    "embedding": entity_embedding
                })

        entities.extend(medical_entities)

        # Create relationships ONLY between medical entities that are semantically related
        for i in range(len(medical_entities) - 1):
            source_entity = medical_entities[i]
            target_entity = medical_entities[i + 1]

            # Only create relationships between entities that could be medically related
            if self._entities_can_be_related(source_entity['type'], target_entity['type']):
                rel_type = self._classify_relationship_with_ontology(source_entity['id'], target_entity['id'])
                relationships.append({
                    "source": source_entity['id'],
                    "target": target_entity['id'],
                    "type": rel_type,
                    "properties": {"description": f"{rel_type} relationship between {source_entity['type']} and {target_entity['type']}"}
                })

        return {
            "entities": entities,
            "relationships": relationships
        }

    def _is_medical_entity_fallback(self, entity_text: str) -> bool:
        """
        Check if an entity is medically relevant for fallback extraction
        """
        entity_lower = entity_text.lower()

        # Skip common non-medical words
        skip_words = [
            'guidelines', 'recommendations', 'committee', 'association', 'society',
            'university', 'institute', 'center', 'department', 'section', 'chapter',
            'table', 'figure', 'page', 'volume', 'issue', 'edition', 'update',
            'limited', 'text', 'update', 'march', 'european', 'american', 'international'
        ]

        if any(word in entity_lower for word in skip_words):
            return False

        # Check for medical keywords
        medical_keywords = [
            'cancer', 'disease', 'treatment', 'therapy', 'drug', 'medication',
            'symptom', 'diagnosis', 'patient', 'physician', 'doctor', 'hospital',
            'clinic', 'surgery', 'procedure', 'tumor', 'disorder', 'syndrome',
            'clinical', 'medical', 'health', 'care', 'therapy', 'medicine',
            'prostate', 'breast', 'lung', 'liver', 'kidney', 'heart', 'brain',
            'bone', 'blood', 'skin', 'lung', 'stomach', 'colon', 'pancreas'
        ]

        return any(keyword in entity_lower for keyword in medical_keywords)

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
        Generate knowledge graph from text with ontology-guided entity extraction
        """
        logging.info("Starting ontology-guided knowledge graph generation")

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
                "ontology_classes": len(self.ontology_classes),
                "ontology_relationships": len(self.ontology_relationships),
                "extraction_method": "ontology_guided_llm",
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
            graph = self._create_neo4j_connection()

            # Create document node
            doc_query = """
            MERGE (d:Document {fileName: $fileName})
            SET d.createdAt = datetime(),
                d.totalChunks = $totalChunks,
                d.totalEntities = $totalEntities,
                d.totalRelationships = $totalRelationships,
                d.ontologyClasses = $ontologyClasses,
                d.ontologyRelationships = $ontologyRelationships
            """
            graph.query(doc_query, {
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

            logging.info(f"Successfully stored ontology-guided knowledge graph for {file_name}")
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
