import json
import re
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph
import os
import sys
import logging

# Import from local kg_utils
from kg_utils.common_functions import load_embedding_model

# Import the ontology-guided creator for unification
from ontology_guided_kg_creator import OntologyGuidedKGCreator

# Import CSV processor for bulk operations
try:
    from csv_processor import MedicalReportCSVProcessor
except ImportError:
    logging.warning("MedicalReportCSVProcessor not available - bulk CSV operations disabled")
    MedicalReportCSVProcessor = None

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
        neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password"),
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
            database=self.neo4j_database,
            refresh_schema=False,
            sanitize=True
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


class UnifiedOntologyGuidedKGCreator(OntologyGuidedKGCreator):
    """
    Unified Knowledge Graph Creator combining ontology-guided LLM extraction with patient-specific parsing.
    Supports both general biomedical documents and patient reports with maximum detail depth.
    """

    def __init__(
        self,
        chunk_size: int = 2000,  # Larger chunks for better context continuity in patient reports
        chunk_overlap: int = 300,
        neo4j_uri: str = os.getenv("NEO4J_URI", "neo4j+s://01ae09e4.databases.neo4j.io"),
        neo4j_user: str = os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password"),
        neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j"),
        embedding_model: str = os.getenv("EMBEDDING_PROVIDER", "openai"),
        ontology_path: Optional[str] = None,
        max_chunks: int = None  # For testing - limit chunks processed per report
    ):
        super().__init__(chunk_size, chunk_overlap, neo4j_uri, neo4j_user, neo4j_password, neo4j_database, embedding_model)

        # Set max_chunks for testing
        self._test_max_chunks = max_chunks

        # Load biomedical ontology if available for enhanced entity classification
        self.ontology_classes = []
        self.ontology_relationships = []
        if ontology_path and os.path.exists(ontology_path):
            self._load_biomedical_ontology(ontology_path)
            logging.info(f"Loaded biomedical ontology: {len(self.ontology_classes)} classes, {len(self.ontology_relationships)} relationships")

    def _load_biomedical_ontology(self, ontology_path: str):
        """Load biomedical ontology classes and relationships from OWL file"""
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(ontology_path)
            root = tree.getroot()

            ns = {
                'owl': 'http://www.w3.org/2002/07/owl#',
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'
            }

            # Extract classes
            for class_elem in root.findall('.//owl:Class', ns):
                class_id = class_elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about', '')
                if class_id:
                    class_name = class_id.split('#')[-1] if '#' in class_id else class_id.split('/')[-1]
                    if class_name:
                        self.ontology_classes.append(class_name)

            # Extract relationships
            for prop_elem in root.findall('.//owl:ObjectProperty', ns):
                prop_id = prop_elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about', '')
                if prop_id:
                    prop_name = prop_id.split('#')[-1] if '#' in prop_id else prop_id.split('/')[-1]
                    if prop_name:
                        self.ontology_relationships.append(prop_name)

            logging.info(f"Loaded {len(self.ontology_classes)} ontology classes and {len(self.ontology_relationships)} relationships")

        except Exception as e:
            logging.warning(f"Failed to load biomedical ontology: {e}")

    def _extract_patient_demographics(self, text: str) -> Dict[str, Any]:
        """Extract patient demographics from text"""
        patient_info = {}

        # Extract name
        name_match = re.search(r'Name:\s*([^\n\r]+)', text, re.IGNORECASE)
        if name_match:
            patient_info['name'] = name_match.group(1).strip()

        # Extract unit/admission info
        unit_match = re.search(r'Unit No:\s*([^\n\r]+)', text, re.IGNORECASE)
        if unit_match:
            patient_info['unit_no'] = unit_match.group(1).strip()

        # Extract dates
        dob_match = re.search(r'Date of Birth:\s*([^\n\r]+)', text, re.IGNORECASE)
        if dob_match:
            patient_info['date_of_birth'] = dob_match.group(1).strip()

        admission_match = re.search(r'Admission Date:\s*([^\n\r]+)', text, re.IGNORECASE)
        if admission_match:
            patient_info['admission_date'] = admission_match.group(1).strip()

        discharge_match = re.search(r'Discharge Date:\s*([^\n\r]+)', text, re.IGNORECASE)
        if discharge_match:
            patient_info['discharge_date'] = discharge_match.group(1).strip()

        # Extract sex and service
        sex_match = re.search(r'Sex:\s*([^\n\r]+)', text, re.IGNORECASE)
        if sex_match:
            patient_info['sex'] = sex_match.group(1).strip()

        service_match = re.search(r'Service:\s*([^\n\r]+)', text, re.IGNORECASE)
        if service_match:
            patient_info['service'] = service_match.group(1).strip()

        # Extract attending physician
        attending_match = re.search(r'Attending:\s*([^\n\r]+)', text, re.IGNORECASE)
        if attending_match:
            patient_info['attending'] = attending_match.group(1).strip()

        return patient_info

    def _extract_medical_conditions(self, text: str) -> List[Dict[str, Any]]:
        """Extract medical conditions from various sections"""
        conditions = []

        # Past Medical History section - extract numbered conditions
        pmh_section = self._extract_section(text, "Past Medical History")
        if pmh_section:
            # Extract numbered list items (1., 2., etc.)
            condition_matches = re.findall(r'\d+\.\s*([^.\n]+)', pmh_section, re.MULTILINE)
            for match in condition_matches:
                condition = match.strip()
                if condition and len(condition) > 3 and not condition.startswith(('No ', 'no ')):
                    conditions.append({
                        'condition': condition,
                        'category': 'past_medical_history',
                        'status': 'chronic'
                    })

            # Also extract conditions from free text
            pmh_conditions = ['cirrhosis', 'HIV', 'IVDU', 'COPD', 'bipolar', 'PTSD', 'cancer', 'diabetes', 'hypertension']
            for condition in pmh_conditions:
                if condition.lower() in pmh_section.lower():
                    conditions.append({
                        'condition': condition,
                        'category': 'past_medical_history',
                        'status': 'chronic'
                    })

        # History of Present Illness - extract current conditions
        hopi_section = self._extract_section(text, "History of Present Illness")
        if hopi_section:
            # Extract conditions mentioned as causes
            condition_matches = re.findall(r'(?:c/b|complicated by|due to|secondary to)\s+([^.,;]+)', hopi_section, re.IGNORECASE)
            for match in condition_matches:
                match = match.strip()
                if len(match) > 2:
                    conditions.append({
                        'condition': match,
                        'category': 'present_illness',
                        'status': 'acute'
                    })

            # Extract other common presenting conditions
            if 'ascites' in hopi_section.lower():
                conditions.append({
                    'condition': 'ascites',
                    'category': 'present_illness',
                    'status': 'acute'
                })

        # Chief Complaint
        chief_complaint_section = self._extract_section(text, "Chief Complaint")
        if chief_complaint_section:
            # Extract the actual complaint
            complaint_text = chief_complaint_section.strip()
            if complaint_text.startswith('-'):
                complaint_text = complaint_text[1:].strip()

            if complaint_text and len(complaint_text) > 3:
                # Check for common conditions in chief complaint
                if 'abdominal' in complaint_text.lower() and 'distension' in complaint_text.lower():
                    conditions.append({
                        'condition': 'abdominal distension',
                        'category': 'chief_complaint',
                        'status': 'presenting'
                    })
                else:
                    conditions.append({
                        'condition': complaint_text,
                        'category': 'chief_complaint',
                        'status': 'presenting'
                    })

        # Discharge Diagnosis
        discharge_dx_section = self._extract_section(text, "Discharge Diagnosis")
        if discharge_dx_section:
            # Extract from discharge diagnosis section
            if 'ascites' in discharge_dx_section.lower():
                conditions.append({
                    'condition': 'ascites from portal hypertension',
                    'category': 'discharge_diagnosis',
                    'status': 'final'
                })

        # Remove duplicates based on condition name
        seen = set()
        unique_conditions = []
        for condition in conditions:
            key = condition['condition'].lower().strip()
            if key not in seen:
                seen.add(key)
                unique_conditions.append(condition)

        return unique_conditions

    def _extract_medications(self, text: str) -> List[Dict[str, Any]]:
        """Extract medications from admission and discharge sections"""
        medications = []

        # Admission medications
        admission_meds_section = self._extract_section(text, "Medications on Admission")
        if admission_meds_section:
            med_lines = [line.strip('- ').strip() for line in admission_meds_section.split('\n') if line.strip() and line[0].isdigit()]
            for med in med_lines:
                if med and len(med) > 2:
                    medications.append({
                        'medication': med,
                        'timing': 'admission',
                        'status': 'chronic'
                    })

        # Discharge medications
        discharge_meds_section = self._extract_section(text, "Discharge Medications")
        if discharge_meds_section:
            med_lines = [line.strip('- ').strip() for line in discharge_meds_section.split('\n') if line.strip() and line[0].isdigit()]
            for med in med_lines:
                if med and len(med) > 2:
                    medications.append({
                        'medication': med,
                        'timing': 'discharge',
                        'status': 'prescribed'
                    })

        # Hospital course medications
        hospital_course = self._extract_section(text, "Brief Hospital Course")
        if hospital_course:
            # Extract diuretic/ACE inhibitor changes, spironolactone, furosemide, etc.
            med_changes = re.findall(r'(?:Furosemide|Lasix|Spironolactone|Aldactone|metoprolol|atorvastatin|simvastatin|omeprazole|lisinopril|enalapril)\s+\d+(?:\s*mg)?', hospital_course, re.IGNORECASE)
            for med_change in med_changes:
                medications.append({
                    'medication': med_change.strip(),
                    'timing': 'hospital_course',
                    'status': 'adjusted'
                })

        return medications

    def _extract_social_history(self, text: str) -> Dict[str, Any]:
        """Extract social history information"""
        social_info = {}

        social_section = self._extract_section(text, "Social History")
        if social_section:
            # Extract smoking history
            if 'smoking' in social_section.lower() or 'smoker' in social_section.lower():
                if 'quit' in social_section.lower() or 'former' in social_section.lower():
                    social_info['smoking_status'] = 'former_smoker'
                elif 'current' in social_section.lower() or 'active' in social_section.lower():
                    social_info['smoking_status'] = 'current_smoker'
                else:
                    social_info['smoking_status'] = 'unknown'

            # Extract alcohol use
            if 'alcohol' in social_section.lower():
                if 'none' in social_section.lower() or 'quit' in social_section.lower():
                    social_info['alcohol_use'] = 'none'
                else:
                    social_info['alcohol_use'] = 'present'

            # Extract drug use
            if 'drug' in social_section.lower():
                if 'none' in social_section.lower() or 'quit' in social_section.lower():
                    social_info['drug_use'] = 'none'
                else:
                    social_info['drug_use'] = 'history'

        return social_info

    def _extract_lab_values(self, text: str) -> List[Dict[str, Any]]:
        """Extract laboratory values and results"""
        lab_values = []

        # Find lab sections
        lab_patterns = [
            r'(\w+)\s*[:-]\s*(\d+(?:\.\d+)?)\*?',
            r'(\w+)\((\w+)\)\s*[:-]\s*(\d+(?:\.\d+)?)\*?',
        ]

        # Look for lab sections like Pertinent Results
        pertinent_results_section = self._extract_section(text, "Pertinent Results")
        if pertinent_results_section:
            for pattern in lab_patterns:
                matches = re.findall(pattern, pertinent_results_section)
                for match in matches:
                    if len(match) == 2:
                        lab_name, value = match
                        lab_values.append({
                            'test': lab_name.strip(),
                            'value': value,
                            'abnormal': '*' in str(match)
                        })
                    elif len(match) == 3:
                        full_name, short_name, value = match
                        lab_values.append({
                            'test': f"{full_name} ({short_name})",
                            'value': value,
                            'abnormal': '*' in str(match)
                        })

        return lab_values

    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific section from the patient report"""
        # Find section start
        section_pattern = rf'{section_name}[:\s]*'
        section_match = re.search(section_pattern, text, re.IGNORECASE)
        if not section_match:
            return ""

        start_pos = section_match.end()

        # Find next section or end of relevant content
        next_section_patterns = [
            r'\n(?:Admission|Discharge|Physical|History|Brief|Medications|Pertinent|Social|Family)\s+(?:Medical\s+)?History[:\(\s]',
            r'\n[A-Z][A-Z\s]+:\s*'
        ]

        text_remaining = text[start_pos:]
        min_end_pos = len(text)

        for pattern in next_section_patterns:
            next_match = re.search(pattern, text_remaining)
            if next_match:
                min_end_pos = min(min_end_pos, next_match.start())

        return text_remaining[:min_end_pos].strip()

    def bulk_process_medical_reports(self, csv_path: str, start_row: int = 0, batch_size: int = 50) -> Dict[str, Any]:
        """
        Process multiple medical reports in bulk from CSV file

        Args:
            csv_path: Path to pipe-delimited CSV file containing medical reports
            start_row: Starting row number (0-based)
            batch_size: Number of reports to process per batch

        Returns:
            Dictionary containing processed knowledge graphs and metadata
        """
        logging.info(f"Starting bulk processing of medical reports from {csv_path}")

        # Initialize CSV processor
        csv_processor = MedicalReportCSVProcessor(delimiter='|')

        # Validate CSV format
        validation = csv_processor.validate_csv_format(csv_path)
        if not validation['is_valid']:
            raise ValueError(f"CSV validation failed: {validation.get('validation_errors', validation.get('error'))}")

        logging.info(f"CSV validated successfully. Schema: {validation}")

        # Process reports in batches
        all_knowledge_graphs = []
        total_processed = 0

        while True:
            try:
                # Load batch of reports
                batch_data = csv_processor.load_reports_bulk(
                    csv_path,
                    start_row=start_row + total_processed,
                    max_rows=batch_size
                )

                reports = batch_data['reports']
                if not reports:
                    break  # No more reports to process

                batch_kgs = []
                for report in reports:
                    try:
                        # Generate knowledge graph for this report
                        report_text = report['sections'].get('full_report_text', '')

                        # If no full text, try to reconstruct from sections
                        if not report_text:
                            sections_text = []
                            for section_name, content in report['sections'].items():
                                if section_name != 'full_report_text':
                                    sections_text.append(f"{section_name.replace('_', ' ').title()}:\n{content}\n\n")
                            report_text = '\n'.join(sections_text)

                        if report_text:
                            # Generate KG using patient-specific method (limited chunks for testing)
                            max_chunks = getattr(self, '_test_max_chunks', None)
                            kg = self.generate_patient_knowledge_graph(report_text, f"report_{report['row_index']}", max_chunks)
                            if kg:
                                batch_kgs.append(kg)

                    except Exception as e:
                        logging.error(f"Failed to process report at row {report['row_index']}: {e}")
                        continue

                all_knowledge_graphs.extend(batch_kgs)
                total_processed += len(reports)

                logging.info(f"Processed batch: {len(reports)} reports, generated {len(batch_kgs)} knowledge graphs")

                if len(reports) < batch_size:
                    break  # Processed all remaining reports

            except Exception as e:
                logging.error(f"Error processing batch starting at row {start_row + total_processed}: {e}")
                break

        # Combine all knowledge graphs (merge entities and relationships)
        merged_kg = self._merge_knowledge_graphs(all_knowledge_graphs)

        return {
            'knowledge_graph': merged_kg,
            'metadata': {
                'total_reports_processed': total_processed,
                'total_knowledge_graphs': len(all_knowledge_graphs),
                'csv_validation': validation,
                'bulk_processing_info': {
                    'start_row': start_row,
                    'batch_size': batch_size,
                    'total_batches': (total_processed + batch_size - 1) // batch_size
                }
            }
        }

    def _merge_knowledge_graphs(self, knowledge_graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple knowledge graphs into a single unified graph

        Args:
            knowledge_graphs: List of individual knowledge graphs

        Returns:
            Merged knowledge graph
        """
        merged = {
            "nodes": [],
            "relationships": [],
            "chunks": [],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "kg_type": "bulk_medical_reports",
                "extraction_method": "patient_specific_hybrid",
                "total_entities": 0,
                "total_relationships": 0,
                "source_knowledge_graphs": len(knowledge_graphs)
            }
        }

        # Collect all entities, relationships, and chunks
        all_nodes = []
        all_relationships = []
        all_chunks = []

        for kg in knowledge_graphs:
            all_nodes.extend(kg.get('nodes', []))
            all_relationships.extend(kg.get('relationships', []))
            all_chunks.extend(kg.get('chunks', []))

        # Harmonize entities and relationships to remove duplicates
        merged["nodes"] = self._harmonize_entities(all_nodes)
        entity_map = {entity['uuid']: entity for entity in merged["nodes"]}
        merged["relationships"] = self._harmonize_relationships(all_relationships, entity_map)
        merged["chunks"] = all_chunks

        # Update metadata
        merged["metadata"]["total_entities"] = len(merged["nodes"])
        merged["metadata"]["total_relationships"] = len(merged["relationships"])

        return merged

    def enhanced_section_parsing_with_field_names(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced section parsing using standardized field names from CSV

        Args:
            report_data: Dictionary containing report sections with field names

        Returns:
            Dictionary with enhanced parsed sections
        """
        enhanced_sections = {}

        # Define section field mappings with expected field names
        section_mappings = {
            'demographics': ['patient_id', 'full_name', 'date_of_birth', 'sex'],
            'admission_info': ['admission_date', 'service', 'attending', 'unit_no'],
            'discharge_info': ['discharge_date'],
            'chief_complaint': ['chief_complaint'],
            'history_present_illness': ['history_present_illness_hopi'],
            'past_medical_history': ['past_medical_history_pmh'],
            'medications': ['medications_admission', 'medications_discharge'],
            'hospital_course': ['brief_hospital_course'],
            'results': ['pertinent_results'],
            'social_history': ['social_history', 'family_history'],
            'discharge_plan': ['discharge_diagnosis', 'discharge_instructions', 'follow_up_instructions'],
            'full_text': ['full_report_text']
        }

        # Extract sections using field names
        for category, field_names in section_mappings.items():
            category_data = {}
            for field in field_names:
                if field in report_data.get('sections', {}):
                    category_data[field] = report_data['sections'][field]
                elif field in report_data.get('metadata', {}):
                    category_data[field] = report_data['metadata'][field]

            if category_data:
                enhanced_sections[category] = category_data

        # Add field name references for traceability
        field_names_used = {}
        for field in report_data.get('field_names', {}):
            field_names_used[field] = {
                'mapped_to': section_mappings.get(
                    next((k for k, v in section_mappings.items() if field in v), 'other'),
                    'other'
                ),
                'content_length': len(report_data.get('sections', {}).get(field, ''))
            }

        enhanced_sections['_field_mapping'] = field_names_used

        return enhanced_sections

    def _extract_entities_and_relationships_for_patients(self, text: str) -> Dict[str, Any]:
        """Extract patient-specific entities and relationships using hybrid approach"""
        entities = []
        relationships = []

        # Extract patient demographics as primary entity
        patient_info = self._extract_patient_demographics(text)
        if patient_info:
            patient_id = f"Patient_{hashlib.md5(str(patient_info).encode()).hexdigest()[:8]}"
            entities.append({
                "id": patient_id,
                "type": "Patient",
                "properties": {
                    "name": patient_info.get('name', 'Unknown'),
                    "type": "Patient",
                    "description": f"Patient with demographics: {', '.join([f'{k}: {v}' for k, v in patient_info.items()])}",
                    **patient_info
                }
            })

        # Extract medical conditions and link to patient
        conditions = self._extract_medical_conditions(text)
        for condition in conditions:
            condition_id = f"Condition_{hashlib.md5(condition['condition'].encode()).hexdigest()[:8]}"
            entities.append({
                "id": condition_id,
                "type": "MedicalCondition",
                "properties": {
                    "name": condition['condition'],
                    "type": "MedicalCondition",
                    "description": f"{condition['condition']} ({condition['status']})",
                    "category": condition['category'],
                    "status": condition['status']
                }
            })

            # Link condition to patient
            if patient_info and condition['category'] != 'presenting':
                relationships.append({
                    "source": patient_id,
                    "target": condition_id,
                    "type": "HAS_CONDITION",
                    "properties": {
                        "description": f"Patient has {condition['condition']}",
                        "category": condition['category'],
                        "status": condition['status']
                    }
                })

        # Extract medications and link to patient and conditions
        medications = self._extract_medications(text)
        for medication in medications:
            med_id = f"Medication_{hashlib.md5(medication['medication'].encode()).hexdigest()[:8]}"
            entities.append({
                "id": med_id,
                "type": "Medication",
                "properties": {
                    "name": medication['medication'],
                    "type": "Medication",
                    "description": f"Medication: {medication['medication']} ({medication['timing']})",
                    "timing": medication['timing'],
                    "status": medication['status']
                }
            })

            # Link medication to patient
            if patient_info:
                relationships.append({
                    "source": patient_id,
                    "target": med_id,
                    "type": "TAKES_MEDICATION",
                    "properties": {
                        "description": f"Patient takes {medication['medication']}",
                        "timing": medication['timing'],
                        "status": medication['status']
                    }
                })

        # Extract lab values and create relationships
        lab_values = self._extract_lab_values(text)
        for lab in lab_values:
            lab_id = f"Lab_{hashlib.md5(lab['test'].encode()).hexdigest()[:8]}"
            entities.append({
                "id": lab_id,
                "type": "LabTest",
                "properties": {
                    "name": lab['test'],
                    "type": "LabTest",
                    "description": f"Lab test: {lab['test']} = {lab['value']}{'*' if lab['abnormal'] else ''}",
                    "value": lab['value'],
                    "abnormal": lab['abnormal']
                }
            })

            # Link lab to patient
            if patient_info:
                relationships.append({
                    "source": patient_id,
                    "target": lab_id,
                    "type": "HAS_LAB_RESULT",
                    "properties": {
                        "description": f"Patient has lab result: {lab['test']} = {lab['value']}",
                        "value": lab['value'],
                        "abnormal": lab['abnormal']
                    }
                })

        # Extract social history and link to patient
        social_info = self._extract_social_history(text)
        if social_info and patient_info:
            for key, value in social_info.items():
                social_id = f"Social_{key}_{hashlib.md5(str(value).encode()).hexdigest()[:8]}"
                entities.append({
                    "id": social_id,
                    "type": "SocialFactor",
                    "properties": {
                        "name": key,
                        "type": "SocialFactor",
                        "description": f"Social history: {key} = {value}",
                        "factor": key,
                        "value": value
                    }
                })

                relationships.append({
                    "source": patient_id,
                    "target": social_id,
                    "type": "HAS_SOCIAL_HISTORY",
                    "properties": {
                        "description": f"Patient has social factor: {key} = {value}",
                        "factor": key,
                        "value": value
                    }
                })

        return {
            "entities": entities,
            "relationships": relationships
        }

    def generate_patient_knowledge_graph(self, text: str, file_name: str = None, max_chunks: int = None) -> Dict[str, Any]:
        """
        Generate detailed knowledge graph from patient report text with patient-specific extraction
        """
        logging.info("Generating patient-specific knowledge graph")

        # Split text into sections to maintain context but allow larger chunks
        chunks = self._chunk_text(text)

        # Limit chunks for testing if specified
        if max_chunks is not None and max_chunks > 0:
            chunks = chunks[:max_chunks]
            logging.info(f"Limited processing to first {max_chunks} chunks (out of {len(chunks)} total)")

        all_entities = []
        all_relationships = []

        for chunk in chunks:
            try:
                # Use patient-specific extraction
                result = self._extract_entities_and_relationships_for_patients(chunk['text'])

                # Also get general LLM extraction for any missed entities
                llm = self._get_simple_llm_for_patient_extraction()
                llm_result = self._extract_entities_and_relationships_with_llm(chunk['text'], llm)

                # Merge results
                all_entities.extend(result.get("entities", []))
                all_entities.extend(llm_result.get("entities", []))
                all_relationships.extend(result.get("relationships", []))
                all_relationships.extend(llm_result.get("relationships", []))

            except Exception as e:
                logging.error(f"Error processing chunk {chunk['chunk_id']}: {e}")
                continue

        # Harmonize entities and relationships to remove duplicates
        harmonized_entities = self._harmonize_entities(all_entities)
        # Create entity map for harmonize_relationships like parent class expects
        entity_map = {entity['uuid']: entity for entity in harmonized_entities}
        harmonized_relationships = self._harmonize_relationships(all_relationships, entity_map)

        # Create KG structure
        kg = {
            "nodes": harmonized_entities,
            "relationships": harmonized_relationships,
            "chunks": chunks,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "file_name": file_name,
                "kg_type": "patient_report",
                "extraction_method": "patient_specific_hybrid",
                "total_entities": len(harmonized_entities),
                "total_relationships": len(harmonized_relationships)
            }
        }

        return kg

    def _get_simple_llm_for_patient_extraction(self):
        """Get a simple LLM instance for basic extraction"""
        # This is a placeholder - in practice you'd pass the actual LLM
        return None
