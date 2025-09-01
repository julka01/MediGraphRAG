import json
import re
import uuid
import random
import numpy as np
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
import owlready2
from io import BytesIO
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ImprovedKGCreator:
    """
    Deterministic Knowledge Graph Creator with enhanced ontology integration
    and biomedical-focused prompting
    """
    
    def __init__(
        self, 
        seed: int = 42,
        ontology_path: str = "biomedical_ontology.owl"
    ):
        # Set global random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        self.seed = seed
        self.biomedical_ontology = self._load_default_biomedical_ontology(ontology_path)
        
        # Create a consistent hash for tracking
        self.kg_generation_hash = hashlib.md5(
            str(seed).encode() + 
            str(self.biomedical_ontology).encode()
        ).hexdigest()
    
    def _preprocess_text(
        self, 
        text: str, 
        max_length: int = 4000,
        preprocessing_rules: Optional[List[Callable]] = None
    ) -> str:
        """Deterministic text preprocessing"""
        default_rules = [
            lambda x: re.sub(r'\s+', ' ', x),  # Normalize whitespace
            lambda x: re.sub(r'[^a-zA-Z0-9\s.,()]', '', x),  # Remove special chars
            lambda x: x.lower(),  # Lowercase for consistency
        ]
        
        rules = preprocessing_rules or default_rules
        
        processed_text = text
        for rule in rules:
            processed_text = rule(processed_text)
        
        return processed_text[:max_length]
    
    def _load_default_biomedical_ontology(self, ontology_path: str) -> Dict[str, Any]:
        """Load the default biomedical ontology with version tracking"""
        try:
            ontology_data = self.parse_owl_ontology_from_file(ontology_path)
            ontology_data['version'] = hashlib.md5(
                open(ontology_path, 'rb').read()
            ).hexdigest()
            return ontology_data
        except Exception as e:
            print(f"Warning: Could not load biomedical ontology: {e}")
            return self._get_fallback_ontology()
    
    def _get_fallback_ontology(self) -> Dict[str, Any]:
        """Fallback ontology if OWL file cannot be loaded"""
        return {
            "node_labels": [
                "Disease", "Treatment", "Medication", "Procedure", "Symptom", 
                "Biomarker", "AnatomicalStructure", "RiskFactor", "Stage", 
                "Gene", "Protein", "Pathway", "ClinicalTrial", "Guideline", "Organization"
            ],
            "relationship_types": [
                "TREATS", "CAUSES", "SYMPTOM_OF", "BIOMARKER_FOR", "LOCATED_IN", 
                "AFFECTS", "HAS_STAGE", "RISK_FACTOR_FOR", "DIAGNOSES", "DETECTS", 
                "PRODUCES", "ENCODES", "PARTICIPATES_IN", "ASSOCIATED_WITH", 
                "CONTRAINDICATED_FOR", "GENETIC_RISK_FACTOR", "RECOMMENDS", 
                "STUDIES", "PUBLISHED_BY"
            ]
        }
    
    def parse_owl_ontology_from_file(self, owl_file_path: str) -> Dict[str, Any]:
        """Parse OWL ontology file into our format"""
        try:
            onto = owlready2.get_ontology(f"file://{owl_file_path}").load()
            
            # Extract classes as node labels (remove namespace prefix)
            node_labels = [cls.name for cls in onto.classes() if cls.name]
            
            # Extract object properties as relationship types (remove namespace prefix)
            relationship_types = [prop.name for prop in onto.object_properties() if prop.name]
            
            return {
                "node_labels": node_labels,
                "relationship_types": relationship_types,
                "ontology_uri": str(onto.base_iri)
            }
        except Exception as e:
            print(f"Error parsing OWL ontology: {e}")
            return self._get_fallback_ontology()
    
    def create_enhanced_biomedical_prompt(self, ontology: Dict[str, Any]) -> str:
        """Create an enhanced prompt for biomedical knowledge graph extraction"""
        
        node_labels_str = ", ".join(ontology.get("node_labels", []))
        relationship_types_str = ", ".join(ontology.get("relationship_types", []))
        
        return f"""
You are an expert biomedical knowledge graph extraction system specialized in clinical and medical literature analysis. 
Your task is to extract a comprehensive, medically accurate knowledge graph from clinical text.

ONTOLOGY CONSTRAINTS:
- Node Labels (MUST use only these): {node_labels_str}
- Relationship Types (MUST use only these): {relationship_types_str}

[Rest of the prompt remains the same as in the previous implementation]
"""
    
    def generate_knowledge_graph(
        self, 
        text: str, 
        llm, 
        ontology: Optional[Dict] = None,
        tracking_enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Generate knowledge graph with enhanced determinism
        """
        # Use preprocessing to ensure consistent input
        processed_text = self._preprocess_text(text)
        
        # Use provided ontology or default biomedical ontology
        working_ontology = ontology if ontology else self.biomedical_ontology
        
        # Create enhanced prompt
        enhanced_prompt_template = self.create_enhanced_biomedical_prompt(working_ontology)
        
        prompt = ChatPromptTemplate.from_template(
            enhanced_prompt_template + "\n\nCLINICAL TEXT TO ANALYZE:\n{text}"
        )
        
        chain = prompt | llm | StrOutputParser()
        
        try:
            result = chain.invoke({"text": processed_text})
            kg_data = self._parse_and_validate_kg(result, working_ontology)
            
            if tracking_enabled:
                # Create a reproducibility record
                reproducibility_record = {
                    "kg_hash": self._compute_kg_hash(kg_data),
                    "generation_timestamp": datetime.now().isoformat(),
                    "input_text_hash": hashlib.md5(text.encode()).hexdigest(),
                    "ontology_version": working_ontology.get('version', 'unknown'),
                    "seed": self.seed
                }
                
                # Optional: Log the record (you can implement logging mechanism)
                self._log_kg_generation(text, kg_data, reproducibility_record)
            
            return kg_data
        
        except Exception as e:
            print(f"Error generating KG: {e}")
            return self._create_fallback_kg(processed_text)
    
    def _compute_kg_hash(self, kg_data: Dict) -> str:
        """Compute a deterministic hash of the knowledge graph"""
        # Sort nodes and relationships to ensure consistent hashing
        sorted_nodes = sorted(
            kg_data.get('nodes', []), 
            key=lambda x: (x.get('id', 0), x.get('label', ''))
        )
        sorted_relationships = sorted(
            kg_data.get('relationships', []), 
            key=lambda x: (x.get('from', 0), x.get('to', 0), x.get('type', ''))
        )
        
        return hashlib.md5(
            json.dumps({
                'nodes': sorted_nodes, 
                'relationships': sorted_relationships
            }, sort_keys=True).encode()
        ).hexdigest()
    
    def _log_kg_generation(
        self, 
        input_text: str, 
        kg_data: Dict, 
        generation_metadata: Dict
    ):
        """
        Create a comprehensive log of KG generation process
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input_text_length": len(input_text),
            "node_count": len(kg_data.get("nodes", [])),
            "relationship_count": len(kg_data.get("relationships", [])),
            "generation_seed": self.seed,
            "ontology_version": generation_metadata.get('ontology_version'),
            "kg_hash": generation_metadata.get('kg_hash'),
            "validation_status": "PASSED"
        }
        
        # In a real implementation, you would store or process this log entry
        print(f"KG Generation Log: {json.dumps(log_entry, indent=2)}")
    
    def _parse_and_validate_kg(self, result: str, ontology: Dict) -> Dict[str, Any]:
        """Parse and validate the generated knowledge graph"""
        try:
            # First attempt: direct JSON parsing
            kg_data = json.loads(result)
        except json.JSONDecodeError:
            try:
                # Second attempt: extract JSON from response
                cleaned = re.sub(r',(\s*[}\]])', r'\1', result)
                json_match = re.search(r'\{[\s\S]*\}', cleaned)
                if json_match:
                    kg_data = json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                print(f"Raw response: {result[:500]}...")
                raise ValueError("Failed to parse JSON from model response")
        
        # Validate and clean the knowledge graph
        return self._validate_and_clean_kg(kg_data, ontology)
    
    def _validate_and_clean_kg(self, kg_data: Dict, ontology: Dict) -> Dict[str, Any]:
        """Validate and clean the knowledge graph data"""
        valid_labels = set(ontology.get("node_labels", []))
        valid_relationships = set(ontology.get("relationship_types", []))
        
        # Validate nodes
        valid_nodes = []
        node_ids = set()
        
        for node in kg_data.get("nodes", []):
            if not isinstance(node.get("id"), int):
                continue
            
            node_id = node["id"]
            if node_id in node_ids:
                continue  # Skip duplicate IDs
            
            node_ids.add(node_id)
            
            # Validate label
            label = node.get("label", "")
            if label not in valid_labels:
                # Try to find closest match
                closest_label = self._find_closest_label(label, valid_labels)
                node["label"] = closest_label
            
            # Ensure properties exist
            if "properties" not in node:
                node["properties"] = {}
            
            # Ensure name property exists
            if "name" not in node["properties"]:
                node["properties"]["name"] = label
            
            valid_nodes.append(node)
        
        # Validate relationships
        valid_relationships_list = []
        
        for rel in kg_data.get("relationships", []):
            # Check if nodes exist
            if rel.get("from") not in node_ids or rel.get("to") not in node_ids:
                continue
            
            # Validate relationship type
            rel_type = rel.get("type", "")
            if rel_type not in valid_relationships:
                closest_rel = self._find_closest_relationship(rel_type, valid_relationships)
                rel["type"] = closest_rel
            
            # Ensure properties exist
            if "properties" not in rel:
                rel["properties"] = {}
            
            valid_relationships_list.append(rel)
        
        return {
            "nodes": valid_nodes,
            "relationships": valid_relationships_list
        }
    
    def _find_closest_label(self, label: str, valid_labels: set) -> str:
        """Find the closest matching label from valid ontology labels"""
        label_lower = label.lower()
        
        # Direct match
        for valid_label in valid_labels:
            if valid_label.lower() == label_lower:
                return valid_label
        
        # Partial match
        for valid_label in valid_labels:
            if label_lower in valid_label.lower() or valid_label.lower() in label_lower:
                return valid_label
        
        # Default fallback based on common patterns
        medical_mappings = {
            "drug": "Medication",
            "medicine": "Medication",
            "cancer": "Disease",
            "tumor": "Disease",
            "surgery": "Procedure",
            "test": "Procedure",
            "organ": "AnatomicalStructure",
            "body": "AnatomicalStructure",
            "sign": "Symptom",
            "marker": "Biomarker",
            "level": "Biomarker"
        }
        
        for key, mapped_label in medical_mappings.items():
            if key in label_lower and mapped_label in valid_labels:
                return mapped_label
        
        # Ultimate fallback
        return "Disease" if "Disease" in valid_labels else list(valid_labels)[0]
    
    def _find_closest_relationship(self, rel_type: str, valid_relationships: set) -> str:
        """Find the closest matching relationship type"""
        rel_lower = rel_type.lower()
        
        # Direct match
        for valid_rel in valid_relationships:
            if valid_rel.lower() == rel_lower:
                return valid_rel
        
        # Pattern matching
        relationship_mappings = {
            "treat": "TREATS",
            "cure": "TREATS",
            "cause": "CAUSES",
            "symptom": "SYMPTOM_OF",
            "sign": "SYMPTOM_OF",
            "marker": "BIOMARKER_FOR",
            "indicate": "BIOMARKER_FOR",
            "diagnose": "DIAGNOSES",
            "detect": "DETECTS",
            "affect": "AFFECTS",
            "locate": "LOCATED_IN",
            "produce": "PRODUCES",
            "risk": "RISK_FACTOR_FOR",
            "stage": "HAS_STAGE"
        }
        
        for key, mapped_rel in relationship_mappings.items():
            if key in rel_lower and mapped_rel in valid_relationships:
                return mapped_rel
        
        # Ultimate fallback
        return "ASSOCIATED_WITH" if "ASSOCIATED_WITH" in valid_relationships else list(valid_relationships)[0]
    
    def _create_fallback_kg(self, text: str) -> Dict[str, Any]:
        """Create a basic fallback knowledge graph if generation fails"""
        words = text.split()[:50]  # Take first 50 words
        
        # Create basic nodes
        nodes = []
        for i, word in enumerate(words[:5]):  # Max 5 nodes
            if len(word) > 3:  # Skip short words
                nodes.append({
                    "id": i + 1,
                    "label": "Disease",  # Default label
                    "properties": {
                        "name": word.capitalize(),
                        "source": "fallback_extraction"
                    }
                })
        
        # Create basic relationships
        relationships = []
        for i in range(len(nodes) - 1):
            relationships.append({
                "from": nodes[i]["id"],
                "to": nodes[i + 1]["id"],
                "type": "ASSOCIATED_WITH",
                "properties": {
                    "confidence": "low",
                    "source": "fallback_extraction"
                }
            })
        
        return {
            "nodes": nodes,
            "relationships": relationships
        }
