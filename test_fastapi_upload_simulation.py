#!/usr/bin/env python3

import tempfile
import uuid
import os

# Test to simulate exactly what the FastAPI endpoint does
print("🔬 Simulating FastAPI ontology upload process...")

ontology_path = "biomedical_ontology.owl"

if not os.path.exists(ontology_path):
    print(f"❌ Ontology file not found: {ontology_path}")
    exit(1)

# Step 1: Read ontology file data (exactly like FastAPI does)
with open(ontology_path, "rb") as f:
    ontology_data = f.read()

print(f"📖 Read {len(ontology_data)} bytes from uploaded ontology file")

# Step 2: Save to temp location (exactly like FastAPI does)
tmp_dir = tempfile.gettempdir()
ontology_filename = f"ontology_{uuid.uuid4()}{os.path.splitext('biomedical_ontology.owl')[1]}"
ontology_path_temp = os.path.join(tmp_dir, ontology_filename)
with open(ontology_path_temp, "wb") as tmpf:
    tmpf.write(ontology_data)

print(f"💾 Saved ontology to temp path: {ontology_path_temp}")
print(f"🧪 File size: {os.path.getsize(ontology_path_temp)} bytes")
print(f"🧪 File exists: {os.path.exists(ontology_path_temp)}")

# Step 3: Create OntologyGuidedKGCreator like the FastAPI endpoint does
print("\n🏗️ Creating OntologyGuidedKGCreator...")
print(f"📋 Using ontology_path: {ontology_path_temp}")

# Mock the constructor call without actually creating dependencies
import xml.etree.ElementTree as ET

def simulate_load_ontology(ontology_path):
    """Simulate the _load_ontology method"""
    try:
        print(f"🔄 Loading ontology from: {ontology_path}")
        tree = ET.parse(ontology_path)
        root = tree.getroot()

        # Define namespaces
        ns = {
            'owl': 'http://www.w3.org/2002/07/owl#',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'
        }

        # Extract classes
        ontology_classes = []
        for class_elem in root.findall('.//owl:Class', ns):
            class_id = class_elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about', '')
            if class_id:
                # Extract local name from URI
                class_name = class_id.split('#')[-1] if '#' in class_id else class_id.split('/')[-1]
                if class_name:
                    ontology_classes.append({
                        'id': class_name,
                        'uri': class_id,
                        'label': class_name.replace('_', ' ').title()
                    })

        # Extract object properties (relationships)
        ontology_relationships = []
        for prop_elem in root.findall('.//owl:ObjectProperty', ns):
            prop_id = prop_elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about', '')
            if prop_id:
                prop_name = prop_id.split('#')[-1] if '#' in prop_id else prop_id.split('/')[-1]
                if prop_name:
                    ontology_relationships.append({
                        'id': prop_name,
                        'uri': prop_id,
                        'label': prop_name.replace('_', ' ').title()
                    })

        print(f"✅ Ontology loaded: {len(ontology_classes)} classes, {len(ontology_relationships)} relationships")
        return ontology_classes, ontology_relationships

    except Exception as e:
        print(f"❌ Ontology loading failed: {e}")
        return [], []

# Call the simulation
classes, relationships = simulate_load_ontology(ontology_path_temp)

print(f"\n📊 Final result:")
print(f"   - Ontology classes loaded: {len(classes)}")
print(f"   - Ontology relationships loaded: {len(relationships)}")

if classes and relationships:
    print("✅ Ontology should be available for entity extraction")
else:
    print("❌ Ontology loading failed - this would trigger 'No ontology provided'")

# Check if ontology is available (like in the LLM extraction method)
has_ontology = bool(classes) or bool(relationships)
if has_ontology:
    print("✅ Ontology-guided extraction should be used")
else:
    print("❌ Basic LLM entity extraction would be used")

# Clean up
os.remove(ontology_path_temp)
print("🧹 Cleaned up temp file")
