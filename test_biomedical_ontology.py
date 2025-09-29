#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import os
import tempfile

def test_biomedical_ontology_parsing():
    """Test parsing the biomedical_ontology.owl file directly"""
    ontology_path = "biomedical_ontology.owl"

    if not os.path.exists(ontology_path):
        print(f"❌ Biomedical ontology file not found: {ontology_path}")
        return

    print(f"📚 Testing biomedical ontology parsing: {ontology_path}")

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

        print(f"✅ Successfully parsed ontology: {len(ontology_classes)} classes, {len(ontology_relationships)} relationships")

        # Show some classes
        print("\n📖 Classes (first 5):")
        for i, cls in enumerate(ontology_classes[:5]):
            print(f"  {i+1}. {cls['id']} ({cls['uri']})")

        # Show some relationships
        print("\n📋 Relationships (all):")
        for i, rel in enumerate(ontology_relationships):
            print(f"  {i+1}. {rel['id']} ({rel['uri']})")

        return ontology_classes, ontology_relationships

    except Exception as e:
        print(f"❌ Error parsing ontology: {e}")
        return [], []

def test_temp_file_simulation(ontology_path="biomedical_ontology.owl"):
    """Test simulating FastAPI temporary file creation"""
    print("\n🔄 Testing temporary file simulation...")
    if not os.path.exists(ontology_path):
        print(f"❌ Source file not found: {ontology_path}")
        return

    try:
        # Read the ontology file
        with open(ontology_path, 'rb') as f:
            ontology_data = f.read()
        print(f"📖 Read {len(ontology_data)} bytes from {ontology_path}")

        # Save to temp location (like FastAPI does)
        tmp_dir = tempfile.gettempdir()
        test_ontology_path = os.path.join(tmp_dir, "biomedical_ontology_temp.owl")
        with open(test_ontology_path, "wb") as tmpf:
            tmpf.write(ontology_data)
        print(f"💾 Saved to temp file: {test_ontology_path}")

        # Verify we can read it back
        if os.path.exists(test_ontology_path):
            size = os.path.getsize(test_ontology_path)
            print(f"✅ Temp file exists and is {size} bytes")
        else:
            print("❌ Temp file creation failed")

        # Parse the temp file
        classes, relationships = test_biomedical_ontology_parsing_by_path(test_ontology_path)

        # Clean up
        os.remove(test_ontology_path)
        print("🧹 Cleaned up temp file")

        return classes, relationships

    except Exception as e:
        print(f"❌ Error in temp file simulation: {e}")
        return [], []

def test_biomedical_ontology_parsing_by_path(ontology_path):
    """Parse ontology by path"""
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

        print(f"✅ Temp file parsed successfully: {len(ontology_classes)} classes, {len(ontology_relationships)} relationships")

        return ontology_classes, ontology_relationships

    except Exception as e:
        print(f"❌ Error parsing temp file: {e}")
        return [], []

if __name__ == "__main__":
    test_biomedical_ontology_parsing()
    test_temp_file_simulation()
