#!/usr/bin/env python3

import os
import sys
import tempfile
import shutil
from ontology_guided_kg_creator import OntologyGuidedKGCreator

# Test ontology loading
print("Testing ontology loading...")

# Test with an existing ontology file
ontology_path = "ProstateCancerOntology.owl"

if os.path.exists(ontology_path):
    print(f"Ontology file exists: {ontology_path}")

    # Test direct constructor
    try:
        creator = OntologyGuidedKGCreator(
            ontology_path=ontology_path,
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            neo4j_database="neo4j"
        )
        print(f"Ontology loaded successfully: {len(creator.ontology_classes)} classes, {len(creator.ontology_relationships)} relationships")
    except Exception as e:
        print(f"Error creating OntologyGuidedKGCreator: {e}")
else:
    print(f"Ontology file not found: {ontology_path}")
    creator = OntologyGuidedKGCreator(
        ontology_path=None,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        neo4j_database="neo4j"
    )
    print(f"Created creator without ontology: {len(creator.ontology_classes)} classes, {len(creator.ontology_relationships)} relationships")

# Test temporary file creation (mimicking FastAPI upload)
print("\nTesting temporary file creation...")

if True:
    try:
        # Read the ontology file
        with open(ontology_path, 'rb') as f:
            ontology_data = f.read()
        print(f"Read {len(ontology_data)} bytes from ontology file")

        # Save to temp location (like FastAPI does)
        tmp_dir = tempfile.gettempdir()
        test_ontology_path = os.path.join(tmp_dir, "test_ontology_debug.owl")
        with open(test_ontology_path, "wb") as tmpf:
            tmpf.write(ontology_data)
        print(f"Saved to temp file: {test_ontology_path}")

        # Verify we can read it back
        if os.path.exists(test_ontology_path):
            size = os.path.getsize(test_ontology_path)
            print(f"Temp file exists and is {size} bytes")
        else:
            print("Temp file creation failed")

        # Try to create creator with temp file
        test_creator = OntologyGuidedKGCreator(
            ontology_path=test_ontology_path,
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            neo4j_database="neo4j"
        )
        print(f"Creator with temp file loaded: {len(test_creator.ontology_classes)} classes, {len(test_creator.ontology_relationships)} relationships")

        # Clean up
        os.remove(test_ontology_path)

    except Exception as e:
        print(f"Error testing temp file: {e}")

print("Ontology debug test complete.")
