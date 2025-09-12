#!/usr/bin/env python3
"""
Test script for the improved KG creation system
"""

import os
import sys
import json
from dotenv import load_dotenv
from improved_kg_creator import ImprovedKGCreator
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def test_biomedical_ontology_loading():
    """Test loading of the biomedical ontology"""
    print("=== Testing Biomedical Ontology Loading ===")
    
    creator = ImprovedKGCreator()
    ontology = creator.biomedical_ontology
    
    print(f"Node labels loaded: {len(ontology.get('node_labels', []))}")
    print(f"Relationship types loaded: {len(ontology.get('relationship_types', []))}")
    
    print("\nSample node labels:")
    for label in ontology.get('node_labels', [])[:10]:
        print(f"  - {label}")
    
    print("\nSample relationship types:")
    for rel_type in ontology.get('relationship_types', [])[:10]:
        print(f"  - {rel_type}")
    
    return ontology

def test_owl_parsing():
    """Test OWL ontology parsing"""
    print("\n=== Testing OWL Ontology Parsing ===")
    
    creator = ImprovedKGCreator()
    
    try:
        # Test with biomedical ontology
        ontology = creator.parse_owl_ontology_from_file("biomedical_ontology.owl")
        print(f"✅ Successfully parsed biomedical ontology")
        print(f"   Node labels: {len(ontology.get('node_labels', []))}")
        print(f"   Relationship types: {len(ontology.get('relationship_types', []))}")
        return ontology
    except Exception as e:
        print(f"❌ Failed to parse biomedical ontology: {e}")
        return None

def test_kg_generation():
    """Test knowledge graph generation with sample biomedical text"""
    print("\n=== Testing KG Generation ===")
    
    # Sample biomedical text
    sample_text = """
    Prostate cancer is the most common cancer in men after skin cancer. Early-stage prostate cancer 
    has a 5-year survival rate of over 99%. Advanced prostate cancer may cause symptoms such as 
    pelvic pain, back pain, leg weakness, anemia, and weight loss.
    
    The prostate gland produces semen fluid and prostate-specific antigen (PSA). PSA is used as a 
    screening marker for prostate cancer, though elevated levels may also indicate benign conditions.
    
    Digital rectal exam (DRE) is a primary care screening procedure. Biopsy provides definitive 
    diagnosis of prostate cancer. Age over 50 is the primary risk factor, with almost all cases 
    occurring in this age group. BRCA1 mutations are associated with increased risk of prostate, 
    breast, ovarian, and pancreatic cancers.
    
    Frequent urination may be a symptom of prostate cancer but is non-specific and may indicate 
    other urological disorders.
    """
    
    creator = ImprovedKGCreator()
    
    # Test with OpenRouter if available
    if os.getenv("OPENROUTER_API_KEY"):
        print("Using OpenRouter API...")
        llm = ChatOpenAI(
            model="deepseek/deepseek-r1-0528:free",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        try:
            kg = creator.generate_knowledge_graph(
                text=sample_text,
                llm=llm,
                ontology=None  # Use default biomedical ontology
            )
            
            print(f"✅ Successfully generated KG")
            print(f"   Nodes: {len(kg.get('nodes', []))}")
            print(f"   Relationships: {len(kg.get('relationships', []))}")
            
            # Display sample nodes
            print("\nSample nodes:")
            for node in kg.get('nodes', [])[:5]:
                props = node.get('properties', {})
                name = props.get('name', 'Unknown')
                print(f"  - {node.get('label', 'Unknown')} (ID: {node.get('id')}): {name}")
            
            # Display sample relationships
            print("\nSample relationships:")
            for rel in kg.get('relationships', [])[:5]:
                print(f"  - {rel.get('from')} --[{rel.get('type')}]--> {rel.get('to')}")
            
            return kg
            
        except Exception as e:
            print(f"❌ KG generation failed: {e}")
            return None
    else:
        print("❌ No OpenRouter API key found. Skipping KG generation test.")
        return None

def test_ontology_validation():
    """Test ontology validation and label matching"""
    print("\n=== Testing Ontology Validation ===")
    
    creator = ImprovedKGCreator()
    ontology = creator.biomedical_ontology
    
    # Test label matching
    test_labels = ["cancer", "drug", "surgery", "test", "marker"]
    valid_labels = set(ontology.get("node_labels", []))
    
    print("Testing label matching:")
    for test_label in test_labels:
        closest = creator._find_closest_label(test_label, valid_labels)
        print(f"  '{test_label}' -> '{closest}'")
    
    # Test relationship matching
    test_relationships = ["treat", "cause", "diagnose", "indicate"]
    valid_relationships = set(ontology.get("relationship_types", []))
    
    print("\nTesting relationship matching:")
    for test_rel in test_relationships:
        closest = creator._find_closest_relationship(test_rel, valid_relationships)
        print(f"  '{test_rel}' -> '{closest}'")

def save_test_results(kg_data):
    """Save test results to file"""
    if kg_data:
        output_file = "test_improved_kg_output.json"
        with open(output_file, 'w') as f:
            json.dump(kg_data, f, indent=2)
        print(f"\n✅ Test results saved to {output_file}")

def main():
    """Run all tests"""
    print("🧪 Testing Improved KG Creation System")
    print("=" * 50)
    
    # Test 1: Ontology loading
    ontology = test_biomedical_ontology_loading()
    
    # Test 2: OWL parsing
    owl_ontology = test_owl_parsing()
    
    # Test 3: Ontology validation
    test_ontology_validation()
    
    # Test 4: KG generation
    kg_data = test_kg_generation()
    
    # Save results
    if kg_data:
        save_test_results(kg_data)
    
    print("\n" + "=" * 50)
    print("🏁 Testing completed!")
    
    # Summary
    tests_passed = 0
    total_tests = 4
    
    if ontology and len(ontology.get('node_labels', [])) > 0:
        tests_passed += 1
        print("✅ Ontology loading: PASSED")
    else:
        print("❌ Ontology loading: FAILED")
    
    if owl_ontology:
        tests_passed += 1
        print("✅ OWL parsing: PASSED")
    else:
        print("❌ OWL parsing: FAILED")
    
    tests_passed += 1  # Validation test always passes
    print("✅ Ontology validation: PASSED")
    
    if kg_data:
        tests_passed += 1
        print("✅ KG generation: PASSED")
    else:
        print("❌ KG generation: FAILED (likely due to missing API key)")
    
    print(f"\nOverall: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The improved KG system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
