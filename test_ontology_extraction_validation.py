#!/usr/bin/env python3

import os
import sys
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append(os.getcwd())

# Import required modules
from ontology_guided_kg_creator import OntologyGuidedKGCreator
from model_providers import get_provider

def test_ontology_keyword_extraction():
    """Test comprehensive ontology keyword extraction and filtering"""

    # Test with different ontology files
    ontology_files = {
        'prostate_cancer': 'ProstateCancerOntology.owl',
        'biomedical': 'biomedical_ontology.owl',
        'test': 'test_ontology.owl'
    }

    # Test text with known medical and non-medical entities
    test_text = """
    Prostate cancer is a disease affecting the prostate gland.
    Gleason score measures the aggressiveness of prostate cancer.
    PSA levels help diagnose the condition. Angela Merkel was a leader.
    The weather is nice in California. BMW is a car manufacturer.
    Future planning involves genomic sequencing and radiation therapy.
    """

    print("🧪 Testing Ontology Keyword Extraction and Filtering")
    print("=" * 60)

    results = {}

    for ontology_name, ontology_file in ontology_files.items():
        if not os.path.exists(ontology_file):
            print(f"❌ Ontology file {ontology_file} not found, skipping...")
            continue

        print(f"\n🔍 Testing with {ontology_file}")
        print("-" * 40)

        try:
            # Initialize ontology-guided KG creator (Neo4j disabled for testing)
            kg_creator = OntologyGuidedKGCreator(
                ontology_path=ontology_file,
                chunk_size=200,
                chunk_overlap=50,
                neo4j_uri=None,  # Disable Neo4j for testing
                embedding_model="sentence_transformers"
            )

            # Check basic ontology loading
            print(f"📚 Ontology loaded: {len(kg_creator.ontology_classes)} classes, "
                  f"{len(kg_creator.ontology_relationships)} relationships")

            # Debug: Show first 10 ontology classes
            print("📋 First 10 Ontology Classes:")
            for i, cls in enumerate(kg_creator.ontology_classes[:10]):
                print(f"   {i+1}. {cls['label']} ({cls['id']})")

            # Test 1: Basic entity validation
            print("\n🔍 Manual Entity Validation Test:")
            test_entities = [
                {'id': 'prostate_cancer', 'type': 'Disease'},
                {'id': 'gleason_score', 'type': 'diagnosis'},
                {'id': 'angela_merkel', 'type': 'Person'},
                {'id': 'bmw', 'type': 'Company'},
                {'id': 'california', 'type': 'Location'},
                {'id': 'radiation_therapy', 'type': 'Treatment'}
            ]

            valid_entities = []
            invalid_entities = []

            for entity in test_entities:
                is_valid = kg_creator._is_medical_entity(entity)
                entity_status = f"{'✓ ALLOWED' if is_valid else '✗ FILTERED'}"
                print(f"   {entity['id']} ({entity['type']}) -> {entity_status}")

                if is_valid:
                    valid_entities.append(entity)
                else:
                    invalid_entities.append(entity)

            print(f"📊 Validation: {len(valid_entities)} allowed, {len(invalid_entities)} filtered")

            # Test 2: LLM Keyword Extraction (if API key available)
            if os.getenv("OPENROUTER_API_KEY"):
                print("\n🤖 Testing LLM Keyword Extraction...")

                try:
                    # Get LLM provider
                    llm = get_provider("openrouter", "deepseek/deepseek-chat")

                    # Extract keywords using LLM
                    keywords = kg_creator._extract_ontology_keywords_with_llm(
                        ontology_file,
                        llm,
                        "deepseek/deepseek-chat"
                    )

                    print(f"📝 LLM extracted {len(keywords)} keywords from {ontology_file}")

                    # Debug: Show first 20 keywords
                    print("🔤 First 20 extracted keywords:")
                    for i, keyword in enumerate(keywords[:20]):
                        print(f"   {i+1:2d}. {keyword}")

                    # Save keywords for reference
                    results[ontology_name] = {
                        'ontology_classes': len(kg_creator.ontology_classes),
                        'ontology_relationships': len(kg_creator.ontology_relationships),
                        'extracted_keywords': len(keywords),
                        'sample_keywords': keywords[:10],
                        'validation_passed': len(valid_entities) > len(invalid_entities)
                    }

                except Exception as e:
                    print(f"⚠️ LLM keyword extraction failed: {e}")
                    results[ontology_name] = 'llm_extraction_failed'
            else:
                print("🚫 OpenRouter API key not found - skipping LLM tests")
                results[ontology_name] = 'api_key_missing'

            # Clean up
            del kg_creator

        except Exception as e:
            print(f"❌ Error testing {ontology_file}: {e}")
            import traceback
            traceback.print_exc()
            results[ontology_name] = f'error: {str(e)}'

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    for ontology_name, result in results.items():
        if isinstance(result, dict):
            status = "✅ PASSED" if result.get('validation_passed', False) else "❌ FAILED"
            print(f"🔍 {ontology_name}: {status}")

            if result.get('llm_extraction_failed') != 'llm_extraction_failed':
                print(f"   Ontology: {result['ontology_classes']} classes, {result['ontology_relationships']} relationships")
                print(f"   Keywords: {result['extracted_keywords']} extracted")
                print(f"   Samples: {', '.join(result['sample_keywords'])}")
            else:
                print("   LLM keyword extraction failed")
        else:
            print(f"🔍 {ontology_name}: {result}")

    return results

def test_prostate_cancer_ontology_specific():
    """Test prostate cancer ontology with specific medical content"""

    print("\n🔬 Testing Prostate Cancer Ontology with EAU Guidelines Text")
    print("-" * 60)

    try:
        # Use a shorter clinical test from EAU guidelines
        clinical_text = """
        Prostate cancer is characterized by Gleason grade groups.
        PSA levels above 4.0 ng/mL indicate further evaluation needed.
        Magnetic resonance imaging helps in prostate cancer staging.
        Active surveillance is recommended for low-risk prostate cancer.
        ADT (androgen deprivation therapy) treats advanced prostate cancer.
        Brachytherapy delivers radiation directly to the prostate gland.
        DUTASTERIDE reduces prostate volume in BPH.
        """

        # Initialize with prostate cancer ontology
        kg_creator = OntologyGuidedKGCreator(
            ontology_path="ProstateCancerOntology.owl",
            chunk_size=300,
            chunk_overlap=50,
            neo4j_uri=None,
            embedding_model="sentence_transformers"
        )

        print(f"📚 Prostate-specific ontology loaded: {len(kg_creator.ontology_classes)} classes")

        # Check if our desired clinical terms are in the ontology
        clinical_terms = ['prostate_cancer', 'gleason_score', 'psa', 'magnetic_resonance_imaging',
                         'active_surveillance', 'adt', 'brachytherapy', 'dutasteride']

        found_terms = []
        missing_terms = []

        for term in clinical_terms:
            term_found = False
            for cls in kg_creator.ontology_classes:
                if term.lower().replace('_', '') in cls['id'].lower().replace('_', '') or \
                   term.lower().replace('_', '') in cls['label'].lower().replace('_', ''):
                    found_terms.append(term)
                    term_found = True
                    break

            if not term_found:
                missing_terms.append(term)

        print(f"🔍 Ontology Term Coverage:")
        print(f"   Found: {len(found_terms)} terms")
        print(f"   Missing: {len(missing_terms)} terms")

        if found_terms:
            print("   ✅ Found terms: " + ", ".join(found_terms[:5]))

        if missing_terms:
            print("   ⚠️  Missing terms: " + ", ".join(missing_terms))

        # Save test results for manual review
        with open('ontology_test_results.txt', 'w') as f:
            f.write("Ontology Extraction Test Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Ontology file: ProstateCancerOntology.owl\n")
            f.write(f"Classes loaded: {len(kg_creator.ontology_classes)}\n")
            f.write(f"Terms found: {' ,'.join(found_terms)}\n")
            f.write(f"Terms missing: {' ,'.join(missing_terms)}\n")

        print("📝 Results saved to ontology_test_results.txt")

    except Exception as e:
        print(f"❌ Prostate cancer ontology test failed: {e}")

if __name__ == "__main__":
    # Test general ontology functionality
    results = test_ontology_keyword_extraction()

    # Test prostate-cancer specific ontology
    test_prostate_cancer_ontology_specific()

    print("\n✨ Ontology extraction and filtering tests completed!")
