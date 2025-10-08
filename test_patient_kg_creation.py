#!/usr/bin/env python3
"""
Test script for patient-specific knowledge graph generation from unstructured reports.
Tests the PatientReportKGCreator with sample patient data.
"""
import os
import sys
import logging
import pandas as pd
from typing import Dict, Any
from enhanced_kg_creator import UnifiedOntologyGuidedKGCreator

# Import model providers
from model_providers import get_provider as get_llm_provider

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_sample_patient_report(csv_path: str, row_index: int = 0) -> str:
    """Load a sample patient report from pipe-delimited CSV"""
    try:
        from csv_processor import MedicalReportCSVProcessor

        # Initialize processor with pipe delimiter validation
        processor = MedicalReportCSVProcessor(delimiter='|')

        # Load reports in bulk batch
        batch_data = processor.load_reports_bulk(csv_path, start_row=row_index, max_rows=1)
        reports = batch_data['reports']

        if not reports:
            logging.warning(f"No reports found at row {row_index}")
            return ""

        # Get the full report text from structured data
        report = reports[0]
        report_text = report['sections'].get('full_report_text', '')

        # If full text not available, construct from sections
        if not report_text.strip():
            sections_text = []
            for section_name, content in report['sections'].items():
                if section_name != 'full_report_text' and content.strip():
                    sections_text.append(f"{section_name.replace('_', ' ').title()}:\n{content}")
            report_text = '\n\n'.join(sections_text)

        logging.info(f"Loaded report from row {row_index}, text length: {len(report_text)}")
        return report_text

    except Exception as e:
        logging.error(f"Failed to load patient report from {csv_path}: {e}")
        # Fallback to simple pandas read with pipe delimiter
        try:
            df = pd.read_csv(csv_path, sep='|')
            if row_index >= len(df):
                row_index = 0
            report_text = str(df.iloc[row_index, -1])  # Last column contains the full report text
            logging.info(f"Fallback successful, loaded report text length: {len(report_text)}")
            return report_text
        except Exception as fallback_e:
            logging.error(f"Fallback also failed: {fallback_e}")
            return ""

def test_patient_kg_generation():
    """Test patient KG generation with sample data"""
    print("🧪 Testing Patient Knowledge Graph Creation")
    print("=" * 50)

    # Load sample patient report
    csv_path = "mimic_iv_summarization_test_dataset_shortened.csv"
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return

    sample_text = load_sample_patient_report(csv_path, row_index=0)
    if not sample_text:
        print("❌ Failed to load sample patient report")
        return

    print(f"📄 Loaded sample report ({len(sample_text)} characters)")
    print("Sample report preview:")
    print(sample_text[:500] + "..." if len(sample_text) > 500 else sample_text)
    print()

    # Initialize patient KG creator
    print("🏗️ Initializing PatientReportKGCreator...")

    # Check if biomedical ontology exists
    ontology_path = "biomedical_ontology.owl"
    if not os.path.exists(ontology_path):
        print(f"⚠️ Biomedical ontology not found at {ontology_path}, proceeding without it")
        ontology_path = None

    kg_creator = UnifiedOntologyGuidedKGCreator(
        chunk_size=2000,
        chunk_overlap=300,
        embedding_model="sentence_transformers",
        ontology_path=ontology_path
    )

    # Generate patient knowledge graph
    print("🔄 Generating patient-specific knowledge graph...")

    try:
        kg = kg_creator.generate_patient_knowledge_graph(sample_text, "sample_patient_report.txt")

        print("✅ Knowledge graph generated successfully!")
        print(f"   📊 Entities: {kg['metadata']['total_entities']}")
        print(f"   🔗 Relationships: {kg['metadata']['total_relationships']}")
        print()

        # Display sample entities
        print("🏥 Sample Entities:")
        entities = kg.get('nodes', [])[:10]  # Show first 10 entities
        for i, entity in enumerate(entities, 1):
            props = entity.get('properties', {})
            print(f"   {i}. {props.get('name', 'Unknown')} ({entity.get('type', 'Unknown Type')})")
            if 'description' in props and len(props['description']) < 100:
                print(f"      └─ {props['description']}")
        print()

        # Display sample relationships
        print("🔗 Sample Relationships:")
        relationships = kg.get('relationships', [])[:5]  # Show first 5 relationships
        for i, rel in enumerate(relationships, 1):
            print(f"   {i}. {rel.get('source', 'Unknown')} →[{rel.get('type', 'RELATED')}]→ {rel.get('target', 'Unknown')}")
        print()

        # Test queries for cohort-level analysis
        print("📈 Testing Cohort-Level Query Capabilities:")

        # Count patients by condition
        condition_counts = {}
        patient_conditions = {}
        for rel in kg.get('relationships', []):
            if rel.get('type') == 'HAS_CONDITION':
                patient = rel.get('source', '')
                condition = rel.get('target', '')
                if patient.startswith('Patient_'):
                    if patient not in patient_conditions:
                        patient_conditions[patient] = []
                    patient_conditions[patient].append(condition)
                    condition_counts[condition] = condition_counts.get(condition, 0) + 1

        print(f"   🏥 Patients analyzed: {len(patient_conditions)}")
        print(f"   🩺 Unique conditions found: {len(condition_counts)}")
        print("   Top conditions by frequency:")
        for condition, count in sorted(condition_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            # Find condition name
            condition_name = condition.replace('Condition_', '').split('_')[0]
            print(f"      • {condition}: {count} patients")

        print()
        print("🎉 Patient KG generation test completed successfully!")

    except Exception as e:
        print(f"❌ KG generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_patient_kg_generation()
