#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append(os.getcwd())

from enhanced_rag_system import EnhancedRAGSystem
from model_providers import get_provider

def test_rag_with_ontology_kg():
    """Test RAG system connection with ontology-guided KG"""

    print("🧪 Testing RAG System Connection with Ontology-Guided KG")
    print("=" * 60)

    try:
        # Initialize RAG system
        print("🤖 Initializing Enhanced RAG System...")
        rag_system = EnhancedRAGSystem(
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USERNAME"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j")
        )

        # Get KG stats
        print("📊 Getting Knowledge Graph Statistics...")
        stats = rag_system.get_knowledge_graph_stats()

        if stats and stats.get('error'):
            print(f"❌ Error getting KG stats: {stats['error']}")
            return False

        if stats:
            print("\n📈 KG Statistics:")
            print(f"   📄 Documents: {stats['documents']}")
            print(f"   📦 Chunks: {stats['chunks']}")
            print(f"   🎯 Entities: {stats['entities']}")
            print(f"   🔗 Relationships: {stats['relationships']}")
            print(f"   📝 Document names: {stats['document_names']}")

            if stats['chunks'] == 0:
                print("⚠️  Warning: No chunks found in KG. KG may be empty.")
                return False
        else:
            print("❌ Failed to get KG statistics")
            return False

        # Test RAG context retrieval
        print("\n🔍 Testing RAG Context Retrieval...")
        questions = [
            "What is prostate cancer?",
            "How is prostate cancer diagnosed?",
            "What are treatment options for prostate cancer?"
        ]

        # Get LLM provider for response generation
        print("🧠 Setting up LLM provider...")
        llm_provider = get_provider("openrouter", "deepseek/deepseek-chat")

        # Enhanced validation with quality checks
        validation_results = []

        for idx, question in enumerate(questions):
            print(f"\n❓ Question {idx+1}: {question}")

            # Get RAG context
            context = rag_system.get_rag_context(question, top_k=3)
            print(f"   📚 Retrieved {len(context['chunks'])} chunks")
            print(f"   🎯 Found {len(context['entities'])} unique entities")
            print(f"   🔗 Found {len(context['relationships'])} relationships")

            # Quality checks
            is_quality_response = False
            quality_notes = []

            if len(context['chunks']) > 0:
                print("   ✅ Context retrieved successfully")
                print(f"      📏 Total relevance score: {context['total_score']:.3f}")

                # Check if context is relevant
                if context['total_score'] > 0.1:
                    quality_notes.append("High relevance score")
                    is_quality_response = True

                # Show sample entities
                if context['entities']:
                    entity_sample = list(context['entities'].keys())[:3]
                    print(f"      🎯 Sample entities: {entity_sample}")

                    # Check for medical context
                    medical_entities = [e for e in entity_sample if any(term.lower() in e.lower()
                                     for term in ['protein', 'cancer', 'prostate', 'treatment', 'diagnosis', 'medical'])]
                    if medical_entities:
                        quality_notes.append(f"Contains medical entities: {medical_entities}")
                        is_quality_response = True

                # Generate response
                try:
                    response = rag_system.generate_response(question, llm_provider, top_k=3)
                    print("   🤖 Response generated successfully")
                    print(f"      📊 Confidence: {response['confidence']:.3f}")

                    # Analyze response quality
                    response_text = response['response'].lower()
                    question_lower = question.lower()

                    # Check if response addresses the question
                    response_addresses_question = False
                    if 'prostate cancer' in question_lower:
                        if 'prostate cancer' in response_text or 'prostate' in response_text:
                            response_addresses_question = True
                            quality_notes.append("Response addresses prostate cancer")
                    elif 'diagnosis' in question_lower:
                        if 'diagnosis' in response_text or 'detect' in response_text or 'test' in response_text:
                            response_addresses_question = True
                            quality_notes.append("Response addresses diagnosis")
                    elif 'treatment' in question_lower:
                        if 'treatment' in response_text or 'therapy' in response_text or 'manage' in response_text:
                            response_addresses_question = True
                            quality_notes.append("Response addresses treatment")

                    if response_addresses_question:
                        is_quality_response = True

                    # Display response preview
                    response_preview = response['response'][:200] + "..." if len(response['response']) > 200 else response['response']
                    print(f"      💬 Response preview: {response_preview}")

                    # Validate response quality
                    if len(response['response']) > 50 and response['confidence'] > 0.3:
                        quality_notes.append("Substantial response length")
                        is_quality_response = True

                except Exception as e:
                    print(f"   ❌ Failed to generate response: {e}")
                    is_quality_response = False
                    quality_notes.append("Response generation failed")

            else:
                print("   ⚠️  No context found - this might indicate an empty or inaccessible KG")
                is_quality_response = False
                quality_notes.append("No context retrieved")

            # Store validation results
            validation_results.append({
                'question': question,
                'is_quality': is_quality_response,
                'notes': quality_notes,
                'chunks_found': len(context['chunks']),
                'entities_found': len(context['entities']),
                'total_score': context.get('total_score', 0)
            })

        # Overall quality assessment
        print("\n🎯 Overall Quality Assessment:"        quality_questions = [r for r in validation_results if r['is_quality']]
        print(f"   ✅ Quality responses: {len(quality_questions)}/{len(questions)}")

        if len(quality_questions) == len(questions):
            print("   🏆 EXCELLENT: All responses show quality indicators!")
        elif len(quality_questions) >= len(questions) // 2:
            print("   👍 GOOD: Majority of responses show quality indicators")
        else:
            print("   🚨 NEEDS IMPROVEMENT: Few quality indicators found")

        # Detailed breakdown
        print("\n📊 Detailed Quality Breakdown:"        for i, result in enumerate(validation_results):
            status = "✅ PASS" if result['is_quality'] else "❌ FAIL"
            print(f"   {i+1}. {status} - {result['question'][:50]}...")
            if result['notes']:
                for note in result['notes'][:2]:  # Show first 2 notes
                    print(f"      ⋅ {note}")

        print("\n🎉 RAG System Testing Complete!")
        print("\n✅ KG-RAG Connection: WORKING" if stats['chunks'] > 0 else "\n❌ KG-RAG Connection: NO DATA")

        return stats['chunks'] > 0

    except Exception as e:
        print(f"❌ Error testing RAG system: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_with_ontology_kg()

    if success:
        print("\n🚀 SUCCESS: RAG system is connected and working with the ontology-guided KG!")
        sys.exit(0)
    else:
        print("\n💥 FAILED: RAG system connection has issues.")
        sys.exit(1)
