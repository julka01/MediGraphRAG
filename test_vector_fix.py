#!/usr/bin/env python3
import sys
import os
from enhanced_rag_system import EnhancedRAGSystem

def test_vector_index_fix():
    """Test that the vector index fix works by attempting a vector search"""
    try:
        print("🔍 Testing vector index fix for RAG system...")

        # Initialize the RAG system
        rag_system = EnhancedRAGSystem()

        # Test different thresholds to see chunk retrieval patterns
        queries_and_thresholds = [
            ("cardiology", 0.5),
            ("cardiology", 0.1),
            ("cardiology", 0.08),
            ("cardiology", 0.05),
            ("diagnosis and treatment", 0.5),
            ("diagnosis and treatment", 0.1),
            ("diagnosis and treatment", 0.08),
            ("diagnosis and treatment", 0.05),
        ]

        print("🧪 Testing chunk retrieval with different similarity thresholds:")
        print("=" * 60)

        max_chunks_seen = 0
        for query, threshold in queries_and_thresholds:
            try:
                context = rag_system.get_rag_context(query, similarity_threshold=threshold)
                chunk_count = len(context.get('chunks', []))

                max_chunks_seen = max(max_chunks_seen, chunk_count)

                status = "✓" if chunk_count > 0 else "⚠️"
                print(".3f")

            except Exception as e:
                print(".3f")

        print("=" * 60)
        print(f"📊 Maximum chunks retrieved: {max_chunks_seen}")
        print(f"🔍 Range tested: {min(t for _, t in queries_and_thresholds)} - {max(t for _, t in queries_and_thresholds)}")

        if max_chunks_seen == 0:
            print("❌ No chunks found across all tests - vector search not working")
            return False
        elif max_chunks_seen <= 5:
            print("🟢 Low chunk counts - threshold might be too restrictive")
            return True
        elif max_chunks_seen <= 20:
            print("🟡 Moderate chunk counts - threshold seems appropriate")
            return True
        else:
            print("🔴 High chunk counts - threshold might be too lenient")
            return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_vector_index_fix()
    sys.exit(0 if success else 1)
