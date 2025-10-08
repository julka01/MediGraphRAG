#!/usr/bin/env python3
"""
Debug script to test what the frontend receives for filtering.
"""

import sys
import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

def test_chat_endpoint_filtering():
    """Test what the /chat endpoint returns for filtering"""

    # Test question
    question = "What are the treatment options for prostate cancer?"

    payload = {
        "question": question,
        "provider_rag": "openrouter",
        "model_rag": "openai/gpt-oss-20b:free"
    }

    print("🔍 Testing /chat endpoint filtering response")
    print(f"❓ Question: {question}")
    print("="*60)

    try:
        # Make request to chat endpoint
        response = requests.post("http://localhost:8004/chat", json=payload)
        response.raise_for_status()

        result = response.json()

        print("✅ Received chat response")
        print(f"📄 Response preview: {result.get('message', '')[100:200]}...")
        print()

        # Check what the frontend should receive
        entities_data = result.get('info', {}).get('entities', {})
        print("🎯 Frontend filtering data:")
        print(f"- result.info.entities.used_entities: {entities_data.get('used_entities', [])}")
        print(f"- result.info.entities.entityids: {entities_data.get('entityids', [])}")
        print()

        # Also check backend filtering data
        backend_used_entities = result.get('used_entities', [])
        backend_used_chunks = result.get('used_chunks', [])
        backend_reasoning_edges = result.get('reasoning_edges', [])
        print("🔧 Backend filtering data:")
        print(f"- result.used_entities: {len(backend_used_entities)} entities")
        print(f"- result.used_chunks: {len(backend_used_chunks)} chunks")
        print(f"- result.reasoning_edges: {len(backend_reasoning_edges)} edges")
        print()

        # Show detailed used entities
        used_entities = entities_data.get('used_entities', [])
        if used_entities:
            print("✅ Used entities found:")
            for entity in used_entities:
                entity_id = entity.get('id', 'MISSING')
                element_id = entity.get('element_id', 'MISSING')
                print(f"  • Entity ID: {entity_id}")
                print(f"    Element ID: {element_id}")
                print(f"    Type: {entity.get('type', 'MISSING')}")
                print()
        elif backend_used_entities:
            print("⚠️ Used entities found in backend but not in frontend format:")
            for entity in backend_used_entities:
                entity_id = entity.get('id', 'MISSING')
                element_id = entity.get('element_id', 'MISSING')
                print(f"  • Entity ID: {entity_id} (Element ID: {element_id})")
                print()
        else:
            print("❌ No used entities found - filtering won't work on frontend")
            print("This explains why the KG visualization doesn't highlight nodes!")

        # Check if response contains entity references
        message = result.get('message', '')
        has_id_refs = '(ID:' in message
        print("\n🔍 Response analysis:")
        print(f"- Contains entity ID references: {has_id_refs}")
        print(f"- Full message preview: {message[:300]}...")

        # Extract IDs from message
        import re
        if has_id_refs:
            entity_id_matches = re.findall(r'\(ID:([^)]+)\)', message)
            print(f"- Referenced element IDs in message: {entity_id_matches}")

            # Show available context entities
            if 'context' in result and 'entities' in result['context']:
                context_entities = result['context']['entities']
                print(f"- Available entity element_ids in context: {list(context_entities.keys()) if context_entities else 'None'}")
                if context_entities:
                    print("  Matching check:")
                    for ref_id in entity_id_matches:
                        found = any(entity.get('element_id') == ref_id for entity in context_entities.values())
                        print(f"    • (ID:{ref_id}) - Found in context: {found}")
            else:
                print("- No context entities available to check")

        if not has_id_refs and not used_entities:
            print("❌ PROBLEM: LLM didn't include ID references, AND no fallback entities were found")
            print("This is exactly why filtering isn't working!")
        elif has_id_refs and used_entities:
            print("✅ SUCCESS: Explicit ID matching working correctly")
        elif not has_id_refs and used_entities:
            print("✅ SUCCESS: Fallback name matching working correctly")

        return result

    except Exception as e:
        print(f"❌ Error testing chat endpoint: {e}")
        return None

def test_frontend_simulation():
    """Simulate what the frontend should do with the filtering data"""

    print("\n🖥️ Simulating Frontend Node Highlighting Logic")
    print("="*50)

    # Get the chat response
    result = test_chat_endpoint_filtering()
    if not result:
        return

    # Simulate frontend logic
    entities_data = result.get('info', {}).get('entities', {})

    highlighted_nodes = set()
    reasoning_edges = set()

    # Process used entities (frontend logic)
    if entities_data.get('used_entities'):
        print("🎯 Processing used_entities for highlighting:")
        for entity in entities_data['used_entities']:
            entity_id = entity.get('id', '')
            element_id = entity.get('element_id', '')

            if entity_id:
                highlighted_nodes.add(entity_id)
                print(f"  • Added entity ID: {entity_id}")

            if element_id:
                highlighted_nodes.add(element_id)
                print(f"  • Added element ID: {element_id}")

    print(f"\n📊 Final highlighted nodes: {len(highlighted_nodes)}")
    print("These should be highlighted in the KG visualization:")
    for node_id in highlighted_nodes:
        print(f"  • {node_id}")

    if not highlighted_nodes:
        print("❌ No nodes to highlight - this is why the KG isn't filtering!")
        print("SOLUTION: Need to implement better LLM prompting or fallback detection")

    return highlighted_nodes

if __name__ == "__main__":
    try:
        test_frontend_simulation()
        print("\n🎯 SUMMARY:")
        print("- If used_entities is empty, the frontend won't highlight any nodes")
        print("- The LLM needs to either:")
        print("  1. Include entity IDs like 'EntityName (ID:actual_id)' in responses")
        print("  2. Use entity names that match the context entities")
        print("- The fallback mechanism should catch most cases now")
    except Exception as e:
        print(f"❌ Complete failure: {e}")
        import traceback
        traceback.print_exc()
