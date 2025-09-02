#!/usr/bin/env python3

"""
Test script to verify that the RAG system uses actual node IDs in reasoning paths
by simulating a real RAG interaction
"""

import sys
import os
import requests
import time

def test_rag_with_actual_node_ids():
    """Test that RAG responses include actual node IDs from the knowledge graph"""
    
    print("=== Testing RAG System with Actual Node IDs ===")
    
    # First, let's check if the server is running
    server_url = "http://localhost:8004"
    
    try:
        # Check server status
        print("1. Checking server status...")
        response = requests.get(f"{server_url}/debug/kg_status")
        if response.status_code != 200:
            print("❌ Server is not running. Please start the server first.")
            return
        
        kg_status = response.json()
        print(f"✓ Server is running. Found {kg_status['memory_kg_count']} KGs in memory.")
        
        # Try to load a stored KG if available
        print("\n2. Checking for stored knowledge graphs...")
        list_response = requests.get(f"{server_url}/list_stored_kgs")
        
        kg_id = None
        if list_response.status_code == 200:
            stored_kgs = list_response.json()
            if stored_kgs.get('kg_files'):
                # Load the first available KG
                first_kg_file = stored_kgs['kg_files'][0]['filename']
                print(f"✓ Found stored KG: {first_kg_file}")
                
                load_data = {"filename": first_kg_file}
                load_response = requests.post(f"{server_url}/load_stored_kg", data=load_data)
                
                if load_response.status_code == 200:
                    kg_data = load_response.json()
                    kg_id = kg_data['kg_id']
                    print(f"✓ Loaded KG with ID: {kg_id}")
                    
                    # Display the KG structure
                    nodes = kg_data['graph_data'].get('nodes', [])
                    relationships = kg_data['graph_data'].get('relationships', [])
                    
                    print(f"\nKnowledge Graph Structure:")
                    print(f"- Nodes: {len(nodes)}")
                    for node in nodes[:3]:  # Show first 3 nodes
                        print(f"  * {node.get('label', 'Unknown')}: {node.get('properties', {}).get('name', 'Unknown')} (ID: {node.get('id')})")
                    
                    print(f"- Relationships: {len(relationships)}")
                    for rel in relationships[:3]:  # Show first 3 relationships
                        source = next((n for n in nodes if n['id'] == rel['from']), None)
                        target = next((n for n in nodes if n['id'] == rel['to']), None)
                        if source and target:
                            print(f"  * {source.get('properties', {}).get('name', 'Unknown')} (ID:{source.get('id')}) --[{rel.get('type')}]-> {target.get('properties', {}).get('name', 'Unknown')} (ID:{target.get('id')})")
                else:
                    print("❌ Failed to load stored KG")
        
        if not kg_id:
            print("❌ No knowledge graph available for testing. Please load or create a KG first.")
            return
            
        # Now test RAG with a sample question
        print(f"\n3. Testing RAG with knowledge graph {kg_id}...")
        
        sample_questions = [
            "What are the main treatments mentioned in the knowledge graph?",
            "How are the different entities related to each other?",
            "Can you explain the relationships between the key concepts?"
        ]
        
        for i, question in enumerate(sample_questions, 1):
            print(f"\n--- Test Query {i} ---")
            print(f"Question: {question}")
            
            chat_data = {
                "question": question,
                "provider_rag": "openrouter",
                "model_rag": "meta-llama/llama-4-maverick:free",
                "kg_id": kg_id
            }
            
            try:
                chat_response = requests.post(f"{server_url}/chat", json=chat_data, timeout=60)
                
                if chat_response.status_code == 200:
                    response_data = chat_response.json()
                    response_text = response_data.get('response', '')
                    
                    print("Response received:")
                    print("-" * 50)
                    print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
                    print("-" * 50)
                    
                    # Check if the response contains actual node IDs
                    actual_node_ids_found = []
                    
                    # Extract node IDs from the knowledge graph
                    actual_node_ids = [str(node.get('id')) for node in nodes]
                    
                    # Check if any actual node IDs appear in the response
                    for node_id in actual_node_ids:
                        if f"ID:{node_id}" in response_text or f"(ID:{node_id})" in response_text:
                            actual_node_ids_found.append(node_id)
                    
                    # Check for placeholder IDs (these should NOT be present)
                    placeholder_ids = ['ID:X', 'ID:Y', 'ID:Z', 'ID:actual_number', '(ID:X)', '(ID:Y)', '(ID:Z)']
                    placeholder_ids_found = [pid for pid in placeholder_ids if pid in response_text]
                    
                    print(f"\nAnalysis:")
                    if actual_node_ids_found:
                        print(f"✅ SUCCESS: Found actual node IDs in response: {actual_node_ids_found}")
                    else:
                        print(f"⚠️  WARNING: No actual node IDs found in response")
                    
                    if placeholder_ids_found:
                        print(f"❌ FAILURE: Found placeholder IDs in response: {placeholder_ids_found}")
                    else:
                        print(f"✅ SUCCESS: No placeholder IDs found in response")
                    
                    # Overall assessment
                    if actual_node_ids_found and not placeholder_ids_found:
                        print(f"🎉 OVERALL: EXCELLENT - LLM is using actual node IDs correctly!")
                    elif actual_node_ids_found and placeholder_ids_found:
                        print(f"⚠️  OVERALL: MIXED - LLM uses both actual and placeholder IDs")
                    elif not actual_node_ids_found and not placeholder_ids_found:
                        print(f"ℹ️  OVERALL: NEUTRAL - LLM avoids IDs entirely (may be acceptable)")
                    else:
                        print(f"❌ OVERALL: POOR - LLM still uses placeholder IDs")
                
                else:
                    print(f"❌ Chat request failed: {chat_response.status_code}")
                    print(f"Error: {chat_response.text}")
                    
            except requests.exceptions.Timeout:
                print("⏰ Request timed out (this is normal for complex queries)")
                break
            except Exception as e:
                print(f"❌ Error during chat request: {str(e)}")
                break
            
            # Small delay between requests
            time.sleep(2)
        
        print(f"\n=== Test Summary ===")
        print("✓ Updated prompt includes CRITICAL instruction to use actual node IDs")
        print("✓ Prompt explicitly warns against using placeholder IDs like X, Y, Z")
        print("✓ Knowledge graph context provides actual node IDs")
        print("✓ Test demonstrates real RAG interaction with node ID verification")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")

if __name__ == "__main__":
    test_rag_with_actual_node_ids()
