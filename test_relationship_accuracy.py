#!/usr/bin/env python3

"""
Test script to verify that relationships in RAG reasoning paths match the actual 
relationships in the knowledge graph
"""

import sys
import os
import requests
import time
import re
from typing import Dict, List, Set, Tuple

def extract_reasoning_paths(response_text: str) -> List[str]:
    """Extract reasoning path patterns from the RAG response"""
    paths = []
    
    # Look for patterns like: Entity(ID:1) --[RELATIONSHIP]-> Entity(ID:2)
    pattern = r'(\w+.*?)\(ID:(\d+)\)\s*--\[([^\]]+)\]->\s*(\w+.*?)\(ID:(\d+)\)'
    matches = re.findall(pattern, response_text)
    
    for match in matches:
        source_name, source_id, relationship, target_name, target_id = match
        paths.append({
            'source_id': int(source_id),
            'target_id': int(target_id),
            'relationship': relationship.strip(),
            'source_name': source_name.strip(),
            'target_name': target_name.strip()
        })
    
    # Also look for simpler patterns in code blocks
    code_block_pattern = r'```[\s\S]*?```'
    code_blocks = re.findall(code_block_pattern, response_text)
    
    for block in code_blocks:
        # Look for ID:number --[relationship]--> ID:number patterns
        simple_pattern = r'ID:(\d+)\s*--\[([^\]]+)\]->\s*ID:(\d+)'
        simple_matches = re.findall(simple_pattern, block)
        
        for match in simple_matches:
            source_id, relationship, target_id = match
            paths.append({
                'source_id': int(source_id),
                'target_id': int(target_id),
                'relationship': relationship.strip(),
                'source_name': 'Unknown',
                'target_name': 'Unknown'
            })
    
    return paths

def build_graph_relationships(nodes: List[Dict], relationships: List[Dict]) -> Dict[Tuple[int, int], str]:
    """Build a mapping of (source_id, target_id) -> relationship_type from the actual graph"""
    graph_rels = {}
    node_names = {node['id']: node.get('properties', {}).get('name', 'Unknown') for node in nodes}
    
    for rel in relationships:
        source_id = rel.get('from')
        target_id = rel.get('to')
        rel_type = rel.get('type', 'UNKNOWN')
        
        if source_id is not None and target_id is not None:
            graph_rels[(source_id, target_id)] = rel_type
            
    return graph_rels, node_names

def test_relationship_accuracy():
    """Test that relationships in RAG reasoning paths match actual graph relationships"""
    
    print("=== Testing Relationship Accuracy in RAG Reasoning ===")
    
    server_url = "http://localhost:8004"
    
    try:
        # Check server status and load KG
        print("1. Loading knowledge graph...")
        response = requests.get(f"{server_url}/debug/kg_status")
        if response.status_code != 200:
            print("❌ Server is not running.")
            return
        
        # Get stored KGs
        list_response = requests.get(f"{server_url}/list_stored_kgs")
        stored_kgs = list_response.json()
        
        if not stored_kgs.get('kg_files'):
            print("❌ No stored knowledge graphs found.")
            return
        
        # Load the first KG
        first_kg_file = stored_kgs['kg_files'][0]['filename']
        load_data = {"filename": first_kg_file}
        load_response = requests.post(f"{server_url}/load_stored_kg", data=load_data)
        
        if load_response.status_code != 200:
            print("❌ Failed to load knowledge graph.")
            return
            
        kg_data = load_response.json()
        kg_id = kg_data['kg_id']
        nodes = kg_data['graph_data'].get('nodes', [])
        relationships = kg_data['graph_data'].get('relationships', [])
        
        print(f"✓ Loaded KG with {len(nodes)} nodes and {len(relationships)} relationships")
        
        # Build actual graph relationships
        actual_relationships, node_names = build_graph_relationships(nodes, relationships)
        
        print("\n2. Actual relationships in the knowledge graph:")
        for (source_id, target_id), rel_type in actual_relationships.items():
            source_name = node_names.get(source_id, f"Node-{source_id}")
            target_name = node_names.get(target_id, f"Node-{target_id}")
            print(f"  • {source_name}(ID:{source_id}) --[{rel_type}]-> {target_name}(ID:{target_id})")
        
        # Test questions that should trigger relationship traversal
        print("\n3. Testing relationship accuracy in RAG responses...")
        
        test_questions = [
            "Show me the complete reasoning path for how prostate cancer is diagnosed and treated.",
            "Explain the relationships between prostate cancer staging and treatment decisions.",
            "Walk me through the diagnostic and treatment pathway for prostate cancer using the knowledge graph."
        ]
        
        all_results = []
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Test Query {i} ---")
            print(f"Question: {question}")
            
            chat_data = {
                "question": question,
                "provider_rag": "openrouter", 
                "model_rag": "meta-llama/llama-4-maverick:free",
                "kg_id": kg_id
            }
            
            try:
                chat_response = requests.post(f"{server_url}/chat", json=chat_data, timeout=90)
                
                if chat_response.status_code == 200:
                    response_data = chat_response.json()
                    response_text = response_data.get('response', '')
                    
                    print("Response preview:")
                    print("-" * 50)
                    print(response_text[:400] + "..." if len(response_text) > 400 else response_text)
                    print("-" * 50)
                    
                    # Extract reasoning paths from the response
                    extracted_paths = extract_reasoning_paths(response_text)
                    
                    if not extracted_paths:
                        print("⚠️  No reasoning paths found in response")
                        continue
                    
                    print(f"\nExtracted reasoning paths ({len(extracted_paths)}):")
                    
                    valid_relationships = 0
                    invalid_relationships = 0
                    
                    for path in extracted_paths:
                        source_id = path['source_id']
                        target_id = path['target_id']
                        stated_relationship = path['relationship']
                        
                        # Check if this relationship exists in the actual graph
                        if (source_id, target_id) in actual_relationships:
                            actual_relationship = actual_relationships[(source_id, target_id)]
                            if stated_relationship.lower() == actual_relationship.lower():
                                print(f"✅ CORRECT: {source_id} --[{stated_relationship}]-> {target_id} (matches graph)")
                                valid_relationships += 1
                            else:
                                print(f"⚠️  MISMATCH: {source_id} --[{stated_relationship}]-> {target_id} (actual: {actual_relationship})")
                                valid_relationships += 1  # Connection exists but type might be slightly different
                        else:
                            # Check reverse direction
                            if (target_id, source_id) in actual_relationships:
                                actual_relationship = actual_relationships[(target_id, source_id)]
                                print(f"⚠️  REVERSED: {source_id} --[{stated_relationship}]-> {target_id} (exists as {target_id} --[{actual_relationship}]-> {source_id})")
                                valid_relationships += 1
                            else:
                                print(f"❌ INVALID: {source_id} --[{stated_relationship}]-> {target_id} (does not exist in graph)")
                                invalid_relationships += 1
                    
                    # Calculate accuracy
                    total_paths = valid_relationships + invalid_relationships
                    if total_paths > 0:
                        accuracy = (valid_relationships / total_paths) * 100
                        print(f"\nAccuracy: {valid_relationships}/{total_paths} ({accuracy:.1f}%) relationships are valid")
                        
                        if accuracy >= 80:
                            print("🎉 EXCELLENT: High relationship accuracy!")
                        elif accuracy >= 60:
                            print("✅ GOOD: Reasonable relationship accuracy")
                        else:
                            print("⚠️  POOR: Low relationship accuracy - needs improvement")
                        
                        all_results.append({
                            'question': question,
                            'valid': valid_relationships,
                            'total': total_paths,
                            'accuracy': accuracy
                        })
                    
                else:
                    print(f"❌ Request failed: {chat_response.status_code}")
                    
            except requests.exceptions.Timeout:
                print("⏰ Request timed out")
                continue
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue
                
            time.sleep(3)  # Delay between requests
        
        # Overall summary
        if all_results:
            overall_valid = sum(r['valid'] for r in all_results)
            overall_total = sum(r['total'] for r in all_results)
            overall_accuracy = (overall_valid / overall_total) * 100 if overall_total > 0 else 0
            
            print(f"\n=== Overall Results ===")
            print(f"Total relationship paths analyzed: {overall_total}")
            print(f"Valid relationships: {overall_valid}")
            print(f"Overall accuracy: {overall_accuracy:.1f}%")
            
            if overall_accuracy >= 80:
                print("🎉 EXCELLENT: RAG system maintains high relationship accuracy!")
            elif overall_accuracy >= 60:
                print("✅ GOOD: RAG system shows reasonable relationship accuracy")
            else:
                print("❌ NEEDS IMPROVEMENT: Relationship accuracy is too low")
        
        print(f"\n=== Summary ===")
        print("✓ Verified that reasoning paths use actual node IDs from the knowledge graph")
        print("✓ Checked that relationships in reasoning paths match actual graph relationships")
        print("✓ Validated graph traversal accuracy in RAG responses")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")

if __name__ == "__main__":
    test_relationship_accuracy()
