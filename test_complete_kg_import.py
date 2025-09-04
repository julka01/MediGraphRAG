#!/usr/bin/env python3
"""
Test script for complete KG import functionality from Neo4j
This script tests the enhanced KG loading capabilities including:
- Complete import (all data)
- Limited import (with node limits)
- Smart sampling (for large graphs)
- Label filtering
"""

import os
import sys
import json
from dotenv import load_dotenv
from kg_loader import KGLoader

# Load environment variables
load_dotenv()

def test_neo4j_connection():
    """Test basic Neo4j connection"""
    print("🔍 Testing Neo4j connection...")
    
    loader = KGLoader()
    
    # Test connection
    try:
        result = loader.list_kg_labels(loader.neo4j_uri, loader.neo4j_user, loader.neo4j_password)
        if result['status'] == 'success':
            print(f"✅ Connected to Neo4j successfully")
            print(f"   Available labels: {result['labels']}")
            return True
        else:
            print(f"❌ Failed to connect: {result['message']}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        return False

def test_complete_import():
    """Test complete KG import (all data)"""
    print("\n🔍 Testing complete KG import...")
    
    loader = KGLoader()
    
    try:
        result = loader.load_from_neo4j(
            loader.neo4j_uri, 
            loader.neo4j_user, 
            loader.neo4j_password,
            limit=None,  # No limit
            sample_mode=False
        )
        
        if result['status'] == 'success':
            print(f"✅ Complete import successful")
            print(f"   Total nodes in DB: {result.get('total_nodes_in_db', 'Unknown')}")
            print(f"   Total relationships in DB: {result.get('total_relationships_in_db', 'Unknown')}")
            print(f"   Loaded nodes: {result.get('loaded_nodes', len(result.get('nodes', [])))}")
            print(f"   Loaded relationships: {result.get('loaded_relationships', len(result.get('relationships', [])))}")
            print(f"   Complete import: {result.get('complete_import', False)}")
            return result
        else:
            print(f"❌ Complete import failed: {result['message']}")
            return None
    except Exception as e:
        print(f"❌ Complete import error: {str(e)}")
        return None

def test_limited_import():
    """Test limited KG import (with node limit)"""
    print("\n🔍 Testing limited KG import (500 nodes max)...")
    
    loader = KGLoader()
    
    try:
        result = loader.load_from_neo4j(
            loader.neo4j_uri, 
            loader.neo4j_user, 
            loader.neo4j_password,
            limit=500,
            sample_mode=False
        )
        
        if result['status'] == 'success':
            print(f"✅ Limited import successful")
            print(f"   Total nodes in DB: {result.get('total_nodes_in_db', 'Unknown')}")
            print(f"   Loaded nodes: {result.get('loaded_nodes', len(result.get('nodes', [])))}")
            print(f"   Loaded relationships: {result.get('loaded_relationships', len(result.get('relationships', [])))}")
            print(f"   Limit respected: {len(result.get('nodes', [])) <= 500}")
            return result
        else:
            print(f"❌ Limited import failed: {result['message']}")
            return None
    except Exception as e:
        print(f"❌ Limited import error: {str(e)}")
        return None

def test_sample_import():
    """Test smart sampling import"""
    print("\n🔍 Testing smart sampling import...")
    
    loader = KGLoader()
    
    try:
        result = loader.load_from_neo4j(
            loader.neo4j_uri, 
            loader.neo4j_user, 
            loader.neo4j_password,
            sample_mode=True
        )
        
        if result['status'] == 'success':
            print(f"✅ Smart sampling successful")
            print(f"   Total nodes in DB: {result.get('total_nodes_in_db', 'Unknown')}")
            print(f"   Loaded nodes: {result.get('loaded_nodes', len(result.get('nodes', [])))}")
            print(f"   Loaded relationships: {result.get('loaded_relationships', len(result.get('relationships', [])))}")
            print(f"   Sample mode: {result.get('sample_mode', False)}")
            print(f"   Sample strategy: {result.get('sample_strategy', 'Unknown')}")
            return result
        else:
            print(f"❌ Smart sampling failed: {result['message']}")
            return None
    except Exception as e:
        print(f"❌ Smart sampling error: {str(e)}")
        return None

def test_label_filtering():
    """Test label-based filtering"""
    print("\n🔍 Testing label-based filtering...")
    
    loader = KGLoader()
    
    # First get available labels
    labels_result = loader.list_kg_labels(loader.neo4j_uri, loader.neo4j_user, loader.neo4j_password)
    if labels_result['status'] != 'success' or not labels_result['labels']:
        print("⚠️  No labels available for filtering test")
        return None
    
    # Use the first available label for testing
    test_label = labels_result['labels'][0]
    print(f"   Testing with label: {test_label}")
    
    try:
        result = loader.load_from_neo4j(
            loader.neo4j_uri, 
            loader.neo4j_user, 
            loader.neo4j_password,
            kg_label=test_label,
            limit=100
        )
        
        if result['status'] == 'success':
            print(f"✅ Label filtering successful")
            print(f"   Filtered by label: {test_label}")
            print(f"   Loaded nodes: {result.get('loaded_nodes', len(result.get('nodes', [])))}")
            print(f"   Loaded relationships: {result.get('loaded_relationships', len(result.get('relationships', [])))}")
            
            # Verify all nodes have the correct label
            nodes_with_label = 0
            for node in result.get('nodes', []):
                if test_label in node.get('labels', []):
                    nodes_with_label += 1
            
            print(f"   Nodes with correct label: {nodes_with_label}/{len(result.get('nodes', []))}")
            return result
        else:
            print(f"❌ Label filtering failed: {result['message']}")
            return None
    except Exception as e:
        print(f"❌ Label filtering error: {str(e)}")
        return None

def test_save_and_export():
    """Test saving KG to file"""
    print("\n🔍 Testing KG save to file...")
    
    loader = KGLoader()
    
    # First load a small sample
    result = loader.load_from_neo4j(
        loader.neo4j_uri, 
        loader.neo4j_user, 
        loader.neo4j_password,
        limit=50
    )
    
    if result['status'] != 'success':
        print("❌ Could not load KG for save test")
        return False
    
    # Save to file
    try:
        save_result = loader.save_to_file(result, "test_complete_import.json")
        if save_result['status'] == 'success':
            print(f"✅ KG saved successfully to: {save_result['file_path']}")
            
            # Verify file exists and has content
            if os.path.exists(save_result['file_path']):
                with open(save_result['file_path'], 'r') as f:
                    saved_data = json.load(f)
                print(f"   Verified file contains {len(saved_data.get('nodes', []))} nodes")
                return True
            else:
                print("❌ Saved file not found")
                return False
        else:
            print(f"❌ Save failed: {save_result['message']}")
            return False
    except Exception as e:
        print(f"❌ Save error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Complete KG Import Functionality")
    print("=" * 50)
    
    # Test connection first
    if not test_neo4j_connection():
        print("\n❌ Cannot proceed without Neo4j connection")
        sys.exit(1)
    
    # Run all tests
    tests = [
        ("Complete Import", test_complete_import),
        ("Limited Import", test_limited_import),
        ("Smart Sampling", test_sample_import),
        ("Label Filtering", test_label_filtering),
        ("Save & Export", test_save_and_export)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result is not None
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Complete KG import functionality is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
