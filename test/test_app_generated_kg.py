import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = "http://localhost:8004"
TEST_PDF = "test_document.pdf"

def test_app_generated_kg():
    print("\n=== Testing App-Generated KG Save to Neo4j ===")
    
    # Step 1: Generate KG from test PDF
    print("Generating KG from test PDF...")
    with open(TEST_PDF, "rb") as f:
        files = {"file": f}
        response = requests.post(
            f"{BASE_URL}/generate_kg",
            files=files,
            data={"provider_kg": "openrouter", "model_kg": "deepseek/deepseek-r1-0528:free"}
        )
    
    if response.status_code != 200:
        print(f"❌ KG generation failed: {response.status_code} - {response.text}")
        return False
    
    kg_data = response.json()
    kg_id = kg_data["kg_id"]
    print(f"Generated KG ID: {kg_id}")
    
    # Step 2: Save KG to Neo4j
    print("Saving KG to Neo4j...")
    # Send as JSON data
    json_data = {
        "kg_id": kg_id,
        "uri": os.getenv("NEO4J_URI"),
        "user": os.getenv("NEO4J_USER"),
        "password": os.getenv("NEO4J_PASSWORD")
    }
    save_response = requests.post(
        f"{BASE_URL}/save_kg_to_neo4j",
        json=json_data
    )
    
    if save_response.status_code != 200:
        print(f"❌ Save to Neo4j failed: {save_response.status_code} - {save_response.text}")
        return False
    
    save_result = save_response.json()
    print(f"Save result: {save_result}")
    
    if save_result.get("status") == "success":
        print("✅ App-generated KG saved successfully to Neo4j")
        return True
    else:
        print(f"❌ Save operation failed: {save_result.get('message')}")
        return False

if __name__ == "__main__":
    test_app_generated_kg()
