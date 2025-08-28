import requests
import json
import os

# Set up test data
TEST_KG_PATH = "test_kg.json"
UPLOAD_URL = "http://localhost:8000/load_kg_from_file"
CHAT_URL = "http://localhost:8000/chat"

def test_kg_import_and_query():
    # Read the test KG file
    with open(TEST_KG_PATH, "rb") as f:
        kg_content = f.read()
    
    # Prepare the request
    files = {
        "file": ("test_kg.json", kg_content, "application/json")
    }
    data = {
        "provider": "openrouter",
        "model": "deepseek/deepseek-r1-0528:free"
    }
    
    # Import the KG
    print("Importing knowledge graph...")
    response = requests.post(UPLOAD_URL, files=files, data=data)
    
    if response.status_code != 200:
        print(f"Import failed: {response.text}")
        return
    
    kg_id = response.json().get("kg_id")
    print(f"KG imported successfully with ID: {kg_id}")
    
    # Query the KG
    print("Querying the KG...")
    chat_data = {
        "question": "What treatments are available for Prostate Cancer?",
        "provider_rag": "openrouter",
        "model_rag": "deepseek/deepseek-r1-0528:free",
        "kg_id": kg_id
    }
    chat_response = requests.post(CHAT_URL, json=chat_data)
    
    if chat_response.status_code != 200:
        print(f"Query failed: {chat_response.text}")
        return
    
    print("Query response:")
    print(json.dumps(chat_response.json(), indent=2))

if __name__ == "__main__":
    # Make sure the test KG file exists
    if not os.path.exists(TEST_KG_PATH):
        print(f"Error: Test KG file not found at {TEST_KG_PATH}")
        print("Please create it first using the previous tool call")
        exit(1)
    
    test_kg_import_and_query()
