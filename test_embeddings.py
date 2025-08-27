import os
import requests
from dotenv import load_dotenv
load_dotenv()

def test_openai_embeddings():
    print("\nTesting OpenAI embeddings directly")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key":
        print("OpenAI API key not set or still using placeholder value")
        return False
        
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": "This is a test",
        "model": "text-embedding-ada-002"
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        print(f"Success! Embedding dimension: {len(result['data'][0]['embedding'])}")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Response: {response.text if 'response' in locals() else 'No response'}")
        return False

def test_openrouter_embeddings():
    print("\nTesting OpenRouter embeddings directly")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OpenRouter API key not set")
        return False
        
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:8005",
        "X-Title": "KG-RAG",
        "Content-Type": "application/json"
    }
    data = {
        "input": "This is a test",
        "model": "text-embedding-ada-002"
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        print(f"Success! Embedding dimension: {len(result['data'][0]['embedding'])}")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Response: {response.text if 'response' in locals() else 'No response'}")
        return False

# Run tests
openai_works = test_openai_embeddings()
openrouter_works = test_openrouter_embeddings()

if not openai_works and not openrouter_works:
    print("\nAll embedding providers failed. Please check your API keys and network connection.")
    exit(1)
