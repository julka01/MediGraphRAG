import os
import httpx
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def ping_openrouter_model(model: str, api_key: str) -> Dict[str, Any]:
    """
    Ping a specific OpenRouter model with a simple test prompt
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/ModelContext/tool-2025-kg-rag",
        "X-Title": "KG RAG Tool"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Hello, can you respond with just 'pong'?"}
        ],
        "max_tokens": 10  # Keep it short for testing
    }

    try:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {
                "model": model,
                "status": "success",
                "response": content.strip(),
                "tokens_used": result.get("usage", {}).get("total_tokens", 0)
            }
        else:
            return {
                "model": model,
                "status": "error",
                "error": f"HTTP {response.status_code}: {response.text}"
            }

    except httpx.TimeoutException:
        return {
            "model": model,
            "status": "error",
            "error": "Request timed out"
        }
    except Exception as e:
        return {
            "model": model,
            "status": "error",
            "error": str(e)
        }

def list_available_models(api_key):
    """List available models from OpenRouter API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/ModelContext/tool-2025-kg-rag",
        "X-Title": "KG RAG Tool"
    }

    try:
        response = httpx.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            # Filter for free models or models that might work
            free_models = [m["id"] for m in models if "free" in m["id"] or any(x in m["id"].lower() for x in ["gemma", "llama", "microsoft", "qwen"])]
            return free_models
        else:
            print(f"❌ Failed to list models: HTTP {response.status_code}: {response.text}")
            return []
    except Exception as e:
        print(f"❌ Failed to list models: {str(e)}")
        return []

def main():
    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        return

    print("🔍 Listing available OpenRouter models...\n")
    available_models = list_available_models(api_key)
    if not available_models:
        print("❌ Could not retrieve model list from OpenRouter API")
        return

    # Filter models to test (free tier, common ones)
    models_to_test = [m for m in available_models if ":free" in m or m in [
        "meta-llama/llama-3.1-8b-instruct:free",
        "microsoft/wizardlm-2-8x22b:free",
        "google/gemma-7b-it:free",
        "qwen/qwen-2-7b-instruct:free"
    ]]

    if not models_to_test:
        models_to_test = available_models[:5]  # Test first 5 if no free ones

    print(f"🏓 Testing {len(models_to_test)} available models...\n")

    results = []

    for model in models_to_test:
        print(f"Testing {model}...")
        result = ping_openrouter_model(model, api_key)
        results.append(result)

        if result["status"] == "success":
            print(f"✅ {model}: {result['response']}")
        else:
            print(f"❌ {model}: {result['error']}")

        print("-" * 50)

    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    total = len(results)

    print(f"\n📊 Summary: {successful}/{total} models responded successfully")

    for result in results:
        status_emoji = "✅" if result["status"] == "success" else "❌"
        print(f"{status_emoji} {result['model']}")

    if successful > 0:
        print("\n✓ Found working models! Update your model lists with these.")
        working_models = [r["model"] for r in results if r["status"] == "success"]
        print(f"Working models: {working_models}")

if __name__ == "__main__":
    main()
