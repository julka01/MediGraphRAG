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

def main():
    # Models to test
    models = [
        "meta-llama/llama-4-maverick:free",
        "deepseek/deepseek-r1-0528:free",
        "microsoft/wizardlm-2-8x22b:free",
        "openai/gpt-oss-20b:free"
    ]

    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        return

    print("🏓 Pinging OpenRouter models...\n")

    results = []

    for model in models:
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

if __name__ == "__main__":
    main()
