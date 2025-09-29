import os
import httpx
import time
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_gpt_oss_different_timeouts(model: str, api_key: str) -> Dict[str, Any]:
    """
    Test GPT-OSS model with different timeouts to understand timeout behavior
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/ModelContext/tool-2025-kg-rag",
        "X-Title": "KG RAG Tool"
    }

    # Test with different timeouts
    timeouts = [10, 30, 60, 120]  # seconds

    results = {}

    for timeout in timeouts:
        print(f"Testing {model} with {timeout}s timeout...")

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that extracts entities and relationships from text. Extract medical entities and relationships in JSON format."},
                {"role": "user", "content": "The patient presented with prostate cancer. He was diagnosed with Gleason score 7 adenocarcinoma affecting the prostate gland. Treatment included radical prostatectomy and adjuvant hormone therapy. Follow-up showedPSA levels decreased significantly."}
            ],
            "max_tokens": 500,
            "temperature": 0.1
        }

        start_time = time.time()
        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout
            )

            end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                results[f"timeout_{timeout}s"] = {
                    "status": "success",
                    "response_length": len(content),
                    "response_preview": content[:200] + "..." if len(content) > 200 else content,
                    "tokens_used": result.get("usage", {}).get("total_tokens", 0),
                    "response_time": round(end_time - start_time, 2)
                }
            else:
                results[f"timeout_{timeout}s"] = {
                    "status": "error",
                    "http_status": response.status_code,
                    "error": response.text[:200],
                    "response_time": round(end_time - start_time, 2)
                }

        except httpx.TimeoutException:
            end_time = time.time()
            results[f"timeout_{timeout}s"] = {
                "status": "timeout",
                "response_time": round(end_time - start_time, 2)
            }
        except Exception as e:
            end_time = time.time()
            results[f"timeout_{timeout}s"] = {
                "status": "exception",
                "error": str(e),
                "response_time": round(end_time - start_time, 2)
            }

        print(f"Result: {results[f'timeout_{timeout}s']['status']} ({results[f'timeout_{timeout}s'].get('response_time', 'N/A')}s)")
        print("-" * 50)

    return results

def test_concurrent_requests(model: str, api_key: str, num_requests: int = 3):
    """
    Test multiple concurrent requests to see if it's a rate limiting issue
    """
    import asyncio
    import aiohttp

    async def make_request(session, request_id):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/ModelContext/tool-2025-kg-rag",
            "X-Title": "KG RAG Tool"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": f"Respond with just the word 'test' followed by {request_id}."}
            ],
            "max_tokens": 10
        }

        start_time = time.time()
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                end_time = time.time()

                if response.status == 200:
                    result = await response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return {
                        "request_id": request_id,
                        "status": "success",
                        "content": content.strip(),
                        "response_time": round(end_time - start_time, 2)
                    }
                else:
                    text = await response.text()
                    return {
                        "request_id": request_id,
                        "status": "error",
                        "http_status": response.status,
                        "error": text[:100],
                        "response_time": round(end_time - start_time, 2)
                    }

        except asyncio.TimeoutError:
            end_time = time.time()
            return {
                "request_id": request_id,
                "status": "timeout",
                "response_time": round(end_time - start_time, 2)
            }
        except Exception as e:
            end_time = time.time()
            return {
                "request_id": request_id,
                "status": "exception",
                "error": str(e),
                "response_time": round(end_time - start_time, 2)
            }

    async def run_concurrent_test():
        async with aiohttp.ClientSession() as session:
            tasks = [make_request(session, i) for i in range(num_requests)]
            results = await asyncio.gather(*tasks)
            return results

    print(f"\nTesting {num_requests} concurrent requests to {model}...")
    results = asyncio.run(run_concurrent_test())

    for result in results:
        status = result['status']
        time_taken = result.get('response_time', 'N/A')
        print(f"Request {result['request_id']}: {status} ({time_taken}s)")

    return results

def main():
    model = "openai/gpt-oss-20b:free"

    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        return

    print(f"🔍 Investigating timeout issues with {model}\n")

    # Test 1: Different timeout values
    print("=== Test 1: Different Timeout Values ===")
    timeout_results = test_gpt_oss_different_timeouts(model, api_key)

    # Test 2: Concurrent requests
    print("\n=== Test 2: Concurrent Requests ===")
    concurrent_results = test_concurrent_requests(model, api_key, 3)

    # Analysis
    print("\n=== Analysis ===")

    # Check timeout test results
    timeout_successes = sum(1 for r in timeout_results.values() if r.get('status') == 'success')
    total_timeouts = len(timeout_results)

    print(f"Timeout Test: {timeout_successes}/{total_timeouts} requests successful")

    # Check if longer timeouts help
    if timeout_results.get('timeout_120s', {}).get('status') == 'success':
        print("✅ Model works with 120s timeout - may be slow processing")
    elif any(r.get('status') == 'success' for r in timeout_results.values()):
        print("⚠️ Model works with shorter timeouts but may be inconsistent")
    else:
        print("❌ Model consistently fails even with long timeouts")

    # Check concurrent test results
    concurrent_successes = sum(1 for r in concurrent_results if r.get('status') == 'success')
    total_concurrent = len(concurrent_results)

    print(f"Concurrent Test: {concurrent_successes}/{total_concurrent} requests successful")

    if concurrent_successes < total_concurrent:
        print("⚠️ Some concurrent requests failed - possible rate limiting")

    # Recommendations
    print("\n=== Recommendations ===")
    if timeout_successes > 0:
        avg_response_time = sum(r.get('response_time', 0) for r in timeout_results.values() if r.get('status') == 'success') / timeout_successes
        print(f"✅ Average response time: {avg_response_time:.1f}s")
        print("  - Increase request timeouts to at least 60-120 seconds")
        print("  - Add retry logic for failed requests")
    else:
        print("❌ This model appears unreliable for KG generation")
        print("- Consider using meta-llama/llama-4-maverick:free instead")

if __name__ == "__main__":
    main()
