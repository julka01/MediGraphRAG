import os
import json
import httpx
from abc import ABC, abstractmethod
from openai import OpenAI as OpenAIClient
import google.generativeai as genai
from huggingface_hub import InferenceClient
import ollama
from typing import Dict, Any

class ModelProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, model: str, **kwargs) -> str:
        pass

class OpenAIProvider(ModelProvider):
    def __init__(self):
        self.client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
    
    def generate(self, system_prompt: str, user_prompt: str, model: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **kwargs
        )
        return response.choices[0].message.content

class OllamaProvider(ModelProvider):
    def generate(self, system_prompt: str, user_prompt: str, model: str, **kwargs) -> str:
        try:
            # Use Ollama's built-in JSON format enforcement
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format='json',
                **kwargs
            )
            content = response['message']['content']
            
            # Validate JSON format
            json.loads(content)
            return content
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response from model: {content}")
        except ollama.ResponseError as e:
            if "not found" in str(e).lower():
                return f"Error: Model '{model}' not found. Please run 'ollama pull {model}' to download it."
            else:
                return f"Ollama error: {str(e)}"
        except Exception as e:
            return f"Error generating response: {str(e)}"

class GeminiProvider(ModelProvider):
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    def generate(self, system_prompt: str, user_prompt: str, model: str, **kwargs) -> str:
        model = genai.GenerativeModel(model)
        response = model.generate_content(
            contents=[system_prompt, user_prompt],
            **kwargs
        )
        return response.text

class HuggingFaceProvider(ModelProvider):
    def generate(self, system_prompt: str, user_prompt: str, model: str, **kwargs) -> str:
        client = InferenceClient(token=os.getenv("HF_API_TOKEN"))
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = client.text_generation(
            prompt=full_prompt,
            model=model,
            **kwargs
        )
        return response

class DeepSeekProvider(ModelProvider):
    def generate(self, system_prompt: str, user_prompt: str, model: str, **kwargs) -> str:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **kwargs
        }
        
        response = httpx.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

class OpenRouterProvider(ModelProvider):
    def generate(self, system_prompt: str, user_prompt: str, model: str, **kwargs) -> str:
        api_key = os.getenv("OPENROUTER_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/ModelContext/tool-2025-kg-rag",  # Optional: replace with your project's URL
            "X-Title": "KG RAG Tool"  # Optional: replace with your project name
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **kwargs
        }
        
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

def get_provider(provider_name: str) -> ModelProvider:
    providers: Dict[str, Any] = {
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
        "gemini": GeminiProvider,
        "huggingface": HuggingFaceProvider,
        "deepseek": DeepSeekProvider,
        "openrouter": OpenRouterProvider
    }
    
    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        raise ValueError(f"Unsupported provider: {provider_name}")
    
    return provider_class()
