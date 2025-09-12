import os
import json
import httpx
from abc import ABC, abstractmethod
from openai import OpenAI as OpenAIClient
import google.generativeai as genai
from huggingface_hub import InferenceClient
import ollama
from typing import Dict, Any

from langchain_core.runnables.base import Runnable

class ModelProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, model: str, **kwargs) -> str:
        pass

class LangChainRunnableAdapter(Runnable):
    def __init__(self, provider: ModelProvider, model: str):
        self.provider = provider
        self.model = model

    def invoke(self, input, config=None) -> str:
        # input may be a ChatPromptValue object, access text attribute
        system_prompt = ""
        user_prompt = ""
        if hasattr(input, "get"):
            system_prompt = input.get("system_prompt", "")
            user_prompt = input.get("text", "")
        else:
            # Try to access attributes for ChatPromptValue
            system_prompt = getattr(input, "system_prompt", "")
            user_prompt = getattr(input, "text", "")
            if not user_prompt and hasattr(input, "text"):
                user_prompt = input.text
        return self.provider.generate(system_prompt, user_prompt, self.model)
    
    def with_structured_output(self, schema, **kwargs):
        """Add structured output support for LLMGraphTransformer compatibility"""
        return self
    
    def bind(self, **kwargs):
        """Add bind method for LangChain compatibility"""
        return self

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
    
    def with_structured_output(self, schema, **kwargs):
        """Add structured output support for LLMGraphTransformer compatibility"""
        return self
    
    def bind(self, **kwargs):
        """Add bind method for LangChain compatibility"""
        return self

def get_provider(provider: str, model: str = None, **kwargs) -> ModelProvider:
    providers: Dict[str, Any] = {
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
        "gemini": GeminiProvider,
        "huggingface": HuggingFaceProvider,
        "deepseek": DeepSeekProvider,
        "openrouter": OpenRouterProvider
    }
    
    provider_class = providers.get(provider.lower())
    if not provider_class:
        raise ValueError(f"Unsupported provider: {provider}")
    
    return provider_class()

# alias for compatibility
get_llm_provider = get_provider

# alias for compatibility
get_llm_provider = get_provider
