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
        # Handle ChatPromptTemplate inputs (system and user messages)
        if isinstance(input, dict):
            # Direct dict input - extract model name if provided
            messages = input.get("messages", [])
            model_name = input.get("model", self.model)
        elif hasattr(input, "to_messages"):
            # ChatPromptValue object
            messages = input.to_messages()
            model_name = self.model
        else:
            # Fallback
            messages = []
            model_name = self.model

        system_prompt = ""
        user_prompt = ""

        for message in messages:
            if hasattr(message, "type") and hasattr(message, "content"):
                if message.type == "system":
                    system_prompt = message.content
                elif message.type == "human":
                    user_prompt = message.content

        # If no explicit messages found, try simple string input
        if not system_prompt and not user_prompt:
            if isinstance(input, dict):
                user_prompt = input.get("input", input.get("text", input.get("chunk_text", "")))
            elif hasattr(input, "content"):
                user_prompt = input.content
            elif isinstance(input, str):
                user_prompt = input
            else:
                user_prompt = str(input)

        return self.provider.generate(system_prompt, user_prompt, model_name)

    def __class_getitem__(cls, item):
        return cls
    
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

def get_embedding_model(provider="huggingface"):
    """Get an embedding model from different providers

    Args:
        provider: Provider to use ('huggingface', 'openai', 'vertexai')

    Returns:
        LangChain embedding model
    """
    if provider == "huggingface":
        try:
            # Try to import from the correct package first
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                except ImportError:
                    try:
                        from langchain.embeddings import HuggingFaceEmbeddings
                    except ImportError:
                        print("❌ No compatible HuggingFace embeddings package found")
                        raise

            print("✓ Importing HuggingFaceEmbeddings...")
            embedder = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},  # Use CPU by default for compatibility
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✓ HuggingFaceEmbeddings initialized successfully")
            return embedder
        except Exception as e:
            print(f"❌ HuggingFace initialization failed: {e}")
            raise ImportError("huggingface embeddings not available. Install with: pip install sentence-transformers transformers torch")
    elif provider == "openai":
        from langchain.embeddings import OpenAIEmbeddings
        return OpenAIEmbeddings()
    elif provider == "vertexai":
        from langchain_google_vertexai import VertexAIEmbeddings
        return VertexAIEmbeddings(model_name="textembedding-gecko")
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}. Choose from 'huggingface', 'openai', 'vertexai'")

def get_embedding_method(provider_name=None):
    """Get the configured embedding method from environment

    Args:
        provider_name: Optional provider override ('huggingface', 'openai', 'vertexai')

    Returns:
        tuple: (provider_name, embedding_model)
    """
    if provider_name is None:
        provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")  # Default to huggingface
    else:
        provider = provider_name

    try:
        embedder = get_embedding_model(provider)
        return provider, embedder
    except Exception as e:
        print(f"Failed to initialize {provider} embeddings, falling back to OpenAI: {e}")
        # Fallback to OpenAI if HuggingFace fails
        try:
            embedder = get_embedding_model("openai")
            return "openai", embedder
        except Exception as e2:
            raise RuntimeError(f"Failed to initialize any embedding model. OpenAI error: {e2}")

# alias for compatibility
get_llm_provider = get_provider
