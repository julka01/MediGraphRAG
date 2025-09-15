"""
Shim module to provide compatibility for code importing from langchain_google_vertexai.
Redirects imports to langchain_google_genai equivalents.
"""

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai._enums import HarmBlockThreshold, HarmCategory

# Expose under expected names
ChatVertexAI = ChatGoogleGenerativeAI
VertexAIEmbeddings = GoogleGenerativeAIEmbeddings

__all__ = [
    "ChatVertexAI",
    "VertexAIEmbeddings",
    "HarmBlockThreshold",
    "HarmCategory",
]
