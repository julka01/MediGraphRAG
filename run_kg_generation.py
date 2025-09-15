#!/usr/bin/env python3
import asyncio
import logging
import os
import sys
import types
from dotenv import load_dotenv

# Stub missing langchain modules required by llm-graph-builder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai._enums import HarmBlockThreshold, HarmCategory

# Stub for Vertex AI modules
stub1 = types.ModuleType("langchain_google_vertexai")
stub1.ChatVertexAI = ChatGoogleGenerativeAI
stub1.VertexAIEmbeddings = GoogleGenerativeAIEmbeddings
stub1.HarmBlockThreshold = HarmBlockThreshold
stub1.HarmCategory = HarmCategory
sys.modules["langchain_google_vertexai"] = stub1

# Stub for other experimental modules
stub2 = types.ModuleType("langchain_groq")
stub2.ChatGroq = object
sys.modules["langchain_groq"] = stub2

stub3 = types.ModuleType("langchain_experimental")
stub3.__path__ = []
stub3.graph_transformers = types.ModuleType("langchain_experimental.graph_transformers")
stub3.graph_transformers.__path__ = []
# Stub DiffbotGraphTransformer and LLMGraphTransformer
stub3.graph_transformers.diffbot = types.ModuleType("langchain_experimental.graph_transformers.diffbot")
from langchain_experimental import graph_transformers  # Ensure names available after patch
stub3.graph_transformers.diffbot.DiffbotGraphTransformer = graph_transformers.LLMGraphTransformer = object
sys.modules["langchain_experimental"] = stub3
sys.modules["langchain_experimental.graph_transformers"] = stub3.graph_transformers
sys.modules["langchain_experimental.graph_transformers.diffbot"] = stub3.graph_transformers.diffbot

# Add llm-graph-builder backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "llm-graph-builder", "backend"))
from src.main import extract_graph_from_file_local_file
import src.llm as llm_module

# Override LLM call to skip actual model invocation
async def dummy_get_graph_from_llm(*args, **kwargs):
    return []
llm_module.get_graph_from_llm = dummy_get_graph_from_llm

# Load environment variables
load_dotenv()

# Configure logging to show debug messages
logging.basicConfig(level=logging.DEBUG)

async def main():
    # Neo4j connection settings
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    # LLM model to use
    llm_model_name = 'openai_gpt-3.5-turbo'

    # File to process
    file_path = 'EAU-EANM-ESTRO-ESUR-ISUP-SIOG-Pocket-on-Prostate-Cancer-2025_updated.pdf'
    merged_file_path = os.path.join(os.getcwd(), file_path)
    file_name = os.path.basename(merged_file_path)

    # Invoke llm-graph-builder's extract logic
    uri_latency, response = await extract_graph_from_file_local_file(
        NEO4J_URI,
        NEO4J_USERNAME,
        NEO4J_PASSWORD,
        NEO4J_DATABASE,
        llm_model_name,
        merged_file_path,
        file_name,
        '', '', 1000, 100, 1, None, None
    )

    # Output results
    print("Extraction response:")
    print(response)
    print("\nLatency information:")
    print(uri_latency)

if __name__ == "__main__":
    asyncio.run(main())
