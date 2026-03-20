import hashlib
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import TransientError
from langchain_neo4j import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument
from typing import List
import re
import os
import time
from pathlib import Path
from urllib.parse import urlparse
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_embedding_model(embedding_model_name: str):
    """
    Load embedding model based on the model name
    Returns embeddings object and dimension
    """
    if embedding_model_name == "openai":
        embeddings = OpenAIEmbeddings()
        dimension = 1536
        logging.info(f"Embedding: Using OpenAI Embeddings , Dimension:{dimension}")
    elif embedding_model_name == "vertexai":
        embeddings = VertexAIEmbeddings(
            model="textembedding-gecko@003"
        )
        dimension = 768
        logging.info(f"Embedding: Using Vertex AI Embeddings , Dimension:{dimension}")
    elif embedding_model_name == "titan":
        embeddings = get_bedrock_embeddings()
        dimension = 1536
        logging.info(f"Embedding: Using bedrock titan Embeddings , Dimension:{dimension}")
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"#, cache_folder="/embedding_model"
        )
        dimension = 384
        logging.info(f"Embedding: Using Langchain HuggingFaceEmbeddings , Dimension:{dimension}")
    return embeddings, dimension

def create_graph_database_connection(uri, userName, password, database):
    """
    Create Neo4j graph database connection
    """
    enable_user_agent = os.environ.get("ENABLE_USER_AGENT", "False").lower() in ("true", "1", "yes")

    driver_config = {}
    if enable_user_agent:
        driver_config['user_agent'] = os.environ.get('NEO4J_USER_AGENT')

    # LangChain Neo4jGraph with direct username/password parameters
    graph = Neo4jGraph(url=uri, database=database, username=userName, password=password,
                       refresh_schema=False, sanitize=True, driver_config=driver_config)
    return graph

def delete_uploaded_local_file(merged_file_path, file_name):
    """
    Delete uploaded local file
    """
    file_path = Path(merged_file_path)
    if file_path.exists():
        file_path.unlink()
        logging.info(f'file {file_name} deleted successfully')

def create_gcs_bucket_folder_name_hashed(uri, file_name):
    """
    Create GCS bucket folder name with hash
    """
    folder_name = uri + file_name
    folder_name_sha1 = hashlib.sha1(folder_name.encode())
    folder_name_sha1_hashed = folder_name_sha1.hexdigest()
    return folder_name_sha1_hashed

def get_bedrock_embeddings():
    """
    Creates and returns a BedrockEmbeddings object using the specified model name.
    Args:
        model (str): The name of the model to use for embeddings.
    Returns:
        BedrockEmbeddings: An instance of the BedrockEmbeddings class.
    """
    try:
        env_value = os.getenv("BEDROCK_EMBEDDING_MODEL")
        if not env_value:
            raise ValueError("Environment variable 'BEDROCK_EMBEDDING_MODEL' is not set.")
        try:
            model_name, aws_access_key, aws_secret_key, region_name = env_value.split(",")
        except ValueError:
            raise ValueError(
                "Environment variable 'BEDROCK_EMBEDDING_MODEL' is improperly formatted. "
                "Expected format: 'model_name,aws_access_key,aws_secret_key,region_name'."
            )
        bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=region_name.strip(),
                aws_access_key_id=aws_access_key.strip(),
                aws_secret_access_key=aws_secret_key.strip(),
            )
        bedrock_embeddings = BedrockEmbeddings(
            model_id=model_name.strip(),
            client=bedrock_client
        )
        return bedrock_embeddings
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
