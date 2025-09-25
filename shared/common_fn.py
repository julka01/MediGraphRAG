import os
import sys

# Add parent directory to path to access model_providers
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def load_embedding_model(embedding_model_name: str = None):
    """
    Load the embedding model based on the given name.
    If no name provided, uses environment variable EMBEDDING_PROVIDER.
    Returns a tuple of (embedding_function, embedding_dimension)
    """
    from model_providers import get_embedding_method

    provider_name, embedding_function = get_embedding_method(embedding_model_name)

    # Determine dimension based on provider
    if provider_name == "openai":
        embedding_dimension = 1536  # OpenAI ada-002 dimension
    elif provider_name == "huggingface":
        embedding_dimension = 384  # all-MiniLM-L6-v2 dimension
    elif provider_name == "vertexai":
        embedding_dimension = 768  # VertexAI textembedding-gecko dimension
    else:
        raise ValueError(f"Unknown embedding provider: {provider_name}")

    print(f"✓ Using {provider_name} embeddings with dimension {embedding_dimension}")
    return embedding_function, embedding_dimension

def wrap_llm_with_model_name(llm):
    '''
    Wrap the given LLM with a model name attribute or wrapper.
    This is a placeholder implementation that adds a "model_name" attribute
    to the LLM instance if it doesn't already have one.
    '''
    if not hasattr(llm, 'model_name'):
        model_name = getattr(llm, 'model', 'unknown_model')
        if not isinstance(model_name, str):
            model_name = str(model_name)
        setattr(llm, 'model_name', model_name)
    return llm

def _add_graph_documents_with_merge(graph, graph_document_list):
    """
    Custom implementation of add_graph_documents using MERGE instead of CREATE
    to handle duplicate IDs gracefully
    """
    for doc in graph_document_list:
        # Process nodes
        for node in doc.nodes:
            # Use MERGE to create or update nodes
            node_query = f"""
            MERGE (n:{node.type} {{id: $id}})
            SET n += $properties
            """
            properties = {}
            if hasattr(node, 'properties') and node.properties:
                properties.update(node.properties)
            # Ensure id is always set
            properties["id"] = node.id

            graph.query(node_query, {
                "id": node.id,
                "properties": properties
            })

        # Process relationships
        for rel in doc.relationships:
            # Use MERGE to create or update relationships
            # Sanitize relationship type to replace spaces with underscores for valid Cypher
            sanitized_rel_type = rel.type.replace(' ', '_').replace('-', '_').upper()

            rel_query = f"""
            MATCH (source {{id: $source_id}})
            MATCH (target {{id: $target_id}})
            MERGE (source)-[r:{sanitized_rel_type}]->(target)
            SET r += $properties
            """
            properties = {}
            if hasattr(rel, 'properties') and rel.properties:
                properties.update(rel.properties)

            graph.query(rel_query, {
                "source_id": rel.source.id,
                "target_id": rel.target.id,
                "properties": properties
            })
