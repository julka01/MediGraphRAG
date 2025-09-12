from langchain_community.embeddings import OpenAIEmbeddings

def load_embedding_model(embedding_model_name: str):
    """
    Load the embedding model based on the given name.
    Supports 'openai' and 'sentence_transformers'.
    Returns a tuple of (embedding_function, embedding_dimension)
    """
    if embedding_model_name.lower() == "openai":
        import os
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embeddings")
        embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
        embedding_dimension = 1536  # typical dimension for OpenAI embeddings
    elif embedding_model_name.lower() == "sentence_transformers":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            embedding_dimension = 384  # dimension for all-MiniLM-L6-v2
        except ImportError:
            raise ValueError("sentence_transformers requires langchain_community with HuggingFaceEmbeddings. Install with: pip install langchain-community")
    else:
        raise ValueError(f"Unsupported embedding model: {embedding_model_name}. Supported: 'openai', 'sentence_transformers'")
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
