import asyncio
import logging
import os
from ontology_guided_kg_creator import OntologyGuidedKGCreator
from model_providers import get_provider as get_llm_provider
from model_providers import LangChainRunnableAdapter
from pypdf import PdfReader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging to show debug messages
logging.basicConfig(level=logging.DEBUG)

async def main():
    llm_provider_name = 'openai'
    llm_model_name = 'openai_gpt-3.5-turbo'

    # Map LLM provider to embedding model
    embedding_model_map = {
        'openai': 'openai',
        'openrouter': 'huggingface',
        'deepseek': 'huggingface',
        'huggingface': 'huggingface',
        'vertexai': 'vertexai'
    }
    embedding_model = embedding_model_map.get(llm_provider_name, 'openai')

    kg_creator = OntologyGuidedKGCreator(ontology_path='biomedical_ontology.owl', embedding_model=embedding_model)
    provider_instance = get_llm_provider(llm_provider_name, llm_model_name)
    llm = LangChainRunnableAdapter(provider_instance, llm_model_name)
    with open('EAU-EANM-ESTRO-ESUR-ISUP-SIOG-Pocket-on-Prostate-Cancer-2025_updated.pdf', 'rb') as f:
        pdf_reader = PdfReader(f)
        text = ''.join(page.extract_text() or '' for page in pdf_reader.pages)
    kg = kg_creator.generate_knowledge_graph(text, llm)
    print("Sample node names from generated KG:")
    for node in kg['nodes'][:10]:
        print(node['id'])

if __name__ == "__main__":
    asyncio.run(main())
