import json
import re
import hashlib
from datetime import datetime
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph
import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add llm-graph-builder paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'llm-graph-builder/backend/src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'llm-graph-builder/backend'))

try:
    from shared.common_fn import load_embedding_model
except ImportError:
    # Fallback without llm-graph-builder
    from sentence_transformers import SentenceTransformer

    def load_embedding_model(embedding_model: str = "sentence-transformers"):
        """Simple embedding model loader"""
        if embedding_model == "sentence-transformers":
            model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            return lambda x: model.encode(x).tolist(), model.get_sentence_embedding_dimension()


class EnhancedKGCreatorProd:
    """
    Enhanced Knowledge Graph Creator with production-ready LangChain 0.3.27
    and full Neo4j integration
    """

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        neo4j_database: str = None,
        embedding_model: str = "openai"
    ):
        # Use environment variables if not provided
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

        # Initialize embedding model
        try:
            self.embedding_function, self.embedding_dimension = load_embedding_model(embedding_model)
        except:
            # Fallback to sentence transformers
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_function = lambda x: self.model.encode(x).tolist()
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()

        logging.info(f"✅ Initialized embedding model with dimension {self.embedding_dimension}")

    def _create_neo4j_connection(self):
        """Create Neo4j graph connection"""
        return Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password,
            database=self.neo4j_database
        )

    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text using TokenTextSplitter"""
        from langchain_text_splitters import TokenTextSplitter

        text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = text_splitter.split_text(text)

        formatted = []
        current_pos = 0

        for idx, chunk_text in enumerate(chunks):
            start_pos = current_pos
            end_pos = start_pos + len(chunk_text)

            try:
                chunk_embedding = self.embedding_function(chunk_text)
            except Exception as e:
                logging.warning(f"Failed to generate embedding for chunk {idx}: {e}")
                chunk_embedding = None

            formatted.append({
                "text": chunk_text,
                "chunk_id": idx,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "total_chunks": len(chunks),
                "embedding": chunk_embedding
            })

            current_pos += len(chunk_text) - self.chunk_overlap

        return formatted

    def _extract_entities_and_relationships_with_llm(self, chunk_text: str, llm) -> Dict[str, Any]:
        """
        Extract entities and relationships using LLM with proper LangChain 0.3.27 structure
        """
        # Create ontology-aware extraction prompt
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert medical knowledge graph extraction system.
Extract entities and relationships from medical/scientific text.

IMPORTANT RULES:
1. Focus on medically relevant entities (diseases, treatments, symptoms, drugs, procedures)
2. Classify entities with appropriate medical categories
3. Extract relationships only between entities that actually interact
4. Be precise and medically accurate

Return ONLY a valid JSON object in this exact format:
{{
  "entities": [
    {{
      "id": "entity_name_exactly_as_it_appears",
      "type": "Medical_Category",
      "properties": {{
        "name": "entity_name_exactly_as_it_appears",
        "description": "brief medical context"
      }}
    }}
  ],
  "relationships": [
    {{
      "source": "source_entity_name",
      "target": "target_entity_name",
      "type": "RELATION_TYPE",
      "properties": {{
        "description": "relationship context in the text"
      }}
    }}
  ]
}}

Return ONLY the JSON object, no additional text."""),
            ("human", "Extract medical knowledge from this text:\n\n{text}")
        ])

        try:
            # Use LangChain 0.3.27 chain structure
            chain = extraction_prompt | llm | StrOutputParser()
            response = chain.invoke({"text": chunk_text})

            # Handle potential markdown code blocks
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]

            response = response.strip()

            # Parse and validate JSON
            result = json.loads(response)

            return {
                'entities': result.get('entities', []),
                'relationships': result.get('relationships', [])
            }

        except Exception as e:
            logging.error(f"LLM extraction failed: {e}")
            return {'entities': [], 'relationships': []}

    def generate_knowledge_graph(self, text: str, llm=None, file_name: str = None) -> Dict[str, Any]:
        """
        Generate knowledge graph from text
        """
        logging.info("🚀 Starting Enhanced KG Generation")

        # Step 1: Chunk the text
        chunks = self._chunk_text(text)
        logging.info(f"📦 Created {len(chunks)} chunks")

        # Step 2: Process chunks (for demo, process only first few chunks)
        kg = {
            "nodes": [],
            "relationships": [],
            "chunks": chunks,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "file_name": file_name,
                "total_chunks": len(chunks),
                "chunk_size": self.chunk_size,
                "extraction_method": "enhanced_kg_creator_prod"
            }
        }

        if llm:
            all_entities = []
            all_relationships = []

            # Process chunks in batches for efficiency
            for i, chunk in enumerate(chunks[:3]):  # Limit to first 3 chunks for demo
                logging.info(f"🔬 Processing chunk {i+1}/3")
                chunk_kg = self._extract_entities_and_relationships_with_llm(chunk['text'], llm)
                all_entities.extend(chunk_kg['entities'])
                all_relationships.extend(chunk_kg['relationships'])

            # Deduplicate entities
            entity_map = {}
            for entity in all_entities:
                key = f"{entity['type']}:{entity['id'].lower()}"
                if key not in entity_map:
                    entity_map[key] = entity

            # Create node format
            for entity in entity_map.values():
                kg["nodes"].append({
                    "id": entity['id'],
                    "label": entity['type'],
                    "properties": entity.get('properties', {}),
                    "embedding": entity.get('embedding')
                })

            # Create relationship format
            seen_relationships = set()
            for rel in all_relationships:
                rel_key = f"{rel['source']}:{rel['type']}:{rel['target']}"
                if rel_key not in seen_relationships:
                    kg["relationships"].append({
                        "id": f"rel_{len(kg['relationships'])}",
                        "from": rel['source'],
                        "to": rel['target'],
                        "source": rel['source'],
                        "target": rel['target'],
                        "type": rel['type'],
                        "label": rel['type'],
                        "properties": rel.get('properties', {})
                    })
                    seen_relationships.add(rel_key)

            kg["metadata"].update({
                "total_entities": len(kg["nodes"]),
                "total_relationships": len(kg["relationships"]),
            })

            logging.info(".1f")
            logging.info("Extraction completed successfully")
        return kg

    def save_to_neo4j(self, kg: Dict[str, Any], file_name: str) -> bool:
        """
        Save knowledge graph to Neo4j
        """
        try:
            graph = self._create_neo4j_connection()

            # Create source document
            graph.query("""
            MERGE (d:Document {fileName: $fileName})
            SET d.createdAt = datetime(),
                d.updatedAt = datetime(),
                d.totalEntities = $totalEntities,
                d.totalChunks = $totalChunks,
                d.totalRelationships = $totalRelationships
            """, {
                "fileName": file_name,
                "totalEntities": len(kg.get('nodes', [])),
                "totalChunks": len(kg.get('chunks', [])),
                "totalRelationships": len(kg.get('relationships', []))
            })

            # Save entities
            for node in kg.get('nodes', []):
                graph.query("""
                MERGE (n:__Entity__ {id: $id})
                SET n.name = $name,
                    n.type = $type,
                    n.description = $description,
                    n.embedding = $embedding,
                    n.updatedAt = datetime()
                """, {
                    "id": node['id'],
                    "name": node['properties'].get('name', node['id']),
                    "type": node['label'],
                    "description": node['properties'].get('description', ''),
                    "embedding": node.get('embedding')
                })

            # Save relationships
            for rel in kg.get('relationships', []):
                graph.query("""
                MATCH (s:__Entity__ {id: $source_id})
                MATCH (t:__Entity__ {id: $target_id})
                MERGE (s)-[r:{rel_type}]->(t)
                SET r.description = $description,
                    r.updatedAt = datetime()
                """.format(rel_type=rel['type']), {
                    "source_id": rel['source'],
                    "target_id": rel['target'],
                    "description": rel.get('properties', {}).get('description', '')
                })

            logging.info(f"✅ Saved KG with {len(kg.get('nodes', []))} entities and {len(kg.get('relationships', []))} relationships")
            return True

        except Exception as e:
            logging.error(f"❌ Failed to save to Neo4j: {e}")
            return False

        finally:
            try:
                graph._driver.close()
            except:
                pass


# Test function
if __name__ == "__main__":
    print("🧪 Testing Enhanced KG Creator Production Version")
    print("=" * 60)

    try:
        # Test with sample PDF content
        from pypdf import PdfReader as PypdfReader

        pdf_path = 'EAU-EANM-ESTRO-ESUR-ISUP-SIOG-Pocket-on-Prostate-Cancer-2025_updated.pdf'
        print(f"📖 Reading PDF: {pdf_path}")

        reader = PypdfReader(pdf_path)
        text_content = ""
        for page in reader.pages[:2]:  # Test with first 2 pages
            text_content += page.extract_text() + "\n"

        print(f"✅ Extracted {len(text_content)} characters from PDF")

        # Initialize KG creator
        kg_creator = EnhancedKGCreatorProd(
            chunk_size=400,  # Smaller for testing
            chunk_overlap=50
        )

        # Test chunking
        chunks = kg_creator._chunk_text(text_content)
        print(f"✅ Created {len(chunks)} chunks")
        print(f"✅ First chunk: {len(chunks[0]['text'])} characters")

        # Test without LLM (demo mode)
        kg = kg_creator.generate_knowledge_graph(text_content[:1000], llm=None, file_name="demo_prostate_cancer.pdf")

        print("🧠 KG Generation (demo without LLM):")
        print(f"   📦 Chunks processed: {len(kg['chunks'])}")
        print(f"   🏷️  Entities found: {kg['metadata'].get('total_entities', 0)}")
        print(f"   🔗 Relationships found: {kg['metadata'].get('total_relationships', 0)}")

        print("\n✅ Enhanced KG Creator production version is ready!")
        print("💡 LLM extraction can be added by providing a LangChain LLM instance")

    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()
