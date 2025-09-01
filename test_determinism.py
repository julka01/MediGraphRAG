import json
import hashlib
from improved_kg_creator import ImprovedKGCreator
import ollama

def test_kg_determinism():
    # Sample medical text for testing
    sample_text = "Patient presents with type 2 diabetes mellitus and hypertension."
    
    # Use a consistent LLM for testing
    def llm(system_prompt, user_prompt):
        response = ollama.chat(
            model='llama3', 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            format='json'
        )
        return response['message']['content']
    
    # Create multiple KG creators with the same seed
    kg_creator1 = ImprovedKGCreator(seed=42)
    kg_creator2 = ImprovedKGCreator(seed=42)
    
    # Prepare a comprehensive system prompt
    system_prompt = kg_creator1.create_enhanced_biomedical_prompt(kg_creator1.biomedical_ontology)
    
    # Generate KGs
    kg1 = kg_creator1.generate_knowledge_graph(
        sample_text, 
        lambda system_prompt, user_prompt: llm(system_prompt, user_prompt), 
        tracking_enabled=False
    )
    kg2 = kg_creator2.generate_knowledge_graph(
        sample_text, 
        lambda system_prompt, user_prompt: llm(system_prompt, user_prompt), 
        tracking_enabled=False
    )
    
    # Compute hashes
    kg1_hash = kg_creator1._compute_kg_hash(kg1)
    kg2_hash = kg_creator2._compute_kg_hash(kg2)
    
    print("KG1 Hash:", kg1_hash)
    print("KG2 Hash:", kg2_hash)
    
    # Detailed comparison
    print("\nKG1 Nodes:", json.dumps(kg1['nodes'], indent=2))
    print("\nKG2 Nodes:", json.dumps(kg2['nodes'], indent=2))
    
    print("\nKG1 Relationships:", json.dumps(kg1['relationships'], indent=2))
    print("\nKG2 Relationships:", json.dumps(kg2['relationships'], indent=2))
    
    # Assert determinism
    assert kg1_hash == kg2_hash, "KG generation is not deterministic"
    assert kg1 == kg2, "KG contents are not identical"
    
    print("Determinism test passed successfully!")

if __name__ == "__main__":
    test_kg_determinism()
