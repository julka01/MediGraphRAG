"""
Analyze hallucination detection correlation with correctness.

This script analyzes existing PubMedQA results to compute AUROC
and correlation between hallucination scores and correctness.
"""

import json
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def load_results():
    """Load existing results."""
    with open('results/mirage_pubmedqa_results.json', 'r') as f:
        return json.load(f)

def analyze_results():
    """Analyze hallucination scores vs correctness."""
    results = load_results()
    
    print("=" * 80)
    print("HALLUCINATION DETECTION ANALYSIS")
    print("=" * 80)
    
    # Extract details
    details = results['config_results'][0]['details']
    
    print(f"\nTotal questions: {len(details)}")
    
    # Collect data for analysis
    metrics_to_analyze = [
        'hallucination_score',
        'semantic_entropy',
        'semantic_entropy_nli', 
        'semantic_entropy_confidence',
        'semantic_entropy_combined',
        'context_consistency',
        'lexical_hallucination',
        'semantic_hallucination',
        'cluster_entropy'
    ]
    
    # Separate by vanilla vs KG
    for system in ['vanilla', 'kg']:
        print(f"\n{'='*40}")
        print(f"SYSTEM: {system.upper()}")
        print(f"{'='*40}")
        
        # Binary correctness
        is_correct = [d[f'{system}_correct'] for d in details]
        is_incorrect = [1 - int(c) for c in is_correct]  # 1 = incorrect/hallucination
        
        print(f"Correct: {sum(is_correct)}/{len(is_correct)} ({100*sum(is_correct)/len(is_correct):.1f}%)")
        
        # Analyze each metric
        for metric in metrics_to_analyze:
            key = f'{system}_{metric}'
            scores = []
            for d in details:
                val = d.get(key, 0)
                # Handle negative values (some metrics have -0.0)
                scores.append(max(0, val))
            
            scores = np.array(scores)
            
            # Compute metrics
            if len(set(is_incorrect)) > 1:  # Need both classes for AUROC
                try:
                    auroc = roc_auc_score(is_incorrect, scores)
                except:
                    auroc = 0.5
            else:
                auroc = "N/A (only one class)"
            
            # Correlation
            correlation = np.corrcoef(is_incorrect, scores)[0, 1]
            
            # Mean scores: correct vs incorrect
            correct_scores = [scores[i] for i in range(len(scores)) if is_correct[i]]
            incorrect_scores = [scores[i] for i in range(len(scores)) if not is_correct[i]]
            
            mean_correct = np.mean(correct_scores) if correct_scores else 0
            mean_incorrect = np.mean(incorrect_scores) if incorrect_scores else 0
            
            print(f"\n{metric}:")
            print(f"  AUROC: {auroc}")
            print(f"  Correlation: {correlation:.3f}")
            print(f"  Mean (correct): {mean_correct:.3f}")
            print(f"  Mean (incorrect): {mean_incorrect:.3f}")
            if mean_incorrect > mean_correct:
                print(f"  ✓ Higher score for incorrect (good!)")
            else:
                print(f"  ✗ Lower score for incorrect (bad!)")
    
    # Compare vanilla vs KG hallucination
    print(f"\n{'='*40}")
    print("VANILLA vs KG COMPARISON")
    print(f"{'='*40}")
    
    for i, d in enumerate(details):
        print(f"\nQ{i+1}: {d['question'][:60]}...")
        print(f"  Expected: {d['expected']}")
        print(f"  Vanilla: {d['vanilla_correct']} (score: {d.get('vanilla_hallucination_score', 0):.3f})")
        print(f"  KG: {d['kg_correct']} (score: {d.get('kg_hallucination_score', 0):.3f})")
    
    # Key insight
    print(f"\n{'='*80}")
    print("KEY INSIGHT")
    print(f"{'='*80}")
    print("""
The problem: Most questions have LOW hallucination scores (near 0) regardless of 
whether they're correct or incorrect. The metrics are not differentiating.

This suggests:
1. Semantic entropy requires MORE samples (3 is too few)
2. The entropy thresholds need tuning
3. Context consistency might be more important than semantic entropy

Let's test with more samples and different weights!
""")

if __name__ == '__main__':
    analyze_results()
