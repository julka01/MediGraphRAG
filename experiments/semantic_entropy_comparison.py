"""
Semantic Entropy Comparison Metrics

This module implements multiple variants of semantic entropy for hallucination detection
in RAG systems, allowing for comparison between different clustering approaches.

Key metrics:
1. semantic_entropy_nli - Original NLI-based clustering (Frol et al., 2024)
2. semantic_entropy_embedding - SBERT cosine similarity clustering  
3. semantic_entropy_context_weighted - Embedding + RAG context weighting
4. rag_hallucination_score - Novel RAG-specific hallucination detection

AUROC evaluation for measuring correlation with correctness.

Scientific contributions:
- First comparison of NLI vs embedding clustering for semantic entropy
- Novel RAG-specific hallucination metric leveraging retrieved context
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional, Callable
from collections import Counter
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Embedding-based methods will use fallback.")

from nltk.tokenize import word_tokenize

# Import from existing rag_metrics
from experiments.rag_metrics import (
    shannon_entropy, 
    weighted_shannon_entropy, 
    compute_nli_entailment,
    compute_lexical_entailment,
    cluster_assignment_entropy,
    predictive_entropy_mc,
    mutual_information
)

logger = logging.getLogger(__name__)

# =============================================================================
# NLI-BASED SEMANTIC CLUSTERING (Original Method from Frol et al.)
# =============================================================================

def nli_based_clustering(
    responses: List[str],
    nli_model: str = "roberta-large-mnli",
    strict_entailment: bool = False,
    example: Optional[Dict] = None
) -> Tuple[List[int], Dict[int, List[str]]]:
    """
    Original NLI-based clustering from Frol et al. (2024).
    
    Uses bidirectional NLI entailment to determine semantic equivalence:
    - Response A and B are semantically equivalent if A→B AND B→A are both entailment/neutral
    - Uses strict mode: both must be "entailment"
    
    This is O(n²) in NLI calls - expensive but theoretically grounded.
    
    Args:
        responses: List of text responses to cluster
        nli_model: HuggingFace NLI model name
        strict_entailment: If True, both directions must be "entailment"
                         If False, one "neutral" is allowed
        example: Example with question for LLM-based NLI
        
    Returns:
        Tuple of (cluster_ids, cluster_to_responses)
    """
    if not responses:
        return [], {}
    
    n = len(responses)
    
    # Initialize all ids with -1
    semantic_set_ids = [-1] * n
    next_id = 0
    
    def check_equivalence(text1: str, text2: str) -> bool:
        """Check if two responses are semantically equivalent using NLI."""
        # Get entailment scores in both directions
        score1 = compute_nli_entailment(text1, text2, nli_model)
        score2 = compute_nli_entailment(text2, text1, nli_model)
        
        if strict_entailment:
            # Both must be high entailment
            return score1 > 0.7 and score2 > 0.7
        else:
            # Allow neutral in at least one direction (original paper logic)
            # Both must NOT be contradiction
            return not (score1 < 0.3 or score2 < 0.3)
    
    # Greedy clustering: O(n²) comparisons
    for i in range(n):
        if semantic_set_ids[i] != -1:
            continue
            
        # Assign new cluster ID
        semantic_set_ids[i] = next_id
        
        # Find all equivalent responses
        for j in range(i + 1, n):
            if semantic_set_ids[j] == -1:
                if check_equivalence(responses[i], responses[j]):
                    semantic_set_ids[j] = next_id
        
        next_id += 1
    
    # Build cluster mapping
    cluster_to_responses = {}
    for idx, cid in enumerate(semantic_set_ids):
        if cid not in cluster_to_responses:
            cluster_to_responses[cid] = []
        cluster_to_responses[cid].append(responses[idx])
    
    return semantic_set_ids, cluster_to_responses


# =============================================================================
# EMBEDDING-BASED SEMANTIC CLUSTERING (Faster Alternative)
# =============================================================================

def embedding_based_clustering(
    responses: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    threshold: float = 0.75
) -> Tuple[List[int], Dict[int, List[str]]]:
    """
    Embedding-based semantic clustering using SBERT cosine similarity.
    
    Advantages over NLI:
    - O(n) for embedding generation + O(n²) for similarity matrix
    - Continuous similarity scores instead of categorical
    - Tunable threshold
    - Much faster (local model vs API calls)
    
    Args:
        responses: List of text responses to cluster
        model_name: Sentence transformer model
        threshold: Cosine similarity threshold for same cluster
        
    Returns:
        Tuple of (cluster_ids, cluster_to_responses)
    """
    if not responses:
        return [], {}
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        # Fallback: each response is its own cluster
        return list(range(len(responses))), {i: [r] for i, r in enumerate(responses)}
    
    try:
        # Encode all responses
        model = SentenceTransformer(model_name)
        embeddings = model.encode(responses)
        
        # Compute cosine similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings / norms
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Connected components clustering
        cluster_ids = [-1] * len(responses)
        next_cluster_id = 0
        
        for i in range(len(responses)):
            if cluster_ids[i] != -1:
                continue
            
            # BFS to find all similar responses
            queue = [i]
            cluster_ids[i] = next_cluster_id
            
            while queue:
                current = queue.pop(0)
                for j in range(len(responses)):
                    if cluster_ids[j] == -1 and similarity_matrix[current, j] >= threshold:
                        cluster_ids[j] = next_cluster_id
                        queue.append(j)
            
            next_cluster_id += 1
        
    except Exception as e:
        logger.warning(f"Embedding clustering failed: {e}. Using fallback.")
        return list(range(len(responses))), {i: [r] for i, r in enumerate(responses)}
    
    # Build cluster mapping
    cluster_to_responses = {}
    for idx, cid in enumerate(cluster_ids):
        if cid not in cluster_to_responses:
            cluster_to_responses[cid] = []
        cluster_to_responses[cid].append(responses[idx])
    
    return cluster_ids, cluster_to_responses


# =============================================================================
# CONTEXT-WEIGHTED SEMANTIC CLUSTERING (RAG-Specific)
# =============================================================================

def context_weighted_clustering(
    responses: List[str],
    retrieved_chunks: List[str],
    embedding_threshold: float = 0.75,
    context_weight: float = 0.3
) -> Tuple[List[int], Dict[int, List[str]], Dict[int, float]]:
    """
    Embedding-based clustering with context-aware weighting.
    
    Novel contribution: Weight semantic clusters by their entailment with
    retrieved RAG context. Clusters better supported by context get higher weight.
    
    Args:
        responses: List of text responses to cluster
        retrieved_chunks: Retrieved context chunks from RAG
        embedding_threshold: Threshold for embedding similarity
        context_weight: Weight for context alignment in final score
        
    Returns:
        Tuple of (cluster_ids, cluster_to_responses, cluster_context_scores)
    """
    if not responses or not retrieved_chunks:
        # Fallback to regular embedding clustering
        cluster_ids, cluster_map = embedding_based_clustering(responses)
        context_scores = {cid: 0.5 for cid in cluster_map.keys()}
        return cluster_ids, cluster_map, context_scores
    
    # First, cluster using embeddings
    cluster_ids, cluster_map = embedding_based_clustering(
        responses, threshold=embedding_threshold
    )
    
    # Then, compute context entailment for each cluster
    context_scores = {}
    for cluster_id, cluster_responses in cluster_map.items():
        # Use representative (most common) response
        rep_response = Counter(cluster_responses).most_common(1)[0][0]
        
        # Compute entailment with each retrieved chunk
        entailment_scores = []
        for chunk in retrieved_chunks:
            score = compute_nli_entailment(chunk, rep_response)
            entailment_scores.append(score)
        
        # Average entailment across chunks
        context_scores[cluster_id] = np.mean(entailment_scores) if entailment_scores else 0.5
    
    return cluster_ids, cluster_map, context_scores


# =============================================================================
# CONTRADICTION DETECTION (Novel Contribution)
# =============================================================================

def compute_contradiction_score(
    responses: List[str],
    cluster_ids: Optional[List[int]] = None,
    nli_model: str = "roberta-large-mnli",
    threshold: float = 0.3
) -> Dict[str, float]:
    """
    Detect contradictions between semantic clusters.
    
    If multiple clusters contain contradictory claims, that's a stronger
    signal of hallucination than just high entropy.
    
    For example:
    - Cluster 1: "Aspirin treats pain"
    - Cluster 2: "Aspirin causes pain" (contradiction!)
    
    Args:
        responses: List of model-generated responses
        cluster_ids: Pre-computed cluster IDs (optional)
        nli_model: NLI model for contradiction detection
        threshold: NLI score below which is considered contradiction
        
    Returns:
        Dictionary with contradiction metrics
    """
    if not responses or len(responses) < 2:
        return {
            "contradiction_score": 0.0,
            "num_contradicting_pairs": 0,
            "total_pairs": 0,
            "contradiction_ratio": 0.0
        }
    
    # If no cluster IDs provided, use embedding clustering
    if cluster_ids is None:
        cluster_ids, cluster_map = embedding_based_clustering(responses)
    else:
        # Build cluster map from IDs
        cluster_map = {}
        for idx, cid in enumerate(cluster_ids):
            if cid not in cluster_map:
                cluster_map[cid] = []
            cluster_map[cid].append(responses[idx])
    
    cluster_ids_list = list(cluster_map.keys())
    n_clusters = len(cluster_ids_list)
    
    if n_clusters < 2:
        # No contradictions possible with only 1 cluster
        return {
            "contradiction_score": 0.0,
            "num_contradicting_pairs": 0,
            "total_pairs": n_clusters * (n_clusters - 1) // 2,
            "contradiction_ratio": 0.0
        }
    
    # Get representative response from each cluster
    cluster_representatives = {}
    for cid, resp_list in cluster_map.items():
        # Use most common response as representative
        cluster_representatives[cid] = Counter(resp_list).most_common(1)[0][0]
    
    # Check all pairs for contradictions
    num_contradictions = 0
    total_pairs = 0
    
    for i, cid1 in enumerate(cluster_ids_list):
        for cid2 in cluster_ids_list[i+1:]:
            total_pairs += 1
            
            rep1 = cluster_representatives[cid1]
            rep2 = cluster_representatives[cid2]
            
            # Check both directions for contradiction
            # NLI model returns [contradiction, neutral, entailment]
            # We want the contradiction score
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch
                
                if not hasattr(compute_contradiction_score, '_nli_model_cache'):
                    compute_contradiction_score._nli_model_cache = {}
                
                if nli_model not in compute_contradiction_score._nli_model_cache:
                    compute_contradiction_score._nli_model_cache[nli_model] = (
                        AutoModelForSequenceClassification.from_pretrained(nli_model),
                        AutoTokenizer.from_pretrained(nli_model)
                    )
                
                model, tokenizer = compute_contradiction_score._nli_model_cache[nli_model]
                
                # Check rep1 -> rep2
                inputs = tokenizer(rep1, rep2, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    # Index 0 = contradiction
                    score1 = probs[0, 0].item()
                
                # Check rep2 -> rep1
                inputs = tokenizer(rep2, rep1, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    score2 = probs[0, 0].item()
                
                # If either direction shows contradiction
                if score1 > threshold or score2 > threshold:
                    num_contradictions += 1
                    
            except Exception as e:
                # Fallback: use our NLI entailment function inversely
                # High entailment one way but not the other suggests contradiction
                entail_1to2 = compute_nli_entailment(rep1, rep2, nli_model)
                entail_2to1 = compute_nli_entailment(rep2, rep1, nli_model)
                
                # If one direction has high entailment but not the other, possible contradiction
                if (entail_1to2 > 0.7 and entail_2to1 < 0.3) or (entail_2to1 > 0.7 and entail_1to2 < 0.3):
                    # This is more like inconsistency than contradiction, weight lower
                    num_contradictions += 0.5
    
    contradiction_ratio = num_contradictions / total_pairs if total_pairs > 0 else 0.0
    
    return {
        "contradiction_score": contradiction_ratio,
        "num_contradicting_pairs": num_contradictions,
        "total_pairs": total_pairs,
        "contradiction_ratio": contradiction_ratio,
        "n_clusters": n_clusters
    }


# =============================================================================
# SEMANTIC ENTROPY COMPUTATION (All Variants)
# =============================================================================

def compute_semantic_entropy_nli(
    responses: List[str],
    strict_entailment: bool = False,
    nli_model: str = "roberta-large-mnli"
) -> Dict[str, float]:
    """
    Compute semantic entropy using NLI-based clustering.
    
    Original method from Frol et al. (2024).
    
    Args:
        responses: List of model-generated responses
        strict_entailment: Whether to require strict entailment
        nli_model: NLI model to use
        
    Returns:
        Dictionary with entropy values
    """
    if len(responses) < 2:
        return {
            "semantic_entropy_nli": 0.0,
            "num_clusters_nli": 1,
            "cluster_entropy_nli": 0.0
        }
    
    # Cluster using NLI
    semantic_ids, cluster_map = nli_based_clustering(
        responses, 
        nli_model=nli_model,
        strict_entailment=strict_entailment
    )
    
    # Compute entropy over clusters
    unique_ids = sorted(set(semantic_ids))
    counts = np.array([semantic_ids.count(uid) for uid in unique_ids])
    probabilities = counts / len(responses)
    
    entropy = shannon_entropy(probabilities)
    cluster_entropy = cluster_assignment_entropy(semantic_ids)
    
    return {
        "semantic_entropy_nli": entropy,
        "num_clusters_nli": len(unique_ids),
        "cluster_entropy_nli": cluster_entropy
    }


def compute_semantic_entropy_embedding(
    responses: List[str],
    threshold: float = 0.75,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Dict[str, float]:
    """
    Compute semantic entropy using embedding-based clustering.
    
    Faster alternative to NLI-based clustering.
    
    Args:
        responses: List of model-generated responses
        threshold: Cosine similarity threshold for clustering
        model_name: Sentence transformer model
        
    Returns:
        Dictionary with entropy values
    """
    if len(responses) < 2:
        return {
            "semantic_entropy_embedding": 0.0,
            "num_clusters_embedding": 1,
            "cluster_entropy_embedding": 0.0
        }
    
    # Cluster using embeddings
    semantic_ids, cluster_map = embedding_based_clustering(
        responses,
        model_name=model_name,
        threshold=threshold
    )
    
    # Compute entropy over clusters
    unique_ids = sorted(set(semantic_ids))
    counts = np.array([semantic_ids.count(uid) for uid in unique_ids])
    probabilities = counts / len(responses)
    
    entropy = shannon_entropy(probabilities)
    cluster_entropy = cluster_assignment_entropy(semantic_ids)
    
    return {
        "semantic_entropy_embedding": entropy,
        "num_clusters_embedding": len(unique_ids),
        "cluster_entropy_embedding": cluster_entropy
    }


def compute_semantic_entropy_context_weighted(
    responses: List[str],
    retrieved_chunks: List[str],
    embedding_threshold: float = 0.75,
    context_weight: float = 0.3
) -> Dict[str, float]:
    """
    Compute semantic entropy with context-aware weighting.
    
    Novel RAG-specific variant: weights clusters by entailment with retrieved context.
    
    Args:
        responses: List of model-generated responses
        retrieved_chunks: Retrieved context from RAG
        embedding_threshold: Threshold for embedding clustering
        context_weight: Weight for context alignment
        
    Returns:
        Dictionary with entropy values including context-weighted variant
    """
    if len(responses) < 2:
        return {
            "semantic_entropy_context_weighted": 0.0,
            "num_clusters_context": 1,
            "context_entailment_avg": 0.0,
            "cluster_entropy_context": 0.0
        }
    
    # Cluster with context weighting
    semantic_ids, cluster_map, context_scores = context_weighted_clustering(
        responses,
        retrieved_chunks,
        embedding_threshold=embedding_threshold,
        context_weight=context_weight
    )
    
    # Compute standard entropy
    unique_ids = sorted(set(semantic_ids))
    counts = np.array([semantic_ids.count(uid) for uid in unique_ids])
    probabilities = counts / len(responses)
    
    raw_entropy = shannon_entropy(probabilities)
    cluster_entropy = cluster_assignment_entropy(semantic_ids)
    
    # Compute context-weighted entropy
    if context_scores:
        context_weights = np.array([context_scores.get(uid, 0.5) for uid in unique_ids])
        weighted_entropy = weighted_shannon_entropy(
            probabilities,
            np.ones_like(probabilities),
            context_weights
        )
    else:
        weighted_entropy = raw_entropy
    
    avg_context_entailment = np.mean(list(context_scores.values())) if context_scores else 0.5
    
    return {
        "semantic_entropy_context_weighted": weighted_entropy,
        "semantic_entropy_raw_context": raw_entropy,  # For comparison
        "num_clusters_context": len(unique_ids),
        "context_entailment_avg": avg_context_entailment,
        "cluster_entropy_context": cluster_entropy
    }


# =============================================================================
# NOVEL RAG HALLUCINATION SCORE
# =============================================================================

def compute_rag_hallucination_score(
    responses: List[str],
    retrieved_chunks: List[str],
    alpha: float = 0.35,
    beta: float = 0.35,
    gamma: float = 0.2,
    delta: float = 0.1,  # NEW: contradiction weight
    embedding_threshold: float = 0.75
) -> Dict[str, float]:
    """
    Novel RAG-specific hallucination detection metric.
    
    Combines four signals:
    1. Semantic entropy (internal consistency)
    2. Context entailment (grounding in retrieved context)
    3. Lexical support (token overlap with context)
    4. Contradiction (mutually exclusive clusters)
    
    Formula:
        Hallucination = α * semantic_entropy + 
                       β * (1 - context_entailment) + 
                       γ * (1 - lexical_support) +
                       δ * contradiction
                       
    This is scientifically novel - leverages RAG context in a way the
    original semantic entropy paper could not.
    
    Args:
        responses: List of model-generated responses
        retrieved_chunks: Retrieved context from RAG
        alpha: Weight for semantic entropy
        beta: Weight for context entailment
        gamma: Weight for lexical support
        delta: Weight for contradiction (default 0.1 - can be tuned)
        embedding_threshold: Threshold for embedding clustering
        
    Returns:
        Dictionary with hallucination score and components
    """
    if not responses or len(responses) < 2:
        # Fallback for single response
        if responses and retrieved_chunks:
            return _compute_single_response_hallucination(
                responses[0], retrieved_chunks, alpha, beta, gamma
            )
        return {
            "rag_hallucination_score": 0.0,
            "semantic_entropy_component": 0.0,
            "context_entailment_component": 0.0,
            "lexical_support_component": 0.0,
            "is_hallucination": False
        }
    
    # 1. Semantic entropy component
    entropy_result = compute_semantic_entropy_embedding(
        responses, threshold=embedding_threshold
    )
    semantic_entropy = entropy_result["semantic_entropy_embedding"]
    
    # Normalize to 0-1 (max entropy is log2(n))
    n_clusters = entropy_result["num_clusters_embedding"]
    max_entropy = np.log2(n_clusters) if n_clusters > 1 else 1.0
    normalized_entropy = semantic_entropy / max_entropy if max_entropy > 0 else 0.0
    
    # 2. Context entailment component
    context_entailments = []
    for response in responses:
        for chunk in retrieved_chunks:
            score = compute_nli_entailment(chunk, response)
            context_entailments.append(score)
    
    avg_context_entailment = np.mean(context_entailments) if context_entailments else 0.5
    context_component = 1.0 - avg_context_entailment
    
    # 3. Lexical support component
    lexical_supports = []
    for response in responses:
        response_tokens = set(word_tokenize(response.lower()))
        response_tokens = {t for t in response_tokens if re.match(r'[a-zA-Z]+', t)}
        
        if not response_tokens:
            lexical_supports.append(1.0)
            continue
        
        # Check union of all chunk tokens
        chunk_tokens = set()
        for chunk in retrieved_chunks:
            tokens = set(word_tokenize(chunk.lower()))
            tokens = {t for t in tokens if re.match(r'[a-zA-Z]+', t)}
            chunk_tokens.update(tokens)
        
        supported = response_tokens & chunk_tokens
        lexical_supports.append(len(supported) / len(response_tokens))
    
    avg_lexical_support = np.mean(lexical_supports) if lexical_supports else 0.5
    lexical_component = 1.0 - avg_lexical_support
    
    # 4. Contradiction component (NEW!)
    contradiction_result = compute_contradiction_score(responses)
    contradiction_component = contradiction_result.get("contradiction_score", 0.0)
    
    # Combined score (now with 4 components)
    hallucination_score = (
        alpha * normalized_entropy +
        beta * context_component +
        gamma * lexical_component +
        delta * contradiction_component
    )
    
    return {
        "rag_hallucination_score": hallucination_score,
        "semantic_entropy_component": normalized_entropy,
        "context_entailment_component": context_component,
        "lexical_support_component": lexical_component,
        "contradiction_component": contradiction_component,
        "avg_context_entailment": avg_context_entailment,
        "avg_lexical_support": avg_lexical_support,
        "is_hallucination": hallucination_score > 0.5,
        "confidence": "high" if hallucination_score < 0.3 else "medium" if hallucination_score < 0.6 else "low",
        # Include full contradiction details
        "num_contradicting_pairs": contradiction_result.get("num_contradicting_pairs", 0),
        "total_pairs": contradiction_result.get("total_pairs", 0),
        "n_clusters": contradiction_result.get("n_clusters", 1)
    }


def _compute_single_response_hallucination(
    response: str,
    retrieved_chunks: List[str],
    alpha: float,
    beta: float,
    gamma: float
) -> Dict[str, float]:
    """Fallback for single response without sampling."""
    # Can't compute semantic entropy without multiple samples
    semantic_component = 0.0
    
    # Context entailment
    context_entailments = []
    for chunk in retrieved_chunks:
        score = compute_nli_entailment(chunk, response)
        context_entailments.append(score)
    avg_entailment = np.mean(context_entailments) if context_entailments else 0.5
    context_component = 1.0 - avg_entailment
    
    # Lexical support
    response_tokens = set(word_tokenize(response.lower()))
    response_tokens = {t for t in response_tokens if re.match(r'[a-zA-Z]+', t)}
    
    if not response_tokens:
        lexical_component = 0.0
    else:
        chunk_tokens = set()
        for chunk in retrieved_chunks:
            tokens = set(word_tokenize(chunk.lower()))
            tokens = {t for t in tokens if re.match(r'[a-zA-Z]+', t)}
            chunk_tokens.update(tokens)
        
        supported = response_tokens & chunk_tokens
        lexical_component = 1.0 - (len(supported) / len(response_tokens))
    
    hallucination_score = (
        alpha * semantic_component +
        beta * context_component +
        gamma * lexical_component
    )
    
    return {
        "rag_hallucination_score": hallucination_score,
        "semantic_entropy_component": semantic_component,
        "context_entailment_component": context_component,
        "lexical_support_component": lexical_component,
        "avg_context_entailment": avg_entailment,
        "is_hallucination": hallucination_score > 0.5
    }


# =============================================================================
# AUROC EVALUATION (Correlation with Correctness)
# =============================================================================

def compute_auroc(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> float:
    """
    Compute Area Under ROC Curve.
    
    Measures how well a hallucination score predicts incorrect answers.
    AUC = 0.5 means random, AUC = 1.0 means perfect detection.
    
    Args:
        y_true: Binary ground truth (1 = incorrect/hallucination, 0 = correct)
        y_scores: Hallucination scores (higher = more likely hallucination)
        
    Returns:
        AUROC value
    """
    from sklearn.metrics import roc_auc_score
    
    try:
        auroc = roc_auc_score(y_true, y_scores)
        return auroc
    except Exception as e:
        logger.warning(f"Could not compute AUROC: {e}")
        return 0.5


def compute_correlation(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    method: str = "spearman"
) -> float:
    """
    Compute correlation between hallucination scores and correctness.
    
    Args:
        y_true: Binary correctness (1 = correct, 0 = incorrect)
        y_scores: Hallucination scores (higher = more uncertainty)
        method: "spearman" or "pearson"
        
    Returns:
        Correlation coefficient (negative = higher score → more incorrect)
    """
    from scipy.stats import spearmanr, pearsonr
    
    # Invert scores: high uncertainty should correlate with incorrect (y_true=0)
    # So we use (1 - y_scores) to get: high score → likely correct
    inverted_scores = 1 - y_scores
    
    if method == "spearman":
        corr, _ = spearmanr(inverted_scores, y_true)
    elif method == "pearson":
        corr, _ = pearsonr(inverted_scores, y_true)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return corr if not np.isnan(corr) else 0.0


def evaluate_hallucination_detection(
    questions: List[str],
    responses_list: List[List[str]],  # Multiple responses per question
    retrieved_chunks_list: List[List[str]],  # Context per question
    ground_truths: List[str],  # Correct answer per question
    predictions: List[str],  # Model's final answer per question
    metric_name: str = "rag_hallucination"
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of hallucination detection.
    
    For each question:
    1. Generate multiple samples (or use provided)
    2. Compute hallucination score
    3. Determine if prediction matches ground truth
    
    Then compute AUROC: how well does score predict incorrect answers?
    
    Args:
        questions: List of questions
        responses_list: List of response lists (multiple samples per question)
        retrieved_chunks_list: Retrieved context per question
        ground_truths: Correct answers
        predictions: Model's predictions
        metric_name: Which metric to use
        
    Returns:
        Dictionary with evaluation results
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    # Compute correctness labels (1 = incorrect/hallucination, 0 = correct)
    is_incorrect = []
    hallucination_scores = []
    
    for i, (responses, prediction, ground_truth) in enumerate(
        zip(responses_list, predictions, ground_truths)
    ):
        # Determine if prediction is correct
        correct = prediction.lower().strip() == ground_truth.lower().strip()
        is_incorrect.append(0 if correct else 1)
        
        # Compute hallucination score
        retrieved_chunks = retrieved_chunks_list[i] if i < len(retrieved_chunks_list) else []
        
        if metric_name == "rag_hallucination":
            result = compute_rag_hallucination_score(responses, retrieved_chunks)
        elif metric_name == "semantic_entropy_nli":
            result = compute_semantic_entropy_nli(responses)
            # Normalize
            max_entropy = np.log2(result.get("num_clusters_nli", 2)) or 1.0
            result["rag_hallucination_score"] = result["semantic_entropy_nli"] / max_entropy
        elif metric_name == "semantic_entropy_embedding":
            result = compute_semantic_entropy_embedding(responses)
            max_entropy = np.log2(result.get("num_clusters_embedding", 2)) or 1.0
            result["rag_hallucination_score"] = result["semantic_entropy_embedding"] / max_entropy
        elif metric_name == "semantic_entropy_context_weighted":
            result = compute_semantic_entropy_context_weighted(responses, retrieved_chunks)
            result["rag_hallucination_score"] = result["semantic_entropy_context_weighted"]
        else:
            result = {"rag_hallucination_score": 0.5}
        
        hallucination_scores.append(result["rag_hallucination_score"])
    
    # Convert to arrays
    y_true = np.array(is_incorrect)
    y_scores = np.array(hallucination_scores)
    
    # Compute metrics
    auroc = compute_auroc(y_true, y_scores)
    spearman_corr = compute_correlation(y_true, y_scores, method="spearman")
    pearson_corr = compute_correlation(y_true, y_scores, method="pearson")
    
    # Accuracy at different thresholds
    accuracies = {}
    for threshold in [0.3, 0.5, 0.7]:
        preds = (y_scores > threshold).astype(int)
        acc = accuracy_score(y_true, preds)
        accuracies[f"accuracy_at_{threshold}"] = acc
    
    # Overall accuracy
    overall_accuracy = 1 - np.mean(is_incorrect)
    
    return {
        "metric_name": metric_name,
        "auroc": auroc,
        "spearman_correlation": spearman_corr,
        "pearson_correlation": pearson_corr,
        "overall_accuracy": overall_accuracy,
        "threshold_accuracies": accuracies,
        "mean_hallucination_score": np.mean(y_scores),
        "std_hallucination_score": np.std(y_scores),
        "n_questions": len(questions),
        "n_incorrect": sum(is_incorrect),
        "n_correct": len(is_incorrect) - sum(is_incorrect)
    }


# =============================================================================
# COMBINED METRIC CLASS FOR EASY USE
# =============================================================================

class SemanticEntropyComparator:
    """
    Compare different semantic entropy variants for hallucination detection.
    
    Usage:
        comparator = SemanticEntropyComparator()
        results = comparator.compute_all(
            responses=[samples], 
            retrieved_chunks=[chunks]
        )
        evaluation = comparator.evaluate_auroc(
            ground_truths=[...],
            predictions=[...],
            retrieved_chunks=[...]
        )
    """
    
    def __init__(
        self,
        embedding_threshold: float = 0.75,
        nli_model: str = "roberta-large-mnli",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.embedding_threshold = embedding_threshold
        self.nli_model = nli_model
        self.embedding_model = embedding_model
    
    def compute_all(
        self,
        responses: List[str],
        retrieved_chunks: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute all semantic entropy variants.
        
        Args:
            responses: Multiple sampled responses
            retrieved_chunks: Retrieved context (optional)
            
        Returns:
            Dictionary with all entropy variants
        """
        results = {}
        
        # 1. NLI-based (original)
        nli_result = compute_semantic_entropy_nli(
            responses, 
            nli_model=self.nli_model
        )
        results.update({f"nli_{k}": v for k, v in nli_result.items()})
        
        # 2. Embedding-based
        embed_result = compute_semantic_entropy_embedding(
            responses,
            threshold=self.embedding_threshold,
            model_name=self.embedding_model
        )
        results.update({f"embedding_{k}": v for k, v in embed_result.items()})
        
        # 3. Context-weighted (if context provided)
        if retrieved_chunks:
            context_result = compute_semantic_entropy_context_weighted(
                responses,
                retrieved_chunks,
                embedding_threshold=self.embedding_threshold
            )
            results.update({f"context_{k}": v for k, v in context_result.items()})
            
            # 4. RAG hallucination score
            rag_result = compute_rag_hallucination_score(
                responses,
                retrieved_chunks
            )
            results.update(rag_result)
        
        return results
    
    def evaluate_auroc(
        self,
        responses_list: List[List[str]],
        retrieved_chunks_list: List[List[str]],
        ground_truths: List[str],
        predictions: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate all variants for hallucination detection using AUROC.
        
        Args:
            responses_list: Multiple responses per question
            retrieved_chunks_list: Retrieved context per question
            ground_truths: Correct answers
            predictions: Model predictions
            
        Returns:
            Evaluation results for each metric
        """
        metrics_to_test = [
            "rag_hallucination",
            "semantic_entropy_nli", 
            "semantic_entropy_embedding",
            "semantic_entropy_context_weighted"
        ]
        
        results = {}
        for metric in metrics_to_test:
            try:
                eval_result = evaluate_hallucination_detection(
                    questions=[""] * len(responses_list),
                    responses_list=responses_list,
                    retrieved_chunks_list=retrieved_chunks_list,
                    ground_truths=ground_truths,
                    predictions=predictions,
                    metric_name=metric
                )
                results[metric] = eval_result
            except Exception as e:
                logger.warning(f"Failed to evaluate {metric}: {e}")
                results[metric] = {"error": str(e)}
        
        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_all_semantic_entropies(
    responses: List[str],
    retrieved_chunks: Optional[List[str]] = None,
    sample_confidences: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Convenience function to compute all semantic entropy variants.
    
    Args:
        responses: Model-generated responses (for sampling-based entropy)
        retrieved_chunks: Retrieved context for RAG
        sample_confidences: Optional token-level confidences
        
    Returns:
        Dictionary with all computed metrics
    """
    comparator = SemanticEntropyComparator()
    return comparator.compute_all(responses, retrieved_chunks)


# Backward compatibility
def semantic_entropy_nli(responses: List[str], **kwargs) -> float:
    """Wrapper for NLI-based semantic entropy."""
    result = compute_semantic_entropy_nli(responses)
    return result["semantic_entropy_nli"]


def semantic_entropy_embedding(responses: List[str], threshold: float = 0.75, **kwargs) -> float:
    """Wrapper for embedding-based semantic entropy."""
    result = compute_semantic_entropy_embedding(responses, threshold=threshold)
    return result["semantic_entropy_embedding"]


def semantic_entropy_context_weighted(
    responses: List[str], 
    retrieved_chunks: List[str],
    threshold: float = 0.75,
    **kwargs
) -> float:
    """Wrapper for context-weighted semantic entropy."""
    result = compute_semantic_entropy_context_weighted(
        responses, retrieved_chunks, embedding_threshold=threshold
    )
    return result["semantic_entropy_context_weighted"]
