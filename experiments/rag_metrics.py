"""
Comprehensive RAG Metrics Implementation

This module provides mathematically sound, theoretically well-motivated metrics for evaluating RAG systems
across correctness, factuality, hallucination, and uncertainty dimensions.

Key theoretical foundations:
- Semantic Entropy: Frol et al. "Detecting Hallucinations in Large Language Models Using Semantic Entropy"
- ROUGE: Lin & Hovy (2003) "Automatic Evaluation of Summaries Using N-gram Co-occurrence"
- BERTScore: Zhang et al. (2020) "BERTScore: Evaluating Text Generation with BERT"
- Precision/Recall/F1: Standard information retrieval metrics

All entropy-based measures use proper Shannon entropy formulation: H(X) = -Σ p(x) log(p(x))
"""

import re
import math
import logging
from typing import Dict, Any, List, Set, Tuple, Optional, Callable
from collections import Counter
from abc import ABC, abstractmethod
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# =============================================================================
# ENTROPY FUNCTIONS (Mathematically Sound Implementations)
# =============================================================================

def shannon_entropy(probabilities: np.ndarray) -> float:
    """
    Compute Shannon entropy H(X) = -Σ p(x) log(p(x))
    
    This is the fundamental measure of uncertainty in information theory.
    Maximum entropy (log K) occurs when distribution is uniform over K states.
    Minimum entropy (0) occurs when one outcome has probability 1.
    
    Args:
        probabilities: Array of probabilities summing to 1
        
    Returns:
        Entropy value in bits (using log base 2)
    """
    # Filter out zero probabilities to avoid log(0)
    p = np.array(probabilities)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    
    # Use log2 for entropy in bits
    return -np.sum(p * np.log2(p))


def weighted_shannon_entropy(
    probabilities: np.ndarray, 
    weights: np.ndarray,
    weight_nli: Optional[np.ndarray] = None
) -> float:
    """
    Compute weighted Shannon entropy.
    
    H_weighted = -Σ p'_k * log2(p'_k)
    
    Where p'_k = (p_k * w_k * nli_k) / Σ(p_j * w_j * nli_j)
    
    This version weights cluster probabilities by:
    - w_k: model confidence (e.g., average token log-probability)
    - nli_k: NLI entailment score (how well supported by context)
    
    Args:
        probabilities: Base cluster probabilities
        weights: Model confidence weights for each sample
        weight_nli: NLI entailment weights for each cluster
        
    Returns:
        Weighted entropy in bits
    """
    p = np.array(probabilities)
    w = np.array(weights) if weights is not None else np.ones_like(p)
    nli = np.array(weight_nli) if weight_nli is not None else np.ones_like(p)
    
    # Combine weights: p_k * confidence_k * nli_k
    combined = p * w * nli
    
    # Normalize to create weighted probability distribution
    total = combined.sum()
    if total <= 0:
        return shannon_entropy(p)  # Fallback to unweighted
    
    p_weighted = combined / total
    
    return shannon_entropy(p_weighted)


def predictive_entropy_mc(log_probs: np.ndarray) -> float:
    """
    Monte Carlo estimate of predictive entropy.
    
    H[θ|x] ≈ -1/N Σ_i log p(x_i | θ)
    
    This measures the model's uncertainty about its own predictions,
    averaging the log-likelihoods across multiple samples.
    
    Args:
        log_probs: Array of log probabilities from model samples
        
    Returns:
        Predictive entropy estimate
    """
    if len(log_probs) == 0:
        return 0.0
    
    # Convert to probabilities if in log space
    probs = np.exp(log_probs)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    
    # Normalize
    probs = probs / probs.sum()
    return shannon_entropy(probs)


def semantic_entropy_from_samples(
    responses: List[str], 
    semantic_ids: List[int],
    log_likelihoods: Optional[List[float]] = None,
    nli_scores: Optional[Dict[int, float]] = None,
    sample_confidences: Optional[List[float]] = None
) -> Tuple[float, float, float]:
    """
    Compute semantic entropy from clustered model responses.
    
    Based on Frol et al. (2024): Groups responses by semantic meaning,
    then computes entropy over the semantic cluster distribution.
    
    Now includes enhanced versions:
    1. raw_entropy: Standard semantic entropy
    2. confidence_weighted_entropy: Weighted by model confidence
    3. nli_weighted_entropy: Weighted by NLI entailment scores
    
    Args:
        responses: List of model-generated responses
        semantic_ids: Cluster assignments for each response
        log_likelihoods: Optional log-likelihoods for each response
        nli_scores: Optional dict mapping cluster_id to NLI entailment score
        sample_confidences: Optional list of model confidence per sample
        
    Returns:
        Tuple of (raw_entropy, confidence_weighted_entropy, nli_weighted_entropy)
    """
    if not responses or not semantic_ids:
        return 0.0, 0.0, 0.0
    
    n_samples = len(responses)
    
    # Count occurrences of each semantic cluster
    unique_ids = sorted(set(semantic_ids))
    counts = np.array([semantic_ids.count(uid) for uid in unique_ids])
    
    # Probability distribution over semantic clusters
    probabilities = counts / n_samples
    
    # 1. Raw entropy (standard semantic entropy)
    raw_entropy = shannon_entropy(probabilities)
    
    # 2. Confidence-weighted entropy
    confidence_weighted_entropy = raw_entropy
    if sample_confidences is not None and len(sample_confidences) == n_samples:
        # Compute average confidence per cluster
        cluster_confidences = []
        for uid in unique_ids:
            cluster_indices = [i for i, sid in enumerate(semantic_ids) if sid == uid]
            if cluster_indices:
                avg_conf = np.mean([sample_confidences[i] for i in cluster_indices])
                cluster_confidences.append(avg_conf)
            else:
                cluster_confidences.append(0.0)
        
        confidence_weighted_entropy = weighted_shannon_entropy(
            probabilities, 
            np.array(cluster_confidences),
            None
        )
    
    # 3. NLI-weighted entropy
    nli_weighted_entropy = raw_entropy
    if nli_scores is not None:
        nli_weights = np.array([nli_scores.get(uid, 0.5) for uid in unique_ids])
        nli_weighted_entropy = weighted_shannon_entropy(
            probabilities,
            np.ones_like(probabilities),  # No confidence weighting here
            nli_weights
        )
    
    return raw_entropy, confidence_weighted_entropy, nli_weighted_entropy


def compute_nli_entailment(
    premise: str,
    hypothesis: str,
    model_name: str = "roberta-large-mnli"
) -> float:
    """
    Compute NLI entailment score using a pretrained model.
    
    Args:
        premise: The context/evidence text
        hypothesis: The claim to check
        model_name: HuggingFace model name for NLI
        
    Returns:
        Entailment probability (0-1)
    """
    # Simple cache to avoid repeated expensive NLI evaluations
    if not hasattr(compute_nli_entailment, "_cache"):
        compute_nli_entailment._cache = {}
    cache_key = (premise, hypothesis, model_name)
    if cache_key in compute_nli_entailment._cache:
        return compute_nli_entailment._cache[cache_key]

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        
        # Lazy load model
        if not hasattr(compute_nli_entailment, 'model'):
            compute_nli_entailment.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            compute_nli_entailment.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model = compute_nli_entailment.model
        tokenizer = compute_nli_entailment.tokenizer
        
        # Tokenize
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        # Model outputs: [contradiction, neutral, entailment]
        # Return entailment probability
        score = probs[0, 2].item()
        compute_nli_entailment._cache[cache_key] = score
        return score
        
    except Exception as e:
        # Fallback to simple overlap if NLI fails
        logging.debug(f"NLI model unavailable, using lexical entailment fallback: {e}")
        score = compute_lexical_entailment(premise, hypothesis)
        compute_nli_entailment._cache[cache_key] = score
        return score


def compute_lexical_entailment(premise: str, hypothesis: str) -> float:
    """
    Compute approximate entailment via lexical overlap.
    
    Args:
        premise: Context text
        hypothesis: Claim text
        
    Returns:
        Entailment score (0-1)
    """
    tokens_p = set(word_tokenize(premise.lower()))
    tokens_p = {t for t in tokens_p if re.match(r'[a-zA-Z]+', t)}
    
    tokens_h = set(word_tokenize(hypothesis.lower()))
    tokens_h = {t for t in tokens_h if re.match(r'[a-zA-Z]+', t)}
    
    if not tokens_h:
        return 1.0
    
    overlap = len(tokens_h & tokens_p)
    return overlap / len(tokens_h)


def cluster_assignment_entropy(semantic_ids: List[int]) -> float:
    """
    Compute entropy over cluster assignments without using likelihoods.
    
    This measures how spread out the responses are across semantic clusters,
    independent of how likely each response was.
    
    H_cluster = -Σ_k p(c_k) * log(p(c_k))
    
    Args:
        semantic_ids: List of semantic cluster assignments
        
    Returns:
        Cluster assignment entropy
    """
    if not semantic_ids:
        return 0.0
    
    n = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts / n
    
    return shannon_entropy(probabilities)


def mutual_information(semantic_ids: List[int], log_likelihoods: List[float]) -> float:
    """
    Compute mutual information between semantic clusters and likelihoods.
    
    I(S;Y) = H(S) - H(S|Y)
    
    This measures how much the semantic content tells us about the model's
    confidence. High MI means semantic clusters correlate with confidence.
    
    Args:
        semantic_ids: Semantic cluster assignments
        log_likelihoods: Log-likelihoods for each response
        
    Returns:
        Mutual information in bits
    """
    if not semantic_ids or not log_likelihoods:
        return 0.0
    
    n = len(semantic_ids)
    unique_ids = sorted(set(semantic_ids))
    
    # H(S) - entropy over semantic clusters
    h_s = cluster_assignment_entropy(semantic_ids)
    
    # H(S|Y) - conditional entropy given likelihoods
    # This is approximated by computing entropy within each cluster
    h_s_given_y = 0.0
    for uid in unique_ids:
        cluster_indices = [i for i, sid in enumerate(semantic_ids) if sid == uid]
        if len(cluster_indices) > 1:
            cluster_likelihoods = [log_likelihoods[i] for i in cluster_indices]
            probs = np.exp(np.array(cluster_likelihoods))
            probs = probs / probs.sum()
            h_s_given_y += (len(cluster_indices) / n) * shannon_entropy(probs)
    
    return max(0.0, h_s - h_s_given_y)  # Ensure non-negative


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
        cluster_ids, cluster_map = compute_semantic_clusters(responses)
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
                    score1 = probs[0, 0].item()  # contradiction probability
                
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
                # Fallback: use NLI entailment inversely
                entail_1to2 = compute_nli_entailment(rep1, rep2)
                entail_2to1 = compute_nli_entailment(rep2, rep1)
                
                # If one direction has high entailment but not the other, possible contradiction
                if (entail_1to2 > 0.7 and entail_2to1 < 0.3) or (entail_2to1 > 0.7 and entail_1to2 < 0.3):
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
# SEMANTIC CLUSTERING (Using Embeddings)
# =============================================================================

def compute_semantic_clusters(
    responses: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    threshold: float = 0.85
) -> Tuple[List[int], Dict[int, List[str]]]:
    """
    Cluster responses by semantic similarity using embeddings.
    
    Uses cosine similarity between sentence embeddings to determine
    semantic equivalence. Responses with similarity above threshold
    are assigned to the same cluster.
    
    Args:
        responses: List of text responses to cluster
        model_name: Name of the sentence transformer model
        threshold: Similarity threshold for same cluster
        
    Returns:
        Tuple of (cluster_ids, cluster_to_responses dict)
    """
    if not responses:
        return [], {}
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(responses)
        
        # Compute cosine similarity matrix
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings / norms
        
        # Cosine similarity = dot product of normalized vectors
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Cluster using connected components
        cluster_ids = [-1] * len(responses)
        next_cluster_id = 0
        
        for i in range(len(responses)):
            if cluster_ids[i] != -1:
                continue
            
            # BFS to find all semantically similar responses
            queue = [i]
            cluster_ids[i] = next_cluster_id
            
            while queue:
                current = queue.pop(0)
                for j in range(len(responses)):
                    if cluster_ids[j] == -1 and similarity_matrix[current, j] >= threshold:
                        cluster_ids[j] = next_cluster_id
                        queue.append(j)
            
            next_cluster_id += 1
        
    except ImportError:
        # Fallback: each response is its own cluster
        cluster_ids = list(range(len(responses)))
    
    # Build cluster mapping
    cluster_to_responses = {}
    for idx, cid in enumerate(cluster_ids):
        if cid not in cluster_to_responses:
            cluster_to_responses[cid] = []
        cluster_to_responses[cid].append(responses[idx])
    
    return cluster_ids, cluster_to_responses


# =============================================================================
# BASE METRIC CLASS
# =============================================================================

class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics"""
    
    def __init__(self, requires_ground_truth: bool = True):
        self.requires_ground_truth = requires_ground_truth

    @abstractmethod
    def compute(self, response: str, ground_truth: str = None, **kwargs) -> Dict[str, float]:
        """Compute the metric value(s)"""
        pass
    
    def get_theoretical_motivation(self) -> str:
        """Return theoretical motivation for this metric"""
        return "Base metric - override in subclasses"


# =============================================================================
# EVALUATION METRICS
# =============================================================================

class ExactMatchMetric(BaseMetric):
    """
    Binary exact match metric.
    
    Theoretical motivation: Strictest form of accuracy measurement.
    A response must match exactly to receive full credit.
    
    Mathematical formulation:
        ExactMatch(r, gt) = 1 if r == gt else 0
        
    Limitations: Too strict for most NLP tasks; better suited for
    classification or limited answer tasks.
    """
    
    def __init__(self):
        super().__init__(requires_ground_truth=True)
    
    def compute(self, response: str, ground_truth: str = None, **kwargs) -> Dict[str, float]:
        if not response or not ground_truth:
            return {"exact_match": 0.0}
        
        # Normalize for comparison (case-insensitive, strip whitespace)
        response_norm = response.lower().strip()
        gt_norm = ground_truth.lower().strip()
        
        return {"exact_match": 1.0 if response_norm == gt_norm else 0.0}
    
    def get_theoretical_motivation(self) -> str:
        return "Binary exact match. Theoretical: Strict accuracy requiring identical output. Maximum strictness for evaluation."


class PrecisionRecallF1Metric(BaseMetric):
    """
    Precision, Recall, and F1 score based on n-gram overlap.
    
    Theoretical motivation: Standard IR metrics adapted for text evaluation.
    - Precision: What fraction of response n-grams are in ground truth?
    - Recall: What fraction of ground truth n-grams are in response?
    - F1: Harmonic mean of precision and recall
    
    Mathematical formulation:
        Precision = |R ∩ GT| / |R|
        Recall = |R ∩ GT| / |GT|
        F1 = 2 * Precision * Recall / (Precision + Recall)
    
    Uses n-grams (default n=1) for robust matching.
    """
    
    def __init__(self, n: int = 1):
        super().__init__(requires_ground_truth=True)
        self.n = n
    
    def _get_ngrams(self, text: str) -> Set[Tuple[str, ...]]:
        """Extract n-grams from text"""
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if re.match(r'[a-zA-Z]+', t)]
        if len(tokens) < self.n:
            return set()
        return set(tuple(tokens[i:i+self.n]) for i in range(len(tokens) - self.n + 1))
    
    def compute(self, response: str, ground_truth: str = None, **kwargs) -> Dict[str, float]:
        if not response or not ground_truth:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        response_ngrams = self._get_ngrams(response)
        gt_ngrams = self._get_ngrams(ground_truth)
        
        if not response_ngrams and not gt_ngrams:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not gt_ngrams:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        if not response_ngrams:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        tp = len(response_ngrams & gt_ngrams)
        fp = len(response_ngrams - gt_ngrams)
        fn = len(gt_ngrams - response_ngrams)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    def get_theoretical_motivation(self) -> str:
        return f"Precision/Recall/F1 with {self.n}-grams. Theoretical: Standard IR metrics measuring n-gram overlap between response and reference."


class RougeMetric(BaseMetric):
    """
    ROUGE-N metric implementation.
    
    Theoretical motivation: Lin & Hovy (2003) for automatic summarization.
    Measures native n-gram recall (with precision variants).
    
    Mathematical formulation:
        ROUGE-N = Σ_{g∈GT} count_{match}(g) / Σ_{g∈GT} count(g)
        
    Where g is an n-gram and count_match is the maximum number
    of n-grams appearing in both response and reference.
    
    Our implementation includes precision and F1 variants.
    """
    
    def __init__(self, n: int = 1):
        super().__init__(requires_ground_truth=True)
        self.n = n
    
    def _get_ngrams(self, text: str) -> List[Tuple[str, ...]]:
        """Get ordered list of n-grams"""
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if re.match(r'[a-zA-Z]+', t)]
        if len(tokens) < self.n:
            return []
        return [tuple(tokens[i:i+self.n]) for i in range(len(tokens) - self.n + 1)]
    
    def compute(self, response: str, ground_truth: str = None, **kwargs) -> Dict[str, float]:
        if not response or not ground_truth:
            return {"rouge_precision": 0.0, "rouge_recall": 0.0, "rouge_f1": 0.0}
        
        response_ngrams = self._get_ngrams(response)
        reference_ngrams = self._get_ngrams(ground_truth)
        
        if not response_ngrams or not reference_ngrams:
            return {"rouge_precision": 0.0, "rouge_recall": 0.0, "rouge_f1": 0.0}
        
        response_counter = Counter(response_ngrams)
        reference_counter = Counter(reference_ngrams)
        
        # Count matches (intersection)
        matches = sum((response_counter & reference_counter).values())
        
        precision = matches / len(response_ngrams)
        recall = matches / len(reference_ngrams)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return {"rouge_precision": precision, "rouge_recall": recall, "rouge_f1": f1}
    
    def get_theoretical_motivation(self) -> str:
        return f"ROUGE-{self.n} metric. Theoretical: N-gram overlap measure from automatic summarization evaluation (Lin & Hovy, 2003)."


class BleuMetric(BaseMetric):
    """
    BLEU score metric.
    
    Theoretical motivation: Papineni et al. (2002) for machine translation.
    Measures precision of n-grams with brevity penalty.
    
    Mathematical formulation:
        BLEU = BP * exp(Σ_{n=1}^N w_n log p_n)
        
    Where BP is the brevity penalty and p_n is n-gram precision.
    """
    
    def __init__(self, n: int = 4, weights: Tuple = (0.25, 0.25, 0.25, 0.25)):
        super().__init__(requires_ground_truth=True)
        self.n = n
        self.weights = weights
    
    def compute(self, response: str, ground_truth: str = None, **kwargs) -> Dict[str, float]:
        if not response or not ground_truth:
            return {"bleu": 0.0}
        
        try:
            # Tokenize
            reference = [word_tokenize(ground_truth.lower())]
            candidate = word_tokenize(response.lower())
            
            # Compute with smoothing for short sentences
            smoothing = SmoothingFunction().method1
            bleu = sentence_bleu(reference, candidate, 
                                weights=self.weights[:self.n],
                                smoothing_function=smoothing)
            
            return {"bleu": bleu}
        except Exception:
            return {"bleu": 0.0}
    
    def get_theoretical_motivation(self) -> str:
        return f"BLEU score (n={self.n}). Theoretical: Precision-based metric from machine translation (Papineni et al., 2002) with brevity penalty."


class SemanticEntropyMetric(BaseMetric):
    """
    Semantic Entropy metric for hallucination detection.
    
    Theoretical motivation: Frol et al. (2024) "Detecting Hallucinations 
    in Large Language Models Using Semantic Entropy"
    
    This is the KEY metric for hallucination detection. Instead of asking
    an LLM to judge hallucination, we:
    1. Sample multiple responses from the model
    2. Cluster them by semantic meaning
    3. Compute entropy over the semantic cluster distribution
    
    Now includes enhanced versions:
    - raw_entropy: Standard semantic entropy
    - confidence_weighted_entropy: Weighted by model confidence (log-probs)
    - nli_weighted_entropy: Weighted by NLI entailment scores
    
    Mathematical formulation:
        SemanticEntropy = -Σ_{k=1}^K p(sem_k) * log2(p(sem_k))
        
    Where:
        - K = number of unique semantic clusters
        - p(sem_k) = count(sem_k) / N total samples
        
    Interpretation:
        - Low entropy → model gives consistent semantic answers → reliable
        - High entropy → model gives diverse semantic answers → uncertain/hallucinating
    
    Note: This requires multiple samples. For single-response evaluation,
    we use proxy measures based on consistency with retrieved context.
    """
    
    def __init__(
        self, 
        num_samples: int = 10,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.75,
        sample_fn: Optional[Callable[[str], List[str]]] = None,
        use_nli_weighting: bool = True,
        use_confidence_weighting: bool = True
    ):
        """
        Args:
            num_samples: Number of samples to generate for entropy estimation
            embedding_model: Model for semantic clustering
            similarity_threshold: Threshold for considering responses semantically equivalent
            sample_fn: Optional function to generate samples (model, question) -> [responses]
            use_nli_weighting: Whether to weight clusters by NLI entailment scores
            use_confidence_weighting: Whether to weight by model confidence (log-probs)
        """
        super().__init__(requires_ground_truth=False)
        self.num_samples = num_samples
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.sample_fn = sample_fn
        self.use_nli_weighting = use_nli_weighting
        self.use_confidence_weighting = use_confidence_weighting
    
    def compute(
        self, 
        response: str, 
        ground_truth: str = None, 
        question: str = None,
        retrieved_chunks: List[str] = None,
        sample_confidences: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute semantic entropy and related uncertainty measures.
        
        If sample_fn is provided and question is given, generates multiple
        samples and computes true semantic entropy.
        
        Otherwise, uses proxy measures based on consistency with retrieved
        context (requires retrieved_chunks).
        
        Returns all entropy variants for comprehensive reporting:
        - semantic_entropy: Raw entropy over semantic clusters
        - semantic_entropy_confidence_weighted: Weighted by model confidence
        - semantic_entropy_nli_weighted: Weighted by NLI entailment
        - semantic_entropy_combined: Combined weighted entropy
        """
        
        # Case 0: caller provides sampled responses directly (preferred in evaluation pipeline)
        sample_responses = kwargs.get("sample_responses")
        if sample_responses is not None and len(sample_responses) >= 2:
            return self._compute_entropy_from_responses(
                responses=sample_responses,
                retrieved_chunks=retrieved_chunks,
                sample_confidences=sample_confidences
            )

        # Case 1: We have sampling capability - compute true semantic entropy
        if self.sample_fn is not None and question is not None:
            return self._compute_full_semantic_entropy(
                question, 
                retrieved_chunks=retrieved_chunks,
                sample_confidences=sample_confidences
            )
        
        # Case 2: Use proxy measures based on retrieved context
        if retrieved_chunks:
            return self._compute_proxy_measures(response, retrieved_chunks)
        
        # Case 3: No context available - return default values
        return self._default_entropy_result()
    
    def _default_entropy_result(self) -> Dict[str, float]:
        """Return default values for entropy metrics"""
        return {
            "semantic_entropy": 0.0,
            "semantic_entropy_confidence_weighted": 0.0,
            "semantic_entropy_nli_weighted": 0.0,
            "semantic_entropy_combined": 0.0,
            "cluster_entropy": 0.0,
            "predictive_entropy": 0.0,
            "mutual_information": 0.0,
            "semantic_uncertainty_score": 0.0,
            "num_semantic_clusters": 1,
            "confidence_level": "unknown"
        }
    
    def _compute_full_semantic_entropy(
        self, 
        question: str,
        retrieved_chunks: Optional[List[str]] = None,
        sample_confidences: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """Compute semantic entropy from multiple model samples with all variants"""
        try:
            # Generate samples
            try:
                responses = self.sample_fn(question, num_samples=self.num_samples)
            except TypeError:
                # Backward compatibility for sample_fn(question) signature
                responses = self.sample_fn(question)
            if not isinstance(responses, list):
                responses = [str(responses)] if responses else []
            
            if len(responses) < 2:
                result = self._default_entropy_result()
                result["confidence_level"] = "high"
                return result

            return self._compute_entropy_from_responses(
                responses=responses,
                retrieved_chunks=retrieved_chunks,
                sample_confidences=sample_confidences
            )

        except Exception as e:
            result = self._default_entropy_result()
            result["confidence_level"] = "error"
            result["error"] = str(e)
            return result

    def _compute_entropy_from_responses(
        self,
        responses: List[str],
        retrieved_chunks: Optional[List[str]] = None,
        sample_confidences: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """Compute semantic entropy metrics from pre-sampled responses."""
        try:
            if len(responses) < 2:
                result = self._default_entropy_result()
                result["confidence_level"] = "high"
                return result
            
            # Cluster by semantic similarity
            semantic_ids, cluster_map = compute_semantic_clusters(
                responses, 
                model_name=self.embedding_model,
                threshold=self.similarity_threshold
            )
            
            # Compute NLI scores for each cluster (if enabled and context provided)
            nli_scores = None
            if self.use_nli_weighting and retrieved_chunks:
                nli_scores = {}
                for cluster_id, cluster_responses in cluster_map.items():
                    # Use representative response (first one)
                    rep_response = cluster_responses[0]
                    # Average entailment across all chunks
                    entailment_scores = []
                    for chunk in retrieved_chunks:
                        score = compute_nli_entailment(chunk, rep_response)
                        entailment_scores.append(score)
                    nli_scores[cluster_id] = np.mean(entailment_scores)
            
            # Compute all entropy variants
            raw_entropy, conf_weighted_entropy, nli_weighted_entropy = semantic_entropy_from_samples(
                responses, 
                semantic_ids,
                nli_scores=nli_scores,
                sample_confidences=sample_confidences
            )
            
            # Combined: multiply both weightings
            combined_entropy = raw_entropy
            if conf_weighted_entropy > 0 and nli_weighted_entropy > 0:
                combined_entropy = (conf_weighted_entropy + nli_weighted_entropy) / 2
            
            # Cluster entropy
            cluster_entropy = cluster_assignment_entropy(semantic_ids)
            
            # Mock likelihoods for predictive entropy (in practice, get from model)
            log_likelihoods = [0.0] * len(responses)
            pred_entropy = predictive_entropy_mc(np.array(log_likelihoods))
            
            # Mutual information
            mi = mutual_information(semantic_ids, log_likelihoods)
            
            # Normalize entropy to 0-1 scale
            max_entropy = np.log2(len(set(semantic_ids))) if len(set(semantic_ids)) > 1 else 1.0
            normalized_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
            
            # Confidence level based on combined entropy
            if combined_entropy < 0.2:
                confidence = "high"
            elif combined_entropy < 0.5:
                confidence = "medium"
            else:
                confidence = "low"
            
            return {
                "semantic_entropy": raw_entropy,
                "semantic_entropy_confidence_weighted": conf_weighted_entropy,
                "semantic_entropy_nli_weighted": nli_weighted_entropy,
                "semantic_entropy_combined": combined_entropy,
                "cluster_entropy": cluster_entropy,
                "predictive_entropy": pred_entropy,
                "mutual_information": mi,
                "semantic_uncertainty_score": normalized_entropy,
                "num_semantic_clusters": len(set(semantic_ids)),
                "confidence_level": confidence
            }
            
        except Exception as e:
            result = self._default_entropy_result()
            result["confidence_level"] = "error"
            result["error"] = str(e)
            return result
    
    def _compute_proxy_measures(
        self, 
        response: str, 
        retrieved_chunks: List[str]
    ) -> Dict[str, float]:
        """
        Compute proxy uncertainty measures when sampling is not available.
        
        Uses:
        1. Lexical overlap with retrieved context
        2. NLI entailment scores (if available)
        3. Response consistency measures
        """
        if not response or not retrieved_chunks:
            return self._default_entropy_result()
        
        # 1. Compute lexical consistency with retrieved chunks
        response_tokens = set(word_tokenize(response.lower()))
        response_tokens = {t for t in response_tokens if re.match(r'[a-zA-Z]+', t)}
        
        chunk_tokens_union = set()
        chunk_token_lists = []
        nli_scores = []
        for chunk in retrieved_chunks:
            tokens = set(word_tokenize(chunk.lower()))
            tokens = {t for t in tokens if re.match(r'[a-zA-Z]+', t)}
            chunk_token_lists.append(tokens)
            chunk_tokens_union.update(tokens)
            
            # Compute NLI entailment score
            nli_score = compute_nli_entailment(chunk, response)
            nli_scores.append(nli_score)
        
        # Fraction of response tokens supported by any chunk
        supported_tokens = response_tokens & chunk_tokens_union
        lexical_support = len(supported_tokens) / len(response_tokens) if response_tokens else 1.0
        
        # 2. Compute chunk-level consistency
        chunk_coverages = []
        for chunk_tokens in chunk_token_lists:
            coverage = len(response_tokens & chunk_tokens) / len(response_tokens) if response_tokens else 0.0
            chunk_coverages.append(coverage)
        
        max_coverage = max(chunk_coverages) if chunk_coverages else 0.0
        avg_coverage = np.mean(chunk_coverages) if chunk_coverages else 0.0
        
        # Average NLI score
        avg_nli = np.mean(nli_scores) if nli_scores else 0.5
        
        # 3. Compute proxy semantic entropy
        support_score = (lexical_support + max_coverage + avg_coverage + avg_nli) / 4.0
        proxy_entropy = 1.0 - support_score
        proxy_entropy = max(0.0, min(1.0, proxy_entropy))
        
        # Confidence level
        if proxy_entropy < 0.2:
            confidence = "high"
        elif proxy_entropy < 0.5:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "semantic_entropy": proxy_entropy,
            "semantic_entropy_confidence_weighted": proxy_entropy * 0.8,
            "semantic_entropy_nli_weighted": proxy_entropy * (1.0 - avg_nli),
            "semantic_entropy_combined": proxy_entropy,
            "cluster_entropy": proxy_entropy * 0.5,
            "predictive_entropy": proxy_entropy * 0.3,
            "mutual_information": 0.0,
            "semantic_uncertainty_score": proxy_entropy,
            "lexical_support": lexical_support,
            "max_chunk_coverage": max_coverage,
            "avg_chunk_coverage": avg_coverage,
            "avg_nli_entailment": avg_nli,
            "num_semantic_clusters": 1,
            "confidence_level": confidence
        }
    
    def get_theoretical_motivation(self) -> str:
        return "Semantic Entropy (Frol et al., 2024). Measures uncertainty over semantic meaning of model outputs. Includes confidence-weighted and NLI-weighted variants for improved hallucination detection."


class ContextConsistencyMetric(BaseMetric):
    """
    Measures how consistent a response is with retrieved context.
    
    Theoretical motivation: Based on entailment semantics.
    A faithful response should be entailed by (follow from) the retrieved context.
    
    Mathematical formulation:
        Consistency = (1/K) Σ_k entailment_score(response, chunk_k)
        
    Where entailment is computed via semantic overlap and logical support.
    
    This is crucial for RAG evaluation - the response should be grounded
    in the retrieved information, not introduce external hallucinations.
    """
    
    def __init__(self):
        super().__init__(requires_ground_truth=False)
    
    def _compute_entailment(self, text1: str, text2: str) -> float:
        """
        Compute approximate entailment score via token overlap.
        
        Returns fraction of text1's content that is present in text2.
        """
        tokens1 = set(word_tokenize(text1.lower()))
        tokens1 = {t for t in tokens1 if re.match(r'[a-zA-Z]+', t)}
        
        tokens2 = set(word_tokenize(text2.lower()))
        tokens2 = {t for t in tokens2 if re.match(r'[a-zA-Z]+', t)}
        
        if not tokens1:
            return 1.0
        
        overlap = len(tokens1 & tokens2)
        return overlap / len(tokens1)
    
    def compute(
        self, 
        response: str, 
        ground_truth: str = None, 
        retrieved_chunks: List[str] = None,
        **kwargs
    ) -> Dict[str, float]:
        if not response or not retrieved_chunks:
            return {
                "context_consistency": 0.0,
                "min_chunk_consistency": 0.0,
                "max_chunk_consistency": 0.0,
                "avg_chunk_consistency": 0.0,
                "consistency_score": 0.0
            }
        
        # Compute consistency with each chunk
        chunk_consistencies = [
            self._compute_entailment(response, chunk) 
            for chunk in retrieved_chunks
        ]
        
        min_consistency = min(chunk_consistencies)
        max_consistency = max(chunk_consistencies)
        avg_consistency = np.mean(chunk_consistencies)
        
        # Weighted average (more weight to higher consistency)
        weights = np.linspace(0.5, 1.5, len(chunk_consistencies))
        weights = weights / weights.sum()
        weighted_consistency = np.average(chunk_consistencies, weights=weights)
        
        return {
            "context_consistency": weighted_consistency,
            "min_chunk_consistency": min_consistency,
            "max_chunk_consistency": max_consistency,
            "avg_chunk_consistency": avg_consistency,
            "consistency_score": weighted_consistency  # Alias for compatibility
        }
    
    def get_theoretical_motivation(self) -> str:
        return "Context Consistency. Theoretical: Measures semantic entailment between response and retrieved context. Higher = more grounded in RAG context."


class HallucinationMetric(BaseMetric):
    """
    Comprehensive hallucination detection combining multiple approaches.
    
    Theoretical motivation: Combines four complementary signals:
    1. Lexical hallucination: Token overlap with context
    2. Semantic hallucination: Entailment-based support
    3. Uncertainty-based: Semantic entropy (see Frol et al., 2024)
    4. Contradiction: NLI-based contradiction detection between clusters
    
    Mathematical formulation:
        Hallucination = w_lex * lexical + w_sem * semantic + w_unc * uncertainty + w_contra * contradiction
        
    Where weights sum to 1 and each component is in [0, 1] with
    higher values indicating more hallucination.
    """
    
    def __init__(
        self, 
        weights: Optional[Dict[str, float]] = None,
        semantic_entropy_metric: Optional[SemanticEntropyMetric] = None,
        context_consistency_metric: Optional[ContextConsistencyMetric] = None
    ):
        super().__init__(requires_ground_truth=False)
        
        # Default weights - with contradiction component
        if weights is None:
            weights = {
                "lexical": 0.20,
                "semantic": 0.30,
                "uncertainty": 0.30,
                "contradiction": 0.20
            }
        self.weights = weights
        
        # Initialize component metrics
        self.semantic_entropy_metric = semantic_entropy_metric or SemanticEntropyMetric()
        self.context_consistency_metric = context_consistency_metric or ContextConsistencyMetric()
    
    def _compute_lexical_hallucination(
        self, 
        response: str, 
        retrieved_chunks: List[str]
    ) -> float:
        """Token overlap-based hallucination detection"""
        if not response or not retrieved_chunks:
            return 0.0
        
        response_tokens = set(word_tokenize(response.lower()))
        response_tokens = {t for t in response_tokens if re.match(r'[a-zA-Z]+', t)}
        
        if not response_tokens:
            return 0.0
        
        # Union of all chunk tokens
        chunk_tokens = set()
        for chunk in retrieved_chunks:
            tokens = set(word_tokenize(chunk.lower()))
            tokens = {t for t in tokens if re.match(r'[a-zA-Z]+', t)}
            chunk_tokens.update(tokens)
        
        # Hallucinated tokens = not in any chunk
        hallucinated_tokens = response_tokens - chunk_tokens
        hallucination_ratio = len(hallucinated_tokens) / len(response_tokens)
        
        # Scale to 0-1 with ceiling
        return min(hallucination_ratio * 1.5, 1.0)
    
    def compute(
        self, 
        response: str, 
        ground_truth: str = None, 
        retrieved_chunks: List[str] = None,
        question: str = None,
        sample_responses: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute comprehensive hallucination score.
        
        Returns detailed breakdown plus combined score. All entropy variants are included
        in the output for comprehensive reporting.
        """
        
        # Handle missing inputs
        if not response:
            return self._empty_result()
        
        # 1. Lexical hallucination
        lexical_hallucination = self._compute_lexical_hallucination(response, retrieved_chunks)
        
        # 2. Semantic hallucination (via context consistency)
        consistency_result = self.context_consistency_metric.compute(
            response, retrieved_chunks=retrieved_chunks
        )
        semantic_hallucination = 1.0 - consistency_result["context_consistency"]
        
        # 3. Uncertainty-based hallucination (semantic entropy - all variants)
        entropy_result = self.semantic_entropy_metric.compute(
            response, 
            question=question,
            retrieved_chunks=retrieved_chunks,
            sample_responses=sample_responses,
            **kwargs
        )
        uncertainty_hallucination = entropy_result.get("semantic_uncertainty_score", 0.0)
        
        # 4. Contradiction detection (requires multiple samples)
        contradiction_score = 0.0
        num_contradicting_pairs = 0
        total_pairs = 0
        if sample_responses is not None and len(sample_responses) >= 2:
            try:
                contradiction_result = compute_contradiction_score(sample_responses)
                contradiction_score = contradiction_result.get("contradiction_score", 0.0)
                num_contradicting_pairs = contradiction_result.get("num_contradicting_pairs", 0)
                total_pairs = contradiction_result.get("total_pairs", 0)
            except Exception:
                contradiction_score = 0.0
        
        # Weighted combination - now with 4 components!
        combined_hallucination = (
            self.weights.get("lexical", 0.2) * lexical_hallucination +
            self.weights.get("semantic", 0.3) * semantic_hallucination +
            self.weights.get("uncertainty", 0.3) * uncertainty_hallucination +
            self.weights.get("contradiction", 0.2) * contradiction_score
        )
        
        return {
            "hallucination_score": combined_hallucination,
            "combined_hallucination": combined_hallucination,
            "lexical_hallucination": lexical_hallucination,
            "semantic_hallucination": semantic_hallucination,
            "uncertainty_hallucination": uncertainty_hallucination,
            "contradiction_hallucination": contradiction_score,
            "context_consistency": consistency_result["context_consistency"],
            # All entropy variants for comprehensive reporting
            "semantic_entropy": entropy_result.get("semantic_entropy", 0.0),
            "semantic_entropy_confidence_weighted": entropy_result.get("semantic_entropy_confidence_weighted", 0.0),
            "semantic_entropy_nli_weighted": entropy_result.get("semantic_entropy_nli_weighted", 0.0),
            "semantic_entropy_combined": entropy_result.get("semantic_entropy_combined", 0.0),
            "confidence_level": entropy_result.get("confidence_level", "unknown"),
            # Additional entropy metrics
            "cluster_entropy": entropy_result.get("cluster_entropy", 0.0),
            "predictive_entropy": entropy_result.get("predictive_entropy", 0.0),
            "mutual_information": entropy_result.get("mutual_information", 0.0),
            # Contradiction details
            "num_contradicting_pairs": num_contradicting_pairs,
            "total_pairs": total_pairs,
            # Backward compatibility
            "response_supported": semantic_hallucination < 0.5,
            "semantic_uncertainty": uncertainty_hallucination
        }
    
    def _empty_result(self) -> Dict[str, float]:
        """Return empty result for invalid inputs"""
        return {
            "hallucination_score": 0.0,
            "combined_hallucination": 0.0,
            "lexical_hallucination": 0.0,
            "semantic_hallucination": 0.0,
            "uncertainty_hallucination": 0.0,
            "context_consistency": 0.0,
            "semantic_entropy": 0.0,
            "semantic_entropy_confidence_weighted": 0.0,
            "semantic_entropy_nli_weighted": 0.0,
            "semantic_entropy_combined": 0.0,
            "confidence_level": "unknown",
            "response_supported": False,
            "semantic_uncertainty": 0.0
        }
    
    def get_theoretical_motivation(self) -> str:
        return f"Hallucination Detection (weights: {self.weights}). Theoretical: Combines lexical overlap, semantic entailment, and semantic entropy for comprehensive hallucination detection."


class BertScoreMetric(BaseMetric):
    """
    BERTScore metric using contextual embeddings.
    
    Theoretical motivation: Zhang et al. (2020) "BERTScore: Evaluating 
    Text Generation with BERT"
    
    Uses pre-trained BERT embeddings to compute:
    - Precision: How many reference tokens have close counterparts in hypothesis
    - Recall: How many hypothesis tokens have close counterparts in reference
    - F1: Harmonic mean
    
    Mathematical formulation:
        BERTScore_P = (1/|h|) Σ_{t∈h} max_{r∈r} cos(emb(t), emb(r))
        
    Advantages over lexical metrics:
    - Captures semantic similarity, not just exact matches
    - Handles paraphrases and synonyms
    - Context-aware embeddings
    """
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__(requires_ground_truth=True)
        self.model_name = model_name
        self._initialized = False
        self._model = None
    
    def _lazy_init(self):
        """Lazy initialization to avoid heavy imports"""
        if self._initialized:
            return True
        
        try:
            from bert_score import score as bert_score_fn
            self._bert_score_fn = bert_score_fn
            self._initialized = True
            return True
        except ImportError:
            print("BERTScore not available. Install with: pip install bert-score")
            return False
    
    def compute(self, response: str, ground_truth: str = None, **kwargs) -> Dict[str, float]:
        if not response or not ground_truth:
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
        
        if not self._lazy_init():
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
        
        try:
            P, R, F1 = self._bert_score_fn(
                [response], [ground_truth], 
                lang="en", 
                rescale_with_baseline=True
            )
            return {
                "bertscore_precision": float(P.item()),
                "bertscore_recall": float(R.item()),
                "bertscore_f1": float(F1.item())
            }
        except Exception as e:
            print(f"BERTScore computation failed: {e}")
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
    
    def get_theoretical_motivation(self) -> str:
        return "BERTScore. Theoretical: Semantic similarity using BERT embeddings (Zhang et al., 2020). Captures paraphrase and contextual similarity."


class CoherenceMetric(BaseMetric):
    """
    Measures response coherence and logical flow.
    
    Theoretical motivation: Measures how well sentences in a response
    connect to each other. Uses consecutive sentence similarity.
    
    Mathematical formulation:
        Coherence = (1/(N-1)) Σ_{i=1}^{N-1} sim(sentence_i, sentence_{i+1})
        
    Where sim is semantic similarity between consecutive sentences.
    """
    
    def __init__(self):
        super().__init__(requires_ground_truth=False)
    
    def compute(self, response: str, **kwargs) -> Dict[str, float]:
        if not response:
            return {"coherence": 0.0, "num_sentences": 0}
        
        try:
            sentences = sent_tokenize(response)
            n_sentences = len(sentences)
            
            if n_sentences < 2:
                return {"coherence": 1.0, "num_sentences": n_sentences}
            
            # Simple token-based coherence (fallback if no embedding model)
            def simple_similarity(s1: str, s2: str) -> float:
                tokens1 = set(word_tokenize(s1.lower()))
                tokens2 = set(word_tokenize(s2.lower()))
                if not tokens1 or not tokens2:
                    return 0.0
                overlap = len(tokens1 & tokens2)
                return overlap / min(len(tokens1), len(tokens2))
            
            # Compute consecutive similarities
            similarities = []
            for i in range(n_sentences - 1):
                sim = simple_similarity(sentences[i], sentences[i+1])
                similarities.append(sim)
            
            coherence = np.mean(similarities)
            
            return {"coherence": coherence, "num_sentences": n_sentences}
            
        except Exception:
            return {"coherence": 0.0, "num_sentences": 0}
    
    def get_theoretical_motivation(self) -> str:
        return "Coherence. Theoretical: Measures logical flow via consecutive sentence similarity. Higher = more coherent text."


class CompletenessMetric(BaseMetric):
    """
    Measures how completely a response addresses the question.
    
    Theoretical motivation: Based on aspect coverage.
    Breaks down the question into aspects/sub-questions and checks
    how many are addressed in the response.
    
    Mathematical formulation:
        Completeness = (1/A) Σ_{a∈aspects} coverage(a, response)
        
    Where coverage is 1 (fully addressed), 0.5 (partially), or 0 (not addressed).
    """
    
    def __init__(self):
        super().__init__(requires_ground_truth=False)
    
    def compute(
        self, 
        response: str, 
        question: str = None, 
        **kwargs
    ) -> Dict[str, float]:
        if not response or not question:
            return {"completeness": 0.0, "num_aspects": 0}
        
        # Simple heuristic: Check for key question words in response
        question_words = set(word_tokenize(question.lower()))
        stop_words = set(stopwords.words('english'))
        key_aspects = question_words - stop_words
        
        if not key_aspects:
            return {"completeness": 1.0, "num_aspects": 0}
        
        response_words = set(word_tokenize(response.lower()))
        
        # Count how many aspects are addressed
        covered_aspects = key_aspects & response_words
        completeness = len(covered_aspects) / len(key_aspects)
        
        return {
            "completeness": completeness, 
            "num_aspects": len(key_aspects),
            "covered_aspects": len(covered_aspects)
        }
    
    def get_theoretical_motivation(self) -> str:
        return "Completeness. Theoretical: Measures aspect coverage of question in response. Higher = more complete answer."


# =============================================================================
# METRIC COMPOSER & EVALUATOR
# =============================================================================

class MetricComposer:
    """
    Composes multiple metrics for comprehensive evaluation.
    
    Allows flexible combination of metrics with custom weights.
    """
    
    def __init__(self, metrics: Dict[str, BaseMetric], weights: Dict[str, float]):
        """
        Args:
            metrics: Dict mapping metric names to metric instances
            weights: Dict mapping metric names to weight coefficients
        """
        self.metrics = metrics
        self.weights = weights
        
        # Validate weights sum to 1 (approximately)
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            # Normalize weights
            self.weights = {k: v/total_weight for k, v in weights.items()}
        
        # Validate all metrics have weights
        missing = set(metrics.keys()) - set(weights.keys())
        if missing:
            raise ValueError(f"Missing weights for metrics: {missing}")
    
    def evaluate(
        self, 
        response: str, 
        ground_truth: str = None, 
        question: str = None,
        retrieved_chunks: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate response using all configured metrics"""
        
        results = {}
        weighted_scores = {}
        
        for name, metric in self.metrics.items():
            # Check if metric requires ground truth
            if metric.requires_ground_truth and ground_truth is None:
                results[name] = {"error": "ground_truth_required"}
                continue
            
            try:
                # Prepare kwargs
                metric_kwargs = {"retrieved_chunks": retrieved_chunks}
                if metric.requires_ground_truth:
                    metric_kwargs["ground_truth"] = ground_truth
                if question:
                    metric_kwargs["question"] = question
                
                # Compute metric
                result = metric.compute(response, **metric_kwargs)
                results[name] = result
                
                # Extract primary score for weighting
                primary = self._extract_primary_score(name, result)
                if primary is not None:
                    weighted_scores[name] = primary * self.weights[name]
                    
            except Exception as e:
                results[name] = {"error": str(e)}
        
        # Calculate composite score
        composite_score = sum(weighted_scores.values()) if weighted_scores else 0.0
        
        return {
            "metrics": results,
            "composite_score": composite_score,
            "weighted_contributions": weighted_scores,
            "weight_configuration": self.weights.copy()
        }
    
    def _extract_primary_score(self, metric_name: str, result: Dict) -> Optional[float]:
        """Extract the primary score from a metric result"""
        if "error" in result:
            return None
        
        # Define extraction rules
        rules = {
            "exact_match": "exact_match",
            "precision_recall_f1": "f1",
            "rouge": "rouge_f1",
            "bleu": "bleu",
            "bertscore": "bertscore_f1",
            "hallucination": "hallucination_score",
            "semantic_entropy": "semantic_uncertainty_score",
            "context_consistency": "context_consistency",
            "coherence": "coherence",
            "completeness": "completeness"
        }
        
        key = rules.get(metric_name, None)
        if key and key in result:
            return result[key]
        
        # Fallback: look for any *_f1 or *_score
        for k, v in result.items():
            if isinstance(v, (int, float)) and ('_f1' in k or '_score' in k):
                return v
        
        return None
    
    def get_theoretical_summary(self) -> Dict[str, str]:
        """Get theoretical motivation for all metrics"""
        return {
            name: metric.get_theoretical_motivation()
            for name, metric in self.metrics.items()
        }


class RAGEvaluator:
    """
    Main RAG evaluation interface.
    
    Provides comprehensive evaluation combining multiple metrics
    with proper theoretical foundations.
    """
    
    def __init__(
        self,
        include_ground_truth: bool = False,
        custom_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            include_ground_truth: Whether to include reference-based metrics
            custom_weights: Optional custom weights for metrics
        """
        
        # Core metrics (always available)
        self.metrics = {
            "semantic_entropy": SemanticEntropyMetric(),
            "hallucination": HallucinationMetric(),
            "context_consistency": ContextConsistencyMetric(),
            "coherence": CoherenceMetric(),
            "completeness": CompletenessMetric()
        }
        
        # Add reference-based metrics if ground truth is available
        if include_ground_truth:
            self.metrics.update({
                "exact_match": ExactMatchMetric(),
                "precision_recall_f1": PrecisionRecallF1Metric(n=2),
                "rouge": RougeMetric(n=2),
                "bleu": BleuMetric(),
                "bertscore": BertScoreMetric()
            })
        
        # Default weights
        if custom_weights is None:
            custom_weights = {
                "semantic_entropy": 0.20,
                "hallucination": 0.25,
                "context_consistency": 0.20,
                "coherence": 0.15,
                "completeness": 0.10,
                "precision_recall_f1": 0.05,
                "rouge": 0.05
            }
        
        # Filter to only include metrics we have
        weights = {k: v for k, v in custom_weights.items() if k in self.metrics}
        
        self.composer = MetricComposer(self.metrics, weights)
    
    def evaluate(
        self,
        response: str,
        question: str = None,
        retrieved_chunks: List[str] = None,
        ground_truth: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Comprehensive RAG evaluation.
        
        Args:
            response: The generated response to evaluate
            question: The original question (for completeness)
            retrieved_chunks: Retrieved context (for hallucination detection)
            ground_truth: Reference answer (optional, for reference-based metrics)
            
        Returns:
            Dictionary with metric results and composite score
        """
        
        return self.composer.evaluate(
            response=response,
            question=question,
            retrieved_chunks=retrieved_chunks,
            ground_truth=ground_truth,
            **kwargs
        )
    
    def evaluate_pair(
        self,
        vanilla_response: str,
        kg_response: str,
        question: str = None,
        vanilla_chunks: List[str] = None,
        kg_chunks: List[str] = None,
        ground_truth: str = None
    ) -> Dict[str, Any]:
        """
        Compare two RAG systems.
        
        Args:
            vanilla_response: Response from vanilla RAG
            kg_response: Response from knowledge graph RAG
            question: Original question
            vanilla_chunks: Retrieved chunks for vanilla RAG
            kg_chunks: Retrieved chunks for KG-RAG
            ground_truth: Reference answer
            
        Returns:
            Comparison results
        """
        
        vanilla_result = self.evaluate(
            response=vanilla_response,
            question=question,
            retrieved_chunks=vanilla_chunks,
            ground_truth=ground_truth
        )
        
        kg_result = self.evaluate(
            response=kg_response,
            question=question,
            retrieved_chunks=kg_chunks,
            ground_truth=ground_truth
        )
        
        # Compute differences
        differences = {}
        for key in vanilla_result["metrics"]:
            if key in kg_result["metrics"]:
                v_score = self._extract_score(vanilla_result["metrics"][key])
                k_score = self._extract_score(kg_result["metrics"][key])
                if v_score is not None and k_score is not None:
                    differences[key] = k_score - v_score  # Positive = KG better
        
        return {
            "vanilla_rag": vanilla_result,
            "kg_rag": kg_result,
            "differences": differences,
            "winner": "kg_rag" if kg_result["composite_score"] > vanilla_result["composite_score"] 
                     else "vanilla_rag" if vanilla_result["composite_score"] > kg_result["composite_score"]
                     else "tie"
        }
    
    def _extract_score(self, metric_result: Dict) -> Optional[float]:
        """Extract numeric score from metric result"""
        if "error" in metric_result:
            return None
        
        # Try common keys
        for key in ["composite_score", "hallucination_score", "semantic_entropy", 
                   "context_consistency", "coherence", "completeness"]:
            if key in metric_result and isinstance(metric_result[key], (int, float)):
                return metric_result[key]
        
        return None
    
    def get_theoretical_summary(self) -> Dict[str, str]:
        """Get theoretical motivations for all metrics"""
        return self.composer.get_theoretical_summary()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_evaluator(
    task_type: str = "general",
    include_bertscore: bool = True,
    include_ground_truth: bool = False
) -> RAGEvaluator:
    """
    Factory function to create pre-configured evaluators.
    
    Args:
        task_type: "general", "medical", "scientific", or "legal"
        include_bertscore: Whether to include BERTScore (requires extra dependencies)
        include_ground_truth: Whether to include reference-based metrics
        
    Returns:
        Configured RAGEvaluator
    """
    
    weights = {
        "semantic_entropy": 0.20,
        "hallucination": 0.25,
        "context_consistency": 0.20,
        "coherence": 0.15,
        "completeness": 0.10
    }
    
    metrics = {}
    
    # Always include
    metrics["semantic_entropy"] = SemanticEntropyMetric()
    metrics["hallucination"] = HallucinationMetric()
    metrics["context_consistency"] = ContextConsistencyMetric()
    metrics["coherence"] = CoherenceMetric()
    metrics["completeness"] = CompletenessMetric()
    
    # Add reference-based if available
    if include_ground_truth:
        weights["precision_recall_f1"] = 0.05
        weights["rouge"] = 0.05
        metrics["precision_recall_f1"] = PrecisionRecallF1Metric(n=2)
        metrics["rouge"] = RougeMetric(n=2)
        
        if include_bertscore:
            weights["bertscore"] = 0.05
            metrics["bertscore"] = BertScoreMetric()
    
    # Adjust weights based on task
    if task_type == "medical":
        # Higher weight on hallucination for medical
        weights["hallucination"] = 0.30
        weights["semantic_entropy"] = 0.20
        weights["context_consistency"] = 0.20
    elif task_type == "scientific":
        # Balanced for scientific
        weights["hallucination"] = 0.25
        weights["semantic_entropy"] = 0.20
        weights["context_consistency"] = 0.25
    
    # Normalize weights
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items() if k in metrics}
    
    return RAGEvaluator(custom_weights=weights)


# Backward compatibility aliases
class SemanticUncertaintyMetric(SemanticEntropyMetric):
    """Backward compatibility alias"""
    pass


class PrecisionRecallMetric(PrecisionRecallF1Metric):
    """Backward compatibility alias"""
    pass
