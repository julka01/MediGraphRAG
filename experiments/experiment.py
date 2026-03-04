"""
MIRAGE Dataset Evaluation Pipeline (Experiment Mode)

This script runs sequential evaluation on MIRAGE datasets in experiment mode:
For each dataset:
   1. Clear Neo4j
   2. Build KG from that dataset's contexts using proper entity extraction
   3. Run RAG comparison experiments (Vanilla vs KG-RAG)
   4. Compute hallucination metrics
   5. Save results

Usage:
    python experiments/run_mirage_evaluation.py --num-samples 10
"""

import sys
import os
import json
import logging
import time
import argparse
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()
os.environ["EMBEDDING_PROVIDER"] = "sentence_transformers"

from medigraphrag_x.rag.systems.vanilla_rag_system import VanillaRAGSystem
from medigraphrag_x.rag.systems.enhanced_rag_system import EnhancedRAGSystem
from medigraphrag_x.providers.model_providers import get_provider as get_model_provider, LangChainRunnableAdapter
from medigraphrag_x.kg.builders.enhanced_kg_creator import UnifiedOntologyGuidedKGCreator
from neo4j import GraphDatabase

# Import hallucination metrics from rag_metrics
from experiments.rag_metrics import HallucinationMetric, SemanticEntropyMetric

import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class MIRAGEEvaluationPipeline:
    """Sequential evaluation pipeline for MIRAGE datasets"""
    
    def __init__(
        self,
        num_samples: int = None,
        entropy_samples: int = 10,
        llm_provider: str = "openrouter",
        llm_model: str = "openai/gpt-oss-120b:free",
        eval_configs: Optional[List[Dict[str, Any]]] = None,
        skip_kg_build: bool = False,
    ):
        self.num_samples = num_samples  # None means use all questions
        self.entropy_samples = max(1, min(entropy_samples, 20))  # safety cap
        self.llm_provider_name = llm_provider
        self.llm_model = llm_model
        self.skip_kg_build = skip_kg_build

        # Retrieval/eval configurations (supports per-config comparisons)
        self.eval_configs = eval_configs or [
            {
                "name": "default",
                "similarity_threshold": 0.1,
                "max_chunks": 10,
            }
        ]
        
        # Initialize LLM
        provider = get_model_provider(self.llm_provider_name, model=self.llm_model)
        self.llm = LangChainRunnableAdapter(provider, model=self.llm_model)
        # KG extraction uses the raw provider interface (generate), same as app pipeline
        self.kg_llm_provider = provider
        
        # Get embedding provider for consistent use across all components
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
        
        # RAG systems - use explicit embedding model for consistency
        self.vanilla_rag = VanillaRAGSystem(embedding_model=self.embedding_provider)
        self.kg_rag = EnhancedRAGSystem(embedding_model=self.embedding_provider)
        
        # Neo4j connection
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        # W&B run
        self.wandb_run = None

    def _log_question_table_to_wandb(
        self,
        dataset_name: str,
        config_name: str,
        details: List[Dict[str, Any]],
    ):
        """Log per-question table (question + responses + metrics) to W&B."""
        if not self.wandb_run:
            return

        table = wandb.Table(columns=[
            "dataset",
            "config",
            "question_id",
            "question",
            "expected",
            "vanilla_response",
            "kg_response",
            "vanilla_correct",
            "kg_correct",
            "vanilla_hallucination_score",
            "kg_hallucination_score",
            "vanilla_semantic_entropy",
            "kg_semantic_entropy",
            "vanilla_semantic_entropy_nli",
            "kg_semantic_entropy_nli",
        ])

        for d in details:
            table.add_data(
                dataset_name,
                config_name,
                d.get("question_id", ""),
                d.get("question", ""),
                d.get("expected", ""),
                d.get("vanilla_response", ""),
                d.get("kg_response", ""),
                int(bool(d.get("vanilla_correct", False))),
                int(bool(d.get("kg_correct", False))),
                float(d.get("vanilla_hallucination_score", 0.0)),
                float(d.get("kg_hallucination_score", 0.0)),
                float(d.get("vanilla_semantic_entropy", 0.0)),
                float(d.get("kg_semantic_entropy", 0.0)),
                float(d.get("vanilla_semantic_entropy_nli", 0.0)),
                float(d.get("kg_semantic_entropy_nli", 0.0)),
            )

        self.wandb_run.log({
            f"tables/{dataset_name}/{config_name}/questions_and_responses": table
        })

    def _log_config_summary_to_wandb(
        self,
        dataset_name: str,
        config_results: List[Dict[str, Any]],
    ):
        """Log per-config summary table and grouped bar charts to W&B."""
        if not self.wandb_run or not config_results:
            return

        summary_table = wandb.Table(columns=[
            "dataset",
            "config",
            "vanilla_accuracy",
            "kg_accuracy",
            "accuracy_gain_kg_minus_vanilla",
            "vanilla_hallucination",
            "kg_hallucination",
            "vanilla_semantic_entropy",
            "kg_semantic_entropy",
            "vanilla_semantic_entropy_nli",
            "kg_semantic_entropy_nli",
        ])

        for r in config_results:
            config = r.get("config", {})
            config_name = config.get("name", "default")

            v_acc = float(r.get("vanilla_accuracy", 0.0))
            k_acc = float(r.get("kg_accuracy", 0.0))
            v_h = float(r.get("vanilla_avg_hallucination_score", 0.0))
            k_h = float(r.get("kg_avg_hallucination_score", 0.0))
            v_se = float(r.get("vanilla_avg_semantic_entropy", 0.0))
            k_se = float(r.get("kg_avg_semantic_entropy", 0.0))
            v_se_nli = float(r.get("vanilla_avg_semantic_entropy_nli", 0.0))
            k_se_nli = float(r.get("kg_avg_semantic_entropy_nli", 0.0))

            summary_table.add_data(
                dataset_name,
                config_name,
                v_acc,
                k_acc,
                k_acc - v_acc,
                v_h,
                k_h,
                v_se,
                k_se,
                v_se_nli,
                k_se_nli,
            )

        self.wandb_run.log({f"tables/{dataset_name}/config_summary": summary_table})

        # Grouped bar charts per measure by configuration
        if HAS_MATPLOTLIB:
            config_names = [r.get("config", {}).get("name", "default") for r in config_results]
            x = list(range(len(config_names)))
            width = 0.36

            metric_specs = [
                ("accuracy", "vanilla_accuracy", "kg_accuracy"),
                ("hallucination", "vanilla_avg_hallucination_score", "kg_avg_hallucination_score"),
                ("semantic_entropy", "vanilla_avg_semantic_entropy", "kg_avg_semantic_entropy"),
                ("semantic_entropy_nli", "vanilla_avg_semantic_entropy_nli", "kg_avg_semantic_entropy_nli"),
            ]

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()

            for i, (title, v_key, k_key) in enumerate(metric_specs):
                ax = axes[i]
                v_vals = [float(r.get(v_key, 0.0)) for r in config_results]
                k_vals = [float(r.get(k_key, 0.0)) for r in config_results]

                ax.bar([p - width/2 for p in x], v_vals, width, label="Vanilla", color="#1f77b4")
                ax.bar([p + width/2 for p in x], k_vals, width, label="KG-RAG", color="#ff7f0e")
                ax.set_title(title.replace("_", " ").title())
                ax.set_xticks(x)
                ax.set_xticklabels(config_names, rotation=20, ha="right")
                ax.grid(axis="y", alpha=0.3)
                ax.legend()

            plt.tight_layout()
            self.wandb_run.log({f"charts/{dataset_name}/metrics_by_config": wandb.Image(fig)})
            plt.close(fig)

    def _extract_chunk_texts_from_result(self, rag_result: Dict[str, Any]) -> List[str]:
        """Extract retrieved chunk texts robustly from a RAG result."""
        chunk_texts: List[str] = []

        # Prefer used_chunks when available
        used_chunks = rag_result.get("used_chunks", [])
        for c in used_chunks:
            if isinstance(c, dict):
                text = c.get("text", "")
            else:
                text = str(c)
            if text:
                chunk_texts.append(text)

        # Fall back to context.chunks
        if not chunk_texts:
            context = rag_result.get("context", {}) if isinstance(rag_result, dict) else {}
            context_chunks = context.get("chunks", []) if isinstance(context, dict) else []
            for c in context_chunks:
                if isinstance(c, dict):
                    text = c.get("text", "")
                else:
                    text = str(c)
                if text:
                    chunk_texts.append(text)

        # Deduplicate while preserving order
        unique_chunk_texts = []
        seen = set()
        for t in chunk_texts:
            key = t.strip()
            if key and key not in seen:
                seen.add(key)
                unique_chunk_texts.append(key)

        return unique_chunk_texts

    def _collect_sample_responses(
        self,
        rag_system,
        question: str,
        base_result: Optional[Dict[str, Any]] = None,
        document_names: Optional[List[str]] = None,
        similarity_threshold: float = 0.1,
        max_chunks: int = 10,
        extra_context_texts: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """Collect multiple responses for proper semantic-entropy estimation.

        extra_context_texts is forwarded to each generate_response call so that
        entropy samples receive the same ground-truth context as the main response,
        preventing the artefact where samples without context always return a
        fixed fallback string (which creates a spurious 2-cluster, fixed-entropy
        distribution of exactly 0.9183 bits).
        """
        responses: List[str] = []
        retrieved_chunk_texts: List[str] = []

        # Use already computed response as first sample (avoids duplicate call)
        if base_result:
            base_response = str(base_result.get("response", "")).strip()
            if base_response:
                responses.append(base_response)
            retrieved_chunk_texts = self._extract_chunk_texts_from_result(base_result)

        # Generate additional samples if requested
        remaining = max(0, self.entropy_samples - len(responses))
        for _ in range(remaining):
            try:
                sampled_result = rag_system.generate_response(
                    question=question,
                    llm=self.llm,
                    document_names=document_names,
                    similarity_threshold=similarity_threshold,
                    max_chunks=max_chunks,
                    extra_context_texts=extra_context_texts,
                )
                sampled_response = str(sampled_result.get("response", "")).strip()
                if sampled_response:
                    responses.append(sampled_response)

                # If we still don't have chunk context, take from sampled call
                if not retrieved_chunk_texts:
                    retrieved_chunk_texts = self._extract_chunk_texts_from_result(sampled_result)
            except Exception as e:
                logging.warning(f"Failed to collect semantic-entropy sample: {e}")

        # Ensure at least one response exists
        if not responses and base_result:
            responses = [str(base_result.get("response", "")).strip()]

        return responses, retrieved_chunk_texts
        
    def _get_neo4j_driver(self):
        """Get Neo4j driver"""
        return GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
    def _clear_neo4j(self):
        """Clear all data from Neo4j"""
        logging.info("Clearing Neo4j database...")
        driver = self._get_neo4j_driver()
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logging.info("Neo4j cleared")
        driver.close()
        
    def _load_mirage_raw_data(self, dataset_name: str) -> Dict[str, Any]:
        """Load MIRAGE raw data (with CONTEXTS) for a specific dataset"""
        raw_data_files = {
            "pubmedqa": "MIRAGE/rawdata/pubmedqa/data/test_set.json",
            "bioasq": "MIRAGE/rawdata/bioasq/Task10BGoldenEnriched/10B1_golden.json",
            "medqa": "MIRAGE/rawdata/medqa/data/test.json",
            "medmcqa": "MIRAGE/rawdata/medmcqa/data/test.json",
            "mmlu": "MIRAGE/rawdata/mmlu/data/test/professional_medicine_test.csv"
        }
        
        file_path = raw_data_files.get(dataset_name)
        if not file_path or not os.path.exists(file_path):
            logging.warning(f"Raw data file not found for {dataset_name}: {file_path}")
            return {}
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # For BioASQ, restructure to match other datasets (dict with id -> question_data)
        if dataset_name == "bioasq" and "questions" in data:
            restructured = {}
            for idx, q in enumerate(data["questions"]):
                q_id = q.get("id", f"bioasq_{idx}")
                restructured[q_id] = q
            return restructured
        
        return data
    
    def _extract_contexts_from_question(self, question_data: Dict, dataset_name: str = None) -> List[str]:
        """Extract context passages from a question"""
        # Try PubMedQA style (CONTEXTS field)
        contexts = question_data.get("CONTEXTS", [])
        if isinstance(contexts, list) and contexts:
            return [c if isinstance(c, str) else c.get("text", "") for c in contexts]
        
        # Try BioASQ style (snippets field)
        snippets = question_data.get("snippets", [])
        if isinstance(snippets, list) and snippets:
            return [s.get("text", "") for s in snippets if s.get("text")]
        
        return []
    
    def _get_answer_from_question(self, question_data: Dict) -> str:
        """Extract ground truth answer from question data"""
        # Try PubMedQA style
        if "final_decision" in question_data:
            return question_data["final_decision"].lower()
        
        # Try BioASQ style
        if "exact_answer" in question_data:
            return str(question_data["exact_answer"]).lower()
        
        # Try generic
        if "answer" in question_data:
            return str(question_data["answer"]).lower()
        if "Answer" in question_data:
            return str(question_data["Answer"]).lower()
        return ""
    
    def _get_question_text(self, question_data: Dict) -> str:
        """Extract question text from question data"""
        # Try BioASQ style
        if "body" in question_data:
            return question_data["body"]
        # Try PubMedQA style
        if "QUESTION" in question_data:
            return question_data["QUESTION"]
        if "question" in question_data:
            return question_data["question"]
        return ""

    def _normalize_decision_label(self, text: str) -> str:
        """Normalize explicit labels into yes/no/maybe when possible."""
        if not text:
            return ""

        t = str(text).strip().lower()

        # 1) Exact/compact labels (safe, no ambiguity).
        if t in {"yes", "y", "true"}:
            return "yes"
        if t in {"no", "n", "false"}:
            return "no"
        if t in {"maybe", "uncertain", "unknown"}:
            return "maybe"

        # 2) Label at start of answer or in explicit "answer is ..." pattern.
        lead = t[:120]
        if re.search(r"^\s*(yes|true)\b", lead) or re.search(r"\b(answer|conclusion|final answer)\s*(is|:)\s*(yes|true)\b", t):
            return "yes"
        if re.search(r"^\s*(no|false)\b", lead) or re.search(r"\b(answer|conclusion|final answer)\s*(is|:)\s*(no|false)\b", t):
            return "no"
        if re.search(r"^\s*(maybe|uncertain)\b", lead) or re.search(r"\b(answer|conclusion|final answer)\s*(is|:)\s*(maybe|uncertain)\b", t):
            return "maybe"

        return ""

    def _infer_decision_from_response(self, response: str, question: str = "") -> str:
        """
        Infer yes/no/maybe from free-form response text.

        This is intentionally heuristic for PubMedQA-style outputs where the
        model may not explicitly say "yes"/"no" but still provides a clear
        conclusion (e.g., "methods were comparable" for a suitability question).
        """
        if not response:
            return ""

        t = re.sub(r"\s+", " ", str(response).strip().lower())
        q = re.sub(r"\s+", " ", str(question or "").strip().lower())
        lead = t[:220]

        # 1) Explicit labels first.
        if re.search(r"^\s*(yes|true)\b", lead) or re.search(r"\b(answer|conclusion)\s*(is|:)\s*(yes|true)\b", t):
            return "yes"
        if re.search(r"^\s*(no|false)\b", lead) or re.search(r"\b(answer|conclusion)\s*(is|:)\s*(no|false)\b", t):
            return "no"
        if re.search(r"^\s*(maybe|uncertain)\b", lead) or re.search(r"\b(answer|conclusion)\s*(is|:)\s*(maybe|uncertain)\b", t):
            return "maybe"

        # 2) Insufficient/inconclusive context => maybe.
        insufficient_patterns = [
            r"\bnot enough information\b",
            r"\binsufficient (data|evidence|information)\b",
            r"\bcannot (determine|conclude|say)\b",
            r"\binconclusive\b",
            r"\bunclear\b",
            r"\bno direct information\b",
            r"\bcontext does not provide\b",
            r"\bsubject of investigation\b",
        ]
        if any(re.search(p, t) for p in insufficient_patterns):
            return "maybe"

        # 3) Evidence-based polarity cues.
        positive_patterns = [
            r"\b(correlated|associated|connected)\b",
            r"\bcorrelated closely\b",
            r"\bcomparable\b",
            r"\bviable alternative\b",
            r"\bsuitable as an alternative\b",
            r"\b(beneficial|valuable|effective)\b",
            r"\bindicat(es|ing)\b",
            r"\bsupport(s|ed|ive)\b",
            r"\bsignificant (association|correlation|benefit)\b",
        ]
        negative_patterns = [
            r"\bno evidence\b",
            r"\bno (association|correlation|connection|benefit)\b",
            r"\bnot (associated|correlated|connected|beneficial|effective|suitable)\b",
            r"\bfailed to (show|demonstrate)\b",
            r"\bdoes not (support|indicate|show|demonstrate)\b",
        ]

        pos_score = sum(1 for p in positive_patterns if re.search(p, t))
        neg_score = sum(1 for p in negative_patterns if re.search(p, t))

        if pos_score > 0 and neg_score == 0:
            return "yes"
        if neg_score > 0 and pos_score == 0:
            return "no"
        if pos_score > neg_score:
            return "yes"
        if neg_score > pos_score:
            return "no"

        # 4) Question-aware fallback for yes/no prompts.
        if q.startswith(("is ", "are ", "does ", "do ", "can ", "should ", "was ", "were ", "has ", "have ")):
            if re.search(r"\b(comparable|viable alternative|correlated|associated|beneficial|effective)\b", t):
                return "yes"
            if re.search(r"\b(no evidence|not associated|not correlated|not beneficial|not effective)\b", t):
                return "no"

        return ""

    def _is_answer_correct(self, expected_answer: str, model_response: str, question: str = "") -> bool:
        """Flexible correctness check for different question types:
        - Binary (yes/no): checks if response starts with explicit yes/no
        - Factoid: checks if key answer appears in response
        - List: checks if all expected items appear in response
        - Multiple choice: checks if correct option appears
        """
        expected_answer = str(expected_answer) if expected_answer else ""
        model_response = str(model_response) if model_response else ""
        
        # Normalize for comparison
        expected_lower = expected_answer.lower().strip()
        response_lower = model_response.lower().strip()
        
        # 1. First try binary (yes/no/maybe) matching - existing logic
        expected_binary = self._normalize_decision_label(expected_answer)
        predicted_binary = self._normalize_decision_label(model_response)
        if expected_binary and predicted_binary:
            return expected_binary == predicted_binary
        
        # 2. If not binary, try to extract expected answer(s) from nested list format
        # Handle formats like: "[['xia']]", "[['casirivimab'], ['imdevimab']]", "['answer']"
        expected_items = []
        
        # Try parsing as Python literal (list of lists)
        try:
            import ast
            parsed = ast.literal_eval(expected_answer)
            if isinstance(parsed, list):
                # Flatten nested lists: [['a'], ['b']] -> ['a', 'b']
                for item in parsed:
                    if isinstance(item, list):
                        expected_items.extend([str(x).lower().strip() for x in item])
                    else:
                        expected_items.append(str(item).lower().strip())
        except (ValueError, SyntaxError):
            # If parsing fails, treat as plain string
            # Remove brackets and quotes for simple factoid answers like "xia"
            cleaned = expected_answer.strip("[]'\"").lower()
            if cleaned:
                expected_items = [cleaned]
        
        if not expected_items:
            # Fallback: use the raw expected answer
            expected_items = [expected_lower]
        
        # 3. Check if expected items are present in the response
        # For list questions, ALL items must be present (AND logic)
        # For factoid questions, at least one item must be present (OR logic)
        
        # Check if response contains any of the expected items
        items_found = [item in response_lower for item in expected_items]
        
        if not items_found:
            return False
        
        # If there's only one expected item, use OR logic (factoid)
        if len(expected_items) == 1:
            return items_found[0]
        
        # If there are multiple expected items, require ALL to be present (list questions)
        # e.g., both "casirivimab" AND "imdevimab" for REGEN-COV
        return all(items_found)
    
    def _build_kg_for_dataset(self, dataset_name: str) -> bool:
        """
        Build knowledge graph from dataset contexts using the SAME pipeline class
        as the app endpoint (`UnifiedOntologyGuidedKGCreator`).
        """
        logging.info(f"Building KG for dataset: {dataset_name}")

        # Step 1: Clear Neo4j
        self._clear_neo4j()

        # Step 2: Load dataset and extract contexts
        dataset = self._load_mirage_raw_data(dataset_name)
        if not dataset:
            logging.error(f"No raw data found for {dataset_name}")
            return False

        # Extract all contexts
        all_contexts = []
        for q_id, q_data in dataset.items():
            if isinstance(q_data, dict):
                contexts = self._extract_contexts_from_question(q_data)
                all_contexts.extend(contexts)

        # Remove duplicates
        unique_contexts = list(set(all_contexts))
        logging.info(f"Found {len(unique_contexts)} unique contexts in {dataset_name}")

        if not unique_contexts:
            logging.warning(f"No contexts found for {dataset_name}")
            return False

        # Step 3: Build KG through unified creator pipeline (same as app)
        try:
            corpus_text = "\n\n".join(unique_contexts)

            kg_creator = UnifiedOntologyGuidedKGCreator(
                chunk_size=1500,
                chunk_overlap=200,
                neo4j_uri=self.neo4j_uri,
                neo4j_user=self.neo4j_user,
                neo4j_password=self.neo4j_password,
                neo4j_database="neo4j",
                embedding_model=self.embedding_provider,
            )

            # Use same KG generation path + LLM extraction interface as app endpoint.
            kg = kg_creator.generate_knowledge_graph(
                text=corpus_text,
                llm=self.kg_llm_provider,
                file_name=dataset_name,
                model_name=self.llm_model,
                kg_name=dataset_name,
            )

            stored = bool(kg.get("metadata", {}).get("stored_in_neo4j", False))
            if not stored:
                logging.error(f"Unified KG pipeline did not store graph for {dataset_name}")
                return False

            logging.info(
                f"Successfully populated Neo4j with unified pipeline for {dataset_name} | "
                f"chunks={kg.get('metadata', {}).get('total_chunks', 0)}, "
                f"entities={kg.get('metadata', {}).get('total_entities', 0)}, "
                f"relationships={kg.get('metadata', {}).get('total_relationships', 0)}"
            )
            return True
            
        except Exception as e:
            logging.error(f"Failed to populate Neo4j for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    
    def _run_evaluation_on_dataset(self, dataset_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run RAG comparison: Vanilla RAG vs KG-RAG using Neo4j database"""
        config = config or {"name": "default", "similarity_threshold": 0.1, "max_chunks": 10}
        config_name = config.get("name", "default")
        similarity_threshold = float(config.get("similarity_threshold", 0.1))
        max_chunks = int(config.get("max_chunks", 10))

        logging.info(
            f"Running RAG comparison on {dataset_name} | config={config_name} "
            f"(threshold={similarity_threshold}, max_chunks={max_chunks})"
        )
        
        # Load dataset
        dataset = self._load_mirage_raw_data(dataset_name)
        if not dataset:
            logging.error(f"No raw data found for {dataset_name}")
            return {"dataset": dataset_name, "error": "No raw data found"}
        
        # Get valid questions
        valid_questions = []
        for q_id, q_data in dataset.items():
            if isinstance(q_data, dict):
                contexts = self._extract_contexts_from_question(q_data)
                if contexts:
                    valid_questions.append((q_id, q_data))
        
        questions = valid_questions[:self.num_samples] if self.num_samples else valid_questions
        
        results = {
            "dataset": dataset_name,
            "config": {
                "name": config_name,
                "similarity_threshold": similarity_threshold,
                "max_chunks": max_chunks,
            },
            "total_questions": len(questions),
            "vanilla_rag_correct": 0,
            "kg_rag_correct": 0,
            "details": []
        }
        
        # Accumulate hallucination scores for averaging
        vanilla_hallucination_scores = []
        kg_hallucination_scores = []
        
        for q_idx, (q_id, q_data) in enumerate(questions):
            question = self._get_question_text(q_data)
            expected_answer = self._get_answer_from_question(q_data)
            
            if not question or not expected_answer:
                continue
            
            logging.info(f"[{q_idx+1}/{len(questions)}] Processing: {question[:50]}...")

            # Extract the question's own contexts (ground-truth source material).
            # These are passed as extra_context_texts so the LLM always has the
            # relevant abstracts available, even when vector retrieval misses them.
            question_contexts = self._extract_contexts_from_question(q_data)

            # Run Vanilla RAG
            vanilla_result = {}
            try:
                vanilla_result = self.vanilla_rag.generate_response(
                    question=question,
                    llm=self.llm,
                    similarity_threshold=similarity_threshold,
                    max_chunks=max_chunks,
                    extra_context_texts=question_contexts,
                )
                vanilla_response = vanilla_result.get("response", "").lower()
            except Exception as e:
                logging.error(f"Vanilla RAG error: {e}")
                vanilla_response = ""
            
            # Run KG-RAG
            kg_result = {}
            try:
                kg_result = self.kg_rag.generate_response(
                    question=question,
                    llm=self.llm,
                    similarity_threshold=similarity_threshold,
                    max_chunks=max_chunks,
                    extra_context_texts=question_contexts,
                )
                kg_response = kg_result.get("response", "").lower()
            except Exception as e:
                logging.error(f"KG-RAG error: {e}")
                kg_response = ""
            
            # Check correctness
            vanilla_correct = self._is_answer_correct(expected_answer, vanilla_response, question=question)
            kg_correct = self._is_answer_correct(expected_answer, kg_response, question=question)
            
            # Use proper HallucinationMetric for hallucination + semantic entropy detection
            # Pass the embedding model used by RAG to ensure consistency
            # The embedding model should match what was used for KG storage and RAG retrieval
            # HallucinationMetric expects full model name like "sentence-transformers/all-MiniLM-L6-v2"
            if self.embedding_provider == "sentence_transformers":
                embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            elif self.embedding_provider == "openai":
                embedding_model_name = "openai/text-embedding-ada-002"
            else:
                embedding_model_name = f"sentence-transformers/{self.embedding_provider}"
            # Use NLI weighting disabled to keep NLI separate from raw semantic entropy
            # This computes raw semantic entropy (as per the original paper)
            # Create SemanticEntropyMetric with the embedding model, then pass to HallucinationMetric
            semantic_entropy = SemanticEntropyMetric(
                embedding_model=embedding_model_name,
                use_nli_weighting=False
            )
            hallucination_metric = HallucinationMetric(
                semantic_entropy_metric=semantic_entropy
            )

            # Collect multiple responses for true semantic entropy estimation.
            # Pass question_contexts as extra_context_texts so entropy samples
            # receive the same ground-truth context as the main response (prevents
            # the artefact where context-free fallback responses create a spurious
            # 2-cluster entropy of exactly 0.9183 bits).
            vanilla_samples, vanilla_chunk_texts = self._collect_sample_responses(
                rag_system=self.vanilla_rag,
                question=question,
                base_result=vanilla_result,
                similarity_threshold=similarity_threshold,
                max_chunks=max_chunks,
                extra_context_texts=question_contexts,
            )
            kg_samples, kg_chunk_texts = self._collect_sample_responses(
                rag_system=self.kg_rag,
                question=question,
                base_result=kg_result,
                similarity_threshold=similarity_threshold,
                max_chunks=max_chunks,
                extra_context_texts=question_contexts,
            )

            # When Neo4j returned no chunks (e.g. empty KG), fall back to the
            # question's own source contexts so that lexical hallucination, NLI
            # entailment, and semantic entropy are computed against the actual
            # texts the LLM used — not against an empty list.
            effective_vanilla_chunks = vanilla_chunk_texts or question_contexts
            effective_kg_chunks = kg_chunk_texts or question_contexts

            # Compute hallucination + semantic entropy (including NLI-weighted entropy)
            vanilla_hallucination_result = hallucination_metric.compute(
                response=vanilla_response,
                question=question,
                retrieved_chunks=effective_vanilla_chunks,
                sample_responses=vanilla_samples,
            )
            kg_hallucination_result = hallucination_metric.compute(
                response=kg_response,
                question=question,
                retrieved_chunks=effective_kg_chunks,
                sample_responses=kg_samples,
            )
            
            # Track correctness
            if vanilla_correct:
                results["vanilla_rag_correct"] += 1
            if kg_correct:
                results["kg_rag_correct"] += 1
            
            # Track hallucination scores for averaging
            vanilla_hallucination_scores.append(vanilla_hallucination_result.get("combined_hallucination", 0))
            kg_hallucination_scores.append(kg_hallucination_result.get("combined_hallucination", 0))
            
            # NOTE: per-question scalar logging intentionally removed to reduce noisy W&B line charts.
            # Per-question visibility is provided via W&B tables.
            
            # Store all metrics in detail - NO THRESHOLD, just raw values
            detail_entry = {
                "question_id": q_id,
                "question": question[:500],
                "expected": expected_answer,
                "vanilla_correct": vanilla_correct,
                "kg_correct": kg_correct,
                "vanilla_response": vanilla_response[:500],
                "kg_response": kg_response[:500],
                # Combined hallucination score (raw, no threshold)
                "vanilla_hallucination_score": vanilla_hallucination_result.get("combined_hallucination", 0),
                "kg_hallucination_score": kg_hallucination_result.get("combined_hallucination", 0),
                # Lexical hallucination (raw)
                "vanilla_lexical_hallucination": vanilla_hallucination_result.get("lexical_hallucination", 0),
                "kg_lexical_hallucination": kg_hallucination_result.get("lexical_hallucination", 0),
                # Semantic hallucination (raw)
                "vanilla_semantic_hallucination": vanilla_hallucination_result.get("semantic_hallucination", 0),
                "kg_semantic_hallucination": kg_hallucination_result.get("semantic_hallucination", 0),
                # ALL SEMANTIC ENTROPY VARIANTS (raw, no threshold)
                # Semantic entropy (raw)
                "vanilla_semantic_entropy": vanilla_hallucination_result.get("semantic_entropy", 0),
                "kg_semantic_entropy": kg_hallucination_result.get("semantic_entropy", 0),
                # Semantic entropy with NLI weighting (raw)
                "vanilla_semantic_entropy_nli": vanilla_hallucination_result.get("semantic_entropy_nli_weighted", 0),
                "kg_semantic_entropy_nli": kg_hallucination_result.get("semantic_entropy_nli_weighted", 0),
                # Confidence-weighted semantic entropy (raw)
                "vanilla_semantic_entropy_confidence": vanilla_hallucination_result.get("semantic_entropy_confidence_weighted", 0),
                "kg_semantic_entropy_confidence": kg_hallucination_result.get("semantic_entropy_confidence_weighted", 0),
                # Combined semantic entropy (raw)
                "vanilla_semantic_entropy_combined": vanilla_hallucination_result.get("semantic_entropy_combined", 0),
                "kg_semantic_entropy_combined": kg_hallucination_result.get("semantic_entropy_combined", 0),
                # Additional entropy metrics
                "vanilla_cluster_entropy": vanilla_hallucination_result.get("cluster_entropy", 0),
                "kg_cluster_entropy": kg_hallucination_result.get("cluster_entropy", 0),
                "vanilla_predictive_entropy": vanilla_hallucination_result.get("predictive_entropy", 0),
                "kg_predictive_entropy": kg_hallucination_result.get("predictive_entropy", 0),
                "vanilla_mutual_information": vanilla_hallucination_result.get("mutual_information", 0),
                "kg_mutual_information": kg_hallucination_result.get("mutual_information", 0),
                # Context consistency (raw)
                "vanilla_context_consistency": vanilla_hallucination_result.get("context_consistency", 0),
                "kg_context_consistency": kg_hallucination_result.get("context_consistency", 0),
                # Contradiction hallucination (NEW!)
                "vanilla_contradiction_hallucination": vanilla_hallucination_result.get("contradiction_hallucination", 0),
                "kg_contradiction_hallucination": kg_hallucination_result.get("contradiction_hallucination", 0),
                "vanilla_num_contradicting_pairs": vanilla_hallucination_result.get("num_contradicting_pairs", 0),
                "kg_num_contradicting_pairs": kg_hallucination_result.get("num_contradicting_pairs", 0),
                # Confidence level
                "vanilla_confidence_level": vanilla_hallucination_result.get("confidence_level", "unknown"),
                "kg_confidence_level": kg_hallucination_result.get("confidence_level", "unknown"),
                # Debugging/traceability for entropy
                "vanilla_entropy_num_samples": len(vanilla_samples),
                "kg_entropy_num_samples": len(kg_samples),
            }
            
            results["details"].append(detail_entry)

            logging.info(f"  Vanilla: {vanilla_response[:40]}... (correct: {vanilla_correct})")
            logging.info(f"  KG-RAG:  {kg_response[:40]}... (correct: {kg_correct})")
        
        # Calculate metrics (using raw scores, no threshold-based classification)
        total = len(questions)
        if total > 0:
            results["vanilla_accuracy"] = results["vanilla_rag_correct"] / total
            results["kg_accuracy"] = results["kg_rag_correct"] / total
            # Average hallucination scores (not rates - report raw values)
            results["vanilla_avg_hallucination_score"] = sum(vanilla_hallucination_scores) / len(vanilla_hallucination_scores) if vanilla_hallucination_scores else 0
            results["kg_avg_hallucination_score"] = sum(kg_hallucination_scores) / len(kg_hallucination_scores) if kg_hallucination_scores else 0
            # Average semantic entropy (raw values)
            vanilla_se = [d.get("vanilla_semantic_entropy", 0) for d in results["details"]]
            kg_se = [d.get("kg_semantic_entropy", 0) for d in results["details"]]
            results["vanilla_avg_semantic_entropy"] = sum(vanilla_se) / len(vanilla_se) if vanilla_se else 0
            results["kg_avg_semantic_entropy"] = sum(kg_se) / len(kg_se) if kg_se else 0
            # Average semantic entropy with NLI (raw values)
            vanilla_se_nli = [d.get("vanilla_semantic_entropy_nli", 0) for d in results["details"]]
            kg_se_nli = [d.get("kg_semantic_entropy_nli", 0) for d in results["details"]]
            results["vanilla_avg_semantic_entropy_nli"] = sum(vanilla_se_nli) / len(vanilla_se_nli) if vanilla_se_nli else 0
            results["kg_avg_semantic_entropy_nli"] = sum(kg_se_nli) / len(kg_se_nli) if kg_se_nli else 0
        
        logging.info(f"{dataset_name} Results: Vanilla={results.get('vanilla_accuracy', 0):.2%}, KG-RAG={results.get('kg_accuracy', 0):.2%}")

        # Log per-question details table for this dataset/config
        self._log_question_table_to_wandb(
            dataset_name=dataset_name,
            config_name=config_name,
            details=results.get("details", []),
        )
        
        return results
    
    def run_pipeline(self, datasets: List[str] = None):
        """Run the full evaluation pipeline"""
        if datasets is None:
            datasets = ["pubmedqa", "bioasq", "medqa", "medmcqa", "mmlu"]
        
        # Initialize W&B
        wandb_entity = os.getenv("WANDB_ENTITY", "julka01")
        wandb.init(
            entity=wandb_entity,
            project="mirage-kg-evaluation",
            name=f"eval_{int(time.time())}",
            mode='online'
        )
        self.wandb_run = wandb
        
        all_results = []
        
        for dataset_name in datasets:
            logging.info(f"\n{'='*50}")
            logging.info(f"Processing dataset: {dataset_name}")
            logging.info(f"{'='*50}")
            
            # Step 1: Build KG for this dataset (clears Neo4j first)
            # Skip if --skip-kg-build was passed (reuse the existing Neo4j KG).
            if self.skip_kg_build:
                # Verify that the existing KG actually has data before proceeding.
                driver = self._get_neo4j_driver()
                try:
                    with driver.session() as session:
                        result = session.run("MATCH (c:Chunk) RETURN count(c) AS n LIMIT 1")
                        chunk_count = result.single()["n"]
                except Exception:
                    chunk_count = 0
                finally:
                    driver.close()

                if chunk_count == 0:
                    logging.error(
                        f"--skip-kg-build was set for '{dataset_name}' but Neo4j is EMPTY "
                        f"(0 chunks found). The previous experiment likely wiped the database "
                        f"without rebuilding it. Re-run WITHOUT --skip-kg-build to rebuild the KG."
                    )
                    continue  # skip this dataset rather than produce meaningless results

                logging.info(
                    f"Skipping KG build for {dataset_name} (--skip-kg-build set). "
                    f"Using existing Neo4j KG ({chunk_count} chunks)."
                )
            else:
                kg_built = self._build_kg_for_dataset(dataset_name)
                if not kg_built:
                    logging.error(f"Skipping {dataset_name} due to KG build failure")
                    continue
            
            # Step 2: Run evaluation for each configuration
            dataset_config_results = []
            for config in self.eval_configs:
                dataset_results = self._run_evaluation_on_dataset(dataset_name, config=config)
                dataset_config_results.append(dataset_results)

                config_name = dataset_results.get("config", {}).get("name", "default")
                # Log final dataset metrics by config
                self.wandb_run.log({
                    f"final/{dataset_name}/{config_name}/vanilla_accuracy": dataset_results.get("vanilla_accuracy", 0),
                    f"final/{dataset_name}/{config_name}/kg_accuracy": dataset_results.get("kg_accuracy", 0),
                    f"final/{dataset_name}/{config_name}/vanilla_hallucination_score": dataset_results.get("vanilla_avg_hallucination_score", 0),
                    f"final/{dataset_name}/{config_name}/kg_hallucination_score": dataset_results.get("kg_avg_hallucination_score", 0),
                    f"final/{dataset_name}/{config_name}/vanilla_semantic_entropy": dataset_results.get("vanilla_avg_semantic_entropy", 0),
                    f"final/{dataset_name}/{config_name}/kg_semantic_entropy": dataset_results.get("kg_avg_semantic_entropy", 0),
                    f"final/{dataset_name}/{config_name}/vanilla_semantic_entropy_nli": dataset_results.get("vanilla_avg_semantic_entropy_nli", 0),
                    f"final/{dataset_name}/{config_name}/kg_semantic_entropy_nli": dataset_results.get("kg_avg_semantic_entropy_nli", 0),
                })

            # Log config-level table + grouped bar chart for this dataset
            self._log_config_summary_to_wandb(dataset_name, dataset_config_results)

            dataset_block = {
                "dataset": dataset_name,
                "config_results": dataset_config_results,
            }
            all_results.append(dataset_block)

            # Save intermediate results
            output_path = f"results/mirage_{dataset_name}_results.json"
            os.makedirs("results", exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(dataset_block, f, indent=2)
        
        # Summary
        wandb.finish()
        
        summary = {
            "datasets": datasets,
            "num_samples_per_dataset": self.num_samples,
            "results": all_results
        }
        
        # Save final summary
        with open("results/mirage_evaluation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info("\n" + "="*50)
        logging.info("EVALUATION COMPLETE")
        logging.info("="*50)
        for dataset_block in all_results:
            dataset_name = dataset_block.get("dataset", "unknown")
            for cfg_res in dataset_block.get("config_results", []):
                cfg_name = cfg_res.get("config", {}).get("name", "default")
                logging.info(
                    f"{dataset_name} [{cfg_name}]: "
                    f"Vanilla={cfg_res.get('vanilla_accuracy', 0):.2%}, "
                    f"KG={cfg_res.get('kg_accuracy', 0):.2%}"
                )
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Run MIRAGE dataset evaluation pipeline (Experiment Mode)")
    parser.add_argument("--num-samples", type=int, default=None, 
                        help="Number of samples per dataset (default: all)")
    parser.add_argument("--entropy-samples", type=int, default=10,
                        help="Number of generations per question for semantic entropy (default: 10, max: 20)")
    parser.add_argument("--similarity-thresholds", nargs="+", type=float, default=[0.1],
                        help="One or more similarity thresholds to evaluate as configurations")
    parser.add_argument("--max-chunks-values", nargs="+", type=int, default=[10],
                        help="One or more max_chunks values to evaluate as configurations")
    parser.add_argument("--llm-provider", type=str, default="openrouter",
                        help="LLM provider to use for KG extraction and response generation")
    parser.add_argument("--llm-model", type=str, default="openai/gpt-oss-120b:free",
                        help="LLM model name for the selected provider")
    parser.add_argument("--datasets", nargs="+", 
                        default=["pubmedqa", "bioasq"],
                        help="Datasets to evaluate")
    parser.add_argument("--skip-kg-build", action="store_true", default=False,
                        help="Skip KG construction and use the existing Neo4j database as-is")
    
    args = parser.parse_args()
    
    eval_configs = []
    for threshold in args.similarity_thresholds:
        for max_chunks in args.max_chunks_values:
            eval_configs.append({
                "name": f"thr{threshold:g}_k{max_chunks}",
                "similarity_threshold": float(threshold),
                "max_chunks": int(max_chunks),
            })

    pipeline = MIRAGEEvaluationPipeline(
        num_samples=args.num_samples,
        entropy_samples=args.entropy_samples,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        eval_configs=eval_configs,
        skip_kg_build=args.skip_kg_build,
    )
    results = pipeline.run_pipeline(datasets=args.datasets)
    
    print(f"\nResults saved to results/mirage_evaluation_summary.json")


if __name__ == "__main__":
    main()
