import pytest


MIRAGEEvaluationPipeline = pytest.importorskip(
    "experiments.experiment",
    reason="experiment module dependencies not installed in this test environment",
).MIRAGEEvaluationPipeline


def _make_benchmark():
    benchmark = MIRAGEEvaluationPipeline.__new__(MIRAGEEvaluationPipeline)
    benchmark.QUESTION_SCOPED_DATASETS = frozenset({"musique", "hotpotqa", "2wikimultihopqa", "pubmedqa"})
    return benchmark


def test_validate_retrieval_scope_rejects_cross_question_chunks():
    benchmark = _make_benchmark()
    rag_result = {
        "context": {
            "chunks": [
                {"chunk_id": "c1", "kg_name": "musique", "question_id": "q_other", "document": "doc1"},
            ]
        }
    }

    assert benchmark._validate_retrieval_scope(
        rag_result=rag_result,
        dataset_name="musique",
        system_name="KG-RAG",
        question_id="q_target",
    ) is False


def test_validate_retrieval_scope_allows_matching_question_scope():
    benchmark = _make_benchmark()
    rag_result = {
        "context": {
            "chunks": [
                {"chunk_id": "c1", "kg_name": "musique", "question_id": "q_target", "document": "doc1"},
                {"chunk_id": "c2", "kg_name": "musique", "question_id": "q_target", "document": "doc1"},
            ]
        }
    }

    assert benchmark._validate_retrieval_scope(
        rag_result=rag_result,
        dataset_name="musique",
        system_name="KG-RAG",
        question_id="q_target",
    ) is True
