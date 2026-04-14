import pytest


MIRAGEEvaluationPipeline = pytest.importorskip(
    "experiments.experiment",
    reason="experiment module dependencies not installed in this test environment",
).MIRAGEEvaluationPipeline


def _make_pipeline():
    return MIRAGEEvaluationPipeline.__new__(MIRAGEEvaluationPipeline)


def test_exact_insufficient_information_is_failure_when_gold_is_specific():
    pipeline = _make_pipeline()

    assert pipeline._is_generation_failure(
        {"response": "Insufficient Information."},
        "insufficient information.",
        expected_answer="atorvastatin",
    ) is True


def test_exact_insufficient_information_is_not_failure_when_gold_matches():
    pipeline = _make_pipeline()

    assert pipeline._is_generation_failure(
        {"response": "Insufficient Information."},
        "insufficient information.",
        expected_answer="insufficient information.",
    ) is False

