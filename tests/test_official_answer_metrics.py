from experiments.official_answer_metrics import (
    compute_answer_em_f1,
    supports_official_answer_metrics,
)


def test_wrapped_answer_gets_partial_f1_but_not_exact_match():
    em, f1 = compute_answer_em_f1(
        "The answer is Sam Bankman-Fried.",
        "Sam Bankman-Fried",
    )
    assert em == 0.0
    assert 0.0 < f1 < 1.0


def test_alias_can_rescue_em_and_f1():
    em, f1 = compute_answer_em_f1(
        "acetylsalicylic acid",
        "aspirin",
        aliases=["acetylsalicylic acid"],
    )
    assert em == 1.0
    assert f1 == 1.0


def test_partial_overlap_scores_fractional_f1():
    em, f1 = compute_answer_em_f1(
        "sam bankman",
        "sam bankman fried",
    )
    assert em == 0.0
    assert 0.0 < f1 < 1.0


def test_supports_official_answer_metrics_only_for_selected_datasets():
    assert supports_official_answer_metrics("hotpotqa") is True
    assert supports_official_answer_metrics("2wikimultihopqa") is True
    assert supports_official_answer_metrics("musique") is True
    assert supports_official_answer_metrics("multihoprag") is True
    assert supports_official_answer_metrics("pubmedqa") is False
