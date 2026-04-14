from experiments.answer_formatting import (
    build_answer_instructions,
    normalize_answer_to_contract,
)


def test_pubmedqa_instructions_include_maybe():
    text = build_answer_instructions("pubmedqa", "binary")
    assert "yes, no, or maybe" in text
    assert "exactly one word" in text.lower()
    assert "study conclusion" in text.lower()


def test_mcq_instructions_include_options():
    text = build_answer_instructions(
        "medhop",
        "mcq",
        options={"A": "Protein X", "B": "Protein Y"},
    )
    assert "multiple-choice" in text.lower()
    assert "A. Protein X" in text
    assert "B. Protein Y" in text


def test_multihoprag_instructions_include_insufficient_information():
    text = build_answer_instructions("multihoprag", "free_text")
    assert "Insufficient Information" in text
    assert "shortest correct entity" in text


def test_2wiki_free_text_instructions_are_short_answer_only():
    text = build_answer_instructions("2wikimultihopqa", "free_text")
    assert "short-answer multi-hop qa task" in text.lower()
    assert "shortest correct entity" in text.lower()
    assert "do not write an explanatory sentence" in text.lower()


def test_realmedqa_instructions_expect_concise_recommendation():
    text = build_answer_instructions("realmedqa", "free_text")
    assert "clinical recommendation qa" in text.lower()
    assert "1 to 3 sentences" in text.lower()
    assert "insufficient information" in text.lower()


def test_pubmedqa_answer_contract_normalizes_leading_label():
    assert (
        normalize_answer_to_contract(
            "pubmedqa",
            "binary",
            "Yes, the abstract overall supports the claim.",
        )
        == "yes"
    )
    assert (
        normalize_answer_to_contract(
            "pubmedqa",
            "binary",
            "maybe. the evidence is mixed across subgroups.",
        )
        == "maybe"
    )


def test_bioasq_binary_contract_normalizes_leading_label():
    assert (
        normalize_answer_to_contract(
            "bioasq",
            "binary",
            "No, the evidence does not support that conclusion.",
        )
        == "no"
    )


def test_free_text_answers_are_left_unchanged():
    text = "Lost and Delirious"
    assert normalize_answer_to_contract("musique", "free_text", text) == text


def test_short_answer_wrapper_is_removed_for_multihop_datasets():
    assert (
        normalize_answer_to_contract(
            "2wikimultihopqa",
            "free_text",
            "Final answer: Lost and Delirious",
        )
        == "Lost and Delirious"
    )
