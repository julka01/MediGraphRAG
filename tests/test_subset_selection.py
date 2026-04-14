"""
Regression tests for deterministic dataset subset selection.

Pure unit tests — no live Neo4j, LLM, or experiment pipeline imports required.
"""

from pathlib import Path

import pytest

from experiments.subset_selection import (
    deterministic_subset_ids,
    resolve_question_subset,
    selection_file_path,
)


def test_deterministic_subset_ids_is_seeded_and_order_preserving():
    ordered_ids = [f"q{i}" for i in range(10)]

    selected_once = deterministic_subset_ids(ordered_ids, num_samples=4, subset_seed=7)
    selected_twice = deterministic_subset_ids(ordered_ids, num_samples=4, subset_seed=7)

    assert selected_once == selected_twice
    assert len(selected_once) == 4
    # The returned subset should preserve original dataset order.
    assert selected_once == [question_id for question_id in ordered_ids if question_id in set(selected_once)]


def test_selection_file_path_is_seed_specific(tmp_path: Path):
    seed_1 = selection_file_path(tmp_path, "musique", num_samples=100, subset_seed=1)
    seed_2 = selection_file_path(tmp_path, "musique", num_samples=100, subset_seed=2)

    assert seed_1 != seed_2
    assert "seed1" in seed_1.name
    assert "seed2" in seed_2.name


def test_resolve_question_subset_creates_and_then_reuses_saved_ids(tmp_path: Path):
    selection_path = selection_file_path(
        tmp_path,
        "musique",
        num_samples=3,
        subset_seed=11,
    )
    available_ids = [f"q{i}" for i in range(8)]

    created = resolve_question_subset(
        dataset_name="musique",
        available_question_ids=available_ids,
        num_samples=3,
        subset_seed=11,
        selection_path=selection_path,
    )
    reused = resolve_question_subset(
        dataset_name="musique",
        available_question_ids=available_ids,
        num_samples=3,
        subset_seed=11,
        selection_path=selection_path,
    )

    assert created.created is True
    assert reused.created is False
    assert created.question_ids == reused.question_ids
    assert selection_path.exists()
    assert created.payload["selection_key"] == "n3_seed11"
    assert created.payload["subset_id"].startswith("musique__n3_seed11_h")


def test_resolve_question_subset_reuses_persisted_ids_when_params_change(tmp_path: Path):
    selection_path = selection_file_path(
        tmp_path,
        "hotpotqa",
        num_samples=5,
        subset_seed=3,
    )
    available_ids = [f"q{i}" for i in range(12)]

    initial = resolve_question_subset(
        dataset_name="hotpotqa",
        available_question_ids=available_ids,
        num_samples=5,
        subset_seed=3,
        selection_path=selection_path,
    )
    reused = resolve_question_subset(
        dataset_name="hotpotqa",
        available_question_ids=available_ids,
        num_samples=7,
        subset_seed=99,
        selection_path=selection_path,
    )

    assert reused.created is False
    assert reused.question_ids == initial.question_ids
    assert len(reused.warnings) == 2
    assert "subset_seed=3" in reused.warnings[0]
    assert "num_samples=5" in reused.warnings[1]


def test_resolve_question_subset_force_resample_overwrites_existing_selection(tmp_path: Path):
    selection_path = selection_file_path(
        tmp_path,
        "pubmedqa",
        num_samples=4,
        subset_seed=1,
    )
    available_ids = [f"q{i}" for i in range(10)]

    first = resolve_question_subset(
        dataset_name="pubmedqa",
        available_question_ids=available_ids,
        num_samples=4,
        subset_seed=1,
        selection_path=selection_path,
    )
    second = resolve_question_subset(
        dataset_name="pubmedqa",
        available_question_ids=available_ids,
        num_samples=4,
        subset_seed=2,
        selection_path=selection_path,
        force_resample=True,
    )

    assert second.created is True
    assert second.question_ids != first.question_ids


def test_resolve_question_subset_requires_existing_file_when_requested(tmp_path: Path):
    selection_path = selection_file_path(
        tmp_path,
        "bioasq",
        num_samples=1,
        subset_seed=42,
    )

    with pytest.raises(FileNotFoundError):
        resolve_question_subset(
            dataset_name="bioasq",
            available_question_ids=["q1", "q2"],
            num_samples=1,
            subset_seed=42,
            selection_path=selection_path,
            require_existing=True,
        )


def test_resolve_question_subset_errors_on_missing_saved_ids(tmp_path: Path):
    selection_path = selection_file_path(
        tmp_path,
        "2wikimultihopqa",
        num_samples=2,
        subset_seed=5,
    )
    resolve_question_subset(
        dataset_name="2wikimultihopqa",
        available_question_ids=["q1", "q2", "q3"],
        num_samples=2,
        subset_seed=5,
        selection_path=selection_path,
    )

    with pytest.raises(ValueError):
        resolve_question_subset(
            dataset_name="2wikimultihopqa",
            available_question_ids=["q1", "q2"],
            num_samples=2,
            subset_seed=5,
            selection_path=selection_path,
        )
