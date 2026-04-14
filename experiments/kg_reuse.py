"""Helpers for deciding whether an existing dataset KG can be safely reused."""

from typing import Any, Dict, List, Tuple


def assess_dataset_kg_compatibility(
    existing_meta: Dict[str, Any],
    expected_meta: Dict[str, Any],
    *,
    evaluation_subset_scope: str = "evaluation_subset",
) -> Tuple[bool, List[str]]:
    """Compare stored KG metadata against the current expected build contract."""
    if not existing_meta or not expected_meta:
        return True, []

    reasons: List[str] = []

    def _compare(
        existing_key: str,
        expected_key: str,
        label: str,
        *,
        normalizer=None,
    ) -> None:
        existing_value = existing_meta.get(existing_key)
        expected_value = expected_meta.get(expected_key)
        if existing_value in (None, "") or expected_value in (None, ""):
            return
        if normalizer is not None:
            existing_value = normalizer(existing_value)
            expected_value = normalizer(expected_value)
        if existing_value != expected_value:
            reasons.append(
                f"{label} mismatch (stored={existing_value!r}, expected={expected_value!r})"
            )

    _compare("uses_global_corpus", "usesGlobalCorpus", "usesGlobalCorpus", normalizer=bool)
    _compare("corpus_source", "corpusSource", "corpusSource")
    _compare("question_context_role", "questionContextRole", "questionContextRole")
    _compare("dataset_kg_scope", "datasetKgScope", "datasetKgScope")
    _compare("content_hash", "contentHash", "contentHash")
    _compare(
        "kg_build_fingerprint_version",
        "kgBuildFingerprintVersion",
        "kgBuildFingerprintVersion",
        normalizer=int,
    )
    _compare("kg_builder", "kgBuilder", "kgBuilder")
    _compare("kg_chunk_size", "kgChunkSize", "kgChunkSize", normalizer=int)
    _compare("kg_chunk_overlap", "kgChunkOverlap", "kgChunkOverlap", normalizer=int)
    _compare("kg_extraction_provider", "kgExtractionProvider", "kgExtractionProvider")
    _compare("kg_extraction_model", "kgExtractionModel", "kgExtractionModel")
    _compare("kg_embedding_provider", "kgEmbeddingProvider", "kgEmbeddingProvider")

    if (
        expected_meta.get("datasetKgScope") == evaluation_subset_scope
        and not bool(expected_meta.get("usesGlobalCorpus"))
    ):
        _compare("selection_key", "selectionKey", "selectionKey")

    return len(reasons) == 0, reasons
