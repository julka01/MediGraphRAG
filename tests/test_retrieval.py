"""
Regression tests for retrieval-layer contracts.

Pure unit tests — no live Neo4j or LLM calls.
Covers:
  - classify_question_type routing
  - _grounding_quality scoring
  - _decompose_question fallback behaviour
  - format_context_for_llm output shape
"""

import sys
import os
import json

import pytest

pytest.importorskip("langchain_neo4j", reason="langchain_neo4j not installed — skipping retrieval tests")
pytest.importorskip("langchain", reason="langchain not installed — skipping retrieval tests")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ontographrag.rag.systems.enhanced_rag_system import EnhancedRAGSystem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_system() -> EnhancedRAGSystem:
    """Instantiate with no graph / embedding dependencies."""
    s = EnhancedRAGSystem.__new__(EnhancedRAGSystem)
    return s


def _minimal_context(chunks=None, entities=None, relationships=None, traversal_paths=None):
    return {
        "chunks": chunks or [],
        "entities": entities or {},
        "relationships": relationships or [],
        "traversal_paths": traversal_paths or [],
    }


# ---------------------------------------------------------------------------
# 1. classify_question_type
# ---------------------------------------------------------------------------

class TestClassifyQuestionType:
    sys_ = _make_system()

    def test_statistical_term_detected(self):
        assert self.sys_.classify_question_type("What is the incidence rate of diabetes?") == "statistical"

    def test_how_many_is_statistical(self):
        assert self.sys_.classify_question_type("How many patients were enrolled?") == "statistical"

    def test_what_percentage_is_statistical(self):
        assert self.sys_.classify_question_type("What percentage of cases respond to metformin?") == "statistical"

    def test_explain_is_semantic(self):
        assert self.sys_.classify_question_type("Explain the mechanism of insulin resistance.") == "semantic"

    def test_what_is_is_semantic(self):
        assert self.sys_.classify_question_type("What is the role of TBK1 in signalling?") == "semantic"

    def test_generic_relational_question(self):
        qt = self.sys_.classify_question_type("Does aspirin treat fever?")
        assert qt in ("generic", "semantic", "statistical"), f"Unexpected type: {qt}"

    def test_case_insensitive(self):
        assert self.sys_.classify_question_type("EXPLAIN THE MECHANISM") == "semantic"

    def test_returns_string(self):
        result = self.sys_.classify_question_type("Some random text.")
        assert isinstance(result, str)
        assert result in ("statistical", "semantic", "generic")


# ---------------------------------------------------------------------------
# 2. _grounding_quality
# ---------------------------------------------------------------------------

class TestGroundingQuality:
    def test_perfect_grounding_capped_at_1(self):
        # 3 content words, 5 matched → capped at 1.0
        score = EnhancedRAGSystem._grounding_quality("Metformin treats diabetes", 5)
        assert score == 1.0

    def test_partial_grounding(self):
        # "Metformin treats diabetes" has 3 content words (>=4 chars: Metformin, treats, diabetes)
        # 1 matched → 1/3
        score = EnhancedRAGSystem._grounding_quality("Metformin treats diabetes", 1)
        assert abs(score - 1 / 3) < 1e-9

    def test_zero_matched_returns_0(self):
        score = EnhancedRAGSystem._grounding_quality("Metformin treats diabetes", 0)
        assert score == 0.0

    def test_empty_query_returns_0(self):
        score = EnhancedRAGSystem._grounding_quality("", 3)
        assert score == 0.0

    def test_short_words_only_returns_0(self):
        # All words < 4 chars → no content words → 0.0
        score = EnhancedRAGSystem._grounding_quality("do it now", 1)
        assert score == 0.0

    def test_output_in_unit_interval(self):
        for n in range(6):
            score = EnhancedRAGSystem._grounding_quality("Aspirin reduces fever inflammation", n)
            assert 0.0 <= score <= 1.0


class TestQuestionLocalGraphScope:
    def test_entity_support_clause_contains_question_local_constraints(self):
        clause = EnhancedRAGSystem._question_local_entity_support_clause(
            "e",
            kg_name="musique",
            question_id="q1",
        )
        assert "c.questionId = $question_id" in clause
        assert "d.kgName = $kg_name" in clause

    def test_pair_support_clause_is_true_without_question_id(self):
        clause = EnhancedRAGSystem._question_local_pair_support_clause(
            "e1",
            "e2",
            kg_name="musique",
            question_id=None,
        )
        assert clause == "true"

    def test_pair_support_clause_contains_question_bundle_constraints(self):
        clause = EnhancedRAGSystem._question_local_pair_support_clause(
            "e1",
            "e2",
            kg_name="musique",
            question_id="q1",
            relationship_var="r",
        )
        assert "$question_id IN coalesce(r.questionIds, [])" in clause
        assert "c1.questionId = $question_id" in clause
        assert "c2.questionId = $question_id" in clause
        assert "coalesce(c1.passageIndex, -1) = coalesce(c2.passageIndex, -1)" in clause
        assert "d1.kgName = $kg_name" in clause

    def test_expand_entities_via_graph_adds_question_local_path_scope(self):
        sys_ = _make_system()

        class FakeGraph:
            def __init__(self):
                self.query_text = None
                self.params = None

            def query(self, query, params):
                self.query_text = query
                self.params = params
                return []

        graph = FakeGraph()
        result = sys_._expand_entities_via_graph(
            graph,
            ["e1"],
            kg_name="musique",
            max_hops=2,
            question_id="q1",
        )

        assert result == {"neighbors": {}, "paths": []}
        assert graph.params["question_id"] == "q1"
        assert "ALL(idx IN range(0, length(path) - 1)" in graph.query_text
        assert "c1.questionId = $question_id" in graph.query_text
        assert "c2.questionId = $question_id" in graph.query_text

    def test_final_chunk_selection_prunes_off_chunk_relationships(self):
        context = {
            "chunks": [
                {
                    "text": "A",
                    "score": 1.0,
                    "linked_entity_ids": ["e1", "e2"],
                    "position": 10,
                    "question_id": "q1",
                    "passage_index": 0,
                }
            ],
            "entities": {
                "e1": {"id": "e1"},
                "e2": {"id": "e2"},
            },
            "relationships": [
                {
                    "source": "e1",
                    "target": "e2",
                    "type": "REL",
                    "provenance_positions": [10],
                },
                {
                    "source": "e1",
                    "target": "e2",
                    "type": "REL2",
                    "provenance_positions": [999],
                },
            ],
            "traversal_paths": [],
        }

        pruned = EnhancedRAGSystem._apply_final_chunk_selection(
            query="test",
            context=context,
            max_chunks=1,
            retrieval_temperature=0.0,
            retrieval_shortlist_factor=1,
            retrieval_sample_id=0,
            kg_name="musique",
        )

        assert len(pruned["relationships"]) == 1
        assert pruned["relationships"][0]["type"] == "REL"


# ---------------------------------------------------------------------------
# 3. _decompose_question — fallback and cap behaviour
# ---------------------------------------------------------------------------

class TestDecomposeQuestion:
    sys_ = _make_system()

    def _llm(self, response: str):
        # RunnableLambda is a proper LangChain Runnable — works with prompt | llm | parser.
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(lambda _: response)

    def test_valid_json_array_returned(self):
        llm = self._llm('["Who founded CRISPR?", "What country are they from?"]')
        result = self.sys_._decompose_question("Multi hop question?", llm, max_hops=2)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(q, str) for q in result)

    def test_malformed_json_falls_back_to_original(self):
        llm = self._llm("I cannot decompose this.")
        result = self.sys_._decompose_question("Original question?", llm, max_hops=2)
        assert result == ["Original question?"]

    def test_empty_response_falls_back(self):
        llm = self._llm("")
        result = self.sys_._decompose_question("Some question?", llm, max_hops=2)
        assert result == ["Some question?"]

    def test_single_item_array_falls_back(self):
        # Needs >= 2 sub-questions; single-item signals LLM didn't decompose
        llm = self._llm('["Only one question?"]')
        result = self.sys_._decompose_question("Some question?", llm, max_hops=2)
        assert result == ["Some question?"]

    def test_result_capped_to_max_hops(self):
        many = '["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"]'
        llm = self._llm(many)
        result = self.sys_._decompose_question("Multi hop question?", llm, max_hops=3)
        assert len(result) <= 3

    def test_always_returns_list(self):
        llm = self._llm("garbage [] more garbage")
        result = self.sys_._decompose_question("X?", llm)
        assert isinstance(result, list)
        assert len(result) >= 1


class TestExtractQueryEntities:
    sys_ = _make_system()

    def _llm(self, response: str):
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(lambda _: response)

    def test_dedupes_and_caps_entities(self):
        self.sys_._entity_extraction_cache = {}
        llm = self._llm(
            json.dumps([
                "Marie Curie",
                " Poland ",
                "radioactivity",
                "Marie Curie",
                "",
                "Pierre Curie",
                "Nobel Prize",
                "Paris",
                "France",
                "University of Paris",
                "Marie Curie",
            ])
        )
        result = self.sys_._extract_query_entities("Who was Marie Curie?", llm)

        assert result == [
            "Marie Curie",
            "Poland",
            "radioactivity",
            "Pierre Curie",
            "Nobel Prize",
            "Paris",
            "France",
            "University of Paris",
        ][: EnhancedRAGSystem._MAX_EXTRACTED_QUERY_ENTITIES]
        assert len(result) == EnhancedRAGSystem._MAX_EXTRACTED_QUERY_ENTITIES


class TestPprEntityScores:
    sys_ = _make_system()

    def test_bidirectional_walk_and_dangling_mass_preserved(self):
        scores = self.sys_._ppr_entity_scores(
            seed_ids=["a"],
            all_entity_ids=["a", "b", "c"],
            edges=[("a", "b")],
        )

        assert set(scores) == {"a", "b", "c"}
        assert abs(sum(scores.values()) - 1.0) < 1e-9
        assert scores["b"] > scores["c"]


class TestMergeRetrievalContexts:
    def test_preserves_nonzero_seed_entity_count_from_secondary(self):
        merged = EnhancedRAGSystem._merge_retrieval_contexts(
            {
                "query": "test",
                "chunks": [{"chunk_id": "c1", "text": "a", "score": 0.8}],
                "entities": {},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
                "documents": [],
                "seed_entity_count": 0,
                "grounding_quality": 0.0,
            },
            {
                "chunks": [{"chunk_id": "c2", "text": "b", "score": 0.6}],
                "entities": {},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
                "documents": [],
                "seed_entity_count": 5,
                "grounding_quality": 0.4,
            },
            max_chunks=4,
            search_method="hybrid",
        )

        assert merged["seed_entity_count"] == 5


# ---------------------------------------------------------------------------
# 4. format_context_for_llm — output shape
# ---------------------------------------------------------------------------

class TestFormatContextForLLM:
    sys_ = _make_system()

    def test_returns_three_strings(self):
        ctx = _minimal_context()
        result = self.sys_.format_context_for_llm(ctx)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(s, str) for s in result)

    def test_chunks_rendered_in_evidence_block(self):
        ctx = _minimal_context(chunks=[{"text": "Aspirin reduces fever.", "document": "paper.txt",
                                        "chunk_id": "c1", "linked_entity_ids": []}])
        evidence_block, _, _ = self.sys_.format_context_for_llm(ctx)
        assert "Aspirin reduces fever." in evidence_block

    def test_no_chunks_returns_no_evidence_message(self):
        ctx = _minimal_context()
        evidence_block, _, _ = self.sys_.format_context_for_llm(ctx)
        assert evidence_block  # must be non-empty (shows placeholder)

    def test_relationships_rendered_in_paths(self):
        ctx = _minimal_context(
            entities={"e1": {"id": "Aspirin", "type": "Drug", "description": "Aspirin"}},
            relationships=[{"source": "e1", "target": "e2", "type": "TREATS"}],
        )
        _, _, paths_str = self.sys_.format_context_for_llm(ctx)
        assert "TREATS" in paths_str

    def test_no_graph_data_shows_fallback_message(self):
        ctx = _minimal_context()
        _, _, paths_str = self.sys_.format_context_for_llm(ctx)
        assert "No graph" in paths_str

    def test_traversal_paths_rendered(self):
        ctx = _minimal_context(
            traversal_paths=[{"path": "Aspirin -> TREATS -> Fever", "hops": 1}]
        )
        _, _, paths_str = self.sys_.format_context_for_llm(ctx)
        assert "Aspirin -> TREATS -> Fever" in paths_str

    def test_seed_vs_neighbor_entity_labelling(self):
        ctx = _minimal_context(entities={
            "e1": {"id": "Aspirin", "type": "Drug", "source": "entity_lookup", "description": "Aspirin"},
            "e2": {"id": "Fever",   "type": "Symptom", "source": "graph_traversal", "description": "Fever"},
        })
        _, entities_str, _ = self.sys_.format_context_for_llm(ctx)
        assert "Seed entities" in entities_str
        assert "Graph-traversal neighbors" in entities_str
