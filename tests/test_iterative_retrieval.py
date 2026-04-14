"""
Regression tests for _iterative_hop_retrieval contracts.

Pure unit tests — no live Neo4j or LLM calls.
The system under test is EnhancedRAGSystem._iterative_hop_retrieval.

Contracts locked down:
  - Bridge answer rejection: bad-phrase answers do not modify the next sub-question
  - Cross-hop dedup: all_chunk_ids (pre-dedup) activates a hop even when its chunk
    was already seen in an earlier hop
  - Grounding anchored to hop-0: later-hop entity matches do not inflate grounding_quality
  - Entity pruning: entities not linked to any retained chunk are excluded from output
  - Path pruning: traversal paths referencing nodes outside retained_eids are dropped
  - Returns None when all hops find no chunks
"""

import sys
import os
import hashlib

import pytest

pytest.importorskip("langchain_neo4j", reason="langchain_neo4j not installed — skipping iterative retrieval tests")
pytest.importorskip("langchain", reason="langchain not installed — skipping iterative retrieval tests")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.runnables import RunnableLambda
from ontographrag.rag.systems.enhanced_rag_system import EnhancedRAGSystem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_system() -> EnhancedRAGSystem:
    s = EnhancedRAGSystem.__new__(EnhancedRAGSystem)
    return s


def _cid(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def _chunk(text: str, score: float = 1.0, doc: str = "doc.txt",
           linked_entity_ids=None) -> dict:
    return {
        "text": text,
        "chunk_id": _cid(text),
        "score": score,
        "document": doc,
        "linked_entity_ids": linked_entity_ids or [],
        "entities": [],
    }


def _hop_ctx(chunks, entities=None, relationships=None, traversal_paths=None,
             grounding_quality=None, seed_entity_count=None) -> dict:
    ctx = {
        "chunks": chunks,
        "entities": entities or {},
        "relationships": relationships or [],
        "graph_neighbors": {},
        "traversal_paths": traversal_paths or [],
    }
    if grounding_quality is not None:
        ctx["grounding_quality"] = grounding_quality
    if seed_entity_count is not None:
        ctx["seed_entity_count"] = seed_entity_count
    return ctx


def _stub_llm(response: str):
    """LangChain Runnable returning a fixed string."""
    return RunnableLambda(lambda _: response)


# StubGraph that returns pre-configured ctx per call
class _StubGraph:
    def __init__(self, hop_ctxs):
        self._ctxs = list(hop_ctxs)

    def query(self, q, params=None):
        return []


# ---------------------------------------------------------------------------
# Stub _entity_first_search, _vector_similarity_search, etc.
# to inject controlled per-hop contexts without Neo4j.
# ---------------------------------------------------------------------------

def _run_iterative(system, hop_contexts, sub_questions, max_chunks=10,
                   kg_name="test_kg", max_hops=2, bridge_response="Paris",
                   question="Original question?", vector_hop_contexts=None):
    """
    Monkey-patch entity-first search to return items from hop_contexts sequentially,
    then run _iterative_hop_retrieval.
    """
    iter_ = iter(hop_contexts)
    vec_iter = iter(vector_hop_contexts or [])

    def _fake_entity_first(graph, query, max_chunks=10, kg_name=None, max_hops=1, **kwargs):
        try:
            return next(iter_)
        except StopIteration:
            return None

    original = system._entity_first_search
    system._entity_first_search = _fake_entity_first

    # Disable fallback search paths — keep tests clean
    def _fake_vec(*a, **kw):
        try:
            return next(vec_iter)
        except StopIteration:
            return None

    def _no_vec(*a, **kw): return None

    def _no_text(*a, **kw): return None

    orig_vec  = getattr(system, "_vector_similarity_search", None)
    orig_text = getattr(system, "_text_similarity_search", None)
    orig_check = getattr(system, "check_vector_index", None)

    system._vector_similarity_search = _fake_vec if vector_hop_contexts is not None else _no_vec
    system._text_similarity_search   = _no_text
    system.check_vector_index        = lambda: vector_hop_contexts is not None

    graph = _StubGraph([])
    llm = _stub_llm(bridge_response)

    try:
        result = system._iterative_hop_retrieval(
            graph=graph,
            question=question,
            sub_questions=sub_questions,
            max_chunks=max_chunks,
            kg_name=kg_name,
            max_hops=max_hops,
            llm=llm,
            similarity_threshold=0.1,
            document_names=[],
        )
    finally:
        system._entity_first_search = original
        if orig_vec:  system._vector_similarity_search = orig_vec
        if orig_text: system._text_similarity_search   = orig_text
        if orig_check: system.check_vector_index       = orig_check

    return result


# ---------------------------------------------------------------------------
# 1. Returns None when all hops find nothing
# ---------------------------------------------------------------------------

class TestReturnsNoneOnEmptyHops:
    sys_ = _make_system()

    def test_none_when_no_chunks(self):
        result = _run_iterative(self.sys_, hop_contexts=[None, None],
                                sub_questions=["Q1?", "Q2?"])
        assert result is None

    def test_none_when_empty_chunk_lists(self):
        hop1 = _hop_ctx(chunks=[])
        hop2 = _hop_ctx(chunks=[])
        result = _run_iterative(self.sys_, hop_contexts=[hop1, hop2],
                                sub_questions=["Q1?", "Q2?"])
        assert result is None


# ---------------------------------------------------------------------------
# 2. Bridge answer rejection
# ---------------------------------------------------------------------------

class TestBridgeAnswerRejection:
    sys_ = _make_system()

    def _run(self, bridge_response, sub_questions=None):
        c1 = _chunk("Aspirin reduces fever.", score=0.9)
        c2 = _chunk("Fever is a common symptom.", score=0.8)
        hops = [_hop_ctx([c1]), _hop_ctx([c2])]
        if sub_questions is None:
            sub_questions = ["What treats fever?", "Where is [BRIDGE] used?"]
        return _run_iterative(
            self.sys_, hop_contexts=hops,
            sub_questions=sub_questions,
            bridge_response=bridge_response,
        )

    def test_valid_bridge_returns_result(self):
        result = self._run("Aspirin")
        assert result is not None

    def test_bad_phrase_i_dont_know_still_returns_result(self):
        # Bridge is rejected but retrieval completes with both hops
        result = self._run("I don't know")
        assert result is not None
        assert len(result["chunks"]) >= 1

    def test_bad_phrase_unknown_still_returns_result(self):
        result = self._run("unknown")
        assert result is not None

    def test_empty_bridge_still_returns_result(self):
        result = self._run("")
        assert result is not None

    def test_bridge_exceeding_200_chars_rejected(self):
        # Overly long "answer" is rejected — retrieval still succeeds
        long_bridge = "X" * 201
        result = self._run(long_bridge)
        assert result is not None


# ---------------------------------------------------------------------------
# 3. Cross-hop dedup: all_chunk_ids activates hop even for deduplicated chunk
# ---------------------------------------------------------------------------

class TestCrossHopDedup:
    sys_ = _make_system()

    def test_shared_chunk_activates_both_hops(self):
        # Hop 0 retrieves chunk A (linked to entity e1).
        # Hop 1 retrieves the same chunk A again (deduplicated as new_chunks=[])
        # plus a fresh chunk B (linked to e2).
        # After truncation, both chunk A and chunk B are in the final set.
        # Hop 1's entity e2 must appear in the output because hop 1's all_chunk_ids
        # includes chunk A's ID, which is in retained_cids.
        c_a = _chunk("Aspirin reduces fever.", score=0.9, linked_entity_ids=["e1"])
        c_b = _chunk("Fever is a symptom.", score=0.8, linked_entity_ids=["e2"])

        hop0_ctx = _hop_ctx(
            chunks=[c_a],
            entities={"e1": {"id": "Aspirin", "type": "Drug", "source": "entity_lookup"}},
        )
        # Hop 1 returns the same chunk A and fresh chunk B
        hop1_ctx = _hop_ctx(
            chunks=[c_a, c_b],  # c_a will be deduped; c_b is new
            entities={
                "e1": {"id": "Aspirin", "type": "Drug", "source": "entity_lookup"},
                "e2": {"id": "Fever",   "type": "Symptom", "source": "entity_lookup"},
            },
            relationships=[{"source": "e1", "target": "e2", "type": "CAUSES",
                            "key": "e1-CAUSES-e2"}],
        )

        result = _run_iterative(
            self.sys_,
            hop_contexts=[hop0_ctx, hop1_ctx],
            sub_questions=["Q1?", "Q2?"],
            max_chunks=10,
        )

        assert result is not None
        chunk_ids_out = {c["chunk_id"] for c in result["chunks"]}
        # Both chunks in the final set
        assert _cid("Aspirin reduces fever.") in chunk_ids_out
        assert _cid("Fever is a symptom.") in chunk_ids_out
        # e2 retained because hop 1 activated (shared chunk A in all_chunk_ids)
        assert "e2" in result["entities"]


# ---------------------------------------------------------------------------
# 4. Grounding anchored to hop-0
# ---------------------------------------------------------------------------

class TestGroundingAnchoredToHop0:
    sys_ = _make_system()

    def test_grounding_from_hop0_not_later_hops(self):
        # Hop 0: grounding_quality=0.3, seed_entity_count=1
        # Hop 1: grounding_quality=0.9, seed_entity_count=10 (should be ignored)
        c1 = _chunk("Aspirin is a drug.", linked_entity_ids=["e1"])
        c2 = _chunk("Fever is a symptom.", linked_entity_ids=["e2"])
        hop0 = _hop_ctx([c1], grounding_quality=0.3, seed_entity_count=1)
        hop1 = _hop_ctx([c2], grounding_quality=0.9, seed_entity_count=10)

        result = _run_iterative(
            self.sys_, hop_contexts=[hop0, hop1],
            sub_questions=["Q1?", "Q2?"],
        )

        assert result is not None
        assert result["grounding_quality"] == 0.3
        assert result["seed_entity_count"] == 1

    def test_grounding_not_overwritten_by_empty_hop(self):
        c1 = _chunk("Aspirin is a drug.", linked_entity_ids=["e1"])
        hop0 = _hop_ctx([c1], grounding_quality=0.5, seed_entity_count=2)
        hop1 = _hop_ctx([])   # empty hop

        result = _run_iterative(
            self.sys_, hop_contexts=[hop0, hop1],
            sub_questions=["Q1?", "Q2?"],
        )
        assert result is not None
        assert result["grounding_quality"] == 0.5


# ---------------------------------------------------------------------------
# 5. Dataset-specific per-subquestion hop caps
# ---------------------------------------------------------------------------

class TestDatasetSpecificSubquestionHopCaps:
    sys_ = _make_system()

    def test_2wiki_is_capped_to_one_hop_per_subquestion(self):
        assert self.sys_._iterative_subquestion_max_hops_for_kg("2wikimultihopqa", 2) == 1
        assert self.sys_._iterative_subquestion_max_hops_for_kg("2wikimultihopqa", 4) == 1

    def test_other_datasets_keep_relaxed_default_cap(self):
        assert self.sys_._iterative_subquestion_max_hops_for_kg("musique", 4) == 3
        assert self.sys_._iterative_subquestion_max_hops_for_kg("multihoprag", 4) == 3
        assert self.sys_._iterative_subquestion_max_hops_for_kg("hotpotqa", 2) == 2


class TestIterativeRetrievalQueryAnchoring:
    sys_ = _make_system()

    def test_musique_later_hops_keep_original_question_anchor(self):
        query = self.sys_._iterative_retrieval_query(
            "Where is the headquarters of the Radio Television of the country whose co-official language is the same as the one Politika is written in?",
            "Which country has [BRIDGE] as a co-official language?",
            "musique",
            1,
            next_sub_question="Where is the headquarters of the Radio Television of [BRIDGE]?",
        )
        assert "Original question:" in query
        assert "Politika" in query
        assert "Next hop target:" in query
        assert "Radio Television" in query

    def test_2wiki_later_hops_remain_local(self):
        query = self.sys_._iterative_retrieval_query(
            "Who is the father-in-law of Helena Palaiologina, Despotess of Serbia?",
            "Who is [BRIDGE]'s father?",
            "2wikimultihopqa",
            1,
            next_sub_question="What office did [BRIDGE] hold?",
        )
        assert query == "Who is [BRIDGE]'s father?"

    def test_anchor_datasets_get_larger_later_hop_budget(self):
        assert self.sys_._iterative_hop_retrieval_budget("musique", 1, 6) == 12
        assert self.sys_._iterative_hop_retrieval_budget("multihoprag", 2, 8) == 12
        assert self.sys_._iterative_hop_retrieval_budget("2wikimultihopqa", 1, 6) == 6


# ---------------------------------------------------------------------------
# 5b. Hybrid retrieval should prefer vector evidence when graph signal is weak
# ---------------------------------------------------------------------------

class TestHybridRetrievalRouting:
    sys_ = _make_system()

    def test_comparison_query_with_single_branch_graph_context_prefers_vector(self):
        graph_chunk = _chunk(
            "Riding the California Trail is a 1947 American Western film directed by William Nigh.",
            score=1.0,
            linked_entity_ids=["e_cal"],
        )
        vector_chunk = _chunk(
            "Lost and Delirious is directed by Lea Pool, born in 1950.",
            score=0.42,
            linked_entity_ids=["e_lost"],
        )

        graph_ctx = _hop_ctx(
            [graph_chunk],
            entities={"e_cal": {"id": "Riding the California Trail", "description": "Riding the California Trail"}},
            traversal_paths=[{"path": "William Nigh --DIRECTED_BY--> Riding the California Trail"}],
            grounding_quality=0.9,
            seed_entity_count=2,
        )
        vector_ctx = _hop_ctx(
            [vector_chunk],
            entities={"e_lost": {"id": "Lost and Delirious", "description": "Lost and Delirious"}},
        )

        result = _run_iterative(
            self.sys_,
            hop_contexts=[graph_ctx],
            vector_hop_contexts=[vector_ctx],
            sub_questions=["Which film has the director born later, Riding The California Trail or Lost And Delirious?"],
            question="Which film has the director born later, Riding The California Trail or Lost And Delirious?",
            max_chunks=6,
        )

        assert result is not None
        texts = [chunk["text"] for chunk in result["chunks"]]
        assert any("Lost and Delirious" in text for text in texts)

    def test_graph_context_meaningfulness_requires_both_comparison_branches(self):
        graph_ctx = _hop_ctx(
            [_chunk("Riding the California Trail is directed by William Nigh.", linked_entity_ids=["e_cal"])],
            entities={"e_cal": {"id": "Riding the California Trail", "description": "Riding the California Trail"}},
            traversal_paths=[{"path": "William Nigh --DIRECTED_BY--> Riding the California Trail"}],
            grounding_quality=0.9,
            seed_entity_count=2,
        )
        assert not self.sys_._graph_context_is_meaningful(
            "Which film has the director born later, Riding The California Trail or Lost And Delirious?",
            graph_ctx,
        )

    def test_comparison_chunk_rerank_prioritizes_explicit_branches(self):
        chunks = [
            _chunk("Fred F. Sears was an American film director.", score=1.0, doc="2wikimultihopqa"),
            _chunk("Lost and Delirious is directed by Lea Pool.", score=0.72, doc="2wikimultihopqa"),
            _chunk("Riding the California Trail is directed by William Nigh.", score=0.69, doc="2wikimultihopqa"),
        ]
        sorted_chunks = self.sys_._sort_chunks_for_query(
            "Which film has the director born later, Riding The California Trail or Lost And Delirious?",
            chunks,
        )
        top_texts = [chunk["text"] for chunk in sorted_chunks[:2]]
        assert any("Lost and Delirious" in text for text in top_texts)
        assert any("Riding the California Trail" in text for text in top_texts)


# ---------------------------------------------------------------------------
# 6. Entity and path pruning after chunk truncation
# ---------------------------------------------------------------------------

class TestEntityAndPathPruning:
    sys_ = _make_system()

    def test_entity_not_linked_to_retained_chunk_is_dropped(self):
        # We set max_chunks=1, so only the higher-score chunk survives.
        c_high = _chunk("Aspirin reduces fever.", score=1.0, linked_entity_ids=["e_keep"])
        c_low  = _chunk("Metformin treats diabetes.", score=0.1, linked_entity_ids=["e_drop"])

        hop_ctx = _hop_ctx(
            chunks=[c_high, c_low],
            entities={
                "e_keep": {"id": "Aspirin",   "type": "Drug", "source": "entity_lookup"},
                "e_drop": {"id": "Metformin", "type": "Drug", "source": "entity_lookup"},
            },
        )

        result = _run_iterative(
            self.sys_, hop_contexts=[hop_ctx],
            sub_questions=["Q1?"],
            max_chunks=1,
        )

        assert result is not None
        assert "e_keep" in result["entities"]
        assert "e_drop" not in result["entities"]

    def test_relationship_dropped_when_either_endpoint_pruned(self):
        c_high = _chunk("Aspirin reduces fever.", score=1.0, linked_entity_ids=["e1"])
        c_low  = _chunk("Metformin treats diabetes.", score=0.1, linked_entity_ids=["e2"])

        hop_ctx = _hop_ctx(
            chunks=[c_high, c_low],
            entities={
                "e1": {"id": "Aspirin",   "type": "Drug", "source": "entity_lookup"},
                "e2": {"id": "Metformin", "type": "Drug", "source": "entity_lookup"},
            },
            relationships=[
                {"source": "e1", "target": "e2", "type": "UNRELATED", "key": "e1-UNRELATED-e2"},
            ],
        )

        result = _run_iterative(
            self.sys_, hop_contexts=[hop_ctx],
            sub_questions=["Q1?"],
            max_chunks=1,
        )

        assert result is not None
        # e2 not retained → relationship must be absent
        assert not any(r.get("target") == "e2" for r in result["relationships"])

    def test_path_without_node_ids_is_kept(self):
        # Paths without node_ids are legacy; they must not be dropped
        c = _chunk("Aspirin reduces fever.", linked_entity_ids=["e1"])
        hop_ctx = _hop_ctx(
            chunks=[c],
            traversal_paths=[{"path": "A -> B -> C", "hops": 2}],  # no node_ids
        )
        result = _run_iterative(
            self.sys_, hop_contexts=[hop_ctx], sub_questions=["Q1?"],
        )
        assert result is not None
        assert any("A -> B" in p["path"] for p in result["traversal_paths"])

    def test_path_with_unknown_node_id_is_dropped(self):
        c = _chunk("Aspirin reduces fever.", linked_entity_ids=["e1"])
        hop_ctx = _hop_ctx(
            chunks=[c],
            traversal_paths=[{
                "path": "Aspirin -> Fever",
                "hops": 1,
                "node_ids": ["e1", "e_unknown"],  # e_unknown not in any chunk
            }],
        )
        result = _run_iterative(
            self.sys_, hop_contexts=[hop_ctx], sub_questions=["Q1?"],
        )
        assert result is not None
        # Path referencing unknown node must be pruned
        assert not any("Aspirin -> Fever" in p["path"] for p in result["traversal_paths"])

    def test_path_with_all_retained_node_ids_kept(self):
        c = _chunk("Aspirin reduces fever.", linked_entity_ids=["e1", "e2"])
        hop_ctx = _hop_ctx(
            chunks=[c],
            traversal_paths=[{
                "path": "Aspirin -> Fever",
                "hops": 1,
                "node_ids": ["e1", "e2"],  # both in linked_entity_ids
            }],
        )
        result = _run_iterative(
            self.sys_, hop_contexts=[hop_ctx], sub_questions=["Q1?"],
        )
        assert result is not None
        assert any("Aspirin -> Fever" in p["path"] for p in result["traversal_paths"])
