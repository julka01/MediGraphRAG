"""
Regression tests for KG construction contracts.

These are pure unit tests — no live Neo4j or LLM calls.
They lock down the behaviours that must survive refactoring:

  - chunk_id is scoped to (kg_name, file_name, text)
  - _verify_triple_confidence scoring tiers
  - _verify_triple_confidence alias coverage
  - _harmonize_entities deduplication by normalised text
  - relation MERGE key includes negated and only includes qualifiers when present
  - relationship store failures fail small/broken builds but tiny misses are tolerated on large graphs
  - synonym merge type guard reads e.type / e.ontology_class
"""

import hashlib
import re
import sys
import os

import pytest

# Skip the whole module gracefully if heavy KG dependencies aren't installed.
# The project venv includes all of these; this guard is for CI environments
# that only install a subset of requirements.
pytest.importorskip("langchain_neo4j", reason="langchain_neo4j not installed — skipping KG build tests")
pytest.importorskip("langchain", reason="langchain not installed — skipping KG build tests")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ontographrag.kg.builders.ontology_guided_kg_creator import OntologyGuidedKGCreator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_creator() -> OntologyGuidedKGCreator:
    """Instantiate with minimal config — no Neo4j, no embeddings required."""
    c = OntologyGuidedKGCreator.__new__(OntologyGuidedKGCreator)
    c.chunk_size = 500
    c.chunk_overlap = 50
    c.ontology_classes = []
    c.ontology_relationships = []
    c.schema_card = {}
    c.ontology_path = None
    c.enable_coreference_resolution = False
    c._ontology_schema = None
    # Stub embedding so _chunk_text works without a model
    emb = type("Emb", (), {"embed_query": lambda self, t: [0.0] * 384})()
    c.embedding_function = emb
    return c


def _chunk_id(kg_name: str, file_name: str, text: str) -> str:
    return hashlib.sha1(f"{kg_name}:{file_name}:{text}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# 1. chunk_id scoping
# ---------------------------------------------------------------------------

class TestChunkIdScoping:
    """chunk_id must be unique across different (kg_name, file_name, text) tuples."""

    def test_same_text_different_kg_gives_different_id(self):
        t = "Aspirin reduces fever."
        id1 = _chunk_id("kg_a", "doc.txt", t)
        id2 = _chunk_id("kg_b", "doc.txt", t)
        assert id1 != id2

    def test_same_text_different_file_gives_different_id(self):
        t = "Aspirin reduces fever."
        id1 = _chunk_id("kg_a", "doc1.txt", t)
        id2 = _chunk_id("kg_a", "doc2.txt", t)
        assert id1 != id2

    def test_same_inputs_gives_stable_id(self):
        id1 = _chunk_id("kg_a", "doc.txt", "text")
        id2 = _chunk_id("kg_a", "doc.txt", "text")
        assert id1 == id2

    def test_id_is_40_char_hex(self):
        cid = _chunk_id("kg_a", "doc.txt", "hello")
        assert re.fullmatch(r"[0-9a-f]{40}", cid)


# ---------------------------------------------------------------------------
# 1b. Passage provenance survives storage
# ---------------------------------------------------------------------------

class TestPassageProvenanceStorage:
    def _kg(self) -> dict:
        return {
            "nodes": [
                {
                    "id": "musique_entity_alice",
                    "label": "Person",
                    "properties": {
                        "name": "Alice",
                        "type": "Person",
                        "original_id": "Alice",
                    },
                    "embedding": None,
                }
            ],
            "relationships": [],
            "chunks": [
                {
                    "text": "Alice wrote a book.",
                    "chunk_id": 0,
                    "chunk_local_index": 0,
                    "position": 7,
                    "start_pos": 0,
                    "end_pos": 19,
                    "source": "musique/q1/p0",
                    "dataset": "musique",
                    "question_id": "q1",
                    "passage_index": 0,
                    "embedding": None,
                }
            ],
            "metadata": {
                "total_chunks": 1,
                "total_entities": 1,
                "total_relationships": 0,
                "ontology_classes": 0,
                "ontology_relationships": 0,
                "kg_name": "musique",
            },
        }

    def test_chunk_storage_includes_passage_provenance(self, stub_graph):
        creator = _make_creator()
        creator._create_neo4j_connection = lambda: stub_graph
        creator._create_vector_indexes = lambda graph: None

        ok = creator.store_knowledge_graph_with_embeddings(self._kg(), "musique")

        assert ok is True
        _, params = next(
            (query, params)
            for query, params in stub_graph.queries
            if "MERGE (c:Chunk {id: $chunk_id})" in query
        )
        assert params["position"] == 7
        assert params["chunk_local_index"] == 0
        assert params["source"] == "musique/q1/p0"
        assert params["dataset"] == "musique"
        assert params["question_id"] == "q1"
        assert params["passage_index"] == 0

    def test_mentions_use_global_chunk_index_and_source(self, stub_graph):
        creator = _make_creator()
        creator._create_neo4j_connection = lambda: stub_graph
        creator._create_vector_indexes = lambda graph: None

        ok = creator.store_knowledge_graph_with_embeddings(self._kg(), "musique")

        assert ok is True
        _, params = next(
            (query, params)
            for query, params in stub_graph.queries
            if "MERGE (m:Mention {id: $mention_id})" in query
        )
        assert params["chunk_index"] == 7
        assert params["chunk_local_index"] == 0
        assert params["chunk_source"] == "musique/q1/p0"

    def test_relationship_verification_uses_local_provenance_chunks(self, stub_graph):
        creator = _make_creator()
        creator._create_neo4j_connection = lambda: stub_graph
        creator._create_vector_indexes = lambda graph: None

        captured = {}

        def fake_verify(source_name, target_name, rel_type, chunks, **kwargs):
            captured["chunks"] = chunks
            return 1.0

        creator._verify_triple_confidence = fake_verify

        kg = {
            "nodes": [
                {
                    "id": "musique_entity_alice",
                    "label": "Person",
                    "properties": {"name": "Alice", "type": "Person", "original_id": "Alice"},
                    "embedding": None,
                },
                {
                    "id": "musique_entity_bob",
                    "label": "Person",
                    "properties": {"name": "Bob", "type": "Person", "original_id": "Bob"},
                    "embedding": None,
                },
            ],
            "relationships": [
                {
                    "id": "musique_rel_alice_knows_bob_0",
                    "from": "musique_entity_alice",
                    "to": "musique_entity_bob",
                    "source": "musique_entity_alice",
                    "target": "musique_entity_bob",
                    "type": "KNOWS",
                    "label": "KNOWS",
                    "negated": False,
                    "properties": {},
                    "provenance_positions": [7],
                }
            ],
            "chunks": [
                {
                    "text": "Alice knows Bob from school.",
                    "chunk_id": 0,
                    "chunk_local_index": 0,
                    "position": 7,
                    "start_pos": 0,
                    "end_pos": 27,
                    "source": "musique/q1/p0",
                    "dataset": "musique",
                    "question_id": "q1",
                    "passage_index": 0,
                    "embedding": None,
                },
                {
                    "text": "Irrelevant extra passage.",
                    "chunk_id": 1,
                    "chunk_local_index": 0,
                    "position": 8,
                    "start_pos": 0,
                    "end_pos": 24,
                    "source": "musique/q2/p0",
                    "dataset": "musique",
                    "question_id": "q2",
                    "passage_index": 0,
                    "embedding": None,
                },
            ],
            "metadata": {
                "total_chunks": 2,
                "total_entities": 2,
                "total_relationships": 1,
                "ontology_classes": 0,
                "ontology_relationships": 0,
                "kg_name": "musique",
            },
        }

        ok = creator.store_knowledge_graph_with_embeddings(kg, "musique")

        assert ok is True
        assert [chunk["position"] for chunk in captured["chunks"]] == [7]

    def test_relationship_store_failure_returns_false(self, stub_graph):
        creator = _make_creator()
        creator._create_neo4j_connection = lambda: stub_graph
        creator._create_vector_indexes = lambda graph: None
        creator._verify_triple_confidence = lambda *args, **kwargs: 1.0

        original_query = stub_graph.query

        def flaky_query(query, params=None):
            if "MERGE (source)-[r:" in query:
                raise RuntimeError("forced relationship store failure")
            return original_query(query, params)

        stub_graph.query = flaky_query

        kg = {
            "nodes": [
                {
                    "id": "musique_entity_alice",
                    "label": "Person",
                    "properties": {"name": "Alice", "type": "Person", "original_id": "Alice"},
                    "embedding": None,
                },
                {
                    "id": "musique_entity_bob",
                    "label": "Person",
                    "properties": {"name": "Bob", "type": "Person", "original_id": "Bob"},
                    "embedding": None,
                },
            ],
            "relationships": [
                {
                    "id": "musique_rel_alice_knows_bob_0",
                    "from": "musique_entity_alice",
                    "to": "musique_entity_bob",
                    "source": "musique_entity_alice",
                    "target": "musique_entity_bob",
                    "type": "KNOWS",
                    "label": "KNOWS",
                    "negated": False,
                    "properties": {},
                    "provenance_positions": [7],
                }
            ],
            "chunks": [
                {
                    "text": "Alice knows Bob from school.",
                    "chunk_id": 0,
                    "chunk_local_index": 0,
                    "position": 7,
                    "start_pos": 0,
                    "end_pos": 27,
                    "source": "musique/q1/p0",
                    "dataset": "musique",
                    "question_id": "q1",
                    "passage_index": 0,
                    "embedding": None,
                }
            ],
            "metadata": {
                "total_chunks": 1,
                "total_entities": 2,
                "total_relationships": 1,
                "ontology_classes": 0,
                "ontology_relationships": 0,
                "kg_name": "musique",
            },
        }

        ok = creator.store_knowledge_graph_with_embeddings(kg, "musique")

        assert ok is False
        assert kg["metadata"]["relationship_store_failures"] == 1
        assert kg["metadata"]["stored_relationships"] == 0

    def test_large_graph_tolerates_single_relationship_store_failure(self, stub_graph):
        creator = _make_creator()
        creator._create_neo4j_connection = lambda: stub_graph
        creator._create_vector_indexes = lambda graph: None
        creator._verify_triple_confidence = lambda *args, **kwargs: 1.0

        original_query = stub_graph.query
        failed_once = {"done": False}

        def flaky_query(query, params=None):
            if "MERGE (source)-[r:" in query and not failed_once["done"]:
                failed_once["done"] = True
                raise RuntimeError("forced relationship store failure")
            return original_query(query, params)

        stub_graph.query = flaky_query

        kg = {
            "nodes": [
                {
                    "id": f"bioasq_entity_{idx}",
                    "label": "Person",
                    "properties": {
                        "name": f"Entity {idx}",
                        "type": "Person",
                        "original_id": f"Entity {idx}",
                    },
                    "embedding": None,
                }
                for idx in range(201)
            ],
            "relationships": [
                {
                    "id": f"bioasq_rel_{idx}",
                    "from": f"bioasq_entity_{idx}",
                    "to": f"bioasq_entity_{idx + 1}",
                    "source": f"bioasq_entity_{idx}",
                    "target": f"bioasq_entity_{idx + 1}",
                    "type": "KNOWS",
                    "label": "KNOWS",
                    "negated": False,
                    "properties": {},
                    "provenance_positions": [7],
                }
                for idx in range(200)
            ],
            "chunks": [
                {
                    "text": " ".join(f"Entity {idx} knows Entity {idx + 1}." for idx in range(200)),
                    "chunk_id": 0,
                    "chunk_local_index": 0,
                    "position": 7,
                    "start_pos": 0,
                    "end_pos": 8192,
                    "source": "bioasq/doc0",
                    "dataset": "bioasq",
                    "question_id": None,
                    "passage_index": 0,
                    "embedding": None,
                }
            ],
            "metadata": {
                "total_chunks": 1,
                "total_entities": 201,
                "total_relationships": 200,
                "ontology_classes": 0,
                "ontology_relationships": 0,
                "kg_name": "bioasq",
            },
        }

        ok = creator.store_knowledge_graph_with_embeddings(kg, "bioasq")

        assert ok is True
        assert kg["metadata"]["relationship_store_failures"] == 1
        assert kg["metadata"]["stored_relationships"] == 199
        assert kg["metadata"]["relationship_store_degraded"] is True
        assert kg["metadata"]["relationship_store_failure_ratio"] == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# 2. _verify_triple_confidence — scoring tiers
# ---------------------------------------------------------------------------

class TestVerifyTripleConfidence:
    creator = _make_creator()

    def _chunks(self, *texts):
        return [{"text": t} for t in texts]

    def test_same_sentence_returns_1_0(self):
        chunks = self._chunks("Metformin treats type 2 diabetes in patients.")
        score = self.creator._verify_triple_confidence(
            "Metformin", "type 2 diabetes", "TREATS", chunks
        )
        assert score == 1.0

    def test_same_chunk_different_sentence_returns_0_7(self):
        # Two sentences in one chunk — not the same sentence
        text = "Metformin is a biguanide drug. Type 2 diabetes affects millions."
        chunks = self._chunks(text)
        score = self.creator._verify_triple_confidence(
            "Metformin", "type 2 diabetes", "TREATS", chunks
        )
        assert score == 0.7

    def test_cross_chunk_both_found_returns_0_4(self):
        chunks = self._chunks("Metformin is a biguanide.", "Type 2 diabetes affects millions.")
        score = self.creator._verify_triple_confidence(
            "Metformin", "type 2 diabetes", "TREATS", chunks
        )
        assert score == 0.4

    def test_only_source_found_returns_0_3(self):
        chunks = self._chunks("Metformin is a biguanide drug.")
        score = self.creator._verify_triple_confidence(
            "Metformin", "type 2 diabetes", "TREATS", chunks
        )
        assert score == 0.3

    def test_neither_found_returns_0_1(self):
        chunks = self._chunks("Paracetamol reduces inflammation.")
        score = self.creator._verify_triple_confidence(
            "Metformin", "type 2 diabetes", "TREATS", chunks
        )
        assert score == 0.1

    def test_empty_chunks_returns_neutral(self):
        score = self.creator._verify_triple_confidence(
            "Metformin", "type 2 diabetes", "TREATS", []
        )
        assert score == 0.5

    def test_score_in_unit_interval(self):
        chunks = self._chunks("Some irrelevant text about nothing.")
        score = self.creator._verify_triple_confidence("A", "B", "REL", chunks)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 3. _verify_triple_confidence — alias coverage
# ---------------------------------------------------------------------------

class TestVerifyTripleConfidenceAliases:
    creator = _make_creator()

    def test_alias_used_for_same_sentence_match(self):
        chunks = [{"text": "MET lowers blood glucose in diabetic patients."}]
        score = self.creator._verify_triple_confidence(
            "Metformin", "blood glucose", "LOWERS",
            chunks,
            source_aliases=["MET", "metformin hydrochloride"],
        )
        assert score == 1.0

    def test_canonical_name_used_when_alias_absent(self):
        chunks = [{"text": "Metformin lowers blood glucose."}]
        score = self.creator._verify_triple_confidence(
            "Metformin", "blood glucose", "LOWERS",
            chunks,
            source_aliases=["MET"],
        )
        assert score == 1.0

    def test_target_alias_triggers_same_chunk(self):
        # Source in chunk 1, target alias in chunk 2 → cross-chunk (0.4)
        chunks = [
            {"text": "Metformin is prescribed widely."},
            {"text": "T2DM affects insulin sensitivity."},
        ]
        score = self.creator._verify_triple_confidence(
            "Metformin", "type 2 diabetes", "TREATS",
            chunks,
            target_aliases=["T2DM"],
        )
        assert score == 0.4


class TestEntityAppearsInText:
    creator = _make_creator()

    def test_alias_match_works_with_cross_chunk_candidate_filter(self):
        entity = {
            "id": "type_2_diabetes",
            "properties": {
                "name": "type_2_diabetes",
                "all_names": ["T2DM", "type 2 diabetes"],
            },
        }

        assert self.creator._entity_appears_in_text(
            entity,
            "Patients with T2DM often require monitoring.",
        )


# ---------------------------------------------------------------------------
# 4. _harmonize_entities — deduplication contract
# ---------------------------------------------------------------------------

class TestHarmonizeEntities:
    creator = _make_creator()

    def _ent(self, name: str, etype: str = "Disease") -> dict:
        # _harmonize_entities reads entity['type'] (not 'label') — match production shape.
        return {
            "id": name,
            "type": etype,
            "properties": {"name": name, "description": ""},
        }

    def test_identical_names_deduplicated(self):
        entities = [self._ent("Aspirin"), self._ent("Aspirin")]
        result = self.creator._harmonize_entities(entities)
        ids = [e["id"] for e in result]
        assert ids.count(ids[0]) == 1

    def test_case_insensitive_dedup(self):
        entities = [self._ent("Aspirin"), self._ent("aspirin"), self._ent("ASPIRIN")]
        result = self.creator._harmonize_entities(entities)
        assert len(result) == 1

    def test_different_names_kept_separate(self):
        entities = [self._ent("Aspirin"), self._ent("Metformin")]
        result = self.creator._harmonize_entities(entities)
        assert len(result) == 2

    def test_more_specific_type_wins(self):
        # "Disease" beats generic "Entity"
        entities = [self._ent("Flu", "Entity"), self._ent("Flu", "Disease")]
        result = self.creator._harmonize_entities(entities)
        assert len(result) == 1
        assert result[0]["type"] in ("Disease", "disease")

    def test_same_surface_different_specific_types_kept_separate(self):
        # "depression" as a Disease and as a GeologicalFeature must not collapse.
        entities = [
            self._ent("depression", "Disease"),
            self._ent("depression", "GeologicalFeature"),
        ]
        result = self.creator._harmonize_entities(entities)
        assert len(result) == 2
        types = {e["type"] for e in result}
        assert "Disease" in types
        assert "GeologicalFeature" in types

    def test_generic_type_drift_still_merges(self):
        # Same entity tagged Disease in one chunk, generic Concept in another
        # (LLM drift) — must collapse into one Disease node, not split.
        entities = [
            self._ent("Prostate Cancer", "Disease"),
            self._ent("Prostate Cancer", "Concept"),
        ]
        result = self.creator._harmonize_entities(entities)
        assert len(result) == 1
        assert result[0]["type"] in ("Disease", "disease")

    def test_generic_assigned_to_dominant_bucket_on_split(self):
        # Three occurrences: Disease x2, GeologicalFeature x1, Concept x1 (generic).
        # Generics should go to Disease (largest bucket); result: Disease node + GeologicalFeature node.
        entities = [
            self._ent("depression", "Disease"),
            self._ent("depression", "Disease"),
            self._ent("depression", "GeologicalFeature"),
            self._ent("depression", "Concept"),
        ]
        result = self.creator._harmonize_entities(entities)
        assert len(result) == 2
        types = {e["type"] for e in result}
        assert "Disease" in types
        assert "GeologicalFeature" in types

    def test_split_entities_have_distinct_uuids(self):
        # End-to-end UUID check: same surface form, different specific types must
        # survive ID generation as two nodes with different UUIDs — not collapse at write.
        entities = [
            self._ent("depression", "Disease"),
            self._ent("depression", "GeologicalFeature"),
        ]
        result = self.creator._harmonize_entities(entities)
        assert len(result) == 2
        uuids = [e["uuid"] for e in result]
        assert uuids[0] != uuids[1], (
            "Same surface form with different specific types must get distinct UUIDs; "
            "if equal, _generate_entity_id is not including type in the seed"
        )

    def test_merged_entity_stable_uuid(self):
        # LLM type-drift case: Disease + Concept for the same surface form must merge
        # into one node AND produce the same UUID regardless of input order.
        fwd = self.creator._harmonize_entities([
            self._ent("Prostate Cancer", "Disease"),
            self._ent("Prostate Cancer", "Concept"),
        ])
        rev = self.creator._harmonize_entities([
            self._ent("Prostate Cancer", "Concept"),
            self._ent("Prostate Cancer", "Disease"),
        ])
        assert len(fwd) == 1
        assert len(rev) == 1
        assert fwd[0]["uuid"] == rev[0]["uuid"], (
            "Merged entity UUID must be stable regardless of input order"
        )

    def test_relationship_resolves_to_dominant_type(self):
        # Disease x2, GeologicalFeature x1 → Disease is dominant.
        # The relationship target "depression" must resolve to the Disease UUID,
        # not GeologicalFeature, regardless of extraction order.
        entities = [
            self._ent("fluoxetine", "Drug"),
            self._ent("depression", "Disease"),
            self._ent("depression", "Disease"),       # dominant: 2 occurrences
            self._ent("depression", "GeologicalFeature"),
        ]
        relationships = [
            {"source": "fluoxetine", "target": "depression",
             "type": "TREATS", "negated": False, "properties": {}},
        ]
        result, entity_map = self.creator._harmonize_entities(entities, return_id_map=True)
        harmonized_rels = self.creator._harmonize_relationships(relationships, entity_map)

        assert len(harmonized_rels) == 1, "Relationship must not be dropped after entity split"
        disease_uuid = next(e["uuid"] for e in result if e.get("type") == "Disease")
        geo_uuid = next(e["uuid"] for e in result if e.get("type") == "GeologicalFeature")
        assert harmonized_rels[0]["target"] == disease_uuid, (
            f"Expected target={disease_uuid} (Disease, dominant), got {harmonized_rels[0]['target']}; "
            f"GeologicalFeature uuid={geo_uuid}"
        )

    def test_relationship_resolution_stable_across_order(self):
        # Same as above but with extraction order reversed — resolution must be identical.
        entities_fwd = [
            self._ent("fluoxetine", "Drug"),
            self._ent("depression", "Disease"),
            self._ent("depression", "Disease"),
            self._ent("depression", "GeologicalFeature"),
        ]
        entities_rev = [
            self._ent("fluoxetine", "Drug"),
            self._ent("depression", "GeologicalFeature"),
            self._ent("depression", "Disease"),
            self._ent("depression", "Disease"),
        ]
        relationships = [
            {"source": "fluoxetine", "target": "depression",
             "type": "TREATS", "negated": False, "properties": {}},
        ]

        _, map_fwd = self.creator._harmonize_entities(entities_fwd, return_id_map=True)
        _, map_rev = self.creator._harmonize_entities(entities_rev, return_id_map=True)
        rels_fwd = self.creator._harmonize_relationships(relationships, map_fwd)
        rels_rev = self.creator._harmonize_relationships(relationships, map_rev)

        assert len(rels_fwd) == 1 and len(rels_rev) == 1
        assert rels_fwd[0]["target"] == rels_rev[0]["target"], (
            "Relationship target UUID must be the same regardless of entity extraction order"
        )

    def test_resolve_relationship_endpoint_handles_short_form(self):
        entity = self._ent("TBK1 kinase", "Protein")
        entity["properties"]["all_names"] = ["TBK1", "TANK-binding kinase 1"]

        resolved = self.creator._resolve_relationship_endpoint("TBK1", [entity])

        assert resolved == "TBK1 kinase"

    def test_relationship_dedup_keeps_qualifier_variants(self):
        entities = [
            self._ent("Aspirin", "Drug"),
            self._ent("Fever", "Disease"),
        ]
        relationships = [
            {
                "source": "Aspirin",
                "target": "Fever",
                "type": "TREATS",
                "negated": False,
                "properties": {"condition": "in adults"},
            },
            {
                "source": "Aspirin",
                "target": "Fever",
                "type": "TREATS",
                "negated": False,
                "properties": {"condition": "in children"},
            },
        ]

        _, entity_map = self.creator._harmonize_entities(entities, return_id_map=True)
        rels = self.creator._harmonize_relationships(relationships, entity_map)

        assert len(rels) == 2, "Condition-specific variants must not collapse before storage"


# ---------------------------------------------------------------------------
# 5. relation MERGE key contract (negated always; qualifiers only when present)
# ---------------------------------------------------------------------------

class TestRelationMergeKey:
    """
    The Cypher MERGE for relationships must always include negated and must
    only include optional qualifiers when present. This avoids Neo4j rejecting
    the query with null-valued MERGE properties.
    """

    def setup_method(self):
        self.creator = _make_creator()

    def test_merge_key_includes_negated(self):
        q = self.creator._build_relationship_merge_query(
            "TREATS",
            include_condition=False,
            include_quantitative=False,
        )
        assert "negated: $negated" in q, "MERGE key must include 'negated'"

    def test_merge_key_omits_absent_condition(self):
        q = self.creator._build_relationship_merge_query(
            "TREATS",
            include_condition=False,
            include_quantitative=False,
        )
        assert "condition: $condition" not in q, (
            "MERGE key must omit absent 'condition' to avoid null MERGE errors"
        )

    def test_merge_key_includes_present_condition(self):
        q = self.creator._build_relationship_merge_query(
            "TREATS",
            include_condition=True,
            include_quantitative=False,
        )
        assert "condition: $condition" in q, (
            "MERGE key must include 'condition' when a qualifier is present"
        )

    def test_merge_key_omits_absent_quantitative(self):
        q = self.creator._build_relationship_merge_query(
            "TREATS",
            include_condition=False,
            include_quantitative=False,
        )
        assert "quantitative: $quantitative" not in q, (
            "MERGE key must omit absent 'quantitative' to avoid null MERGE errors"
        )

    def test_merge_key_includes_present_quantitative(self):
        q = self.creator._build_relationship_merge_query(
            "TREATS",
            include_condition=False,
            include_quantitative=True,
        )
        assert "quantitative: $quantitative" in q, (
            "MERGE key must include 'quantitative' when a qualifier is present"
        )

    def test_merge_query_persists_edge_provenance(self):
        q = self.creator._build_relationship_merge_query(
            "TREATS",
            include_condition=False,
            include_quantitative=False,
        )
        assert "r.provenancePositions" in q
        assert "r.questionIds" in q
        assert "r.passageKeys" in q


class TestRelationshipLocalProvenance:
    def test_relationship_local_provenance_resolves_question_and_passage(self):
        creator = _make_creator()
        rel = {"provenance_positions": [7, 8]}
        chunks = [
            {"position": 7, "question_id": "q1", "passage_index": 0},
            {"position": 8, "question_id": "q1", "passage_index": 0},
            {"position": 30, "question_id": "q2", "passage_index": 1},
        ]

        provenance = creator._relationship_local_provenance(rel, chunks)

        assert provenance["provenance_positions"] == [7, 8]
        assert provenance["question_ids"] == ["q1"]
        assert provenance["passage_keys"] == ["q1::p0"]


# ---------------------------------------------------------------------------
# 6. synonym merge type guard reads e.type / e.ontology_class
# ---------------------------------------------------------------------------

class TestSynonymMergeTypeProperty:
    """
    The fetch query in merge_synonym_entities must read e.type / e.ontology_class,
    not the never-written e.entity_type.
    """

    def _get_fetch_query(self) -> str:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ontographrag", "kg", "builders", "ontology_guided_kg_creator.py"
        )
        with open(path) as f:
            src = f.read()
        # Find the fetch_q assignment inside merge_synonym_entities
        m = re.search(
            r"fetch_q\s*=\s*f?\"\"\"(.*?)\"\"\"",
            src, re.DOTALL
        )
        return m.group(1) if m else ""

    def test_fetch_reads_type_not_entity_type(self):
        q = self._get_fetch_query()
        assert "e.entity_type" not in q, (
            "fetch query must not use e.entity_type (never written); "
            "use e.type or e.ontology_class"
        )

    def test_fetch_reads_type_or_ontology_class(self):
        q = self._get_fetch_query()
        assert "e.type" in q or "e.ontology_class" in q, (
            "fetch query must read e.type or e.ontology_class"
        )
