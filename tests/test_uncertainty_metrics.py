import sys
import types

import numpy as np

from experiments import uncertainty_metrics as um


class _FakeSentenceTransformer:
    init_calls = []

    def __init__(self, model_name):
        self.model_name = model_name
        self.__class__.init_calls.append(model_name)

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=False):
        if isinstance(texts, str):
            texts = [texts]
        vecs = np.ones((len(texts), 3), dtype=float)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / norms
        return vecs


def test_sentence_transformer_helper_reuses_normalized_cache(monkeypatch):
    fake_module = types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(um, "_SENTENCE_TRANSFORMER_CACHE", {})
    _FakeSentenceTransformer.init_calls.clear()

    model_a = um._get_or_load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")
    model_b = um._get_or_load_sentence_transformer("all-MiniLM-L6-v2")

    assert model_a is model_b
    assert _FakeSentenceTransformer.init_calls == ["all-MiniLM-L6-v2"]


def test_evidence_vn_entropy_query_checks_tail_reachability(monkeypatch):
    captured_queries = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, query, params=None):
            captured_queries.append(query)
            if "RETURN DISTINCT e.id AS id" in query:
                return [{"id": "q1"}]
            return [{"head": "Head", "rel": "RELATED_TO", "tail": "Tail"}]

    class FakeDriver:
        def session(self, database=None):
            return FakeSession()

        def close(self):
            return None

    class FakeGraphDatabase:
        @staticmethod
        def driver(uri, auth=None):
            return FakeDriver()

    fake_sentence_module = types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
    fake_neo4j_module = types.SimpleNamespace(GraphDatabase=FakeGraphDatabase)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_module)
    monkeypatch.setitem(sys.modules, "neo4j", fake_neo4j_module)
    monkeypatch.setattr(um, "_SENTENCE_TRANSFORMER_CACHE", {})
    _FakeSentenceTransformer.init_calls.clear()

    score = um.compute_evidence_vn_entropy(
        question="Which county is Hughesville in?",
        neo4j_uri="bolt://unused",
        neo4j_user="neo4j",
        neo4j_password="test",
        max_hops=4,
        n_triples=5,
    )

    assert 0.0 <= score <= 1.0
    triple_query = captured_queries[1]
    assert "MATCH (q_e:__Entity__)-[*1..4]-(t)" in triple_query
