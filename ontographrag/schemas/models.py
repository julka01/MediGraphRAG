"""
Canonical schemas for the ontographrag pipeline.

These Pydantic models define the shapes that flow between the KG build layer,
the retrieval layer, and the generation layer.

Where to use them
-----------------
- At I/O boundaries (API responses, file serialisation): validate with
  ``Model(**raw_dict)`` to catch shape regressions at the earliest point.
- In tests: construct fixtures with the model so missing-field bugs surface
  as Pydantic validation errors rather than silent None values.

Compatibility: all fields that were previously optional in plain dicts are
marked Optional so existing callers that omit them don't break on parse.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ontology schema  (steps 1–2: typed schema models)
# ---------------------------------------------------------------------------

class PropertyType(str, Enum):
    STRING      = "string"
    INTEGER     = "integer"
    DECIMAL     = "decimal"
    DOUBLE      = "double"
    FLOAT       = "float"
    BOOLEAN     = "boolean"
    DATE        = "date"
    DATETIME    = "datetime"
    ENUM        = "enum"
    ID          = "id"


class DataBinding(BaseModel):
    """A typed property bound to an EntityType."""
    name: str
    type: PropertyType = PropertyType.STRING
    description: Optional[str] = None
    identifier: bool = False        # is this the primary lookup key?
    required: bool = False
    enum_values: List[str] = Field(default_factory=list)
    unit: Optional[str] = None      # e.g. "mg", "years", "mmHg"


class RelationshipAttribute(BaseModel):
    """A typed attribute that qualifies a RelationshipType edge."""
    name: str
    type: PropertyType = PropertyType.STRING
    description: Optional[str] = None
    unit: Optional[str] = None


class EntityType(BaseModel):
    """An ontology class with full property schema."""
    id: str
    label: str
    description: Optional[str] = None
    uri: Optional[str] = None
    properties: List[DataBinding] = Field(default_factory=list)


class RelationshipType(BaseModel):
    """An ontology relationship with domain/range/cardinality/attributes."""
    id: str
    label: str
    description: Optional[str] = None
    uri: Optional[str] = None
    domain: Optional[str] = None       # EntityType.id of the source class
    range: Optional[str] = None        # EntityType.id of the target class
    cardinality: Optional[str] = None  # "one_to_one" | "one_to_many" | "many_to_many"
    attributes: List[RelationshipAttribute] = Field(default_factory=list)


class OntologySchema(BaseModel):
    """Full parsed ontology — the single internal representation for both OWL and JSON input."""
    entity_types: List[EntityType] = Field(default_factory=list)
    relationship_types: List[RelationshipType] = Field(default_factory=list)
    # "owl" | "json" | "inferred" — records which parser produced this
    source_format: str = "unknown"
    source_path: Optional[str] = None

    # --------------- convenience helpers ---------------

    def entity_type_by_id(self, eid: str) -> Optional[EntityType]:
        eid_lower = eid.lower()
        for et in self.entity_types:
            if et.id.lower() == eid_lower or et.label.lower() == eid_lower:
                return et
        return None

    def compatible_relationships(
        self,
        source_type: Optional[str] = None,
        target_type: Optional[str] = None,
    ) -> List[RelationshipType]:
        """Return relationship types whose domain/range are compatible with the given entity types.

        A relationship is compatible when:
        - Its domain is None (unconstrained) OR equals source_type
        - Its range  is None (unconstrained) OR equals target_type

        Fully-constrained matches are ranked first.
        """
        if not source_type and not target_type:
            return list(self.relationship_types)

        src = (source_type or "").lower()
        tgt = (target_type or "").lower()

        full_match, partial_match, unconstrained = [], [], []
        for rt in self.relationship_types:
            d = (rt.domain or "").lower()
            r = (rt.range or "").lower()
            dom_ok  = not d or d == src
            rang_ok = not r or r == tgt
            if not dom_ok or not rang_ok:
                continue
            if d and r:
                full_match.append(rt)
            elif d or r:
                partial_match.append(rt)
            else:
                unconstrained.append(rt)
        return full_match + partial_match + unconstrained


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """A text chunk retrieved from the vector/text index or entity-first search."""

    text: str
    chunk_id: Optional[str] = None
    score: float = 0.0
    document: Optional[str] = None
    kg_name: Optional[str] = None

    # Entity IDs directly linked to this chunk (set by entity-first search).
    linked_entity_ids: List[str] = Field(default_factory=list)

    # Inline entity dicts carried by vector/text search results.
    entities: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """A KG entity, as returned by entity-first search and graph traversal."""

    id: str
    name: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None

    # How the entity entered the result set.
    # "entity_lookup"  — matched directly from the question text
    # "graph_traversal" — reached via graph walk from a seed entity
    source: str = "entity_lookup"

    # Minimum graph-hop distance from any seed entity (0 = seed).
    min_hops: int = 0

    # Ontology class (may differ from type in ontology-guided KGs).
    ontology_class: Optional[str] = None


# ---------------------------------------------------------------------------
# Relationship
# ---------------------------------------------------------------------------

class Relationship(BaseModel):
    """A KG relationship between two entities."""

    source: str   # entity ID
    target: str   # entity ID
    type: str = "RELATED_TO"

    # Composite key used for deduplication in the result set.
    # Format: "{source}-{type}-{target}"
    key: Optional[str] = None

    # Claim modifiers — all three are part of the Neo4j MERGE key so that
    # contradictory / variant claims are stored as distinct edges.
    negated: bool = False
    condition: Optional[str] = None
    quantitative: Optional[str] = None

    confidence: float = 1.0
    properties: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# TraversalPath
# ---------------------------------------------------------------------------

class TraversalPath(BaseModel):
    """A single multi-hop traversal path discovered by graph walk."""

    path: str   # human-readable, e.g. "Aspirin --TREATS--> Fever"
    hops: int = 1

    # Entity IDs along the path — used to prune paths after chunk truncation.
    # Paths produced before this field was added will have node_ids=[] and are
    # kept as-is (see _iterative_hop_retrieval path pruning logic).
    node_ids: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# KGContext
# ---------------------------------------------------------------------------

class KGContext(BaseModel):
    """
    The structured context assembled by the retrieval layer and passed to the
    generation layer.  Corresponds to the dict returned by _entity_first_search,
    _vector_similarity_search, _iterative_hop_retrieval, etc.
    """

    query: str

    chunks: List[Chunk] = Field(default_factory=list)

    # Keyed by entity ID.
    entities: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    relationships: List[Relationship] = Field(default_factory=list)

    # Graph-traversal neighbours keyed by entity ID (subset of entities that
    # were discovered via graph walk, not from question-entity matching).
    graph_neighbors: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Multi-hop traversal paths discovered during graph walk.
    traversal_paths: List[TraversalPath] = Field(default_factory=list)

    # Source document names present in the retrieved chunks.
    documents: List[str] = Field(default_factory=list)

    total_score: float = 0.0
    entity_count: int = 0
    relationship_count: int = 0

    # Which retrieval strategy produced this context.
    # One of: "entity_first", "vector_similarity", "text_similarity",
    #         "iterative_hop", "hybrid"
    search_method: str = "entity_first"

    kg_name: Optional[str] = None

    # Grounding metrics (anchored to hop-0 in iterative retrieval).
    seed_entity_count: int = 0
    grounding_quality: float = 0.0


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    """
    The top-level result returned by generate_response.
    Wraps the KGContext with the generated answer and evaluation metadata.
    """

    question: str
    answer: str

    context: KGContext

    # Whether the answer was generated with KG-RAG (True) or vanilla RAG (False).
    used_kg: bool = True

    # LLM model that generated the answer.
    model: Optional[str] = None

    # Milliseconds taken for the full generate_response call.
    latency_ms: Optional[float] = None

    # Optional evaluation scores (filled in by the experiment pipeline).
    scores: Dict[str, Any] = Field(default_factory=dict)
