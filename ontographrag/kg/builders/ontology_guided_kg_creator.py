import json
import re
import hashlib
import difflib
import xml.etree.ElementTree as ET
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain_neo4j import Neo4jGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ontographrag.kg.chunking import chunk_text as _chunk_text_fn
from collections import defaultdict, Counter
import os
import sys
import logging
import time

# Import from local kg_utils
from ontographrag.kg.utils.common_functions import load_embedding_model
from ontographrag.schemas.models import (
    OntologySchema, EntityType as OntEntityType, RelationshipType as OntRelType,
    DataBinding, RelationshipAttribute, PropertyType,
)

class OntologyGuidedKGCreator:
    """
    Ontology-Guided Knowledge Graph Creator that properly extracts entities from PDF content
    using LLM with ontology guidance for better entity classification and relationships
    """

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        embedding_model: str = "sentence_transformers",
        ontology_path: str = None,
        enable_coreference_resolution: bool = False,
        retrieval_chunk_size: Optional[int] = None,
        retrieval_chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        env_retrieval_chunk_size = os.getenv("RETRIEVAL_CHUNK_SIZE")
        env_retrieval_chunk_overlap = os.getenv("RETRIEVAL_CHUNK_OVERLAP")
        resolved_retrieval_chunk_size = retrieval_chunk_size
        if resolved_retrieval_chunk_size is None and env_retrieval_chunk_size:
            try:
                resolved_retrieval_chunk_size = int(env_retrieval_chunk_size)
            except ValueError:
                logging.warning(
                    "Invalid RETRIEVAL_CHUNK_SIZE=%r; falling back to 256",
                    env_retrieval_chunk_size,
                )
        resolved_retrieval_chunk_overlap = retrieval_chunk_overlap
        if resolved_retrieval_chunk_overlap is None and env_retrieval_chunk_overlap:
            try:
                resolved_retrieval_chunk_overlap = int(env_retrieval_chunk_overlap)
            except ValueError:
                logging.warning(
                    "Invalid RETRIEVAL_CHUNK_OVERLAP=%r; falling back to 64",
                    env_retrieval_chunk_overlap,
                )
        self.retrieval_chunk_size = max(64, int(resolved_retrieval_chunk_size or 256))
        default_retrieval_overlap = resolved_retrieval_chunk_overlap
        if default_retrieval_overlap is None:
            default_retrieval_overlap = min(64, max(16, self.retrieval_chunk_size // 4))
        self.retrieval_chunk_overlap = max(
            0,
            min(int(default_retrieval_overlap), self.retrieval_chunk_size - 1),
        )
        self.enable_coreference_resolution = enable_coreference_resolution
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.embedding_model = embedding_model
        self.ontology_path = ontology_path

        # Initialize embedding model
        self.embedding_function, self.embedding_dimension = load_embedding_model(embedding_model)
        logging.info(f"Initialized embedding model: {embedding_model}, dimension: {self.embedding_dimension}")

        # Load ontology if provided
        self.ontology_classes = []
        self.ontology_relationships = []
        self._ontology_schema: Optional[OntologySchema] = None
        if ontology_path and os.path.exists(ontology_path):
            logging.info(f"Ontology file exists at: {ontology_path} (size: {os.path.getsize(ontology_path)} bytes)")
            try:
                self._load_ontology(ontology_path)
                logging.info(f"✅ Successfully loaded ontology: {len(self.ontology_classes)} classes, {len(self.ontology_relationships)} relationships")

                # Validate ontology structure to prevent "string indices must be integers" errors
                self._validate_ontology_structure()

            except Exception as e:
                logging.error(f"❌ Failed to load ontology: {e}")
                # Continue with empty ontology - LLM extraction will still work
                self.ontology_classes = []
                self.ontology_relationships = []
        else:
            if ontology_path:
                logging.warning(f"Ontology file not found: {ontology_path}")
                logging.info(f"Available files in temp dir: {os.listdir(os.path.dirname(ontology_path)) if ontology_path else 'N/A'}")
            else:
                logging.info("No ontology provided - using basic LLM entity extraction")
            # No ontology - use empty lists (will fall back to pattern matching)

        # Pre-compute ontology class label embeddings for semantic classification
        self._ontology_class_embeddings: List[Tuple[str, Any]] = []  # [(class_id, embedding), ...]
        if self.ontology_classes and self.embedding_function:
            try:
                labels = [cls['label'] for cls in self.ontology_classes]
                embeddings = self.embedding_function.embed_documents(labels)
                self._ontology_class_embeddings = [
                    (cls['id'], emb)
                    for cls, emb in zip(self.ontology_classes, embeddings)
                ]
                logging.info(f"Pre-computed embeddings for {len(self._ontology_class_embeddings)} ontology classes")
            except Exception as e:
                logging.warning(f"Could not pre-compute ontology class embeddings: {e}. Falling back to keyword matching.")

    # ------------------------------------------------------------------
    # Ontology loading — supports OWL/RDF and JSON
    # ------------------------------------------------------------------

    @staticmethod
    def _prop_type(raw: str) -> PropertyType:
        """Map a raw type string from JSON to a PropertyType enum value."""
        _map = {
            "string": PropertyType.STRING, "str": PropertyType.STRING,
            "integer": PropertyType.INTEGER, "int": PropertyType.INTEGER,
            "decimal": PropertyType.DECIMAL, "numeric": PropertyType.DECIMAL,
            "double": PropertyType.DOUBLE,
            "float": PropertyType.FLOAT, "number": PropertyType.FLOAT,
            "boolean": PropertyType.BOOLEAN, "bool": PropertyType.BOOLEAN,
            "date": PropertyType.DATE, "datetime": PropertyType.DATETIME,
            "enum": PropertyType.ENUM,
            "id": PropertyType.ID, "identifier": PropertyType.ID,
        }
        return _map.get((raw or "string").strip().lower(), PropertyType.STRING)

    def _load_ontology(self, ontology_path: str):
        """Load ontology from OWL/RDF (XML) or Ontology Playground-style JSON.

        Both paths normalise into:
          self._ontology_schema     — OntologySchema (full typed model)
          self.ontology_classes     — List[dict]  (legacy flat list)
          self.ontology_relationships — List[dict]  (legacy flat list)
        """
        ext = os.path.splitext(ontology_path)[1].lower()
        is_json = ext == '.json'
        if not is_json and ext not in ('.owl', '.rdf', '.ttl', '.xml'):
            # Peek at first byte to detect JSON
            try:
                with open(ontology_path, 'r', encoding='utf-8') as _f:
                    _peek = _f.read(3).lstrip()
                is_json = _peek.startswith('{') or _peek.startswith('[')
            except OSError:
                pass

        try:
            if is_json:
                self._ontology_schema = self._load_ontology_json(ontology_path)
            else:
                self._ontology_schema = self._load_ontology_owl(ontology_path)
        except Exception as e:
            logging.error("Error loading ontology: %s", e)
            raise

        # Populate legacy flat lists for backwards compatibility
        self.ontology_classes = [
            {'id': et.id, 'uri': et.uri or '', 'label': et.label,
             'description': et.description or ''}
            for et in self._ontology_schema.entity_types
        ]
        self.ontology_relationships = [
            {'id': rt.id, 'uri': rt.uri or '', 'label': rt.label,
             'description': rt.description or '',
             'domain': rt.domain or '', 'range': rt.range or '',
             'cardinality': rt.cardinality or ''}
            for rt in self._ontology_schema.relationship_types
        ]
        logging.info(
            "Loaded ontology (%s): %d entity types, %d relationship types",
            self._ontology_schema.source_format,
            len(self.ontology_classes), len(self.ontology_relationships),
        )

    def _load_ontology_json(self, ontology_path: str) -> OntologySchema:
        """Parse an Ontology Playground-style JSON file.

        Accepts layout A: {"classes": [...], "relationships": [...]}
        and layout B:     {"entity_types": [...], "relationship_types": [...]}
        """
        with open(ontology_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        raw_classes = raw.get('classes') or raw.get('entity_types') or []
        raw_rels = raw.get('relationships') or raw.get('relationship_types') or []

        entity_types: List[OntEntityType] = []
        for cls in raw_classes:
            if not isinstance(cls, dict):
                continue
            eid = cls.get('id') or cls.get('name') or ''
            if not eid:
                continue
            props = []
            for p in cls.get('properties') or []:
                pname = (p.get('name') or p.get('id') or '') if isinstance(p, dict) else ''
                if not pname:
                    continue
                props.append(DataBinding(
                    name=pname,
                    type=self._prop_type(p.get('type', 'string')),
                    description=p.get('description') or None,
                    identifier=bool(p.get('identifier', False)),
                    required=bool(p.get('required', False)),
                    enum_values=list(p.get('enum_values') or p.get('values') or []),
                    unit=p.get('unit') or None,
                ))
            entity_types.append(OntEntityType(
                id=eid,
                label=cls.get('label') or eid.replace('_', ' ').title(),
                description=cls.get('description') or None,
                uri=cls.get('uri') or None,
                properties=props,
            ))

        relationship_types: List[OntRelType] = []
        for rel in raw_rels:
            if not isinstance(rel, dict):
                continue
            rid = rel.get('id') or rel.get('name') or rel.get('type') or ''
            if not rid:
                continue
            attrs = []
            for a in rel.get('attributes') or rel.get('properties') or []:
                aname = (a.get('name') or a.get('id') or '') if isinstance(a, dict) else ''
                if not aname:
                    continue
                attrs.append(RelationshipAttribute(
                    name=aname,
                    type=self._prop_type(a.get('type', 'string')),
                    description=a.get('description') or None,
                    unit=a.get('unit') or None,
                ))
            relationship_types.append(OntRelType(
                id=rid,
                label=rel.get('label') or rid.replace('_', ' ').title(),
                description=rel.get('description') or None,
                uri=rel.get('uri') or None,
                domain=rel.get('from') or rel.get('domain') or None,
                range=rel.get('to') or rel.get('range') or None,
                cardinality=rel.get('cardinality') or None,
                attributes=attrs,
            ))

        return OntologySchema(
            entity_types=entity_types, relationship_types=relationship_types,
            source_format='json', source_path=ontology_path,
        )

    def _load_ontology_owl(self, ontology_path: str) -> OntologySchema:
        """Parse an OWL/RDF XML ontology into OntologySchema."""
        tree = ET.parse(ontology_path)
        root = tree.getroot()

        ns = {
            'owl':  'http://www.w3.org/2002/07/owl#',
            'rdf':  'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
        }
        _rdf_about  = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about'
        _rdfs_label = '{http://www.w3.org/2000/01/rdf-schema#}label'
        _rdfs_cmt   = '{http://www.w3.org/2000/01/rdf-schema#}comment'
        _rdfs_dom   = '{http://www.w3.org/2000/01/rdf-schema#}domain'
        _rdfs_rng   = '{http://www.w3.org/2000/01/rdf-schema#}range'
        _rdf_rsrc   = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource'

        def _local(uri: str) -> str:
            return uri.split('#')[-1] if '#' in uri else uri.split('/')[-1]

        def _res_local(elem):
            if elem is None:
                return None
            r = elem.get(_rdf_rsrc, '')
            return _local(r) if r else None

        def _child_text_by_local_name(parent, local_name: str) -> Optional[str]:
            if parent is None:
                return None
            for child in list(parent):
                tag = child.tag
                if isinstance(tag, str):
                    child_local = tag.split('}')[-1] if '}' in tag else tag.split(':')[-1]
                    if child_local == local_name and child.text:
                        return child.text.strip()
            return None

        def _bool_text(value: Optional[str]) -> bool:
            return str(value or "").strip().lower() in {"true", "1", "yes"}

        def _xsd_to_prop_type(range_uri: Optional[str], explicit_type: Optional[str]) -> PropertyType:
            if explicit_type:
                return self._prop_type(explicit_type)
            local = _local(range_uri) if range_uri else ""
            return {
                "string": PropertyType.STRING,
                "integer": PropertyType.INTEGER,
                "int": PropertyType.INTEGER,
                "long": PropertyType.INTEGER,
                "decimal": PropertyType.DECIMAL,
                "float": PropertyType.FLOAT,
                "double": PropertyType.DOUBLE,
                "date": PropertyType.DATE,
                "dateTime": PropertyType.DATETIME,
                "boolean": PropertyType.BOOLEAN,
            }.get(local, PropertyType.STRING)

        entity_types: List[OntEntityType] = []
        for cls_elem in root.findall('.//owl:Class', ns):
            uri = cls_elem.get(_rdf_about, '')
            if not uri:
                continue
            local = _local(uri)
            if not local:
                continue
            lbl_el = cls_elem.find(_rdfs_label)
            cmt_el = cls_elem.find(_rdfs_cmt)
            entity_types.append(OntEntityType(
                id=local, uri=uri,
                label=(lbl_el.text.strip() if lbl_el is not None and lbl_el.text else local.replace('_', ' ').title()),
                description=(cmt_el.text.strip() if cmt_el is not None and cmt_el.text else None),
            ))

        entity_by_id = {et.id: et for et in entity_types}

        relationship_attribute_map: Dict[str, List[RelationshipAttribute]] = defaultdict(list)

        for dt_elem in root.findall('.//owl:DatatypeProperty', ns):
            uri = dt_elem.get(_rdf_about, '')
            if not uri:
                continue

            local = _local(uri)
            label = _child_text_by_local_name(dt_elem, 'label') or local
            description = _child_text_by_local_name(dt_elem, 'comment')
            domain = _res_local(dt_elem.find('.//' + _rdfs_dom))
            range_uri = dt_elem.find('.//' + _rdfs_rng)
            range_local = _res_local(range_uri)
            explicit_type = _child_text_by_local_name(dt_elem, 'propertyType') or _child_text_by_local_name(dt_elem, 'attributeType')
            prop_type = _xsd_to_prop_type(range_local, explicit_type)
            enum_values_text = _child_text_by_local_name(dt_elem, 'enumValues')
            enum_values = [v.strip() for v in (enum_values_text or '').split(',') if v.strip()]
            unit = _child_text_by_local_name(dt_elem, 'unit')
            identifier = _bool_text(_child_text_by_local_name(dt_elem, 'isIdentifier'))
            relationship_attr_of = _child_text_by_local_name(dt_elem, 'relationshipAttributeOf')

            if relationship_attr_of:
                relationship_attribute_map[relationship_attr_of].append(
                    RelationshipAttribute(
                        name=label,
                        type=prop_type,
                        description=description,
                        unit=unit,
                    )
                )
                continue

            if not domain or domain not in entity_by_id:
                continue

            entity_by_id[domain].properties.append(
                DataBinding(
                    name=label,
                    type=prop_type,
                    description=description,
                    identifier=identifier or prop_type == PropertyType.ID,
                    required=False,
                    enum_values=enum_values,
                    unit=unit,
                )
            )

        relationship_types: List[OntRelType] = []
        for prop_elem in root.findall('.//owl:ObjectProperty', ns):
            uri = prop_elem.get(_rdf_about, '')
            if not uri:
                continue
            local = _local(uri)
            if not local:
                continue
            lbl_el = prop_elem.find(_rdfs_label)
            cmt_el = prop_elem.find(_rdfs_cmt)
            dom_el = prop_elem.find('.//' + _rdfs_dom)
            rng_el = prop_elem.find('.//' + _rdfs_rng)
            relationship_types.append(OntRelType(
                id=local, uri=uri,
                label=(lbl_el.text.strip() if lbl_el is not None and lbl_el.text else local.replace('_', ' ').title()),
                description=(cmt_el.text.strip() if cmt_el is not None and cmt_el.text else None),
                domain=_res_local(dom_el),
                range=_res_local(rng_el),
                cardinality=_child_text_by_local_name(prop_elem, 'cardinality'),
                attributes=relationship_attribute_map.get(local, []),
            ))

        return OntologySchema(
            entity_types=entity_types, relationship_types=relationship_types,
            source_format='owl', source_path=ontology_path,
        )

    def _validate_ontology_structure(self):
        """
        Validate and clean ontology class and relationship structures to prevent "string indices must be integers" errors
        """
        # Clean ontology classes
        valid_classes = []
        for cls in self.ontology_classes:
            if isinstance(cls, dict) and 'id' in cls and 'label' in cls:
                valid_classes.append(cls)
            else:
                logging.warning(f"Removing invalid ontology class entry: {cls}")

        self.ontology_classes = valid_classes
        logging.info(f"Validated ontology classes: {len(self.ontology_classes)} valid entries")

        # Clean ontology relationships
        valid_relationships = []
        for rel in self.ontology_relationships:
            if isinstance(rel, dict) and 'id' in rel and 'label' in rel:
                valid_relationships.append(rel)
            else:
                logging.warning(f"Removing invalid ontology relationship entry: {rel}")

        self.ontology_relationships = valid_relationships
        logging.info(f"Validated ontology relationships: {len(self.ontology_relationships)} valid entries")

    def _build_schema_card(self) -> dict:
        """Build a versioned snapshot of the ontology for this KG build.

        Stored on the Document node so future queries can detect ontology drift.
        Includes full property signatures, domain/range, cardinalities, and
        attribute schemas when a typed OntologySchema is available.
        """
        classes = [c.get('id', '') for c in self.ontology_classes if isinstance(c, dict)]
        rels    = [r.get('id', '') for r in self.ontology_relationships if isinstance(r, dict)]

        ontology_file_hash = None
        if self.ontology_path and os.path.exists(self.ontology_path):
            try:
                with open(self.ontology_path, 'rb') as f:
                    ontology_file_hash = hashlib.sha256(f.read()).hexdigest()
            except OSError:
                pass

        card: dict = {
            "ontologyFileHash": ontology_file_hash or "",
            "ontologyPath":     os.path.basename(self.ontology_path) if self.ontology_path else "",
            "sourceFormat":     (self._ontology_schema.source_format if self._ontology_schema else "unknown"),
            "classes":          sorted(classes),
            "relationships":    sorted(rels),
            "classCount":       len(classes),
            "relationshipCount": len(rels),
            "builtAt":          datetime.now().isoformat(),
        }

        # Enrich with typed property signatures and domain/range when available
        schema = self._ontology_schema
        if schema:
            card["entityTypes"] = [
                {
                    "id": et.id,
                    "label": et.label,
                    "description": et.description,
                    "properties": [
                        {
                            "name": p.name, "type": p.type.value,
                            "identifier": p.identifier, "required": p.required,
                            "enum_values": p.enum_values, "unit": p.unit,
                        }
                        for p in et.properties
                    ],
                }
                for et in schema.entity_types
            ]
            card["relationshipTypes"] = [
                {
                    "id": rt.id, "label": rt.label, "description": rt.description,
                    "domain": rt.domain, "range": rt.range, "cardinality": rt.cardinality,
                    "attributes": [
                        {"name": a.name, "type": a.type.value, "unit": a.unit}
                        for a in rt.attributes
                    ],
                }
                for rt in schema.relationship_types
            ]

        fingerprint_payload = {
            "sourceFormat": card.get("sourceFormat", "unknown"),
            "classes": card.get("classes", []),
            "relationships": card.get("relationships", []),
            "entityTypes": card.get("entityTypes", []),
            "relationshipTypes": card.get("relationshipTypes", []),
        }
        fingerprint_str = json.dumps(
            fingerprint_payload,
            sort_keys=True,
            ensure_ascii=False,
        )
        schema_hash = hashlib.sha256(fingerprint_str.encode("utf-8")).hexdigest()
        card["schemaVersion"] = schema_hash[:16]
        card["schemaHash"] = schema_hash

        return card

    def _create_neo4j_connection(self):
        """Create Neo4j graph connection"""
        # Ensure password is not None - use environment variable as fallback
        password = self.neo4j_password
        if password is None or password == "":
            password = os.getenv("NEO4J_PASSWORD", "password")

        # Set environment variables to ensure LangChain Neo4jGraph can read them
        os.environ["NEO4J_URI"] = self.neo4j_uri
        os.environ["NEO4J_USERNAME"] = self.neo4j_user
        os.environ["NEO4J_PASSWORD"] = password
        os.environ["NEO4J_DATABASE"] = self.neo4j_database

        return Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=password,
            database=self.neo4j_database,
            refresh_schema=False,
            sanitize=True
        )

    # ------------------------------------------------------------------
    # Context enrichment helpers (section headers, qualifier sentences,
    # and cross-chunk coreference resolution)
    # ------------------------------------------------------------------

    # Patterns that mark standard biomedical paper sections
    _SECTION_HEADER_RE = re.compile(
        r"(?m)^(?:"
        r"\d+[\.\d]*\s*"                              # optional numbering: "2.", "2.1."
        r")?"
        r"(?P<header>"
        r"abstract|introduction|background|methods?|materials?\s+and\s+methods?|"
        r"experimental\s+(?:procedures?|design)|study\s+design|"
        r"results?(?:\s+and\s+discussion)?|"
        r"discussion|conclusions?|summary|"
        r"statistical\s+analysis|data\s+analysis|"
        r"supplementary|acknowledgements?|references?"
        r")"
        r"[:\s]*$",
        re.IGNORECASE,
    )

    # Keywords that mark qualifier-bearing sentences
    _QUALIFIER_KEYWORDS = re.compile(
        r"\b(?:condition|experiment|treat(?:ed|ment)|knockout|knock[- ]?out|"
        r"mutant|patient|cohort|model|culture|in\s+vitro|in\s+vivo|"
        r"hypox|normox|baseline|control|express(?:ed|ion)|stimulat|"
        r"inhibit|activat|induc|depleted|overexpress|transfect|"
        r"under\s+these|such\s+conditions?|this\s+(?:model|system|context|protocol)|"
        r"these\s+(?:cells?|conditions?|animals?|patients?|mice|rats?))\b",
        re.IGNORECASE,
    )

    # Demonstrative coreference markers that indicate cross-chunk references
    _COREF_MARKERS = re.compile(
        r"\b(?:"
        r"these\s+(?:conditions?|cells?|animals?|mice|rats?|patients?|results?|findings?|data)|"
        r"this\s+(?:model|system|treatment|context|approach|protocol|setup|condition)|"
        r"such\s+conditions?|"
        r"under\s+these\s+(?:conditions?|circumstances?)|"
        r"the\s+(?:treated|knockout|mutant|control)\s+(?:group|cells?|mice|animals?)|"
        r"the\s+above[-\s](?:mentioned\s+)?(?:conditions?|treatment|model|protocol)|"
        r"as\s+(?:described|mentioned|stated)\s+(?:above|previously|earlier)"
        r")\b",
        re.IGNORECASE,
    )

    def _detect_section_headers(self, text: str) -> List[Tuple[int, str]]:
        """Return list of (char_position, normalised_section_name) for every section
        header found in *text*, in document order.

        e.g. [(0, 'Abstract'), (412, 'Introduction'), (2105, 'Methods'), ...]
        """
        headers: List[Tuple[int, str]] = []
        for m in self._SECTION_HEADER_RE.finditer(text):
            raw = m.group("header").strip()
            # Normalise to title case; collapse "materials and methods" variants
            if re.match(r"materials?\s+and\s+methods?", raw, re.I):
                normalised = "Methods"
            elif re.match(r"experimental\s+(?:procedures?|design)|study\s+design", raw, re.I):
                normalised = "Methods"
            elif re.match(r"results?\s+and\s+discussion", raw, re.I):
                normalised = "Results"
            else:
                normalised = raw.title()
            headers.append((m.start(), normalised))
        return headers

    def _get_section_for_position(
        self, pos: int, section_headers: List[Tuple[int, str]]
    ) -> Optional[str]:
        """Return the section name that covers character position *pos*."""
        current = None
        for header_pos, name in section_headers:
            if header_pos <= pos:
                current = name
            else:
                break
        return current

    def _extract_qualifier_sentences(self, text: str, max_sentences: int = 4) -> str:
        """Return up to *max_sentences* sentences from *text* that contain
        qualifier / experimental-context keywords.  Used to build the
        "context from previous chunk" header.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        selected = [s.strip() for s in sentences if self._QUALIFIER_KEYWORDS.search(s)]
        # Prefer sentences near the end of the chunk (most recent context)
        selected = selected[-max_sentences:]
        return " ".join(selected)

    def _has_coreference_markers(self, text: str) -> bool:
        """Return True if *text* contains demonstrative coreference markers."""
        return bool(self._COREF_MARKERS.search(text))

    def _resolve_coreferences_llm(
        self,
        chunk_text: str,
        context_text: str,
        llm,
        model_name: str,
    ) -> str:
        """Use the LLM to resolve demonstrative coreferences in *chunk_text*
        using *context_text* (content from previous chunks) as the lookup source.

        Returns a rewritten version of *chunk_text* with all resolvable references
        replaced by their full referents.  Falls back to the original text on any
        error.

        This is the cross-chunk qualifier coreference step: phrases like
        "these conditions", "this model", "the treated cells" are replaced with
        the specific experimental setup described in *context_text*, so that the
        downstream extraction LLM can attach the correct qualifier to each claim.
        """
        if not context_text.strip():
            return chunk_text

        coref_prompt = f"""You are resolving ambiguous references in a biomedical research text.

CONTEXT (from earlier in the paper — use this to identify what demonstrative phrases refer to):
{context_text}

CURRENT PASSAGE (rewrite this passage, replacing every ambiguous demonstrative reference with its full referent from the CONTEXT):
{chunk_text}

Rules:
- Replace "these conditions" / "this model" / "the treated cells" / "such circumstances" etc. with their specific referents from the CONTEXT.
- If a reference cannot be resolved from the CONTEXT, leave it unchanged.
- Do NOT change any scientific claims, entity names, or factual content.
- Return ONLY the rewritten passage text, nothing else."""

        try:
            resolved = llm.generate(coref_prompt, "", model_name)
            resolved = resolved.strip()
            if len(resolved) < 20 or len(resolved) > len(chunk_text) * 3:
                # Sanity check: resolved text must be plausible
                return chunk_text
            return resolved
        except Exception as e:
            logging.warning("Coreference resolution failed, using original chunk: %s", e)
            return chunk_text

    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text using RecursiveCharacterTextSplitter with tiktoken encoding.

        Delegates to ontographrag.kg.chunking.chunk_text so the logic is
        testable in isolation without instantiating the full creator.
        """
        return _chunk_text_fn(
            text=text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embedding_fn=self.embedding_function.embed_query,
        )

    def _build_retrieval_subchunks(
        self,
        chunk: Dict[str, Any],
        *,
        parent_chunk_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Build smaller retrieval-only spans from a larger KG extraction chunk.

        KG extraction keeps broad context, but vector retrieval should operate on
        spans sized for the embedding model. These RetrievalChunk nodes map back
        to their parent Chunk so graph provenance stays intact.
        """
        chunk_text = str(chunk.get("text") or "")
        if not chunk_text.strip():
            return []

        retrieval_chunk_size = max(64, int(getattr(self, "retrieval_chunk_size", 256) or 256))
        retrieval_chunk_overlap = max(
            0,
            min(
                int(getattr(self, "retrieval_chunk_overlap", 64) or 64),
                retrieval_chunk_size - 1,
            ),
        )

        raw_subchunks = _chunk_text_fn(
            text=chunk_text,
            chunk_size=retrieval_chunk_size,
            chunk_overlap=retrieval_chunk_overlap,
            embedding_fn=self.embedding_function.embed_query,
        )
        retrieval_subchunks: List[Dict[str, Any]] = []
        parent_start = int(chunk.get("start_pos") or 0)
        for subchunk in raw_subchunks:
            retrieval_subchunks.append({
                "id": hashlib.sha1(
                    f"{parent_chunk_id}:{subchunk['position']}:{subchunk['text']}".encode()
                ).hexdigest(),
                "text": subchunk["text"],
                "embedding": subchunk.get("embedding"),
                "retrieval_local_index": subchunk["position"],
                "start_pos": parent_start + int(subchunk.get("start_pos") or 0),
                "end_pos": parent_start + int(subchunk.get("end_pos") or 0),
                "position": int(chunk.get("position") or 0),
                "source": chunk.get("source"),
                "dataset": chunk.get("dataset"),
                "question_id": chunk.get("question_id"),
                "passage_index": chunk.get("passage_index"),
                "chunk_local_index": chunk.get("chunk_local_index", chunk.get("chunk_id", 0)),
                "parent_chunk_id": parent_chunk_id,
            })
        return retrieval_subchunks

    # ------------------------------------------------------------------
    # Schema-aware prompt helpers (step 4)
    # ------------------------------------------------------------------

    def _build_ontology_prompt_section(self, chunk_text: str, max_entity_types: int = 30, max_rel_types: int = 25) -> str:
        """Build the ontology section of the extraction prompt.

        When a typed OntologySchema is available, emits:
          - entity types with description + typed property list
          - relationship types with description + domain/range/cardinality + attributes

        Only includes entity types whose label appears in the chunk text (relevance
        filter), plus a tail of up to (max_entity_types - n_relevant) additional types
        so low-frequency schema types aren't silently ignored.

        Falls back to the flat list format when _ontology_schema is not available.
        """
        schema = self._ontology_schema

        # ---- entity types ----
        if schema and schema.entity_types:
            chunk_lower = chunk_text.lower()
            relevant, rest = [], []
            for et in schema.entity_types:
                if et.label.lower() in chunk_lower or et.id.lower() in chunk_lower:
                    relevant.append(et)
                else:
                    rest.append(et)
            selected_et = relevant + rest[:max(0, max_entity_types - len(relevant))]

            et_lines = []
            for et in selected_et:
                desc = f" — {et.description}" if et.description else ""
                line = f"  {et.id}{desc}"
                if et.properties:
                    prop_parts = []
                    for p in et.properties:
                        pt = p.type.value
                        extra = ""
                        if p.type == PropertyType.ENUM and p.enum_values:
                            extra = f" [{', '.join(p.enum_values)}]"
                        if p.unit:
                            extra += f" ({p.unit})"
                        flag = " [identifier]" if p.identifier else (" [required]" if p.required else "")
                        prop_parts.append(f"{p.name}:{pt}{extra}{flag}")
                    line += f"\n      properties: {', '.join(prop_parts)}"
                et_lines.append(line)
            entity_section = "ENTITY TYPES (id — description; properties with types):\n" + "\n".join(et_lines)
        else:
            lines = [
                f"  {c['id']} — {c.get('description') or c.get('label', '')}"
                for c in self.ontology_classes[:max_entity_types]
                if isinstance(c, dict)
            ]
            entity_section = "ENTITY TYPES:\n" + "\n".join(lines)

        # ---- relationship types ----
        if schema and schema.relationship_types:
            rel_lines = []
            for rt in schema.relationship_types[:max_rel_types]:
                desc = f" — {rt.description}" if rt.description else ""
                dom_rng = ""
                if rt.domain or rt.range:
                    dom_rng = f" ({rt.domain or '?'} → {rt.range or '?'})"
                    if rt.cardinality:
                        dom_rng += f" [{rt.cardinality}]"
                line = f"  {rt.id}{desc}{dom_rng}"
                if rt.attributes:
                    attr_parts = [
                        f"{a.name}:{a.type.value}" + (f" ({a.unit})" if a.unit else "")
                        for a in rt.attributes
                    ]
                    line += f"\n      attributes: {', '.join(attr_parts)}"
                rel_lines.append(line)
            rel_section = "RELATIONSHIP TYPES (id — description; domain→range [cardinality]):\n" + "\n".join(rel_lines)
        else:
            lines = [
                f"  {r['id']} — {r.get('description') or r.get('label', '')}"
                + (f" ({r.get('domain', '')} → {r.get('range', '')})" if r.get('domain') or r.get('range') else "")
                for r in self.ontology_relationships[:max_rel_types]
                if isinstance(r, dict)
            ]
            rel_section = "RELATIONSHIP TYPES:\n" + "\n".join(lines)

        return entity_section + "\n\n" + rel_section

    def _extract_entities_and_relationships_with_llm(
        self,
        chunk_text: str,
        llm,
        model_name: str = "openai/gpt-oss-120b:free",
        context_header: str = None,
        section_header: str = None,
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships using LLM with ontology guidance (if ontology
        available) or natural LLM detection.

        Args:
            chunk_text:      The text chunk to extract from.
            llm:             LLM provider instance.
            model_name:      Model identifier.
            context_header:  Qualifier sentences from the previous chunk, injected
                             before the main text so the LLM can resolve cross-chunk
                             experimental conditions.
            section_header:  Paper section the chunk belongs to (e.g. "Methods",
                             "Results").  Helps the LLM interpret claims correctly.
        """
        if llm is None:
            logging.warning("_extract_entities_and_relationships_with_llm called with llm=None; returning empty.")
            return {"entities": [], "relationships": []}

        # Build context preamble from section header + qualifier context.
        # This is injected before the main chunk text so the LLM can resolve
        # cross-chunk experimental conditions and interpret claims correctly.
        context_preamble = ""
        if section_header:
            context_preamble += (
                f"[DOCUMENT SECTION: {section_header}]\n"
                f"Interpret claims in the context of a {section_header} section "
                f"(e.g. experimental setups in Methods, findings reported in Results).\n\n"
            )
        if context_header:
            context_preamble += (
                f"[QUALIFIER CONTEXT FROM PREVIOUS CHUNK]\n"
                f"The following experimental conditions were established earlier in the paper. "
                f"Use them to resolve any ambiguous references (e.g. 'these conditions', "
                f"'this model') in the text below and attach them as qualifiers where relevant:\n"
                f"{context_header}\n\n"
            )

        # ------------------------------------------------------------------
        # Build the ontology section of the system prompt.
        # When a typed OntologySchema is available we emit schema-aware text
        # (entity descriptions + allowed properties, relationship domain/range/
        # cardinality/attributes).  We limit to the most relevant subset so the
        # prompt stays compact: entity types whose label appears in the chunk
        # text are always included; the rest are included up to a cap.
        # ------------------------------------------------------------------
        has_ontology = bool(self.ontology_classes) or bool(self.ontology_relationships)

        if has_ontology:
            try:
                ontology_section = self._build_ontology_prompt_section(chunk_text)
            except Exception as _oe:
                logging.warning("Failed to build schema-aware ontology prompt section: %s", _oe)
                ontology_section = (
                    "ENTITY TYPES:\n"
                    + "\n".join(f"- {c['label']} ({c['id']})" for c in self.ontology_classes[:100] if isinstance(c, dict))
                    + "\n\nRELATIONSHIP TYPES:\n"
                    + "\n".join(f"- {r['label']} ({r['id']})" for r in self.ontology_relationships[:50] if isinstance(r, dict))
                )

            system_message = f"""
You are an expert ontology-guided knowledge graph extraction system.
Extract entities and relationships from text using the schema below.

{ontology_section}

INSTRUCTIONS:
1. Extract all named entities; classify each using the entity types above ("Entity" if no type fits)
2. For each entity, populate the typed properties defined in the schema (dates, quantities, enums, IDs)
3. Create relationships ONLY between entities that actually interact in the text; use schema relationship types when they match
4. Prefer relationship types whose domain/range match the source/target entity types
5. Include specific named entities and technical terms — these are often the answer to downstream questions
6. Ignore only pure function words, filler phrases, and generic pronouns

Return ONLY a valid JSON object.
IMPORTANT: Output "relationships" FIRST, then "entities".
{{
  "relationships": [
    {{
      "source": "source_entity_id",
      "target": "target_entity_id",
      "type": "RELATIONSHIP_TYPE",
      "negated": false,
      "properties": {{
        "description": "how they are related in the text",
        "condition": "condition constraining this claim, or null",
        "quantitative": "numerical finding attached to this relationship, or null",
        "confidence": "demonstrated|suggested|hypothesized"
      }}
    }}
  ],
  "entities": [
    {{
      "id": "exact_entity_name_from_text",
      "type": "EntityType",
      "properties": {{
        "name": "exact_entity_name_from_text",
        "description": "brief grounded description"
      }}
    }}
  ]
}}

NEGATION RULE: If the text states a relationship does NOT hold, set "negated": true and keep the positive form of the type.
QUALIFIER RULE: Conditions (e.g. "in ALS patients") go in "condition"; numerical findings (e.g. "3-fold increase") go in "quantitative".

{context_preamble}TEXT TO ANALYZE:
{chunk_text}

IMPORTANT: Return ONLY the JSON object, no additional text."""
        else:
            system_message = f"""
You are an expert knowledge graph extraction system.
Extract entities and relationships from text naturally and comprehensively.

INSTRUCTIONS:
1. Extract ALL significant entities and concepts
2. Create relationships between any meaningfully related entities
3. Use descriptive relationship types that capture how entities interact
4. Be comprehensive; include both technical and non-technical concepts

Return ONLY a valid JSON object.
IMPORTANT: Output "relationships" FIRST, then "entities".
{{
  "relationships": [
    {{
      "source": "source_entity_id",
      "target": "target_entity_id",
      "type": "RELATIONSHIP_TYPE",
      "negated": false,
      "properties": {{
        "description": "how they are related in the text",
        "condition": "condition constraining this claim, or null",
        "quantitative": "any numerical finding, or null",
        "confidence": "demonstrated|suggested|hypothesized"
      }}
    }}
  ],
  "entities": [
    {{
      "id": "exact_entity_name_from_text",
      "type": "EntityType",
      "properties": {{
        "name": "exact_entity_name_from_text",
        "description": "contextual description"
      }}
    }}
  ]
}}

NEGATION RULE: If the text states a relationship does NOT hold, set "negated": true and keep the positive form.
QUALIFIER RULE: Conditions go in "condition"; numerical findings go in "quantitative".

{context_preamble}TEXT TO ANALYZE:
{chunk_text}

IMPORTANT: Return ONLY the JSON object, no additional text."""

        try:

            # Get the response directly from LLM provider with timeout handling
            try:
                response = llm.generate(system_message, "", model_name)
            except Exception as timeout_error:
                if "timeout" in str(timeout_error).lower() or "read operation timed out" in str(timeout_error).lower():
                    logging.warning(f"LLM request timed out for chunk. Returning empty result to continue processing.")
                    return {'entities': [], 'relationships': []}
                else:
                    raise timeout_error

            # Debug: Log the raw response
            logging.info(f"Raw LLM response length: {len(response)}")
            logging.info(f"Raw LLM response preview: {response[:200]}...")

            # Clean the response to extract JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            # Robust JSON extraction with multiple fallback strategies
            json_content = ""

            # Strategy 1: Try to extract JSON from response with improved parsing
            json_start = response.find('{')
            if json_start == -1:
                logging.warning(f"No JSON start found in response: {response[:200]}... returning empty.")
                return {"entities": [], "relationships": []}

            # Try to find the complete JSON object by looking for balanced braces
            brace_count = 0
            json_end = json_start
            in_string = False
            escape_next = False

            for i, char in enumerate(response[json_start:], json_start):
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found the matching closing brace
                            json_end = i + 1
                            break

            if brace_count == 0 and json_end > json_start:
                json_content = response[json_start:json_end]
            else:
                logging.warning(f"Incomplete JSON structure (brace_count={brace_count}). Trying alternative extraction.")
                # Strategy 2: Try simple { } extraction
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = response[json_start:json_end]
                else:
                    logging.warning("Could not extract JSON with any strategy; returning empty.")
                    return {"entities": [], "relationships": []}

            # Parse the JSON content
            try:
                result = json.loads(json_content)
            except json.JSONDecodeError as e:
                logging.warning(f"JSON parsing error: {e}. Attempting partial extraction from: {json_content[:300]}...")
                result = self._partial_json_extract(json_content)

            # Validate the result has the expected structure
            if not isinstance(result, dict) or ('entities' not in result and 'relationships' not in result):
                logging.warning(f"Invalid response structure. Returning empty result.")
                return {'entities': [], 'relationships': []}
            result.setdefault('entities', [])
            result.setdefault('relationships', [])

            # Filter out non-medical entities - with better error handling
            medical_entities = []
            entities_raw = result.get('entities', [])

            # Check if entities is the wrong type (LLM returned malformed JSON)
            if not isinstance(entities_raw, list):
                logging.warning(f"LLM returned entities as {type(entities_raw)} instead of list: {entities_raw}. Skipping chunk.")
                return {'entities': [], 'relationships': []}

            for entity in entities_raw:
                # Handle both dict and string formats from LLM response
                if isinstance(entity, str):
                    # Convert string entity to dict format
                    try:
                        entity_type = self._classify_entity_with_ontology(entity)
                        entity = {
                            "id": entity,
                            "type": entity_type,
                            "properties": {
                                "name": entity,
                                "description": f"{entity_type}: {entity}"
                            }
                        }
                    except Exception as e:
                        logging.warning(f"Error processing string entity '{entity}': {e}. Skipping.")
                        continue
                elif isinstance(entity, dict):
                    # Ensure required fields exist
                    if 'id' not in entity:
                        logging.warning(f"Entity missing 'id' field: {entity}. Skipping.")
                        continue
                    # Ensure id is a string
                    if not isinstance(entity['id'], str):
                        logging.warning(f"Entity id is not a string: {type(entity['id'])} = {entity['id']}. Converting to string.")
                        entity['id'] = str(entity['id'])

                    try:
                        if 'type' not in entity:
                            entity['type'] = self._classify_entity_with_ontology(entity['id'])
                        if 'properties' not in entity:
                            entity['properties'] = {
                                "name": entity['id'],
                                "description": f"{entity['type']}: {entity['id']}"
                            }
                    except Exception as e:
                        logging.warning(f"Error processing dict entity {entity}: {e}. Skipping.")
                        continue
                else:
                    logging.warning(f"Entity is neither string nor dict: {type(entity)} = {entity}. Skipping.")
                    continue

                medical_entities.append(entity)

            # Filter relationships — source/target must resolve to known entities.
            # The LLM often abbreviates endpoint names in the relationship list,
            # so we allow safe fuzzy resolution rather than exact-only matching.
            medical_relationships = []
            relationship_drop_count = 0
            relationship_fuzzy_resolutions = 0

            relationships_raw = result.get('relationships', [])

            # Check if relationships is the wrong type (LLM returned malformed JSON)
            if not isinstance(relationships_raw, list):
                logging.warning(f"LLM returned relationships as {type(relationships_raw)} instead of list: {relationships_raw}. Skipping relationships.")
                medical_relationships = []
            else:
                # Process relationships, handling both dict and string formats
                for rel in relationships_raw:
                    if isinstance(rel, str):
                        logging.warning(f"Relationship is string: {rel}. Skipping.")
                        continue
                    elif isinstance(rel, dict):
                        raw_src = rel.get('source', '')
                        raw_tgt = rel.get('target', '')
                        if not isinstance(raw_src, str) or not isinstance(raw_tgt, str):
                            logging.warning(f"Relationship source/target not strings: {rel}. Skipping.")
                            continue
                        canonical_src = self._resolve_relationship_endpoint(raw_src, medical_entities)
                        canonical_tgt = self._resolve_relationship_endpoint(raw_tgt, medical_entities)
                        if canonical_src and canonical_tgt:
                            rel_copy = dict(rel)
                            rel_copy['source'] = canonical_src
                            rel_copy['target'] = canonical_tgt
                            rel_props = dict(rel_copy.get('properties') or {})
                            rel_props.setdefault('source_name', raw_src)
                            rel_props.setdefault('target_name', raw_tgt)
                            rel_copy['properties'] = rel_props
                            if canonical_src != raw_src or canonical_tgt != raw_tgt:
                                relationship_fuzzy_resolutions += 1
                            medical_relationships.append(rel_copy)
                        else:
                            relationship_drop_count += 1
                            logging.warning(f"Invalid relationship (source/target not in entities): {rel}. Skipping.")
                    else:
                        logging.warning(f"Relationship is neither string nor dict: {type(rel)} = {rel}. Skipping.")
                        continue

            if relationships_raw:
                logging.info(
                    "Relationship endpoint resolution kept=%d dropped=%d fuzzy_resolved=%d",
                    len(medical_relationships),
                    relationship_drop_count,
                    relationship_fuzzy_resolutions,
                )

            # ── GraphRAG-style self-reflection pass ──────────────────────────
            # Ask the LLM to review the chunk against its own output and fill gaps.
            # This catches entities the first pass missed (e.g. biological processes
            # that were present in text but not extracted).
            try:
                existing_entity_names = [e['id'] for e in medical_entities]
                reflection_prompt = f"""You are a quality-control agent for a knowledge graph extraction pipeline.

ORIGINAL TEXT:
{chunk_text}

ALREADY-EXTRACTED ENTITIES:
{json.dumps(existing_entity_names, indent=2)}

TASK: Read the original text carefully. Identify any named entities (people, organizations, locations, works, events, concepts, domain-specific terms, quantitative measurements) that are clearly present in the text but MISSING from the already-extracted list above.

Return ONLY a JSON object with the NEW missing entities (do not repeat already-extracted ones):
{{
  "new_entities": [
    {{
      "id": "exact_name_from_text",
      "type": "EntityType",
      "properties": {{
        "name": "exact_name_from_text",
        "description": "brief description"
      }}
    }}
  ]
}}

If there are no missing entities, return: {{"new_entities": []}}
Return ONLY the JSON object."""

                reflection_response = llm.generate(reflection_prompt, "", model_name)
                reflection_response = reflection_response.strip()
                # Strip markdown code fences if present
                if reflection_response.startswith('```'):
                    reflection_response = re.sub(r'^```[a-z]*\n?', '', reflection_response)
                    reflection_response = re.sub(r'\n?```$', '', reflection_response.rstrip())
                r_start = reflection_response.find('{')
                r_end = reflection_response.rfind('}') + 1
                if r_start >= 0 and r_end > r_start:
                    reflection_data = json.loads(reflection_response[r_start:r_end])
                    new_entities_raw = reflection_data.get('new_entities', [])
                    existing_ids_lower = {e['id'].lower() for e in medical_entities}
                    added = 0
                    added_entity_ids = []
                    for entity in new_entities_raw:
                        if not isinstance(entity, dict) or 'id' not in entity:
                            continue
                        if not isinstance(entity['id'], str):
                            entity['id'] = str(entity['id'])
                        if entity['id'].lower() in existing_ids_lower:
                            continue
                        entity.setdefault('type', self._classify_entity_with_ontology(entity['id']))
                        entity.setdefault('properties', {'name': entity['id'], 'description': entity['type']})
                        medical_entities.append(entity)
                        existing_ids_lower.add(entity['id'].lower())
                        added_entity_ids.append(entity['id'])
                        added += 1
                    if added:
                        logging.info(f"Self-reflection added {added} new entities")

                    # ── Relational reconciliation for new reflection entities ──
                    # New entities from reflection are not yet connected to the graph.
                    # Run a targeted relationship extraction pass to find any relationships
                    # between the newly added entities and the rest of the known entities.
                    if added_entity_ids:
                        all_known_ids = [e['id'] for e in medical_entities]
                        recon_prompt = f"""You are extracting relationships for a knowledge graph.

TEXT:
{chunk_text}

NEW ENTITIES (just discovered — need relationships):
{json.dumps(added_entity_ids, indent=2)}

ALL KNOWN ENTITIES IN THIS CHUNK:
{json.dumps(all_known_ids, indent=2)}

TASK: Find ALL relationships in the text that involve at least one NEW ENTITY and any other known entity.

Return ONLY a JSON object:
{{
  "relationships": [
    {{
      "source": "source_entity_id",
      "target": "target_entity_id",
      "type": "RELATIONSHIP_TYPE",
      "negated": false,
      "properties": {{
        "description": "how they are related in the text",
        "condition": "condition constraining this claim, or null",
        "quantitative": "any numerical finding, or null",
        "confidence": "demonstrated|suggested|hypothesized"
      }}
    }}
  ]
}}

Rules:
- source and target MUST be entity ids from the ALL KNOWN ENTITIES list above
- At least one of source or target MUST be from the NEW ENTITIES list
- If no relationships found, return {{"relationships": []}}
- Return ONLY the JSON object."""

                        try:
                            recon_response = llm.generate(recon_prompt, "", model_name).strip()
                            if recon_response.startswith('```'):
                                recon_response = re.sub(r'^```[a-z]*\n?', '', recon_response)
                                recon_response = re.sub(r'\n?```$', '', recon_response.rstrip())
                            rc_start = recon_response.find('{')
                            rc_end = recon_response.rfind('}') + 1
                            if rc_start >= 0 and rc_end > rc_start:
                                recon_data = json.loads(recon_response[rc_start:rc_end])
                                new_rels = recon_data.get('relationships', [])
                                new_rels_added = 0
                                for rel in new_rels:
                                    if isinstance(rel, dict) and rel.get('source') and rel.get('target') and rel.get('type'):
                                        medical_relationships.append(rel)
                                        new_rels_added += 1
                                if new_rels_added:
                                    logging.info(f"Reflection reconciliation added {new_rels_added} relationships for new entities")
                        except Exception as recon_err:
                            logging.debug(f"Reflection reconciliation failed (non-fatal): {recon_err}")
                    # ── end relational reconciliation ────────────────────────

            except Exception as refl_err:
                logging.debug(f"Self-reflection pass failed (non-fatal): {refl_err}")
            # ── end self-reflection ──────────────────────────────────────────

            return {
                'entities': medical_entities,
                'relationships': medical_relationships
            }

        except Exception as e:
            logging.error(f"LLM extraction failed: {e}")
            return {"entities": [], "relationships": []}






    def merge_synonym_entities(self, graph, similarity_threshold: float = 0.82, kg_name: str = None) -> int:
        """
        HippoRAG-style synonym merging: cluster entity nodes that share nearly-identical
        embeddings (surface-form variants like "TBK1" / "TBK1 kinase" / "TBK1 protein")
        and merge them into a single canonical node.

        Uses the existing entity_vector index.  Merges the lower-degree node into the
        higher-degree one so that the richer node survives.

        Returns the number of merges performed.
        """
        try:
            import numpy as np

            # Fetch all entity nodes with their embeddings
            params = {"kg_name": kg_name} if kg_name else {}
            kg_filter = "AND e.kgName = $kg_name" if kg_name else ""
            fetch_q = f"""
            MATCH (e:__Entity__)
            WHERE e.embedding IS NOT NULL {kg_filter}
            RETURN elementId(e) AS eid, e.id AS name, e.embedding AS emb,
                   COUNT {{ (e)--() }} AS degree,
                   coalesce(e.type, e.ontology_class, '') AS etype
            """
            rows = graph.query(fetch_q, params)
            if not rows:
                logging.info("No entity embeddings found; skipping synonym merging.")
                return 0

            rows = [r for r in rows if r.get('emb') is not None]
            if not rows:
                logging.info("No entity embeddings found after filtering; skipping synonym merging.")
                return 0
            eids = [r['eid'] for r in rows]
            names = [r['name'] for r in rows]
            degrees = [r['degree'] for r in rows]
            etypes = [r.get('etype', '') or '' for r in rows]
            embeddings = np.array([r['emb'] for r in rows], dtype=float)

            # Generic types that should not block merging (the LLM falls back to these
            # when it cannot classify an entity, so two nodes might genuinely represent
            # the same concept even though one was labelled "Entity" and the other "Concept").
            _generic_types = {'entity', 'concept', 'unknown', 'other', ''}

            # Normalise for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            normed = embeddings / norms

            # Build equivalence clusters via union-find so that synonym chains
            # (A≈B, B≈C) are merged atomically rather than pairwise.
            # Pairwise sequential merges leave dangling references when later pairs
            # reference nodes already merged away.
            sim_matrix = normed @ normed.T
            n = len(eids)

            parent = list(range(n))

            def _find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]  # path compression
                    x = parent[x]
                return x

            def _union(x: int, y: int) -> None:
                px, py = _find(x), _find(y)
                if px != py:
                    parent[px] = py

            for i in range(n):
                for j in range(i + 1, n):
                    if sim_matrix[i, j] >= similarity_threshold:
                        ti, tj = etypes[i].lower(), etypes[j].lower()
                        if ti not in _generic_types and tj not in _generic_types and ti != tj:
                            continue
                        _union(i, j)

            # Group indices by cluster root
            from collections import defaultdict as _dd
            clusters: dict = _dd(list)
            for i in range(n):
                clusters[_find(i)].append(i)

            # Build one (canonical, [duplicates]) tuple per multi-node cluster.
            # Canonical = highest-degree node in the cluster.
            merge_tasks = []
            for members in clusters.values():
                if len(members) < 2:
                    continue
                canonical_idx = max(members, key=lambda i: degrees[i])
                dups = [i for i in members if i != canonical_idx]
                merge_tasks.append((canonical_idx, dups))

            if not merge_tasks:
                logging.info("No synonym clusters found above threshold %.2f", similarity_threshold)
                return 0

            merged = 0
            for canonical_idx, dup_indices in merge_tasks:
                canonical_name = names[canonical_idx]
                for dup_idx in dup_indices:
                    duplicate_name = names[dup_idx]
                    try:
                        # Stamp synonym alias on canonical BEFORE merge (dup ceases to exist after).
                        # Then use apoc.refactor.mergeNodes to atomically rewire all relationships
                        # and delete the duplicate.  {properties: 'discard'} means the first
                        # node in the list (canonical) wins all property conflicts.
                        merge_q = """
                        MATCH (can:__Entity__ {id: $can_id})
                        MATCH (dup:__Entity__ {id: $dup_id})
                        SET can.synonyms = coalesce(can.synonyms, []) + [dup.id]
                        WITH [can, dup] AS nodes
                        CALL apoc.refactor.mergeNodes(nodes, {properties: 'discard', mergeRels: true})
                        YIELD node
                        RETURN node.id AS merged_id
                        """
                        graph.query(merge_q, {"dup_id": duplicate_name, "can_id": canonical_name})
                        logging.info("Merged synonym '%s' → '%s'", duplicate_name, canonical_name)
                        merged += 1
                    except Exception as merge_err:
                        logging.warning("Failed to merge '%s' → '%s': %s",
                                        duplicate_name, canonical_name, merge_err)

            logging.info(f"Synonym merging complete: {merged}/{len(merge_tasks)} pairs merged")
            return merged

        except Exception as e:
            logging.error(f"Synonym merging failed: {e}")
            return 0

    def _entities_can_be_related(self, source_type: str, target_type: str) -> bool:
        """
        Check if two entity types can be meaningfully related
        """
        # Define which entity types can be related
        medical_relationships = {
            'Disease': ['Treatment', 'Symptom', 'Drug', 'MedicalProcedure', 'Patient'],
            'Treatment': ['Disease', 'Drug', 'MedicalProcedure', 'Patient', 'Physician'],
            'Drug': ['Disease', 'Treatment', 'MedicalProcedure', 'Patient'],
            'Symptom': ['Disease', 'Patient', 'MedicalProcedure'],
            'Patient': ['Disease', 'Treatment', 'Drug', 'Physician', 'Hospital', 'MedicalProcedure'],
            'Physician': ['Patient', 'Treatment', 'MedicalProcedure', 'Hospital'],
            'Hospital': ['Patient', 'Physician', 'MedicalProcedure'],
            'MedicalProcedure': ['Disease', 'Treatment', 'Patient', 'Physician', 'Hospital'],
            'MedicalDevice': ['MedicalProcedure', 'Treatment'],
            'Anatomy': ['Disease', 'MedicalProcedure']
        }

        return target_type in medical_relationships.get(source_type, [])

    def _partial_json_extract(self, text: str) -> Dict[str, Any]:
        """
        Best-effort recovery when json.loads() fails on an LLM response.

        Tries three strategies in order:
        1. Extract the 'entities' array via regex and parse it independently.
        2. Extract the 'relationships' array via regex and parse it independently.
        3. Return empty dict (caller will fall back to empty entities/relationships).

        This prevents silently discarding an entire chunk just because a trailing
        comma or stray character made the top-level JSON invalid.
        """
        def _extract_array(key: str, src: str):
            """Regex-extract the JSON array for a given key from a potentially broken JSON string."""
            pattern = rf'"{key}"\s*:\s*(\[.*?\])'
            match = re.search(pattern, src, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                # Try to clean common issues: trailing commas before ]
                cleaned = re.sub(r',\s*]', ']', match.group(1))
                cleaned = re.sub(r',\s*}', '}', cleaned)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    return None

        entities = _extract_array('entities', text)
        relationships = _extract_array('relationships', text)

        if entities is not None or relationships is not None:
            recovered = {
                'entities': entities if isinstance(entities, list) else [],
                'relationships': relationships if isinstance(relationships, list) else [],
            }
            logging.info(
                f"Partial JSON recovery succeeded: {len(recovered['entities'])} entities, "
                f"{len(recovered['relationships'])} relationships"
            )
            return recovered

        logging.warning("Partial JSON recovery failed; returning empty extraction for this chunk.")
        return {'entities': [], 'relationships': []}

    def _classify_entity_with_ontology(self, entity_text: str) -> str:
        """
        Classify entity using ontology guidance.

        Strategy (in priority order):
        1. Embedding-based cosine similarity against pre-computed ontology class label embeddings
           (threshold ≥ 0.50 required to accept the match).
        2. Exact substring match against ontology class id/label (legacy fallback).
        3. Keyword heuristics (final fallback).
        """
        entity_lower = entity_text.lower()

        # Strategy 1: embedding similarity (preferred — robust to abbreviations / paraphrases)
        if self._ontology_class_embeddings and self.embedding_function:
            try:
                entity_emb = self.embedding_function.embed_query(entity_text)
                best_class, best_score = None, 0.0
                for cls_id, cls_emb in self._ontology_class_embeddings:
                    # Cosine similarity (both vectors are L2-normalised by the embedding model)
                    score = float(
                        sum(a * b for a, b in zip(entity_emb, cls_emb))
                        / (
                            (sum(a * a for a in entity_emb) ** 0.5 + 1e-9)
                            * (sum(b * b for b in cls_emb) ** 0.5 + 1e-9)
                        )
                    )
                    if score > best_score:
                        best_score, best_class = score, cls_id
                if best_class and best_score >= 0.50:
                    return best_class
            except Exception as e:
                logging.debug(f"Embedding classification failed for '{entity_text}': {e}")

        # Strategy 2: exact substring match against ontology class labels
        for cls in self.ontology_classes:
            if cls['id'].lower() in entity_lower or cls['label'].lower() in entity_lower:
                return cls['id']

        # Strategy 3: keyword heuristics
        if any(word in entity_lower for word in ['disease', 'cancer', 'tumor', 'syndrome', 'disorder', 'carcinoma', 'malignancy']):
            return 'Disease'
        elif any(word in entity_lower for word in ['drug', 'medication', 'treatment', 'therapy', 'surgery', 'chemotherapy', 'radiotherapy']):
            return 'Treatment'
        elif any(word in entity_lower for word in ['patient', 'person', 'individual', 'male', 'female']):
            return 'Patient'
        elif any(word in entity_lower for word in ['doctor', 'physician', 'surgeon', 'specialist', 'oncologist', 'urologist']):
            return 'Physician'
        elif any(word in entity_lower for word in ['hospital', 'clinic', 'center', 'institute', 'department']):
            return 'Hospital'
        elif any(word in entity_lower for word in ['symptom', 'sign', 'manifestation', 'pain', 'fever']):
            return 'Symptom'
        elif any(word in entity_lower for word in ['gene', 'mutation', 'protein', 'biomarker', 'receptor', 'marker']):
            return 'Biomarker'
        elif any(word in entity_lower for word in ['score', 'grade', 'stage', 'classification', 'risk']):
            return 'ClinicalFinding'
        else:
            return 'Concept'

    def _classify_relationship_with_ontology(self, source: str, target: str) -> str:
        """
        Classify relationship using ontology guidance
        """
        # ------------------------------------------------------------------
        # Schema-constrained relationship classification (step 5).
        # When OntologySchema is available, find relationship types whose
        # domain/range are compatible with the given entity types, then rank
        # by lexical similarity to the entity names as a tiebreaker.
        # Falls back to hardcoded heuristics only when no schema is loaded.
        # ------------------------------------------------------------------
        schema = self._ontology_schema
        if schema and schema.relationship_types:
            source_type = self._classify_entity_with_ontology(source)
            target_type = self._classify_entity_with_ontology(target)
            candidates = schema.compatible_relationships(source_type, target_type)
            if candidates:
                # Rank by lexical similarity of the relationship label to
                # the concatenated source+target text as a weak domain signal
                combined = (source + " " + target).lower()
                best = max(
                    candidates,
                    key=lambda rt: difflib.SequenceMatcher(
                        None, rt.label.lower(), combined
                    ).ratio(),
                )
                return best.id.replace(' ', '_').replace('-', '_').upper()

        # Heuristic fallback (no schema or no compatible candidates found)
        source_lower = source.lower()
        target_lower = target.lower()
        if any(w in source_lower for w in ['treatment', 'therapy', 'drug']) or \
           any(w in target_lower for w in ['treatment', 'therapy', 'drug']):
            return 'TREATS'
        if any(w in source_lower for w in ['disease', 'cancer']) and \
           any(w in target_lower for w in ['symptom', 'sign']):
            return 'HAS_SYMPTOM'
        if any(w in source_lower for w in ['physician', 'doctor']) and \
           any(w in target_lower for w in ['patient', 'person']):
            return 'DIAGNOSES'
        return 'RELATED_TO'

    def _canonicalize_relationship_type(
        self,
        raw_type: str,
        source_type: Optional[str] = None,
        target_type: Optional[str] = None,
    ) -> str:
        """Map a raw LLM-generated relationship type to the closest ontology relationship.

        Steps:
        1. Exact label/id match (case/space-insensitive) filtered to schema-compatible
           candidates when source_type and target_type are provided.
        2. Fuzzy match (SequenceMatcher ≥ 0.72) with schema-compatibility boost
           (+0.10) for fully-compatible domain/range matches.
        3. Sanitize raw type if it passes regex; else fall back to ASSOCIATED_WITH.
        """
        if not raw_type:
            return 'ASSOCIATED_WITH'

        normalized_raw = raw_type.lower().replace(' ', '_').replace('-', '_')

        schema = self._ontology_schema
        candidates = (
            schema.compatible_relationships(source_type, target_type)
            if schema and (source_type or target_type)
            else (schema.relationship_types if schema else [])
        )
        # Merge with legacy flat list for non-schema path
        ont_rels = self.ontology_relationships

        if schema and candidates:
            # Step 1: exact match within schema-compatible candidates first
            for rt in candidates:
                cand = rt.label.lower().replace(' ', '_')
                if cand == normalized_raw or rt.id.lower() == normalized_raw:
                    logging.debug("Exact schema-compatible rel match: '%s' → '%s'", raw_type, rt.id)
                    return rt.id.replace(' ', '_').replace('-', '_').upper()
            # Step 1b: exact match in full schema (less preferred)
            for rt in schema.relationship_types:
                cand = rt.label.lower().replace(' ', '_')
                if cand == normalized_raw or rt.id.lower() == normalized_raw:
                    return rt.id.replace(' ', '_').replace('-', '_').upper()

            # Step 2: fuzzy match with schema-compatibility boost
            compat_ids = {rt.id for rt in candidates}
            best_match, best_score = None, 0.0
            for rt in schema.relationship_types:
                cand_label = rt.label.lower().replace(' ', '_')
                score = max(
                    difflib.SequenceMatcher(None, normalized_raw, cand_label).ratio(),
                    difflib.SequenceMatcher(None, normalized_raw, rt.id.lower()).ratio(),
                )
                # Boost schema-compatible candidates
                if rt.id in compat_ids:
                    score += 0.10
                if score > best_score:
                    best_score, best_match = score, rt
            if best_match and best_score >= 0.72:
                logging.debug(
                    "Fuzzy schema rel: '%s' → '%s' (score=%.2f)", raw_type, best_match.id, best_score
                )
                return best_match.id.replace(' ', '_').replace('-', '_').upper()

        elif ont_rels:
            # Legacy flat-list path (no OntologySchema)
            for ont_rel in ont_rels:
                candidate = ont_rel['label'].lower().replace(' ', '_')
                if candidate == normalized_raw or ont_rel['id'].lower() == normalized_raw:
                    return ont_rel['id'].replace(' ', '_').replace('-', '_').upper()
            best_match, best_score = None, 0.0
            for ont_rel in ont_rels:
                candidate = ont_rel['label'].lower().replace(' ', '_')
                score = difflib.SequenceMatcher(None, normalized_raw, candidate).ratio()
                id_score = difflib.SequenceMatcher(None, normalized_raw, ont_rel['id'].lower()).ratio()
                best_s = max(score, id_score)
                if best_s > best_score:
                    best_score, best_match = best_s, ont_rel
            if best_match and best_score >= 0.72:
                logging.debug(
                    "Fuzzy rel (legacy): '%s' → '%s' (score=%.2f)", raw_type, best_match['id'], best_score
                )
                return best_match['id'].replace(' ', '_').replace('-', '_').upper()

        # Step 3: sanitize raw type
        sanitized = raw_type.strip().replace(' ', '_').replace('-', '_').upper()
        if len(sanitized) > 50 or not re.match(r'^[A-Z][A-Z0-9_]*$', sanitized):
            logging.debug("Rel type '%s' failed sanitization; using ASSOCIATED_WITH", raw_type)
            return 'ASSOCIATED_WITH'
        return sanitized

    def _extract_relationships_only(self, combined_text: str, known_entities: List[Dict], llm, model_name: str) -> List[Dict]:
        """
        Given a combined text (two adjacent chunks) and entities already known from those chunks,
        ask the LLM only for relationships between those entities — no new entity extraction.

        Returns a list of raw relationship dicts {source, target, type, properties}.
        """
        if llm is None or len(known_entities) < 2:
            return []

        entity_list = "\n".join(
            f"- {e['id']} (type: {e.get('type', 'Unknown')})"
            for e in known_entities[:60]  # cap to avoid over-long prompts
        )

        prompt = f"""You are a knowledge graph extraction expert.
Given the following text and list of known entities, identify ONLY relationships between these entities that are explicitly supported by the text.

KNOWN ENTITIES:
{entity_list}

TEXT:
{combined_text}

Return ONLY a JSON object with a "relationships" array (no "entities" key needed):
{{
  "relationships": [
    {{
      "source": "source_entity_id_from_list",
      "target": "target_entity_id_from_list",
      "type": "RELATIONSHIP_TYPE",
      "negated": false,
      "properties": {{
        "description": "how they relate in the text",
        "condition": "biological/experimental condition constraining this claim, or null",
        "quantitative": "any numerical finding attached to this relationship, or null",
        "confidence": "demonstrated|suggested|hypothesized"
      }}
    }}
  ]
}}

Rules:
- source and target MUST be entity ids from the KNOWN ENTITIES list above
- Only include relationships explicitly supported by the text
- Prefer specific relationship types (TREATS, CAUSES, INDICATES, etc.) over generic ones
- NEGATION RULE: set "negated": true if the text states the relationship does NOT hold
- QUALIFIER RULE: put experimental conditions in "condition", numerical findings in "quantitative"
- Return ONLY the JSON object, no other text"""

        try:
            response = llm.generate(prompt, "", model_name)
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start < 0 or json_end <= json_start:
                return []
            result = json.loads(response[json_start:json_end])
            return result.get('relationships', [])
        except Exception as e:
            logging.warning(f"Cross-chunk relationship extraction failed: {e}")
            return []

    def _generate_entity_id(self, entity: Dict[str, Any]) -> str:
        """Generate a deterministic UUID scoped to both normalized text AND type.

        Including the type means same-surface/different-type entities (e.g. "depression"
        as Disease vs GeologicalFeature) get distinct UUIDs after the harmonization split.
        For merged entities (LLM type-drift already resolved to one specific type by
        _harmonize_entities) the representative carries a single resolved type, so
        chunk-level variants still converge to the same UUID.
        """
        norm_text = self._normalize_entity_text(entity['id'])
        entity_type = (entity.get('type') or '').strip()
        unique_seed = f"{norm_text}:{entity_type}" if entity_type else norm_text
        return str(uuid.uuid5(uuid.NAMESPACE_OID, unique_seed))

    def _normalize_entity_text(self, text: str) -> str:
        """
        Normalize entity text for duplicate detection.

        Applies generic, domain-agnostic normalization:
        - lowercase + collapse whitespace
        - strip leading articles / conjunctions
        - remove punctuation that doesn't affect identity
        - collapse runs of underscores/spaces

        Domain-specific aliases are intentionally NOT hardcoded here.
        The LLM is expected to produce consistent names; UUID5-based
        deduplication in _harmonize_entities handles surface variants.
        """
        normalized = re.sub(r'\s+', ' ', text.lower().strip())

        # Strip leading articles and conjunctions
        normalized = re.sub(r'^(the |an |a |and |or )', '', normalized)

        # Remove punctuation that doesn't carry semantic weight
        normalized = re.sub(r'[,()\[\];:.]', '', normalized)

        # Collapse hyphens/slashes to underscores for stable keys
        normalized = re.sub(r'[\-/]', '_', normalized)

        # Final cleanup
        normalized = re.sub(r'\s+', '_', normalized.strip())
        normalized = re.sub(r'_+', '_', normalized)
        normalized = normalized.strip('_')

        return normalized

    def _entity_name_variants(self, name: str) -> List[str]:
        """Generate robust surface-form variants for entity resolution."""
        if not isinstance(name, str):
            return []
        base = name.strip()
        if not base:
            return []

        variants = {
            base,
            base.lower(),
            base.replace('_', ' '),
            base.replace(' ', '_'),
            self._normalize_entity_text(base),
        }
        return [v for v in variants if isinstance(v, str) and v.strip()]

    def _entity_candidate_names(self, entity: Dict[str, Any]) -> List[str]:
        """Collect plausible surface forms for an extracted entity."""
        if not isinstance(entity, dict):
            return []

        candidates: List[str] = []
        for candidate in (
            entity.get("id"),
            entity.get("name"),
            (entity.get("properties") or {}).get("name"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                candidates.append(candidate)

        all_names = (entity.get("properties") or {}).get("all_names") or []
        if isinstance(all_names, list):
            for alias in all_names:
                if isinstance(alias, str) and alias.strip():
                    candidates.append(alias)

        seen = set()
        deduped: List[str] = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                deduped.append(candidate)
        return deduped

    def _entity_appears_in_text(self, entity: Dict[str, Any], text: str) -> bool:
        """Check whether an entity or any of its aliases appears in text."""
        if not isinstance(text, str) or not text.strip():
            return False

        text_lower = text.lower()
        seen_variants = set()
        for candidate in self._entity_candidate_names(entity):
            for variant in self._entity_name_variants(candidate):
                normalized = variant.strip().lower()
                if len(normalized) < 3 or normalized in seen_variants:
                    continue
                seen_variants.add(normalized)
                prefix = r'(?<!\w)' if not normalized[:1].isalnum() and normalized[:1] != '_' else r'\b'
                suffix = r'(?!\w)' if not normalized[-1:].isalnum() and normalized[-1:] != '_' else r'\b'
                pattern = re.compile(prefix + re.escape(normalized) + suffix)
                if pattern.search(text_lower):
                    return True
        return False

    def _resolve_relationship_endpoint(
        self,
        raw_name: str,
        entities: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Resolve a relationship endpoint to the best matching extracted entity ID.

        The LLM often uses short forms in relationship triples while the entity
        list carries a fuller mention, so we allow safe fuzzy matching here.
        """
        if not isinstance(raw_name, str) or not raw_name.strip():
            return None

        raw_name = raw_name.strip()
        raw_variants = self._entity_name_variants(raw_name)
        raw_norm = self._normalize_entity_text(raw_name)

        direct_matches: List[str] = []
        best_entity_id: Optional[str] = None
        best_score = 0.0
        second_best = 0.0

        for entity in entities:
            entity_id = entity.get("id")
            if not isinstance(entity_id, str) or not entity_id.strip():
                continue

            candidate_variants = set()
            for candidate in self._entity_candidate_names(entity):
                candidate_variants.update(self._entity_name_variants(candidate))

            if any(rv.lower() == cv.lower() for rv in raw_variants for cv in candidate_variants):
                direct_matches.append(entity_id)
                continue

            entity_best = 0.0
            for candidate in candidate_variants:
                candidate_norm = self._normalize_entity_text(candidate)
                if not candidate_norm or not raw_norm:
                    continue
                if candidate_norm == raw_norm:
                    entity_best = 1.0
                    break

                score = 0.0
                if raw_norm in candidate_norm or candidate_norm in raw_norm:
                    if min(len(raw_norm), len(candidate_norm)) >= 4:
                        score = 0.94
                else:
                    score = difflib.SequenceMatcher(None, raw_norm, candidate_norm).ratio()
                    raw_tokens = set(raw_norm.split('_'))
                    candidate_tokens = set(candidate_norm.split('_'))
                    if raw_tokens and candidate_tokens:
                        overlap = len(raw_tokens & candidate_tokens) / max(
                            1, min(len(raw_tokens), len(candidate_tokens))
                        )
                        score = max(score, overlap * 0.92)

                entity_best = max(entity_best, score)

            if entity_best > best_score:
                second_best = best_score
                best_score = entity_best
                best_entity_id = entity_id
            elif entity_best > second_best:
                second_best = entity_best

        unique_direct = list(dict.fromkeys(direct_matches))
        if len(unique_direct) == 1:
            return unique_direct[0]
        if len(unique_direct) > 1:
            for entity_id in unique_direct:
                if entity_id.lower() == raw_name.lower():
                    return entity_id
            return None

        if best_entity_id and best_score >= 0.90 and (best_score - second_best) >= 0.03:
            return best_entity_id
        return None

    def _verify_triple_confidence(
        self,
        source_name: str,
        target_name: str,
        rel_type: str,
        chunks: List[Dict],
        source_aliases: List[str] = None,
        target_aliases: List[str] = None,
    ) -> float:
        """
        Evidence-grounded verification for an extracted triple (inspired by MOSAICX verify).

        Searches all chunks for co-occurrence of the source and target entity names.
        Returns a confidence score in [0.0, 1.0]:
          - 1.0  both names found in the same sentence
          - 0.7  both names found in the same chunk (not same sentence)
          - 0.4  only one name found in any chunk
          - 0.1  neither name found (LLM may have hallucinated this triple)

        This is intentionally cheap (string matching only) so it doesn't add
        meaningful latency. An LLM-based re-verification pass can be added later
        for low-confidence triples as an opt-in upgrade.
        """
        if not chunks or not source_name or not target_name:
            return 0.5  # neutral when no evidence to check

        src_lower = source_name.strip().lower()
        tgt_lower = target_name.strip().lower()

        # Build boundary-aware patterns.
        # Use (?<!\w)/(?!\w) lookarounds instead of \b so that entity names that
        # start or end with non-word characters (parentheses, hyphens, dots) are
        # still matched correctly.  e.g. "gleason score (gs)" ends with ')' which
        # is not a word char — \b would fail there, but (?!\w) does not.
        def _boundary_pattern(name: str) -> re.Pattern:
            prefix = r'(?<!\w)' if not name[:1].isalnum() and name[:1] != '_' else r'\b'
            suffix = r'(?!\w)' if not name[-1:].isalnum() and name[-1:] != '_' else r'\b'
            return re.compile(prefix + re.escape(name) + suffix)

        def _surface_forms(name: str) -> set:
            """Return the name plus underscore↔space variants so LLM IDs like
            'United_States' match chunk text 'United States' and vice versa."""
            forms = {name}
            forms.add(name.replace('_', ' '))
            forms.add(name.replace(' ', '_'))
            return {f for f in forms if f}

        # Build a list of patterns for each entity — canonical name plus all aliases,
        # including underscore/space variants to survive LLM ID normalisation differences.
        _src_forms = list(
            _surface_forms(src_lower)
            | {v for a in (source_aliases or []) if a for v in _surface_forms(a.strip().lower())}
        )
        _tgt_forms = list(
            _surface_forms(tgt_lower)
            | {v for a in (target_aliases or []) if a for v in _surface_forms(a.strip().lower())}
        )
        src_pats = [_boundary_pattern(f) for f in _src_forms if f]
        tgt_pats = [_boundary_pattern(f) for f in _tgt_forms if f]

        # Keep single-pattern aliases for backwards compatibility
        src_pat = src_pats[0] if src_pats else _boundary_pattern(src_lower)
        tgt_pat = tgt_pats[0] if tgt_pats else _boundary_pattern(tgt_lower)

        found_src, found_tgt, same_chunk, same_sentence = False, False, False, False

        for chunk in chunks:
            text = chunk.get('text', '').lower()
            has_src = any(p.search(text) for p in src_pats)
            has_tgt = any(p.search(text) for p in tgt_pats)

            if has_src:
                found_src = True
            if has_tgt:
                found_tgt = True

            if has_src and has_tgt:
                # Both entities present in this single chunk — check for sentence co-occurrence
                same_chunk = True
                sentences = re.split(r'(?<=[.!?])\s+', text)
                if any(
                    any(sp.search(s) for sp in src_pats) and any(tp.search(s) for tp in tgt_pats)
                    for s in sentences
                ):
                    same_sentence = True
                    break  # best possible score — stop scanning

        if same_sentence:
            return 1.0
        if same_chunk:
            # Both found within the same chunk (but not the same sentence)
            return 0.7
        if found_src and found_tgt:
            # Found in separate chunks of the same document — weaker evidence
            return 0.4
        if found_src or found_tgt:
            return 0.3
        return 0.1

    def _harmonize_entities(self, all_entities: List[Dict], return_id_map: bool = False):
        """
        Harmonize entities across chunks to avoid duplicates using improved normalization
        """
        logging.info(f"Starting harmonization of {len(all_entities)} raw entities")

        # Step 1: Build grouping by normalized text, then refine to avoid cross-type collapse.
        #
        # Why text-first, not (text, type):
        #   Using (text, type) directly splits "Prostate Cancer" typed as Disease in chunk 1
        #   and Concept in chunk 2 into two nodes — that is LLM drift, not a real distinction.
        #   Grouping by text first and picking the most specific type handles this correctly.
        #
        # Why we then split by specific type:
        #   Pure text grouping collapses genuinely different entities that happen to share a
        #   surface form — e.g. "depression" (Disease) vs "depression" (GeologicalFeature).
        #   If a text group contains more than one *distinct specific type*, we split it into
        #   per-specific-type buckets; generic-typed occurrences (Concept/Entity/Unknown/Other)
        #   are assigned to the dominant (largest) specific-type bucket so that LLM drift
        #   toward generic labels still merges into the right node rather than floating free.
        _generic_types = {'Concept', 'Entity', 'Unknown', 'Other'}

        text_groups: Dict[str, list] = defaultdict(list)
        for entity in all_entities:
            normalized_text = self._normalize_entity_text(entity['id'])
            text_groups[normalized_text].append(entity)

        # Refine: split text groups that contain multiple distinct specific types.
        # dominant_for_surface records which specific type is the largest bucket for
        # each surface form that was split — used below to make entity_map writes
        # deterministic (dominant type always wins, regardless of iteration order).
        entity_groups: Dict = defaultdict(list)
        dominant_for_surface: Dict[str, str] = {}  # norm_text → dominant specific type

        for norm_text, entities in text_groups.items():
            specific_types = {e.get('type') for e in entities if e.get('type') not in _generic_types}
            if len(specific_types) <= 1:
                # Common case: 0 or 1 distinct specific type — merge as before.
                entity_groups[norm_text].extend(entities)
            else:
                # Multiple distinct specific types → split, generics go to largest bucket.
                type_buckets: Dict[str, list] = defaultdict(list)
                generics = []
                for e in entities:
                    if e.get('type') in _generic_types:
                        generics.append(e)
                    else:
                        type_buckets[e['type']].append(e)
                dominant = max(type_buckets, key=lambda t: len(type_buckets[t]))
                if generics:
                    type_buckets[dominant].extend(generics)
                dominant_for_surface[norm_text] = dominant
                logging.info(
                    "Surface form '%s' has %d distinct specific types %s; "
                    "relationships will resolve to dominant type '%s' (%d occurrences). "
                    "Add source/target type fields to relationship dicts for per-type resolution.",
                    norm_text, len(type_buckets), sorted(type_buckets),
                    dominant, len(type_buckets[dominant]),
                )
                for stype, bucket in type_buckets.items():
                    entity_groups[(norm_text, stype)].extend(bucket)

        # Step 2: Log entity distribution before harmonization
        type_distribution = Counter()
        for entities in entity_groups.values():
            for entity in entities:
                type_distribution[entity['type']] += 1

        logging.info(f"Entity type distribution before harmonization: {dict(type_distribution)}")

        # Step 3: Create harmonized entities
        harmonized_entities = []
        entity_map = {}  # For mapping original IDs to harmonized versions
        total_duplicates_removed = 0

        for normalized_key, entities in entity_groups.items():
            if not entities:
                continue

            # --- Deterministic deduplication rules ---
            # Rule 1: prefer specific ontology types over generic fallbacks (Concept, Entity, Unknown)
            # Rule 2: among equally specific types, prefer the longest description
            # Rule 3: if still tied, prefer the longest entity name (most fully-qualified)
            # (_generic_types defined at Step 1 above)

            def _type_specificity(e):
                return 0 if e.get('type', 'Concept') in _generic_types else 1

            def _desc_len(e):
                return len(e.get('properties', {}).get('description') or '')

            def _name_len(e):
                return len(e.get('id') or '')

            representative_entity = max(
                entities, key=lambda e: (_type_specificity(e), _desc_len(e), _name_len(e))
            ).copy()

            # Merge information from all occurrences
            all_names = set()
            all_descriptions = set()

            for entity in entities:
                # Rule 3: accumulate all surface forms as synonyms
                all_names.add(entity['id'])
                desc = entity.get('properties', {}).get('description')
                if desc:
                    all_descriptions.add(desc if isinstance(desc, str) else str(desc))

                # Keep the best embedding (prefer any available embedding)
                if not representative_entity.get('embedding') and entity.get('embedding'):
                    representative_entity['embedding'] = entity['embedding']

            # Merge all non-description properties from variants into representative
            for entity in entities:
                if entity.get('properties'):
                    for k, v in entity['properties'].items():
                        if k not in ('description',):  # description already resolved above
                            representative_entity.setdefault('properties', {}).setdefault(k, v)

            # Update representative entity with merged information
            if len(all_names) > 1 or len(all_descriptions) > 1:
                representative_entity['properties']['all_names'] = sorted(all_names)
                representative_entity['properties']['all_descriptions'] = sorted(all_descriptions, key=len, reverse=True)
                total_duplicates_removed += len(entities) - 1

            # Generate deterministic UUID
            entity_uuid = self._generate_entity_id(representative_entity)
            representative_entity['uuid'] = entity_uuid

            harmonized_entities.append(representative_entity)

            # Map all original name variations to the harmonized entity.
            # For split surface forms, entity_map[surface_form] must always point to the
            # dominant-type representative so relationship resolution is deterministic
            # regardless of which order entity_groups iterates the split buckets.
            # The dominant type was pre-computed in the split phase above.
            for entity in entities:
                norm = self._normalize_entity_text(entity['id'])
                is_dominant = (
                    dominant_for_surface.get(norm) == representative_entity.get('type')
                    if norm in dominant_for_surface
                    else True  # not a split surface form — always write
                )
                if is_dominant or entity['id'] not in entity_map:
                    entity_map[entity['id']] = representative_entity

        logging.info(f"Harmonization complete: {len(harmonized_entities)} entities (removed {total_duplicates_removed} duplicates)")

        # Log final distribution
        final_type_distribution = Counter(e['type'] for e in harmonized_entities)
        logging.info(f"Entity type distribution after harmonization: {dict(final_type_distribution)}")

        if return_id_map:
            return harmonized_entities, entity_map  # entity_map: original_id → representative
        return harmonized_entities

    def _harmonize_relationships(self, all_relationships: List[Dict], entity_map: Dict) -> List[Dict]:
        """
        Harmonize relationships across chunks and map to UUID-based entity IDs.

        entity_map may be either:
          - original_id → representative entity  (from _harmonize_entities with return_id_map=True)
          - uuid → entity  (legacy call sites)
        In both cases we build original_to_uuid by walking .values().
        """
        harmonized_relationships = []
        seen_relationships = set()

        # Build variant_name → uuid from entity_map.
        # entity_map keys are ALL original variant IDs (every surface form seen during
        # extraction), values are the representative entities with uuid set.
        # Iterating .items() — not .values() — ensures every variant name is covered,
        # so relationships that used a non-canonical spelling are not silently dropped.
        original_to_uuid = {}
        candidate_names_by_uuid: Dict[str, set] = defaultdict(set)
        entity_type_by_uuid: Dict[str, str] = {}
        for variant_name, representative in entity_map.items():
            if 'uuid' not in representative:
                logging.warning(f"Entity '{representative.get('id', '?')}' missing uuid in entity_map — skipping in relationship mapping")
                continue
            uuid_value = representative['uuid']
            if representative.get('type'):
                entity_type_by_uuid.setdefault(uuid_value, representative.get('type'))
            for candidate in self._entity_name_variants(variant_name) if isinstance(variant_name, str) else []:
                original_to_uuid.setdefault(candidate, uuid_value)
                original_to_uuid.setdefault(candidate.lower(), uuid_value)
                candidate_names_by_uuid[uuid_value].add(candidate)
            for candidate in self._entity_candidate_names(representative):
                for variant in self._entity_name_variants(candidate):
                    original_to_uuid.setdefault(variant, uuid_value)
                    original_to_uuid.setdefault(variant.lower(), uuid_value)
                    candidate_names_by_uuid[uuid_value].add(variant)

        def _lookup_uuid(name: str):
            """Try increasingly fuzzy lookups before giving up."""
            if not isinstance(name, str) or not name.strip():
                return None
            exact = (
                original_to_uuid.get(name)
                or original_to_uuid.get(name.lower())
                or original_to_uuid.get(name.replace('_', ' '))
                or original_to_uuid.get(name.replace('_', ' ').lower())
                or original_to_uuid.get(name.replace(' ', '_'))
                or original_to_uuid.get(name.replace(' ', '_').lower())
                or original_to_uuid.get(self._normalize_entity_text(name))
            )
            if exact:
                return exact

            raw_norm = self._normalize_entity_text(name)
            best_uuid = None
            best_score = 0.0
            second_best = 0.0

            for uuid_value, candidates in candidate_names_by_uuid.items():
                entity_best = 0.0
                for candidate in candidates:
                    cand_norm = self._normalize_entity_text(candidate)
                    if not cand_norm or not raw_norm:
                        continue
                    if cand_norm == raw_norm:
                        entity_best = 1.0
                        break
                    if raw_norm in cand_norm or cand_norm in raw_norm:
                        if min(len(raw_norm), len(cand_norm)) >= 4:
                            entity_best = max(entity_best, 0.94)
                    else:
                        entity_best = max(
                            entity_best,
                            difflib.SequenceMatcher(None, raw_norm, cand_norm).ratio(),
                        )

                if entity_best > best_score:
                    second_best = best_score
                    best_score = entity_best
                    best_uuid = uuid_value
                elif entity_best > second_best:
                    second_best = entity_best

            if best_uuid and best_score >= 0.90 and (best_score - second_best) >= 0.03:
                return best_uuid
            return None

        dropped_unmapped = 0
        deduped_relationships = 0
        for rel in all_relationships:
            src = rel.get('source')
            tgt = rel.get('target')
            source_uuid = _lookup_uuid(src)
            target_uuid = _lookup_uuid(tgt)

            if not source_uuid or not target_uuid:
                dropped_unmapped += 1
                logging.warning(
                    "Dropping relationship — entity not found in map: '%s' -[%s]-> '%s'",
                    src, rel.get('type', '?'), tgt,
                )
                continue

            if source_uuid and target_uuid:
                # Create new relationship with UUID-based IDs
                uuid_rel = rel.copy()
                rel_properties = dict(uuid_rel.get('properties') or {})
                if isinstance(src, str):
                    rel_properties.setdefault("source_name", src)
                if isinstance(tgt, str):
                    rel_properties.setdefault("target_name", tgt)
                uuid_rel['properties'] = rel_properties
                uuid_rel['source'] = source_uuid
                uuid_rel['target'] = target_uuid

                # Include negated in the key: A-INHIBITS->B (negated=True) and
                # A-INHIBITS->B (negated=False) are opposite claims and must not collapse.
                _neg = bool(uuid_rel.get('negated', False))
                condition = uuid_rel.get('condition') or rel_properties.get('condition') or ''
                quantitative = uuid_rel.get('quantitative') or rel_properties.get('quantitative') or ''
                rel_type_key = self._canonicalize_relationship_type(
                    uuid_rel.get('type', ''),
                    source_type=entity_type_by_uuid.get(source_uuid),
                    target_type=entity_type_by_uuid.get(target_uuid),
                )
                rel_key = (
                    f"{source_uuid}:{rel_type_key}:{target_uuid}:{_neg}:"
                    f"{str(condition).strip().lower()}:{str(quantitative).strip().lower()}"
                )

                if rel_key not in seen_relationships:
                    harmonized_relationships.append(uuid_rel)
                    seen_relationships.add(rel_key)
                else:
                    deduped_relationships += 1

        logging.info(
            "Relationship harmonization kept=%d dropped_unmapped=%d deduped=%d",
            len(harmonized_relationships),
            dropped_unmapped,
            deduped_relationships,
        )

        return harmonized_relationships

    @staticmethod
    def _attach_relationship_provenance(
        relationships: List[Dict],
        chunk_positions: List[int],
    ) -> List[Dict]:
        """Attach local chunk provenance so triple verification can stay passage-local."""
        annotated: List[Dict] = []
        normalized_positions = [
            int(pos)
            for pos in chunk_positions
            if isinstance(pos, (int, float))
        ]
        for rel in relationships or []:
            if not isinstance(rel, dict):
                continue
            rel_copy = dict(rel)
            existing_positions = rel_copy.get("provenance_positions") or []
            merged_positions = {
                int(pos)
                for pos in existing_positions
                if isinstance(pos, (int, float))
            }
            merged_positions.update(normalized_positions)
            rel_copy["provenance_positions"] = sorted(merged_positions)
            annotated.append(rel_copy)
        return annotated

    @staticmethod
    def _relationship_local_provenance(
        rel: Dict[str, Any],
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, List[Any]]:
        """
        Resolve question-local provenance for a relationship from chunk positions.

        Older KGs only expose chunk positions on relationships. Newer builds also
        stamp question and passage provenance onto the edge so question-scoped
        retrieval can be enforced exactly at query time.
        """
        positions = {
            int(pos)
            for pos in (rel.get("provenance_positions") or [])
            if isinstance(pos, (int, float))
        }
        if not positions:
            return {
                "provenance_positions": [],
                "question_ids": [],
                "passage_keys": [],
            }

        question_ids = set()
        passage_keys = set()
        for chunk in chunks or []:
            pos = chunk.get("position")
            if not isinstance(pos, (int, float)) or int(pos) not in positions:
                continue
            qid = chunk.get("question_id")
            if qid is not None and str(qid).strip():
                question_ids.add(str(qid))
            passage_key = f"{chunk.get('question_id', '')}::p{chunk.get('passage_index', -1)}"
            passage_keys.add(passage_key)

        return {
            "provenance_positions": sorted(positions),
            "question_ids": sorted(question_ids),
            "passage_keys": sorted(passage_keys),
        }

    def generate_knowledge_graph(self, text: str, llm, file_name: str = None, model_name: str = "openai/gpt-oss-120b:free", max_chunks: int = None, kg_name: str = None, doc_metadata: dict = None, doc_hash: str = None) -> Dict[str, Any]:
        """
        Generate knowledge graph from text with ontology-guided entity extraction

        Args:
            text: Input text to process
            llm: LLM provider instance
            file_name: Optional filename for storage
            model_name: LLM model name
            max_chunks: Maximum number of chunks to process (for large documents)
        """
        logging.info("Starting ontology-guided knowledge graph generation")

        # Determine if ontology is available
        has_ontology = bool(self.ontology_classes) or bool(self.ontology_relationships)
        extraction_method = "ontology_guided_llm" if has_ontology else "natural_llm"
        logging.info(f"Extraction method: {extraction_method}")

        # Step 1: Chunk the text
        chunks = self._chunk_text(text)
        logging.info(f"Created {len(chunks)} chunks")

        # Limit chunks if specified (for very large documents)
        if max_chunks and len(chunks) > max_chunks:
            logging.warning(f"Limiting processing to {max_chunks} chunks out of {len(chunks)} total")
            chunks = chunks[:max_chunks]

        # Step 1b: Detect section headers once across the full document.
        # Each chunk is tagged with its section (e.g. "Methods", "Results") so
        # the extraction LLM can interpret claims in the correct context.
        section_headers = self._detect_section_headers(text)
        logging.info("Detected %d section headers: %s", len(section_headers),
                     [name for _, name in section_headers])

        # Step 2: Extract entities and relationships from each chunk
        all_entities = []
        all_relationships = []
        processed_chunks = 0
        failed_chunks = 0
        entities_per_chunk: Dict[int, List[Dict]] = {}  # chunk index → entities (for cross-chunk pass)

        for i, chunk in enumerate(chunks):
            try:
                logging.info(f"Processing chunk {i+1}/{len(chunks)}")

                # --- Context enrichment ---
                # 1. Section header for this chunk
                chunk_section = self._get_section_for_position(
                    chunk.get("start_pos", 0), section_headers
                )

                # 2. Qualifier sentences from the previous chunk
                qualifier_ctx = ""
                if i > 0:
                    qualifier_ctx = self._extract_qualifier_sentences(chunks[i - 1]["text"])

                # 3. Cross-chunk coreference resolution (only when enabled and markers detected)
                extraction_text = chunk["text"]
                if self.enable_coreference_resolution and llm is not None and self._has_coreference_markers(extraction_text):
                    # Build context window: up to 2 previous chunks
                    coref_context = "\n\n".join(
                        chunks[j]["text"] for j in range(max(0, i - 2), i)
                    )
                    if coref_context:
                        logging.info(
                            "Chunk %d/%d: coreference markers detected — running resolution pass",
                            i + 1, len(chunks),
                        )
                        extraction_text = self._resolve_coreferences_llm(
                            extraction_text, coref_context, llm, model_name
                        )
                        time.sleep(0.5)  # extra LLM call — brief pause

                chunk_kg = self._extract_entities_and_relationships_with_llm(
                    extraction_text,
                    llm,
                    model_name,
                    context_header=qualifier_ctx or None,
                    section_header=chunk_section,
                )

                # Ensure chunk_kg is a dictionary with expected keys
                try:
                    if isinstance(chunk_kg, dict) and (chunk_kg.get('entities', []) or chunk_kg.get('relationships', [])):
                        chunk_entities = chunk_kg.get('entities', [])
                        all_entities.extend(chunk_entities)
                        all_relationships.extend(
                            self._attach_relationship_provenance(
                                chunk_kg.get('relationships', []),
                                [chunk.get("position", i)],
                            )
                        )
                        entities_per_chunk[i] = chunk_entities
                        processed_chunks += 1
                        logging.info(f"✓ Chunk {i+1} processed: {len(chunk_entities)} entities, {len(chunk_kg.get('relationships', []))} relationships")
                    else:
                        logging.warning(f"⚠ Chunk {i+1} returned invalid or empty format: {type(chunk_kg)}")
                        failed_chunks += 1
                except Exception as inner_e:
                    logging.error(f"❌ Error processing chunk result: {inner_e}")
                    failed_chunks += 1

            except Exception as e:
                logging.error(f"❌ Failed to process chunk {i+1}: {e}")
                import traceback
                logging.error(f"Traceback: {traceback.format_exc()}")
                failed_chunks += 1
                continue

            # Add small delay between chunks only when an external LLM is in use
            if llm is not None and i < len(chunks) - 1:  # Don't delay after the last chunk
                time.sleep(1.0)  # 1 second delay between API calls

        logging.info(f"Processing complete: {processed_chunks} successful, {failed_chunks} failed")

        # Step 2b: Cross-chunk relationship extraction (sliding window of 2 adjacent chunks).
        # The per-chunk LLM pass can only see entities within one chunk at a time, so relationships
        # that span a chunk boundary are never extracted.  This pass combines adjacent chunk pairs
        # and asks the LLM only for relationships between already-known entities — no new entity
        # extraction, so it is cheap (no dedup work) and adds N-1 LLM calls total.
        if llm is not None and len(chunks) > 1:
            cross_chunk_rels = 0
            for i in range(len(chunks) - 1):
                left_entities = entities_per_chunk.get(i, [])
                right_entities = entities_per_chunk.get(i + 1, [])
                combined_text = chunks[i]['text'] + "\n\n" + chunks[i + 1]['text']
                combined_text_lower = combined_text.lower()
                # Pre-filter: only pass entities whose names actually appear in the combined text.
                # Avoids sending irrelevant entities that the LLM can't ground, and removes the
                # hard 60-entity cap from _extract_relationships_only becoming a problem.
                combined_entities = [
                    e for e in (left_entities + right_entities)
                    if isinstance(e.get('id'), str) and e['id'].lower() in combined_text_lower
                ]
                if len(combined_entities) < 2:
                    continue
                new_rels = self._extract_relationships_only(combined_text, combined_entities, llm, model_name)
                if new_rels:
                    all_relationships.extend(
                        self._attach_relationship_provenance(
                            new_rels,
                            [chunks[i].get("position", i), chunks[i + 1].get("position", i + 1)],
                        )
                    )
                    cross_chunk_rels += len(new_rels)
                    logging.info(f"Cross-chunk pass ({i+1}↔{i+2}): {len(new_rels)} relationships found")
                time.sleep(0.5)  # shorter delay for relationship-only calls
            logging.info(f"Cross-chunk pass complete: {cross_chunk_rels} additional relationships extracted")

        if processed_chunks == 0 and failed_chunks > 0:
            raise RuntimeError(
                f"KG extraction failed: all {failed_chunks} chunk(s) returned errors. "
                "Check LLM connectivity and rate limits."
            )

        # Step 3: Harmonize entities and relationships.
        # _harmonize_entities returns the representative entities AND builds the
        # full original-ID → representative mapping internally (stored on the
        # entities as entity['_all_ids'] is NOT available, so we rebuild it here).
        # We need every original variant ID to map to its representative so that
        # relationships whose source/target used a non-canonical name aren't dropped.
        harmonized_entities, id_to_representative = self._harmonize_entities(all_entities, return_id_map=True)
        harmonized_relationships = self._harmonize_relationships(all_relationships, id_to_representative)

        logging.info(f"Harmonized to {len(harmonized_entities)} entities and {len(harmonized_relationships)} relationships")

        # Step 4: Format the final knowledge graph
        # Use UUID-based IDs to prevent duplicates
        kg_prefix = f"{kg_name}_" if kg_name else ""

        kg = {
            "nodes": [
                {
                    "id": f"{kg_prefix}{entity['uuid']}",
                    "label": entity['type'],
                    "properties": {
                        "name": entity['id'],
                        "type": entity['type'],
                        "original_id": entity['id'],  # Keep original ID for reference
                        **entity.get('properties', {})
                    },
                    "embedding": entity.get('embedding'),
                    "color": self._get_node_color(entity['type']),
                    "size": 30,
                    "font": {"size": 14, "color": "#333333"},
                    "title": f"Entity: {entity['id']}\nType: {entity['type']}\nKG: {kg_name or 'default'}\nClick for details"
                }
                for entity in harmonized_entities
            ],
            "relationships": [
                {
                    "id": f"{kg_prefix}rel_{rel['source']}_{rel['type']}_{rel['target']}_{idx}",
                    "from": f"{kg_prefix}{rel['source']}",
                    "to": f"{kg_prefix}{rel['target']}",
                    "source": f"{kg_prefix}{rel['source']}",
                    "target": f"{kg_prefix}{rel['target']}",
                    "type": rel['type'],
                    "label": rel['type'],
                    "negated": rel.get('negated', False),
                    "properties": rel.get('properties', {}),
                    "arrows": "to",
                    "color": {"color": "#444444"},
                    "font": {"size": 12, "align": "middle"}
                }
                for idx, rel in enumerate(harmonized_relationships)
            ],
            "chunks": chunks,
            "metadata": {
                "total_chunks": len(chunks),
                "total_entities": len(harmonized_entities),
                "total_relationships": len(harmonized_relationships),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": type(self.embedding_function).__name__,
                "embedding_dimension": self.embedding_dimension,
                "ontology_classes": len(self.ontology_classes),
                "ontology_relationships": len(self.ontology_relationships),
                "extraction_method": "ontology_guided_llm" if has_ontology else "natural_llm",
                "kg_name": kg_name,
                "provider": provider,
                "model": model_name,
                "max_chunks_setting": max_chunks,
                "created_at": datetime.now().isoformat(),
                "visualization_ready": True,
                "file_name": file_name,
                "doc_hash": doc_hash,
                "schema_card": self._build_schema_card()
            }
        }

        # Attach any document-level metadata from the source (e.g. CSV columns)
        if doc_metadata:
            kg['metadata']['doc_metadata'] = doc_metadata

        # Step 5: Store in Neo4j if requested
        if file_name:
            success = self.store_knowledge_graph_with_embeddings(
                kg, file_name, doc_metadata=doc_metadata, doc_hash=doc_hash
            )
            kg['metadata']['stored_in_neo4j'] = success

            # Step 5b: HippoRAG-style synonym merging.
            # After embeddings are in Neo4j, cluster near-duplicate entity nodes
            # (e.g. "TBK1" / "TBK1 kinase") and merge them so entity-first search
            # can find all surface-form variants via a single canonical node.
            if success:
                try:
                    graph_for_merge = self._create_neo4j_connection()
                    merges = self.merge_synonym_entities(graph_for_merge, kg_name=kg_name)
                    kg['metadata']['synonym_merges'] = merges
                    logging.info(f"Synonym merging complete: {merges} pairs merged")
                except Exception as syn_err:
                    logging.warning(f"Synonym merging failed (non-fatal): {syn_err}")

                # Compute node specificity (HippoRAG-style IDF weight):
                # s(e) = 1 / |passages containing e|.  Stored on each entity node
                # so that retrieval can down-weight ubiquitous hub entities as seeds.
                try:
                    graph_for_merge = self._create_neo4j_connection()
                    scope_filter = "WHERE c.kgName = $kg_name" if kg_name else ""
                    graph_for_merge.query(
                        f"""
                        MATCH (e:__Entity__)<-[:HAS_ENTITY]-(c:Chunk)
                        {scope_filter}
                        WITH e, count(DISTINCT c) AS passage_count
                        SET e.passage_count = passage_count,
                            e.node_specificity = 1.0 / toFloat(passage_count)
                        """,
                        {"kg_name": kg_name} if kg_name else {},
                    )
                    logging.info("Node specificity weights computed for kg_name=%s", kg_name)
                except Exception as spec_err:
                    logging.warning("Node specificity computation failed (non-fatal): %s", spec_err)

        return kg

    def generate_knowledge_graph_from_passages(
        self,
        passages,           # List[ContextPassage] — avoids circular import; duck-typed
        llm,
        file_name: str = None,
        model_name: str = "openai/gpt-oss-120b:free",
        kg_name: str = None,
        doc_metadata: dict = None,
        doc_hash: str = None,
    ) -> Dict[str, Any]:
        """Generate a KG from a list of ContextPassage objects.

        Unlike generate_knowledge_graph, this method never concatenates passages
        from different records before chunking.  Each passage is chunked
        independently; sub-splitting only occurs when a single passage exceeds
        chunk_size.  Cross-chunk relationship extraction is scoped within each
        passage so the LLM never sees entity pairs that are only co-located
        because two unrelated passages were glued together.

        Harmonisation and Neo4j storage happen once after all passages are
        processed, so the result is equivalent in structure to the single-call
        path but with clean passage-level extraction boundaries.
        """
        logging.info(
            "Starting passage-aware KG generation: %d passages", len(passages)
        )
        has_ontology = bool(self.ontology_classes) or bool(self.ontology_relationships)
        extraction_method = "ontology_guided_llm" if has_ontology else "natural_llm"

        all_chunks: List[Dict] = []
        all_entities: List[Dict] = []
        all_relationships: List[Dict] = []

        global_chunk_offset = 0  # running count for position uniqueness across passages

        for p_idx, passage in enumerate(passages):
            passage_text = passage.text
            source_label = f"{passage.dataset}/{passage.question_id}/p{passage.passage_index}"

            # Chunk this passage independently.  Most passages fit in one chunk;
            # sub-splitting only happens when the passage exceeds chunk_size.
            p_chunks = self._chunk_text(passage_text)

            # Shift position fields so they are globally unique across passages.
            for local_i, ch in enumerate(p_chunks):
                ch["position"] = global_chunk_offset + local_i
                ch["source"] = source_label
                ch["dataset"] = passage.dataset
                ch["question_id"] = passage.question_id
                ch["passage_index"] = passage.passage_index
                ch["chunk_local_index"] = ch.get("chunk_id", local_i)

            # Section headers scoped to this passage (usually empty for short passages).
            section_headers = self._detect_section_headers(passage_text)

            entities_this_passage: Dict[int, List[Dict]] = {}
            processed = 0
            failed = 0

            for local_i, chunk in enumerate(p_chunks):
                global_i = global_chunk_offset + local_i
                try:
                    chunk_section = self._get_section_for_position(
                        chunk.get("start_pos", 0), section_headers
                    )
                    # Coreference context: only previous chunks within the same passage.
                    qualifier_ctx = ""
                    if local_i > 0:
                        qualifier_ctx = self._extract_qualifier_sentences(
                            p_chunks[local_i - 1]["text"]
                        )
                    extraction_text = chunk["text"]
                    if (
                        self.enable_coreference_resolution
                        and llm is not None
                        and self._has_coreference_markers(extraction_text)
                    ):
                        coref_context = "\n\n".join(
                            p_chunks[j]["text"]
                            for j in range(max(0, local_i - 2), local_i)
                        )
                        if coref_context:
                            extraction_text = self._resolve_coreferences_llm(
                                extraction_text, coref_context, llm, model_name
                            )
                            time.sleep(0.5)

                    chunk_kg = self._extract_entities_and_relationships_with_llm(
                        extraction_text, llm, model_name,
                        context_header=qualifier_ctx or None,
                        section_header=chunk_section,
                    )
                    if isinstance(chunk_kg, dict) and (
                        chunk_kg.get("entities") or chunk_kg.get("relationships")
                    ):
                        chunk_entities = chunk_kg.get("entities", [])
                        all_entities.extend(chunk_entities)
                        all_relationships.extend(
                            self._attach_relationship_provenance(
                                chunk_kg.get("relationships", []),
                                [chunk.get("position", global_i)],
                            )
                        )
                        entities_this_passage[local_i] = chunk_entities
                        processed += 1
                        logging.info(
                            "Passage %d/%d chunk %d/%d: %d entities, %d rels [%s]",
                            p_idx + 1, len(passages),
                            local_i + 1, len(p_chunks),
                            len(chunk_entities),
                            len(chunk_kg.get("relationships", [])),
                            source_label,
                        )
                    else:
                        failed += 1
                except Exception as e:
                    logging.error(
                        "Passage %d chunk %d failed: %s", p_idx + 1, local_i + 1, e
                    )
                    failed += 1
                    continue

                if llm is not None and local_i < len(p_chunks) - 1:
                    time.sleep(1.0)

            # Cross-chunk relationship extraction — scoped within this passage only.
            if llm is not None and len(p_chunks) > 1:
                for local_i in range(len(p_chunks) - 1):
                    left_ents = entities_this_passage.get(local_i, [])
                    right_ents = entities_this_passage.get(local_i + 1, [])
                    combined_text = (
                        p_chunks[local_i]["text"] + "\n\n" + p_chunks[local_i + 1]["text"]
                    )
                    candidate_ents = []
                    seen_entity_ids = set()
                    for entity in (left_ents + right_ents):
                        entity_id = entity.get("id")
                        dedupe_key = entity_id if isinstance(entity_id, str) and entity_id else id(entity)
                        if dedupe_key in seen_entity_ids:
                            continue
                        if self._entity_appears_in_text(entity, combined_text):
                            candidate_ents.append(entity)
                            seen_entity_ids.add(dedupe_key)
                    if len(candidate_ents) >= 2:
                        new_rels = self._extract_relationships_only(
                            combined_text, candidate_ents, llm, model_name
                        )
                        if new_rels:
                            all_relationships.extend(
                                self._attach_relationship_provenance(
                                    new_rels,
                                    [
                                        p_chunks[local_i].get("position", global_chunk_offset + local_i),
                                        p_chunks[local_i + 1].get("position", global_chunk_offset + local_i + 1),
                                    ],
                                )
                            )
                        time.sleep(0.5)

            all_chunks.extend(p_chunks)
            global_chunk_offset += len(p_chunks)

            logging.info(
                "Passage %d/%d complete: processed=%d failed=%d [%s]",
                p_idx + 1, len(passages), processed, failed, source_label,
            )

        logging.info(
            "All passages processed: %d chunks, %d raw entities, %d raw relationships",
            len(all_chunks), len(all_entities), len(all_relationships),
        )

        if not all_entities and not all_relationships:
            raise RuntimeError(
                "Passage-aware KG extraction yielded no entities or relationships. "
                "Check LLM connectivity and passage content."
            )

        # Harmonise globally across all passages.
        harmonized_entities, id_to_representative = self._harmonize_entities(
            all_entities, return_id_map=True
        )
        harmonized_relationships = self._harmonize_relationships(
            all_relationships, id_to_representative
        )
        logging.info(
            "Harmonized: %d entities, %d relationships",
            len(harmonized_entities), len(harmonized_relationships),
        )

        kg_prefix = f"{kg_name}_" if kg_name else ""
        kg = {
            "nodes": [
                {
                    "id": f"{kg_prefix}{entity['uuid']}",
                    "label": entity["type"],
                    "properties": {
                        "name": entity["id"],
                        "type": entity["type"],
                        "original_id": entity["id"],
                        **entity.get("properties", {}),
                    },
                    "embedding": entity.get("embedding"),
                    "color": self._get_node_color(entity["type"]),
                    "size": 30,
                    "font": {"size": 14, "color": "#333333"},
                    "title": (
                        f"Entity: {entity['id']}\nType: {entity['type']}"
                        f"\nKG: {kg_name or 'default'}\nClick for details"
                    ),
                }
                for entity in harmonized_entities
            ],
            "relationships": [
                {
                    "id": f"{kg_prefix}rel_{rel['source']}_{rel['type']}_{rel['target']}_{idx}",
                    "from": f"{kg_prefix}{rel['source']}",
                    "to": f"{kg_prefix}{rel['target']}",
                    "source": f"{kg_prefix}{rel['source']}",
                    "target": f"{kg_prefix}{rel['target']}",
                    "type": rel["type"],
                    "label": rel["type"],
                    "negated": rel.get("negated", False),
                    "properties": rel.get("properties", {}),
                    "provenance_positions": rel.get("provenance_positions") or [],
                    "arrows": "to",
                    "color": {"color": "#444444"},
                    "font": {"size": 12, "align": "middle"},
                }
                for idx, rel in enumerate(harmonized_relationships)
            ],
            "chunks": all_chunks,
            "metadata": {
                "total_chunks": len(all_chunks),
                "total_passages": len(passages),
                "total_entities": len(harmonized_entities),
                "total_relationships": len(harmonized_relationships),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": type(self.embedding_function).__name__,
                "embedding_dimension": self.embedding_dimension,
                "ontology_classes": len(self.ontology_classes),
                "ontology_relationships": len(self.ontology_relationships),
                "extraction_method": extraction_method,
                "kg_name": kg_name,
                "created_at": datetime.now().isoformat(),
                "visualization_ready": True,
                "file_name": file_name,
                "doc_hash": doc_hash,
            },
        }

        if doc_metadata:
            kg["metadata"]["doc_metadata"] = doc_metadata

        if file_name:
            success = self.store_knowledge_graph_with_embeddings(
                kg, file_name, doc_metadata=doc_metadata, doc_hash=doc_hash
            )
            kg["metadata"]["stored_in_neo4j"] = success
            if success:
                try:
                    graph_for_merge = self._create_neo4j_connection()
                    merges = self.merge_synonym_entities(
                        graph_for_merge, kg_name=kg_name
                    )
                    kg["metadata"]["synonym_merges"] = merges
                    logging.info("Synonym merging complete: %d pairs merged", merges)
                except Exception as syn_err:
                    logging.warning("Synonym merging failed (non-fatal): %s", syn_err)

                try:
                    graph_for_merge = self._create_neo4j_connection()
                    scope_filter = "WHERE c.kgName = $kg_name" if kg_name else ""
                    graph_for_merge.query(
                        f"""
                        MATCH (e:__Entity__)<-[:HAS_ENTITY]-(c:Chunk)
                        {scope_filter}
                        WITH e, count(DISTINCT c) AS passage_count
                        SET e.passage_count = passage_count,
                            e.node_specificity = 1.0 / toFloat(passage_count)
                        """,
                        {"kg_name": kg_name} if kg_name else {},
                    )
                    logging.info("Node specificity weights computed for kg_name=%s", kg_name)
                except Exception as spec_err:
                    logging.warning("Node specificity computation failed (non-fatal): %s", spec_err)

        return kg

    def _get_node_color(self, entity_type: str) -> str:
        """Get color for node based on entity type"""
        color_map = {
            "Disease": "#ff6666",
            "Treatment": "#66ff66",
            "Drug": "#66ffff",
            "Symptom": "#ffff66",
            "Patient": "#ff9999",
            "Physician": "#ffcc99",
            "Hospital": "#99ccff",
            "MedicalProcedure": "#cc99ff",
            "MedicalDevice": "#ff99cc",
            "Anatomy": "#99ff99",
            "Concept": "#a6cee3"
        }
        return color_map.get(entity_type, "#a6cee3")

    def _build_relationship_merge_query(
        self,
        sanitized_rel_type: str,
        *,
        include_condition: bool,
        include_quantitative: bool,
    ) -> str:
        """Build the relationship MERGE query without null-valued qualifiers.

        Neo4j rejects null property values inside MERGE patterns, so optional
        qualifiers must only participate in the identity key when present.
        """
        merge_key_parts = ["negated: $negated"]
        if include_condition:
            merge_key_parts.append("condition: $condition")
        if include_quantitative:
            merge_key_parts.append("quantitative: $quantitative")

        merge_key = ",\n                        ".join(merge_key_parts)
        return f"""
                    MATCH (source:__Entity__ {{id: $source_id}})
                    MATCH (target:__Entity__ {{id: $target_id}})
                    MERGE (source)-[r:{sanitized_rel_type} {{
                        {merge_key}
                    }}]->(target)
                    SET r += $properties,
                        r.provenancePositions =
                            reduce(acc = [], x IN coalesce(r.provenancePositions, []) + $provenance_positions |
                                CASE WHEN x IN acc THEN acc ELSE acc + x END),
                        r.questionIds =
                            reduce(acc = [], x IN coalesce(r.questionIds, []) + $question_ids |
                                CASE WHEN x IN acc THEN acc ELSE acc + x END),
                        r.passageKeys =
                            reduce(acc = [], x IN coalesce(r.passageKeys, []) + $passage_keys |
                                CASE WHEN x IN acc THEN acc ELSE acc + x END)
                    """

    def store_knowledge_graph_with_embeddings(self, kg: Dict[str, Any], file_name: str, doc_metadata: dict = None, doc_hash: str = None) -> bool:
        """
        Store the knowledge graph in Neo4j database with proper embedding support
        """
        try:
            # Try to create Neo4j connection - handle APOC issues gracefully
            try:
                graph = self._create_neo4j_connection()
            except Exception as conn_error:
                if "APOC" in str(conn_error) or "apoc" in str(conn_error):
                    logging.warning(f"APOC not available, skipping advanced KG storage: {conn_error}")
                    return False
                else:
                    raise conn_error

            # Pre-flight: remove any orphaned entities from a previous failed build
            # for this dataset (entities written before the Document node was committed).
            # This prevents unique-constraint violations on __Entity__.id when the LLM
            # assigns a different type label on a subsequent run.
            import uuid
            kg_name_value = kg['metadata'].get('kg_name') or file_name or "default"
            try:
                graph.query(
                    """
                    MATCH (e:__Entity__)
                    WHERE e.id STARTS WITH $prefix
                      AND NOT EXISTS {
                        MATCH (:Chunk)-[:HAS_ENTITY|MENTIONS]->(e)
                      }
                    DETACH DELETE e
                    """,
                    {"prefix": kg_name_value + "_"},
                )
                logging.info(f"[store_kg] Pre-flight cleanup done for '{kg_name_value}'")
            except Exception as _cleanup_err:
                logging.warning(f"[store_kg] Pre-flight cleanup failed (non-fatal): {_cleanup_err}")

            try:
                graph.query(
                    """
                    MATCH (q:Qualifier {kgName: $kg_name})
                    WHERE NOT EXISTS { MATCH ()-[]->(q) }
                       OR NOT EXISTS { MATCH (q)-[]->() }
                    DETACH DELETE q
                    """,
                    {"kg_name": kg_name_value},
                )
            except Exception as _qual_cleanup_err:
                logging.debug("[store_kg] Orphan qualifier cleanup failed (non-fatal): %s", _qual_cleanup_err)

            # Create document node with versioning
            kg_version = str(uuid.uuid4())
            schema_card = self._build_schema_card()
            schema_card_json = json.dumps(schema_card, ensure_ascii=False)
            doc_query = """
            MERGE (d:Document {fileName: $fileName, kgName: $kgName})
            SET d.kgVersion = $kgVersion,
                d.kgName = $kgName,
                d.createdAt = datetime(),
                d.updatedAt = datetime(),
                d.totalChunks = $totalChunks,
                d.totalEntities = $totalEntities,
                d.totalRelationships = $totalRelationships,
                d.ontologyClasses = $ontologyClasses,
                d.ontologyRelationships = $ontologyRelationships,
                d.contentHash = $contentHash,
                d.schemaCard = $schemaCard,
                d.schemaVersion = $schemaVersion,
                d.schemaHash = $schemaHash,
                d.embeddingModel = $embeddingModel,
                d.provider = $provider,
                d.model = $model,
                d.maxChunks = $maxChunks
            """
            graph.query(doc_query, {
                "kgVersion": kg_version,
                "kgName": kg_name_value,
                "fileName": file_name,
                "totalChunks": kg['metadata']['total_chunks'],
                "totalEntities": kg['metadata']['total_entities'],
                "totalRelationships": kg['metadata']['total_relationships'],
                "ontologyClasses": kg['metadata']['ontology_classes'],
                "ontologyRelationships": kg['metadata']['ontology_relationships'],
                "contentHash": doc_hash or "",
                "schemaCard": schema_card_json,
                "schemaVersion": schema_card["schemaVersion"],
                "schemaHash": schema_card["schemaHash"],
                "embeddingModel": self.embedding_model,
                "provider": kg['metadata'].get('provider', 'openai'),
                "model": kg['metadata'].get('model', ''),
                "maxChunks": kg['metadata'].get('max_chunks_setting'),
            })

            # Store document-level metadata from source (e.g. CSV columns like SUBJECT_ID, HADM_ID)
            if doc_metadata:
                # Sanitise: Neo4j only accepts str/int/float/bool — drop NaN and convert the rest
                import math
                safe_meta = {
                    k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                    for k, v in doc_metadata.items()
                    if v is not None and not (isinstance(v, float) and math.isnan(v))
                }
                if safe_meta:
                    graph.query(
                        "MATCH (d:Document {fileName: $fileName, kgName: $kgName}) SET d += $meta",
                        {"fileName": file_name, "kgName": kg_name_value, "meta": safe_meta},
                    )

            # Create chunk nodes with embeddings.
            # Include kg_name in the hash so identical text in different KGs
            # gets distinct Chunk nodes and retrieval filters are not contaminated.
            for chunk in kg['chunks']:
                chunk_id = hashlib.sha1(f"{kg_name_value}:{file_name}:{chunk['position']}:{chunk['text']}".encode()).hexdigest()
                chunk_query = """
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text,
                    c.position = $position,
                    c.chunkLocalIndex = $chunk_local_index,
                    c.start_pos = $start_pos,
                    c.end_pos = $end_pos,
                    c.source = $source,
                    c.dataset = $dataset,
                    c.questionId = $question_id,
                    c.passageIndex = $passage_index,
                    c.embedding = $embedding
                """
                graph.query(chunk_query, {
                    "chunk_id": chunk_id,
                    "text": chunk['text'],
                    "position": chunk['position'],
                    "chunk_local_index": chunk.get('chunk_local_index', chunk.get('chunk_id', 0)),
                    "start_pos": chunk['start_pos'],
                    "end_pos": chunk['end_pos'],
                    "source": chunk.get('source'),
                    "dataset": chunk.get('dataset'),
                    "question_id": chunk.get('question_id'),
                    "passage_index": chunk.get('passage_index'),
                    "embedding": chunk.get('embedding')
                })

                # Link chunk to document
                chunk_doc_query = """
                MATCH (c:Chunk {id: $chunk_id})
                MATCH (d:Document {fileName: $fileName, kgName: $kgName})
                MERGE (c)-[:PART_OF]->(d)
                """
                graph.query(chunk_doc_query, {
                    "chunk_id": chunk_id,
                    "fileName": file_name,
                    "kgName": kg_name_value,
                })

                # Create retrieval-only subchunks sized for the embedding model.
                # These keep dense retrieval faithful without shrinking the
                # larger parent chunks used for KG extraction.
                retrieval_subchunks = self._build_retrieval_subchunks(
                    chunk,
                    parent_chunk_id=chunk_id,
                )
                for subchunk in retrieval_subchunks:
                    retrieval_chunk_query = """
                    MERGE (rc:RetrievalChunk {id: $retrieval_chunk_id})
                    SET rc.text = $text,
                        rc.position = $position,
                        rc.retrievalLocalIndex = $retrieval_local_index,
                        rc.parentChunkId = $parent_chunk_id,
                        rc.chunkLocalIndex = $chunk_local_index,
                        rc.start_pos = $start_pos,
                        rc.end_pos = $end_pos,
                        rc.source = $source,
                        rc.dataset = $dataset,
                        rc.questionId = $question_id,
                        rc.passageIndex = $passage_index,
                        rc.embedding = $embedding
                    """
                    graph.query(retrieval_chunk_query, {
                        "retrieval_chunk_id": subchunk["id"],
                        "text": subchunk["text"],
                        "position": subchunk["position"],
                        "retrieval_local_index": subchunk["retrieval_local_index"],
                        "parent_chunk_id": subchunk["parent_chunk_id"],
                        "chunk_local_index": subchunk["chunk_local_index"],
                        "start_pos": subchunk["start_pos"],
                        "end_pos": subchunk["end_pos"],
                        "source": subchunk.get("source"),
                        "dataset": subchunk.get("dataset"),
                        "question_id": subchunk.get("question_id"),
                        "passage_index": subchunk.get("passage_index"),
                        "embedding": subchunk.get("embedding"),
                    })
                    graph.query(
                        """
                        MATCH (rc:RetrievalChunk {id: $retrieval_chunk_id})
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (rc)-[:RETRIEVES_FROM]->(c)
                        """,
                        {
                            "retrieval_chunk_id": subchunk["id"],
                            "chunk_id": chunk_id,
                        },
                    )

            # Create entity nodes with embeddings and ontology-based labels.
            # MERGE on node id (kg-prefixed UUID) so each KG's entities are independent.
            # content_hash is stored as a property for reference but is NOT the merge key,
            # which previously caused relationship storage to silently fail: a second KG
            # build would find the existing node by content_hash but leave n.id pointing at
            # the first KG's prefix, so MATCH (source {id: "kg2_uuid"}) never matched.
            for node in kg['nodes']:
                # Generate content-based deduplication hash (stored as property, not merge key)
                original_id = node.get('properties', {}).get('original_id', node['id'])
                normalized_content = self._normalize_entity_text(original_id)
                content_hash = hashlib.md5(f"{node['label']}:{normalized_content}".encode()).hexdigest()
                node['content_hash'] = content_hash

            for node in kg['nodes']:
                properties = node.get('properties', {})
                entity_type = node['label']  # This is the ontology class (Disease, Treatment, etc.)
                # Sanitize entity type for Cypher label compatibility and validate against whitelist
                cypher_safe_entity_type = re.sub(r'[^A-Za-z0-9_]', '_', entity_type).strip('_') or 'Concept'
                # Validate: label must start with a letter, max 64 chars, no injection risk
                if not re.match(r'^[A-Za-z][A-Za-z0-9_]{0,63}$', cypher_safe_entity_type):
                    logging.warning(f"Unsafe entity type '{entity_type}' → falling back to 'Concept'")
                    cypher_safe_entity_type = 'Concept'
                # Blocklist structural/system labels that must not be applied to entity nodes
                _RESERVED_LABELS = {'Document', 'Chunk', 'Mention', 'Entity', '__Entity__',
                                    '__KGDocument__', 'Relationship', 'Node', 'Schema'}
                if cypher_safe_entity_type in _RESERVED_LABELS:
                    logging.warning(f"Entity type '{entity_type}' clashes with structural label — falling back to 'Concept'")
                    cypher_safe_entity_type = 'Concept'

                # Generate embedding for entity if it doesn't have one.
                # Embed the entity name only (not description): the entity_vector
                # index is used for entity→entity ANN matching at query time, where
                # the probe vector is also a short entity-mention string.  Including
                # description here shifts the embedding into description-semantic space,
                # making name-level similarity unreliable (HippoRAG embeds names only).
                entity_embedding = node.get('embedding')
                if entity_embedding is None:
                    entity_text = properties.get('name', node['id'])
                    try:
                        entity_embedding = self.embedding_function.embed_query(entity_text)
                    except Exception as e:
                        logging.warning(f"Failed to generate embedding for entity {node['id']}: {e}")
                        entity_embedding = None

                # MERGE on __Entity__ only (not the specific type label) so that the
                # unique constraint on __Entity__.id is respected even when the LLM assigns
                # a different type label on a re-run.  The specific label is added with SET
                # after the MERGE, which is idempotent on existing nodes.
                node_query = f"""
                MERGE (n:__Entity__ {{id: $id}})
                ON CREATE SET
                    n.name = $name,
                    n.type = $type,
                    n.description = $description,
                    n.embedding = $embedding,
                    n.ontology_class = $entity_type,
                    n.content_hash = $content_hash,
                    n.kgName = $kg_name,
                    n.all_names = $all_names,
                    n.original_ids = $original_ids,
                    n.created_at = datetime()
                ON MATCH SET
                    n.last_accessed = datetime(),
                    n.kgName = $kg_name,
                    n.type = $type,
                    n.ontology_class = $entity_type,
                    n.all_names = coalesce(n.all_names, []) + $all_names,
                    n.original_ids = coalesce(n.original_ids, []) + $original_ids
                SET n:{cypher_safe_entity_type}
                """
                graph.query(node_query, {
                    "id": node['id'],
                    "content_hash": node['content_hash'],
                    "kg_name": kg_name_value,
                    "name": properties.get('name', node['id']),
                    "type": node['label'],
                    "description": properties.get('description', ''),
                    "embedding": entity_embedding,
                    "entity_type": entity_type,
                    "all_names": list(set(properties.get('all_names', [node['id']]))),
                    "original_ids": list(set(properties.get('original_ids', [node['id']])))
                })

            # Build prefixed-UUID → human-readable name lookup for confidence verification.
            # kg['nodes'][i]['id'] is the prefixed UUID (e.g. "kg_abc_<uuid>")
            # kg['nodes'][i]['properties']['name'] is the actual entity name text.
            # rel['from'] / rel['to'] use the same prefixed-UUID format.
            _uuid_to_name = {
                _n['id']: (_n.get('properties', {}).get('name') or _n['id'])
                for _n in kg.get('nodes', [])
                if _n.get('id')
            }

            # Create relationships with improved error handling
            relationships_stored = 0
            relationship_store_failures = 0
            relationships_skipped_low_confidence = 0
            for idx, rel in enumerate(kg['relationships']):
                try:
                    # Filter out fields that are managed explicitly below to avoid duplicates:
                    # 'id' causes duplicate key issues; 'negated'/'condition'/'quantitative'
                    # are promoted to top-level properties_with_confidence below.
                    _exclude = {'id', 'negated', 'condition', 'quantitative'}
                    properties_filtered = {
                        k: v for k, v in rel.get('properties', {}).items()
                        if k not in _exclude
                    }

                    # Resolve UUIDs to entity names for evidence-grounded confidence check.
                    # Prefer explicit source_name/target_name properties; fall back to name lookup.
                    _src_id = rel.get('from') or rel.get('source', '')
                    _tgt_id = rel.get('to') or rel.get('target', '')
                    _src_node = next((n for n in kg.get('nodes', []) if n.get('id') == _src_id), {})
                    _tgt_node = next((n for n in kg.get('nodes', []) if n.get('id') == _tgt_id), {})
                    source_type = (
                        _src_node.get('label')
                        or (_src_node.get('properties') or {}).get('type')
                        or (_src_node.get('properties') or {}).get('ontology_class')
                    )
                    target_type = (
                        _tgt_node.get('label')
                        or (_tgt_node.get('properties') or {}).get('type')
                        or (_tgt_node.get('properties') or {}).get('ontology_class')
                    )

                    # Canonicalize relationship type against ontology using fuzzy matching
                    sanitized_rel_type = self._canonicalize_relationship_type(
                        rel['type'],
                        source_type=source_type,
                        target_type=target_type,
                    )

                    # Use all known surface forms for the entity so canonical-name
                    # mismatches (e.g. "TBK1 kinase" → "TBK1") don't falsely score 0.
                    _src_all_names = _src_node.get('properties', {}).get('all_names') or []
                    _tgt_all_names = _tgt_node.get('properties', {}).get('all_names') or []
                    source_name = (rel.get('properties', {}).get('source_name')
                                   or _uuid_to_name.get(_src_id)
                                   or _src_id)
                    target_name = (rel.get('properties', {}).get('target_name')
                                   or _uuid_to_name.get(_tgt_id)
                                   or _tgt_id)
                    verification_chunks = kg.get('chunks', [])
                    provenance_positions = rel.get('provenance_positions') or []
                    if provenance_positions:
                        scoped_chunks = [
                            chunk for chunk in kg.get('chunks', [])
                            if chunk.get('position') in provenance_positions
                        ]
                        if scoped_chunks:
                            verification_chunks = scoped_chunks

                    triple_confidence = self._verify_triple_confidence(
                        source_name,
                        target_name,
                        sanitized_rel_type,
                        verification_chunks,
                        source_aliases=_src_all_names,
                        target_aliases=_tgt_all_names,
                    )

                    # Reject only clear hallucinations: neither entity found anywhere in the document.
                    # Score 0.1 = neither entity present; 0.4+ = at least one entity grounded in text.
                    # Threshold just above 0.1 avoids discarding relationships where one entity
                    # is confirmed (score 0.4) or entity names have minor surface-form mismatches.
                    if triple_confidence < 0.15:
                        relationships_skipped_low_confidence += 1
                        logging.info(
                            "Skipping hallucinated relationship (confidence=%.2f): %s -[%s]-> %s",
                            triple_confidence, rel.get('from'), sanitized_rel_type, rel.get('to'),
                        )
                        continue

                    # Pull negation and qualifiers extracted by LLM
                    negated    = bool(rel.get('negated', False))
                    condition  = rel.get('properties', {}).get('condition') or None
                    quantitative = rel.get('properties', {}).get('quantitative') or None

                    properties_with_confidence = {
                        **properties_filtered,
                        "confidence": triple_confidence,
                        "negated": negated,
                    }
                    if condition:
                        properties_with_confidence["condition"] = condition
                    if quantitative:
                        properties_with_confidence["quantitative"] = quantitative

                    # Keep negated in the MERGE key so that opposite claims
                    # (A INHIBITS B negated=false vs true) do not overwrite each other.
                    # Optional qualifiers only participate when present; Neo4j forbids
                    # null property values inside MERGE patterns.
                    rel_query = self._build_relationship_merge_query(
                        sanitized_rel_type,
                        include_condition=condition is not None,
                        include_quantitative=quantitative is not None,
                    )

                    edge_provenance = self._relationship_local_provenance(
                        rel,
                        kg.get("chunks", []),
                    )

                    logging.info(
                        "Creating relationship %d/%d: %s -[%s%s]-> %s (confidence=%.2f)",
                        idx + 1, len(kg['relationships']),
                        rel.get('from'),
                        "NOT " if negated else "",
                        sanitized_rel_type,
                        rel.get('to'), triple_confidence,
                    )

                    graph.query(rel_query, {
                        "source_id": rel.get('from'),
                        "target_id": rel.get('to'),
                        "negated": negated,
                        "condition": condition,
                        "quantitative": quantitative,
                        "properties": properties_with_confidence,
                        "provenance_positions": edge_provenance["provenance_positions"],
                        "question_ids": edge_provenance["question_ids"],
                        "passage_keys": edge_provenance["passage_keys"],
                    })

                    # Create QUALIFIED_BY nodes for significant qualifiers so they
                    # can be traversed independently and appear in path strings.
                    for q_type, q_value in [("condition", condition), ("quantitative", quantitative)]:
                        if not q_value:
                            continue
                        try:
                            qual_id = hashlib.sha1(
                                f"{rel.get('from')}|{sanitized_rel_type}|{rel.get('to')}|{q_type}|{q_value}".encode()
                            ).hexdigest()
                            qual_query = f"""
                            MATCH (source:__Entity__ {{id: $source_id}})
                            MATCH (target:__Entity__ {{id: $target_id}})
                            MERGE (q:Qualifier {{id: $qual_id}})
                            SET q.type = $q_type, q.value = $q_value,
                                q.kgName = $kg_name
                            MERGE (source)-[:{sanitized_rel_type}_QUALIFIED {{negated: $negated}}]->(q)
                            MERGE (q)-[:QUALIFIES]->(target)
                            """
                            graph.query(qual_query, {
                                "source_id": rel.get('from'),
                                "target_id": rel.get('to'),
                                "qual_id":   qual_id,
                                "q_type":    q_type,
                                "q_value":   q_value,
                                "kg_name":   kg.get('metadata', {}).get('kg_name', ''),
                                "negated":   negated,
                            })
                        except Exception as _qe:
                            logging.debug("QUALIFIED_BY node creation failed (non-fatal): %s", _qe)

                    relationships_stored += 1

                except Exception as rel_error:
                    relationship_store_failures += 1
                    logging.error(f"Failed to store relationship {idx+1}: {rel} - Error: {rel_error}")
                    continue

            logging.info(f"Successfully stored {relationships_stored} out of {len(kg['relationships'])} relationships")

            extracted_relationships = len(kg['relationships'])
            attempted_relationships = max(
                0,
                extracted_relationships - relationships_skipped_low_confidence,
            )
            relationship_store_ratio = (
                relationships_stored / attempted_relationships
                if attempted_relationships
                else 1.0
            )
            kg.setdefault("metadata", {})
            kg["metadata"]["stored_relationships"] = relationships_stored
            kg["metadata"]["relationship_store_failures"] = relationship_store_failures
            kg["metadata"]["relationships_skipped_low_confidence"] = relationships_skipped_low_confidence
            kg["metadata"]["relationship_store_ratio"] = relationship_store_ratio

            graph.query(
                """
                MATCH (d:Document {fileName: $fileName, kgName: $kgName})
                SET d.totalRelationships = $storedRelationships,
                    d.extractedRelationships = $extractedRelationships,
                    d.relationshipStoreFailures = $relationshipStoreFailures,
                    d.relationshipsSkippedLowConfidence = $relationshipsSkippedLowConfidence
                """,
                {
                    "fileName": file_name,
                    "kgName": kg_name_value,
                    "storedRelationships": relationships_stored,
                    "extractedRelationships": extracted_relationships,
                    "relationshipStoreFailures": relationship_store_failures,
                    "relationshipsSkippedLowConfidence": relationships_skipped_low_confidence,
                },
            )

            if relationship_store_failures > 0:
                # Treat isolated write misses on large graphs as degraded-but-usable.
                # We still fail fast on tiny graphs or when the failure rate suggests
                # a systemic storage problem rather than a one-off edge issue.
                relationship_failure_ratio = (
                    relationship_store_failures / attempted_relationships
                    if attempted_relationships
                    else 1.0
                )
                fail_small_graph = attempted_relationships < 100
                fail_systemic = relationship_failure_ratio > 0.005
                fail_complete_loss = attempted_relationships > 0 and relationships_stored == 0
                should_fail_build = fail_small_graph or fail_systemic or fail_complete_loss

                log_fn = logging.error if should_fail_build else logging.warning
                log_fn(
                    "Relationship storage incomplete for %s: stored=%d attempted=%d skipped_low_confidence=%d failures=%d failure_ratio=%.4f fatal=%s",
                    file_name,
                    relationships_stored,
                    attempted_relationships,
                    relationships_skipped_low_confidence,
                    relationship_store_failures,
                    relationship_failure_ratio,
                    should_fail_build,
                )
                kg["metadata"]["relationship_store_failure_ratio"] = relationship_failure_ratio
                kg["metadata"]["relationship_store_degraded"] = not should_fail_build
                if should_fail_build:
                    return False

            # Link entities to chunks via per-fact provenance Mention nodes
            # Pattern: (Entity)-[:MENTIONED_IN]->(Mention {quote, ...})-[:FROM_CHUNK]->(Chunk)
            def _mention_boundary(name: str) -> re.Pattern:
                """Adaptive boundary pattern — handles names ending in non-word chars like '(' or '-'."""
                prefix = r'(?<!\w)' if not name[:1].isalnum() and name[:1] != '_' else r'\b'
                suffix = r'(?!\w)' if not name[-1:].isalnum() and name[-1:] != '_' else r'\b'
                return re.compile(prefix + re.escape(name) + suffix)

            for chunk in kg['chunks']:
                # Must use the same kg-scoped hash as the chunk CREATE step above.
                chunk_id = hashlib.sha1(f"{kg_name_value}:{file_name}:{chunk['position']}:{chunk['text']}".encode()).hexdigest()
                chunk_text_lower = chunk['text'].lower()

                for node in kg['nodes']:
                    properties = node.get('properties', {})
                    candidate_names = []
                    candidate_names.extend(properties.get('all_names', []) if isinstance(properties.get('all_names', []), list) else [])
                    candidate_names.append(properties.get('name', ''))
                    candidate_names.append(properties.get('original_id', ''))

                    # Keep meaningful normalized names only
                    normalized_names = [n.strip().lower() for n in candidate_names if isinstance(n, str) and len(n.strip()) > 2]
                    matched_name = next(
                        (n for n in normalized_names if _mention_boundary(n).search(chunk_text_lower)),
                        None
                    )
                    if matched_name:
                        # Extract a short quote: the sentence containing the matched name
                        sentences = re.split(r'(?<=[.!?])\s+', chunk['text'])
                        quote = next(
                            (s.strip() for s in sentences if matched_name in s.lower()),
                            chunk['text'][:200]
                        )[:500]  # cap at 500 chars
                        # Seed mention_id from node['id'] (the unique key), not content_hash.
                        # content_hash is no longer unique across KGs (Fix 8 changed MERGE to id).
                        mention_id = hashlib.sha256(
                            f"{node['id']}::{chunk_id}".encode()
                        ).hexdigest()
                        mention_query = """
                        MATCH (c:Chunk {id: $chunk_id})
                        MATCH (e:__Entity__ {id: $entity_id})
                        MERGE (m:Mention {id: $mention_id})
                        SET m.quote = $quote,
                            m.chunkIndex = $chunk_index,
                            m.chunkLocalIndex = $chunk_local_index,
                            m.chunkStart = $chunk_start,
                            m.chunkEnd = $chunk_end,
                            m.chunkSource = $chunk_source,
                            m.entityName = $entity_name,
                            m.createdAt = datetime()
                        MERGE (e)-[:MENTIONED_IN]->(m)
                        MERGE (m)-[:FROM_CHUNK]->(c)
                        MERGE (c)-[:HAS_ENTITY]->(e)
                        """
                        graph.query(mention_query, {
                            "chunk_id": chunk_id,
                            "entity_id": node['id'],
                            "mention_id": mention_id,
                            "quote": quote,
                            "chunk_index": chunk.get('position', chunk.get('chunk_id', 0)),
                            "chunk_local_index": chunk.get('chunk_local_index', chunk.get('chunk_id', 0)),
                            "chunk_start": chunk.get('start_pos', 0),
                            "chunk_end": chunk.get('end_pos', 0),
                            "chunk_source": chunk.get('source', ''),
                            "entity_name": properties.get('name', ''),
                        })

            # Create vector indexes for RAG
            self._create_vector_indexes(graph)

            logging.info(f"Successfully stored ontology-guided knowledge graph for {file_name}")
            return True

        except Exception as e:
            logging.error(f"Error storing knowledge graph: {e}")
            return False

    def _create_vector_indexes(self, graph):
        """
        Create vector indexes and unique constraints for RAG functionality
        """
        try:
            # Create unique constraint for entity IDs to prevent duplicates
            entity_constraint_query = """
            CREATE CONSTRAINT unique_entity_id IF NOT EXISTS
            FOR (e:__Entity__) REQUIRE e.id IS UNIQUE
            """
            graph.query(entity_constraint_query)

            # Create unique constraint for chunk IDs
            chunk_constraint_query = """
            CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS
            FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """
            graph.query(chunk_constraint_query)

            retrieval_chunk_constraint_query = """
            CREATE CONSTRAINT unique_retrieval_chunk_id IF NOT EXISTS
            FOR (rc:RetrievalChunk) REQUIRE rc.id IS UNIQUE
            """
            graph.query(retrieval_chunk_constraint_query)

            # Create composite uniqueness for dataset-scoped documents
            # (same fileName may exist in different kgName datasets)
            doc_constraint_query = """
            CREATE CONSTRAINT unique_document_filename_kgname IF NOT EXISTS
            FOR (d:Document) REQUIRE (d.fileName, d.kgName) IS UNIQUE
            """
            graph.query(doc_constraint_query)

            # Unique constraint for Mention nodes (entity × chunk pair)
            mention_constraint_query = """
            CREATE CONSTRAINT unique_mention_id IF NOT EXISTS
            FOR (m:Mention) REQUIRE m.id IS UNIQUE
            """
            graph.query(mention_constraint_query)

            # Create vector index for chunks
            chunk_index_query = f"""
            CREATE VECTOR INDEX vector IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            graph.query(chunk_index_query)

            retrieval_chunk_index_query = f"""
            CREATE VECTOR INDEX retrieval_vector IF NOT EXISTS
            FOR (rc:RetrievalChunk) ON (rc.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            graph.query(retrieval_chunk_index_query)

            # Create vector index for entities
            entity_index_query = f"""
            CREATE VECTOR INDEX entity_vector IF NOT EXISTS
            FOR (e:__Entity__) ON (e.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            graph.query(entity_index_query)

            # Create keyword index for full-text search
            keyword_index_query = """
            CREATE FULLTEXT INDEX keyword IF NOT EXISTS
            FOR (c:Chunk) ON EACH [c.text]
            """
            graph.query(keyword_index_query)

            retrieval_keyword_index_query = """
            CREATE FULLTEXT INDEX retrieval_keyword IF NOT EXISTS
            FOR (rc:RetrievalChunk) ON EACH [rc.text]
            """
            graph.query(retrieval_keyword_index_query)

            # Index on entity name for fast text-matching and multi-hop traversal
            # lookups (EnhancedRAGSystem._expand_entities_via_graph seeds from e.id)
            entity_id_index_query = """
            CREATE INDEX entity_id_index IF NOT EXISTS
            FOR (e:__Entity__) ON (e.id)
            """
            graph.query(entity_id_index_query)

            entity_name_index_query = """
            CREATE INDEX entity_name_index IF NOT EXISTS
            FOR (e:__Entity__) ON (e.name)
            """
            graph.query(entity_name_index_query)

            # Composite index for chunk→document lookup used in kg_name filtering
            chunk_kg_index_query = """
            CREATE INDEX chunk_document_index IF NOT EXISTS
            FOR ()-[r:PART_OF]-() ON (r)
            """
            try:
                graph.query(chunk_kg_index_query)
            except Exception:
                pass  # Relationship indexes not supported on all Neo4j versions

            logging.info("Created constraints, vector, keyword, and entity lookup indexes for RAG")

        except Exception as e:
            logging.warning(f"Error creating constraints/indexes (may already exist): {e}")

    def get_rag_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Get RAG context for a query using vector similarity search
        """
        try:
            graph = self._create_neo4j_connection()

            # Generate query embedding
            query_embedding = self.embedding_function.embed_query(query)

            # Vector search query
            search_query = """
            CALL db.index.vector.queryNodes('vector', $top_k, $query_vector)
            YIELD node AS chunk, score
            MATCH (chunk)-[:PART_OF]->(d:Document)
            OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
            WITH chunk, score, d, collect(e) AS entities
            RETURN
                chunk.text AS text,
                chunk.id AS chunk_id,
                score,
                d.fileName AS document,
                [entity IN entities | {id: entity.id, type: entity.type, description: entity.description}] AS entities
            ORDER BY score DESC
            """

            results = graph.query(search_query, {
                "top_k": top_k,
                "query_vector": query_embedding
            })

            context = {
                "query": query,
                "chunks": [],
                "entities": set(),
                "documents": set()
            }

            for result in results:
                context["chunks"].append({
                    "text": result["text"],
                    "chunk_id": result["chunk_id"],
                    "score": result["score"],
                    "document": result["document"],
                    "entities": result["entities"]
                })

                context["documents"].add(result["document"])
                for entity in result["entities"]:
                    context["entities"].add(entity["id"])

            context["entities"] = list(context["entities"])
            context["documents"] = list(context["documents"])

            return context

        except Exception as e:
            logging.error(f"Error getting RAG context: {e}")
            return {"query": query, "chunks": [], "entities": [], "documents": []}
