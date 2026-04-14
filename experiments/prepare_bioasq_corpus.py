"""
Prepare a shared BioASQ retrieval corpus from PubMed.

This converts BioASQ's question-local gold file (which only contains question
metadata, PubMed URLs, and expert snippets) into a shared abstract corpus that
can be indexed fairly by both vanilla RAG and KG-RAG.

Output format: JSONL with one record per PMID, e.g.
    {"pmid": "34022131", "title": "...", "abstract": "..."}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from xml.etree import ElementTree as ET

import requests


logger = logging.getLogger(__name__)

NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_BIOASQ_PATH = Path("MIRAGE/rawdata/bioasq/Task10BGoldenEnriched/10B1_golden.json")
DEFAULT_OUTPUT_PATH = Path("MIRAGE/rawdata/bioasq/pubmed_abstracts.jsonl")
PMID_RE = re.compile(r"/pubmed/(\d+)")


def extract_pmid_from_url(url: str) -> Optional[str]:
    """Extract a PMID from a BioASQ PubMed URL."""
    text = str(url or "").strip()
    match = PMID_RE.search(text)
    if match:
        return match.group(1)
    if text.isdigit():
        return text
    return None


def load_bioasq_pmids(bioasq_path: Path) -> List[str]:
    """Collect unique PMIDs referenced by a BioASQ golden file."""
    data = json.loads(bioasq_path.read_text())
    questions = data.get("questions", []) if isinstance(data, dict) else []

    pmids: Set[str] = set()
    for question in questions:
        if not isinstance(question, dict):
            continue
        for url in question.get("documents", []) or []:
            pmid = extract_pmid_from_url(str(url))
            if pmid:
                pmids.add(pmid)
        for snippet in question.get("snippets", []) or []:
            if not isinstance(snippet, dict):
                continue
            pmid = extract_pmid_from_url(str(snippet.get("document", "")))
            if pmid:
                pmids.add(pmid)

    return sorted(pmids, key=int)


def _stringify_xml_text(elem: Optional[ET.Element]) -> str:
    """Flatten an XML element into readable text."""
    if elem is None:
        return ""
    text = "".join(elem.itertext())
    return re.sub(r"\s+", " ", text).strip()


def parse_pubmed_article_xml(xml_text: str) -> Dict[str, Dict[str, str]]:
    """Parse an EFetch PubMedArticleSet XML response into PMID -> record."""
    root = ET.fromstring(xml_text)
    records: Dict[str, Dict[str, str]] = {}

    for article in root.findall(".//PubmedArticle"):
        pmid = _stringify_xml_text(article.find(".//MedlineCitation/PMID"))
        if not pmid:
            continue

        article_title = _stringify_xml_text(article.find(".//Article/ArticleTitle"))
        abstract_nodes = article.findall(".//Article/Abstract/AbstractText")
        abstract_parts: List[str] = []
        for node in abstract_nodes:
            label = str(node.attrib.get("Label", "")).strip()
            part = _stringify_xml_text(node)
            if not part:
                continue
            if label:
                abstract_parts.append(f"{label}: {part}")
            else:
                abstract_parts.append(part)
        abstract = "\n".join(abstract_parts).strip()

        records[pmid] = {
            "pmid": pmid,
            "title": article_title,
            "abstract": abstract,
        }

    return records


def fetch_pubmed_batch(
    pmids: Sequence[str],
    *,
    email: Optional[str] = None,
    api_key: Optional[str] = None,
    tool: str = "ontographrag",
    timeout_seconds: float = 60.0,
) -> Dict[str, Dict[str, str]]:
    """Fetch one PMID batch from PubMed via EFetch XML."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": tool,
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    response = requests.get(
        f"{NCBI_EUTILS_BASE}/efetch.fcgi",
        params=params,
        timeout=timeout_seconds,
        headers={"User-Agent": f"{tool}/1.0"},
    )
    response.raise_for_status()
    return parse_pubmed_article_xml(response.text)


def load_existing_corpus(output_path: Path) -> Dict[str, Dict[str, str]]:
    """Load an existing JSONL corpus file so reruns can resume incrementally."""
    existing: Dict[str, Dict[str, str]] = {}
    if not output_path.exists():
        return existing

    with output_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            pmid = str(record.get("pmid", "")).strip()
            if pmid:
                existing[pmid] = record
    return existing


def _chunked(values: Sequence[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(values), size):
        yield list(values[idx: idx + size])


def build_bioasq_shared_corpus(
    *,
    bioasq_path: Path = DEFAULT_BIOASQ_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    email: Optional[str] = None,
    api_key: Optional[str] = None,
    batch_size: int = 100,
    sleep_seconds: float = 0.34,
    max_pmids: Optional[int] = None,
    overwrite: bool = False,
) -> Tuple[int, int]:
    """
    Build a shared BioASQ abstract corpus.

    Returns:
        (num_records_written, num_missing_records)
    """
    if not bioasq_path.exists():
        raise FileNotFoundError(f"BioASQ file not found: {bioasq_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pmids = load_bioasq_pmids(bioasq_path)
    if max_pmids is not None:
        pmids = pmids[:max_pmids]

    existing = {} if overwrite else load_existing_corpus(output_path)
    pending = [pmid for pmid in pmids if pmid not in existing]

    logger.info(
        "BioASQ corpus prep: %d PMIDs total, %d already cached, %d pending",
        len(pmids),
        len(existing),
        len(pending),
    )

    fetched: Dict[str, Dict[str, str]] = dict(existing)
    missing: Set[str] = set()

    for batch_num, batch in enumerate(_chunked(pending, batch_size), start=1):
        logger.info("Fetching PubMed batch %d (%d PMIDs)", batch_num, len(batch))
        try:
            records = fetch_pubmed_batch(batch, email=email, api_key=api_key)
        except Exception as e:
            logger.error("Failed to fetch batch %d: %s", batch_num, e)
            raise

        fetched.update(records)
        batch_missing = set(batch) - set(records.keys())
        if batch_missing:
            missing.update(batch_missing)
            logger.warning(
                "Batch %d returned no abstract/title for %d PMIDs",
                batch_num,
                len(batch_missing),
            )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    ordered_records: List[Dict[str, str]] = []
    for pmid in pmids:
        record = fetched.get(pmid)
        if not record:
            missing.add(pmid)
            continue
        ordered_records.append(record)

    with output_path.open("w") as f:
        for record in ordered_records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    logger.info(
        "Wrote %d BioASQ abstract records to %s (%d missing)",
        len(ordered_records),
        output_path,
        len(missing),
    )
    return len(ordered_records), len(missing)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a shared PubMed abstract corpus for BioASQ retrieval benchmarking."
    )
    parser.add_argument(
        "--bioasq-path",
        type=Path,
        default=DEFAULT_BIOASQ_PATH,
        help="Path to the BioASQ golden JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path for fetched PubMed abstracts.",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=os.getenv("NCBI_EMAIL") or os.getenv("ENTREZ_EMAIL"),
        help="Contact email sent to NCBI E-utilities.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("NCBI_API_KEY"),
        help="Optional NCBI API key for higher rate limits.",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="PMIDs to fetch per EFetch request.")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.34,
        help="Delay between EFetch batches to stay polite to NCBI.",
    )
    parser.add_argument(
        "--max-pmids",
        type=int,
        default=None,
        help="Optional cap for smoke-testing corpus preparation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Ignore any existing output JSONL and rebuild from scratch.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable info-level logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    build_bioasq_shared_corpus(
        bioasq_path=args.bioasq_path,
        output_path=args.output,
        email=args.email,
        api_key=args.api_key,
        batch_size=max(1, args.batch_size),
        sleep_seconds=max(0.0, args.sleep_seconds),
        max_pmids=args.max_pmids,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
