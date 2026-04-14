from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import importlib.util
import unicodedata
from pathlib import Path
from typing import Optional

import httpx
import typer


app = typer.Typer(
    help="OntographRAG CLI — ontology-guided ingest, graph exploration, Q&A, and evaluation.",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)
explore_app = typer.Typer(help="Inspect named knowledge graphs exposed by a running OntographRAG server.")
app.add_typer(explore_app, name="explore")


def _base_url(server: str) -> str:
    return server.rstrip("/")


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    detail = response.text
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("message") or detail
    except Exception:
        pass
    raise typer.BadParameter(f"{response.status_code} {response.reason_phrase}: {detail}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_experiment_script() -> Optional[Path]:
    candidates = [
        _repo_root() / "experiments" / "experiment.py",
        Path(sys.prefix) / "experiments" / "experiment.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    return api_key or os.getenv("ONTOGRAPHRAG_API_KEY")


def _request_kwargs(api_key: Optional[str]) -> dict:
    resolved = _resolve_api_key(api_key)
    if not resolved:
        return {}
    return {"headers": {"X-API-Key": resolved}}


def _normalize_probe_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower().strip()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host interface to bind."),
    port: int = typer.Option(8004, help="Port for the API and GUI."),
    reload: bool = typer.Option(False, help="Enable auto-reload for development."),
) -> None:
    """Serve the web app and REST API."""
    import uvicorn

    uvicorn.run("ontographrag.api.app:app", host=host, port=port, reload=reload)


@app.command()
def doctor(
    probe_models: bool = typer.Option(False, help="Actively probe configured model providers."),
    write_probe_dir: Optional[Path] = typer.Option(None, exists=False, file_okay=False, dir_okay=True, help="Directory used for the write-permission probe."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Run a readiness check for Neo4j, embeddings, OCR, ontology parsing, and write access."""
    from ontographrag.api.app import run_doctor_checks

    report = asyncio.run(
        run_doctor_checks(
            probe_models=probe_models,
            write_probe_dir=str(write_probe_dir) if write_probe_dir else None,
        )
    )
    if json_output:
        typer.echo(json.dumps(report, indent=2))
    else:
        typer.echo(f"OntographRAG doctor: {report['status'].upper()}")
        typer.echo(
            f"Checks: {report['summary']['ok']} ok, "
            f"{report['summary']['warn']} warn, "
            f"{report['summary']['fail']} fail"
        )
        for check in report["checks"]:
            typer.echo(f"- {check['check']}: {check['status']} — {check['detail']}")
    if report["status"] == "fail":
        raise typer.Exit(code=1)


@app.command()
def ingest(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Document to ingest."),
    server: str = typer.Option("http://127.0.0.1:8004", help="OntographRAG server base URL."),
    kg_name: Optional[str] = typer.Option(None, help="Name for the resulting knowledge graph."),
    ontology: Optional[Path] = typer.Option(None, exists=True, dir_okay=False, readable=True, help="Optional ontology (.json/.owl)."),
    provider: str = typer.Option("openai", help="KG extraction provider."),
    model: str = typer.Option("gpt-4o-mini", help="KG extraction model."),
    embedding_model: str = typer.Option("sentence_transformers", help="Embedding backend."),
    max_chunks: int = typer.Option(20, min=1, help="Max chunks to process."),
    enable_coreference: bool = typer.Option(False, help="Enable cross-chunk coreference resolution."),
    api_key: Optional[str] = typer.Option(None, envvar="ONTOGRAPHRAG_API_KEY", help="API key for protected OntographRAG servers."),
) -> None:
    """Ingest a document into a named knowledge graph through a running server."""
    url = f"{_base_url(server)}/create_ontology_guided_kg"
    data = {
        "provider": provider,
        "model": model,
        "embedding_model": embedding_model,
        "max_chunks": str(max_chunks),
        "enable_coreference_resolution": "true" if enable_coreference else "false",
    }
    if kg_name:
        data["kg_name"] = kg_name

    files = {
        "file": (file.name, file.read_bytes(), "application/octet-stream"),
    }
    if ontology:
        files["ontology_file"] = (ontology.name, ontology.read_bytes(), "application/octet-stream")

    with httpx.Client(timeout=300.0) as client:
        response = client.post(url, data=data, files=files, **_request_kwargs(api_key))
    _raise_for_status(response)
    payload = response.json()
    graph = payload.get("graph_data") or {}
    typer.echo(f"KG name: {payload.get('kg_name') or '(unnamed)'}")
    typer.echo(f"Nodes: {len(graph.get('nodes', []))}")
    typer.echo(f"Relationships: {len(graph.get('relationships', []))}")
    typer.echo("Ingest complete.")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask against the active KG."),
    server: str = typer.Option("http://127.0.0.1:8004", help="OntographRAG server base URL."),
    kg_name: Optional[str] = typer.Option(None, help="Restrict retrieval to a named KG."),
    provider: str = typer.Option("openai", help="RAG provider."),
    model: str = typer.Option("gpt-4o-mini", help="RAG model."),
    json_output: bool = typer.Option(False, "--json", help="Emit full JSON response."),
    api_key: Optional[str] = typer.Option(None, envvar="ONTOGRAPHRAG_API_KEY", help="API key for protected OntographRAG servers."),
) -> None:
    """Ask a grounded question against a running OntographRAG server."""
    payload = {
        "question": question,
        "provider_rag": provider,
        "model_rag": model,
    }
    if kg_name:
        payload["kg_name"] = kg_name

    with httpx.Client(timeout=180.0) as client:
        response = client.post(f"{_base_url(server)}/chat", json=payload, **_request_kwargs(api_key))
    _raise_for_status(response)
    result = response.json()
    if json_output:
        typer.echo(json.dumps(result, indent=2))
        return

    typer.echo(result.get("response") or result.get("message") or "")
    info = result.get("info") or {}
    kg_confidence = info.get("kg_confidence")
    if kg_confidence is not None:
        typer.echo(f"\nKG confidence: {kg_confidence:.3f}")
    used_entities = info.get("entities", {}).get("used_entities") or []
    if used_entities:
        names = [entity.get("description") or entity.get("id") for entity in used_entities[:10]]
        typer.echo("Sources: " + ", ".join(filter(None, names)))


@app.command("runtime-regression")
def runtime_regression(
    server: str = typer.Option("http://127.0.0.1:8004", help="OntographRAG server base URL."),
    provider: str = typer.Option("openrouter", help="RAG provider for the regression pass."),
    model: str = typer.Option("openai/gpt-4o-mini", help="RAG model for the regression pass."),
    cases_file: Path = typer.Option(
        _repo_root() / "experiments" / "runtime_regression_cases.json",
        exists=True,
        dir_okay=False,
        readable=True,
        help="JSON file containing curated runtime regression cases.",
    ),
    api_key: Optional[str] = typer.Option(None, envvar="ONTOGRAPHRAG_API_KEY", help="API key for protected OntographRAG servers."),
    stop_on_failure: bool = typer.Option(False, help="Stop after the first failing case."),
    timeout_seconds: float = typer.Option(180.0, min=10.0, help="Per-request timeout."),
) -> None:
    """Run curated live regression questions against a running server."""
    cases = json.loads(cases_file.read_text())
    if not isinstance(cases, list) or not cases:
        raise typer.BadParameter("Regression cases file must contain a non-empty JSON array.")

    failures = 0
    with httpx.Client(timeout=timeout_seconds) as client:
        for idx, case in enumerate(cases, start=1):
            question = str(case.get("question", "")).strip()
            if not question:
                raise typer.BadParameter(f"Case {idx} is missing a question.")
            payload = {
                "question": question,
                "provider_rag": provider,
                "model_rag": model,
                "runtime_guardrail": True,
            }
            if case.get("kg_name"):
                payload["kg_name"] = case["kg_name"]

            response = client.post(
                f"{_base_url(server)}/chat",
                json=payload,
                **_request_kwargs(api_key),
            )
            _raise_for_status(response)
            result = response.json()
            answer = result.get("message") or result.get("response") or ""
            info = result.get("info") or {}
            guardrail = info.get("guardrail") or {}

            expected_any = [str(item) for item in (case.get("expected_any") or []) if str(item).strip()]
            normalized_answer = _normalize_probe_text(answer)
            matched = any(_normalize_probe_text(expected) in normalized_answer for expected in expected_any)

            if matched:
                typer.echo(f"PASS [{idx}/{len(cases)}] {case.get('id', f'case-{idx}')}: {question}")
                continue

            failures += 1
            typer.echo(f"FAIL [{idx}/{len(cases)}] {case.get('id', f'case-{idx}')}: {question}")
            typer.echo(f"  Expected any of: {expected_any}")
            typer.echo(f"  Got: {answer}")
            if guardrail:
                typer.echo(f"  Guardrail: {guardrail.get('final_decision')}")
            if stop_on_failure:
                raise typer.Exit(code=1)

    if failures:
        typer.echo(f"\nRuntime regression suite failed: {failures}/{len(cases)} cases.")
        raise typer.Exit(code=1)
    typer.echo(f"\nRuntime regression suite passed: {len(cases)}/{len(cases)} cases.")


@explore_app.command("list")
def explore_list(
    server: str = typer.Option("http://127.0.0.1:8004", help="OntographRAG server base URL."),
    api_key: Optional[str] = typer.Option(None, envvar="ONTOGRAPHRAG_API_KEY", help="API key for protected OntographRAG servers."),
) -> None:
    """List known KGs on the server."""
    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"{_base_url(server)}/kg/list", **_request_kwargs(api_key))
    _raise_for_status(response)
    payload = response.json()
    for kg in payload.get("kgs", []):
        typer.echo(f"- {kg.get('kg_name') or kg.get('name')}: {kg.get('documentCount', '?')} documents")


@explore_app.command("show")
def explore_show(
    kg_name: str = typer.Argument(..., help="Knowledge graph name."),
    server: str = typer.Option("http://127.0.0.1:8004", help="OntographRAG server base URL."),
    json_output: bool = typer.Option(False, "--json", help="Emit full JSON payload."),
    api_key: Optional[str] = typer.Option(None, envvar="ONTOGRAPHRAG_API_KEY", help="API key for protected OntographRAG servers."),
) -> None:
    """Show summary information for a named KG."""
    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"{_base_url(server)}/kg/{kg_name}", **_request_kwargs(api_key))
    _raise_for_status(response)
    payload = response.json()
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    stats = payload.get("stats") or {}
    typer.echo(f"KG: {kg_name}")
    for key, value in stats.items():
        typer.echo(f"- {key}: {value}")


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def evaluate(ctx: typer.Context) -> None:
    """Run the benchmark pipeline with the existing experiment runner."""
    if importlib.util.find_spec("experiments.experiment") is not None:
        raise typer.Exit(code=subprocess.call([sys.executable, "-m", "experiments.experiment", *ctx.args]))
    script = _find_experiment_script()
    if script is None:
        raise typer.BadParameter(
            "Could not find experiments/experiment.py. Use a source checkout for the evaluate workflow."
        )
    cmd = [sys.executable, str(script), *ctx.args]
    raise typer.Exit(code=subprocess.call(cmd))


@app.command("prepare-bioasq-corpus")
def prepare_bioasq_corpus(
    bioasq_path: Path = typer.Option(
        Path("MIRAGE/rawdata/bioasq/Task10BGoldenEnriched/10B1_golden.json"),
        exists=True,
        dir_okay=False,
        readable=True,
        help="BioASQ golden JSON file containing PubMed document URLs.",
    ),
    output: Path = typer.Option(
        Path("MIRAGE/rawdata/bioasq/pubmed_abstracts.jsonl"),
        file_okay=True,
        dir_okay=False,
        help="Output JSONL path for the shared PubMed abstract corpus.",
    ),
    email: Optional[str] = typer.Option(
        None,
        help="Contact email to send with NCBI E-utilities requests (recommended).",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        help="Optional NCBI API key for higher E-utilities rate limits.",
    ),
    batch_size: int = typer.Option(100, min=1, help="PMIDs per EFetch batch."),
    sleep_seconds: float = typer.Option(
        0.34,
        min=0.0,
        help="Delay between EFetch batches to stay polite to NCBI.",
    ),
    max_pmids: Optional[int] = typer.Option(
        None,
        min=1,
        help="Optional cap for smoke-testing corpus preparation.",
    ),
    overwrite: bool = typer.Option(
        False,
        help="Ignore an existing output JSONL and rebuild the corpus from scratch.",
    ),
) -> None:
    """Build a fair shared BioASQ retrieval corpus from PubMed abstracts."""
    from experiments.prepare_bioasq_corpus import build_bioasq_shared_corpus

    written, missing = build_bioasq_shared_corpus(
        bioasq_path=bioasq_path,
        output_path=output,
        email=email,
        api_key=api_key,
        batch_size=batch_size,
        sleep_seconds=sleep_seconds,
        max_pmids=max_pmids,
        overwrite=overwrite,
    )
    typer.echo(f"Wrote {written} BioASQ abstract records to {output}")
    if missing:
        typer.echo(f"Missing abstracts/titles for {missing} PMIDs")


def main() -> None:
    app()
