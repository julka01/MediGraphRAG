"""Small production-facing helpers for API runtime behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def parse_request_timeout_seconds(
    raw_value: Any,
    *,
    default: int = 120,
    minimum: int = 15,
    maximum: int = 300,
) -> int:
    """Parse and clamp a request timeout from config."""
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def configured_provider_names_from_env(env: Mapping[str, str]) -> list[str]:
    """Return model providers that appear configured from environment variables."""
    providers: list[str] = []
    if env.get("OPENAI_API_KEY"):
        providers.append("openai")
    if env.get("OPENROUTER_API_KEY"):
        providers.append("openrouter")
    if env.get("GEMINI_API_KEY"):
        providers.append("gemini")
    if env.get("DEEPSEEK_API_KEY"):
        providers.append("deepseek")
    if env.get("OLLAMA_HOST") or env.get("OLLAMA_BASE_URL"):
        providers.append("ollama")
    return providers


def guardrail_forces_abstention(guardrail: Mapping[str, Any] | None) -> bool:
    """True when a runtime guardrail verdict means the response should be treated as abstained."""
    if not guardrail:
        return False
    final_decision = str(guardrail.get("final_decision", "")).strip().lower()
    return bool(guardrail.get("enabled")) and final_decision not in {"keep", "retry_keep"}


def filesystem_write_probe_ok(probe_root: Path) -> tuple[bool, str]:
    """Check whether the process can create and delete a small file under probe_root."""
    try:
        probe_root.mkdir(parents=True, exist_ok=True)
        probe_file = probe_root / ".ontographrag_write_probe"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink(missing_ok=True)
        return True, f"Can write to {probe_root}"
    except Exception as e:  # pragma: no cover - platform-specific exception shapes
        return False, str(e)


def build_readiness_report(
    *,
    neo4j_ready: bool,
    rag_runtime_ready: bool,
    write_probe_ok: bool,
    write_probe_detail: str,
    configured_providers: Sequence[str],
) -> Dict[str, Any]:
    """Create a compact readiness payload for fast infrastructure checks."""
    checks = [
        {"name": "neo4j", "ok": bool(neo4j_ready)},
        {"name": "rag_runtime", "ok": bool(rag_runtime_ready)},
        {
            "name": "write_permissions",
            "ok": bool(write_probe_ok),
            "detail": write_probe_detail,
        },
        {
            "name": "model_provider",
            "ok": len(configured_providers) > 0,
            "detail": list(configured_providers),
        },
    ]
    ready = all(check["ok"] for check in checks)
    return {
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "checks": checks,
    }
