import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ontographrag.api.runtime_helpers import (
    build_readiness_report,
    configured_provider_names_from_env,
    filesystem_write_probe_ok,
    guardrail_forces_abstention,
    parse_request_timeout_seconds,
)


def test_parse_request_timeout_seconds_clamps_values():
    assert parse_request_timeout_seconds("not-a-number") == 120
    assert parse_request_timeout_seconds("5") == 15
    assert parse_request_timeout_seconds("900") == 300
    assert parse_request_timeout_seconds("60") == 60


def test_configured_provider_names_from_env_detects_keys():
    providers = configured_provider_names_from_env(
        {
            "OPENAI_API_KEY": "x",
            "OPENROUTER_API_KEY": "y",
            "OLLAMA_HOST": "http://localhost:11434",
        }
    )
    assert providers == ["openai", "openrouter", "ollama"]


def test_guardrail_forces_abstention_only_on_non_keep_verdicts():
    assert guardrail_forces_abstention({"enabled": True, "final_decision": "abstain"}) is True
    assert guardrail_forces_abstention({"enabled": True, "final_decision": "retry_keep"}) is False
    assert guardrail_forces_abstention({"enabled": False, "final_decision": "abstain"}) is False
    assert guardrail_forces_abstention(None) is False


def test_filesystem_write_probe_ok(tmp_path):
    ok, detail = filesystem_write_probe_ok(tmp_path)
    assert ok is True
    assert "Can write to" in detail


def test_build_readiness_report_flags_missing_dependencies():
    report = build_readiness_report(
        neo4j_ready=True,
        rag_runtime_ready=False,
        write_probe_ok=True,
        write_probe_detail="ok",
        configured_providers=[],
    )
    assert report["ready"] is False
    assert report["status"] == "not_ready"
