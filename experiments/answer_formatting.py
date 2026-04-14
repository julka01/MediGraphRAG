"""Task-aware answer formatting instructions for experiment-time generation."""

import re
from typing import Dict, Iterable, Optional

SHORT_ANSWER_DATASETS = {
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "multihoprag",
}


def _extract_leading_label(response: str, labels: Iterable[str]) -> str:
    """Return a normalized leading decision label when one is explicitly present."""
    text = str(response or "").strip().lower()
    if not text:
        return ""

    allowed = [str(label).strip().lower() for label in labels if str(label).strip()]
    if not allowed:
        return ""

    # Exact single-token label.
    if text in allowed:
        return text

    # Leading label followed by punctuation or explanation.
    pattern = r"^\s*(" + "|".join(re.escape(label) for label in allowed) + r")\b"
    match = re.search(pattern, text)
    if match:
        return match.group(1)

    # Explicit conclusion pattern.
    pattern = (
        r"\b(?:answer|final answer|conclusion)\s*(?:is|:)\s*("
        + "|".join(re.escape(label) for label in allowed)
        + r")\b"
    )
    match = re.search(pattern, text)
    if match:
        return match.group(1)

    return ""


def _strip_short_answer_wrapper(response: str) -> str:
    """Remove explicit answer wrappers without paraphrasing the answer itself."""
    text = str(response or "").strip()
    if not text:
        return ""

    # Prefer the first non-empty line for wrapper-style outputs.
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text)

    wrapper_patterns = [
        r"^\s*(?:final answer|answer)\s*[:\-]\s*(.+?)\s*$",
        r"^\s*the answer is\s+(.+?)\s*$",
    ]
    for pattern in wrapper_patterns:
        match = re.match(pattern, first_line, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().strip(" .")

    return text


def build_answer_instructions(
    dataset_name: str,
    task_type: str,
    *,
    options: Optional[Dict[str, str]] = None,
) -> str:
    """Return task-aware answer instructions for generation prompts.

    The goal is to make the generation format match the evaluation contract.
    """
    dataset = str(dataset_name or "").strip().lower()
    task = str(task_type or "").strip().lower()
    normalized_options = {
        str(k).strip().upper(): str(v).strip()
        for k, v in (options or {}).items()
        if str(v).strip()
    }

    lines = []

    if dataset == "pubmedqa":
        lines.extend([
            "This is a 3-label scientific inference task, not open-ended QA.",
            "Respond with exactly one word: yes, no, or maybe.",
            "Do not add any explanation, hedging phrase, or second sentence.",
            "Choose yes when the abstract overall supports the claim.",
            "Choose no when the abstract overall rejects the claim.",
            "IMPORTANT: 'maybe' is rare — use it only when the abstract explicitly states the evidence is inconclusive or contradictory and no overall conclusion can be drawn. Most abstracts have a clear directional conclusion; choose yes or no in those cases.",
            "If a study shows mixed results but the authors draw a net conclusion (e.g. 'overall, treatment X was effective'), that counts as yes or no, not maybe.",
            "When graph paths are present, treat them as auxiliary support only; they must not override the study conclusion stated in the text chunks.",
        ])
        return "\n".join(lines)

    if dataset == "bioasq" and task == "binary":
        lines.extend([
            "This is a binary biomedical QA task.",
            "Respond with exactly one word: yes or no.",
            "Do not answer with maybe.",
            "Do not add explanation unless the task-specific context explicitly requires it.",
        ])
        return "\n".join(lines)

    if dataset == "realmedqa":
        lines.extend([
            "This is clinical recommendation QA grounded in guideline text.",
            "Answer with a concise clinical recommendation in 1 to 3 sentences.",
            "Do not answer with a single entity, label, or fragment.",
            "Stay close to the retrieved guideline wording and do not invent extra recommendations.",
            "If the retrieved guideline text is insufficient, respond exactly: Insufficient Information.",
        ])
        return "\n".join(lines)

    if task == "mcq" and normalized_options:
        lines.extend([
            "This is a multiple-choice question.",
            "Choose exactly one option from the list below.",
            "Begin your response with the option letter, then the option text.",
            "Do not invent a new option or answer outside the list.",
            "Options:",
        ])
        for key, value in normalized_options.items():
            lines.append(f"{key}. {value}")
        return "\n".join(lines)

    if dataset == "multihoprag":
        lines.extend([
            "Answer with the shortest correct entity, title, date, number, or phrase grounded in the retrieved documents.",
            "Do not write an explanatory sentence.",
            "If the question compares two candidates, return only the selected candidate.",
            "If the provided context is insufficient, respond exactly: Insufficient Information.",
        ])
        if task == "binary":
            lines.append("For binary questions, begin your response with exactly Yes or No.")
        return "\n".join(lines)

    if dataset in {"hotpotqa", "2wikimultihopqa", "musique"} and task == "free_text":
        lines.extend([
            "This is a short-answer multi-hop QA task.",
            "Respond with the shortest correct entity, title, date, number, or phrase only.",
            "Do not write an explanatory sentence or reasoning chain after the answer.",
            "Do not restate the question.",
            "For comparison questions, return only the winning entity/title/person, not an explanation.",
            "If the answer is a date, location, person, or title, return exactly that span.",
        ])
        return "\n".join(lines)

    if task == "binary":
        lines.extend([
            "This is a binary question.",
            "Begin your response with exactly Yes or No as the first word.",
        ])
        return "\n".join(lines)

    if task == "free_text":
        return "Answer with a short entity or phrase when possible."

    return ""


def normalize_answer_to_contract(
    dataset_name: str,
    task_type: str,
    response: str,
) -> str:
    """
    Canonicalize model outputs for strict-label tasks without changing semantics.

    This is intentionally conservative: it only compresses responses when an
    explicit label is already present. Free-text answers are left untouched.
    """
    dataset = str(dataset_name or "").strip().lower()
    task = str(task_type or "").strip().lower()
    text = str(response or "").strip()
    if not text:
        return text

    if dataset == "pubmedqa":
        label = _extract_leading_label(text, ("yes", "no", "maybe"))
        return label or text

    if dataset == "bioasq" and task == "binary":
        label = _extract_leading_label(text, ("yes", "no"))
        return label or text

    if dataset in SHORT_ANSWER_DATASETS and task == "free_text":
        return _strip_short_answer_wrapper(text)

    return text
