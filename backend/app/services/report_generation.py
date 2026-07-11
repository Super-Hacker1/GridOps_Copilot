"""Conservative engineer-facing report generation."""

import json
import re
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import Settings
from app.safety import SAFETY_STATEMENT

FireworksRequest = Callable[[dict[str, Any], Settings], str | None]

_UNSAFE_NARRATIVE_PATTERNS = (
    re.compile(
        r"\b(?:trip|operate|reclose|bypass|override|disable|switch|disconnect|isolate|"
        r"open|close|energize|de[- ]?energize|lockout|reset|actuate|shutdown)\b"
    ),
    re.compile(r"\bshut\s+down\b"),
    re.compile(r"\btake\b[^.]{0,80}\boffline\b"),
    re.compile(r"\bwithout\s+(?:human|engineer)\b"),
    re.compile(r"\b(?:automated|autonomous)\s+control\b"),
    re.compile(r"\b(?:certified diagnosis|confirmed fault|final fault certification)\b"),
)


def _is_safe_supplemental_narrative(content: str) -> bool:
    normalized = " ".join(content.lower().split())
    return not any(pattern.search(normalized) for pattern in _UNSAFE_NARRATIVE_PATTERNS)


def _render_grounded_sections(diagnosis: dict[str, Any]) -> str:
    evidence = diagnosis.get("evidence") or ["No diagnostic evidence was provided."]
    evidence_lines = "\n".join(f"- {item}" for item in evidence)

    return (
        "# GridOps Copilot Diagnostic Report\n\n"
        f"**Asset:** {diagnosis.get('asset_id', 'unknown')}\n\n"
        f"**Diagnostic type:** {diagnosis.get('diagnostic_type', 'unknown')}\n\n"
        f"**Suspected condition:** {diagnosis.get('fault_class', 'needs_human_review')}\n\n"
        f"**Confidence:** {float(diagnosis.get('confidence', 0.0)):.2f}\n\n"
        f"**Risk:** {diagnosis.get('risk_level', 'Unknown')} "
        f"({int(diagnosis.get('risk_score', 0))}/100)\n\n"
        "## Evidence\n\n"
        f"{evidence_lines}\n\n"
        "## Recommended action\n\n"
        f"{diagnosis.get('recommended_action', 'Request qualified engineering review.')}\n"
    )


def render_deterministic_report(diagnosis: dict[str, Any]) -> str:
    """Render an evidence-only Markdown report without an LLM."""

    return (
        f"{_render_grounded_sections(diagnosis).rstrip()}\n\n"
        "## Safety and limitations\n\n"
        f"{SAFETY_STATEMENT}\n"
    )


def request_fireworks_report(
    diagnosis: dict[str, Any],
    settings: Settings,
) -> str | None:
    """Request a grounded report from Fireworks, returning `None` on failure."""

    if not settings.use_fireworks or not settings.fireworks_api_key or not settings.fireworks_model:
        return None

    prompt = (
        "Generate a concise Markdown maintenance report using only the structured "
        "evidence below. Use conservative language, never claim certified diagnosis, "
        "and require qualified human confirmation.\n\n"
        f"{json.dumps(diagnosis, indent=2)}"
    )
    payload = {
        "model": settings.fireworks_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an evidence-grounded substation maintenance report assistant. "
                    "Never invent evidence or authorize control actions."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1200,
    }
    request = Request(
        f"{settings.fireworks_base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.fireworks_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=settings.fireworks_timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        return str(content).strip() or None
    except (HTTPError, URLError, TimeoutError, KeyError, IndexError, ValueError, OSError):
        return None


def generate_report(
    diagnosis: dict[str, Any],
    settings: Settings,
    *,
    fireworks_request: FireworksRequest = request_fireworks_report,
) -> tuple[str, str]:
    """Generate a Fireworks report when configured, otherwise use the template."""

    if settings.use_fireworks and settings.fireworks_api_key and settings.fireworks_model:
        generated = fireworks_request(diagnosis, settings)
        if generated and _is_safe_supplemental_narrative(generated):
            report = (
                f"{_render_grounded_sections(diagnosis).rstrip()}\n\n"
                "## AI-generated narrative (Fireworks; verify against evidence)\n\n"
                f"{generated.strip()}\n\n"
                "## Safety and limitations\n\n"
                f"{SAFETY_STATEMENT}\n"
            )
            return report, "fireworks"

    return render_deterministic_report(diagnosis), "template"
