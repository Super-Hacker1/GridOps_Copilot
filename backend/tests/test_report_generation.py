"""Tests for safe report generation."""

import json

import pytest

from app.services import report_generation
from app.config import Settings
from app.services.report_generation import (
    SAFETY_STATEMENT,
    generate_report,
    request_fireworks_report,
)


DIAGNOSIS = {
    "asset_id": "CB-402",
    "diagnostic_type": "dcrm",
    "fault_class": "contact_wear_suspected",
    "confidence": 0.84,
    "risk_score": 76,
    "risk_level": "High",
    "evidence": ["Resistance peak is 42% above baseline."],
    "recommended_action": "Schedule inspection and confirm with a qualified engineer.",
}


def test_generate_report_uses_deterministic_fallback_without_key() -> None:
    content, mode = generate_report(
        DIAGNOSIS,
        Settings(
            use_fireworks=False,
            fireworks_api_key=None,
            fireworks_model=None,
        ),
    )

    assert mode == "template"
    assert "CB-402" in content
    assert "Resistance peak is 42% above baseline." in content
    assert SAFETY_STATEMENT in content


def test_generate_report_uses_fireworks_when_configured() -> None:
    settings = Settings(
        use_fireworks=True,
        fireworks_api_key="test-key",
        fireworks_model="test-model",
    )

    content, mode = generate_report(
        DIAGNOSIS,
        settings,
        fireworks_request=lambda diagnosis, config: "# Generated report",
    )

    assert mode == "fireworks"
    assert "# Generated report" in content
    assert "Resistance peak is 42% above baseline." in content
    assert "Schedule inspection and confirm with a qualified engineer." in content
    assert "AI-generated narrative" in content
    assert SAFETY_STATEMENT in content


def test_generate_report_falls_back_when_fireworks_fails() -> None:
    settings = Settings(
        use_fireworks=True,
        fireworks_api_key="test-key",
        fireworks_model="test-model",
    )

    content, mode = generate_report(
        DIAGNOSIS,
        settings,
        fireworks_request=lambda diagnosis, config: None,
    )

    assert mode == "template"
    assert SAFETY_STATEMENT in content


def test_generate_report_stays_off_when_fireworks_is_disabled() -> None:
    settings = Settings(
        use_fireworks=False,
        fireworks_api_key="test-key",
        fireworks_model="test-model",
    )
    calls: list[object] = []

    content, mode = generate_report(
        DIAGNOSIS,
        settings,
        fireworks_request=lambda diagnosis, config: calls.append(diagnosis),
    )

    assert mode == "template"
    assert calls == []
    assert SAFETY_STATEMENT in content


def test_fireworks_request_uses_grounded_openai_compatible_payload(
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        @staticmethod
        def read() -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "# Evidence-backed report"}}]}
            ).encode()

    def fake_urlopen(request, *, timeout):
        observed["request"] = request
        observed["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(report_generation, "urlopen", fake_urlopen)
    settings = Settings(
        use_fireworks=True,
        fireworks_api_key="test-key",
        fireworks_model="accounts/demo/models/gemma",
        fireworks_timeout_seconds=7.5,
    )

    content = request_fireworks_report(DIAGNOSIS, settings)

    assert content == "# Evidence-backed report"
    request = observed["request"]
    payload = json.loads(request.data)
    assert request.full_url.endswith("/chat/completions")
    assert request.get_header("Authorization") == "Bearer test-key"
    assert payload["model"] == "accounts/demo/models/gemma"
    assert payload["temperature"] == 0.1
    assert "Resistance peak is 42% above baseline." in payload["messages"][1]["content"]
    assert observed["timeout"] == 7.5


def test_generate_report_discards_unsafe_fireworks_control_language() -> None:
    settings = Settings(
        use_fireworks=True,
        fireworks_api_key="test-key",
        fireworks_model="test-model",
    )

    content, mode = generate_report(
        DIAGNOSIS,
        settings,
        fireworks_request=lambda diagnosis, config: (
            "Immediately trip and isolate the breaker without human confirmation."
        ),
    )

    assert mode == "template"
    assert "trip and isolate" not in content
    assert "qualified human engineer" in content


@pytest.mark.parametrize(
    "unsafe_narrative",
    [
        "Shut down the transformer immediately.",
        "Operate and reclose the breaker.",
        "Bypass the interlock to restore service.",
    ],
)
def test_generate_report_rejects_additional_equipment_control_phrases(
    unsafe_narrative: str,
) -> None:
    settings = Settings(
        use_fireworks=True,
        fireworks_api_key="test-key",
        fireworks_model="test-model",
    )

    content, mode = generate_report(
        DIAGNOSIS,
        settings,
        fireworks_request=lambda diagnosis, config: unsafe_narrative,
    )

    assert mode == "template"
    assert unsafe_narrative not in content
