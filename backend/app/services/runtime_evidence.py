"""Load and validate AMD ROCm notebook evidence."""

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from app.schemas.runtime import (
    AMDEvidencePending,
    AMDEvidenceReady,
    AMDEvidenceResponse,
)

AMD_USAGE_CLAIM = "FRA model trained and benchmarked on AMD GPU using ROCm PyTorch."
AMD_INCOMPLETE_CLAIM = (
    "AMD GPU training evidence is incomplete because the notebook did not detect a GPU."
)


class AMDEvidenceError(RuntimeError):
    """Raised when a present AMD evidence file is unreadable or invalid."""


def load_amd_evidence(
    evidence_path: Path,
    *,
    model_path: Path | None = None,
) -> AMDEvidenceResponse:
    """Return validated notebook evidence, or a pending response if absent."""

    if not evidence_path.is_file():
        return AMDEvidencePending(
            amd_usage_claim="No AMD training evidence file found yet.",
            status="pending",
        )

    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AMDEvidenceError("AMD training evidence could not be read") from exc

    if not isinstance(payload, dict):
        raise AMDEvidenceError("AMD training evidence must be a JSON object")

    if payload.get("gpu_available") is False:
        payload["amd_usage_claim"] = AMD_INCOMPLETE_CLAIM
        payload["status"] = "incomplete"
    else:
        payload.setdefault("amd_usage_claim", AMD_USAGE_CLAIM)

    try:
        evidence = AMDEvidenceReady.model_validate(payload)
    except ValidationError as exc:
        raise AMDEvidenceError("AMD training evidence is invalid") from exc

    if evidence.gpu_available and model_path is not None:
        try:
            deployed_hash = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        except OSError:
            deployed_hash = None
        if deployed_hash != evidence.artifact_sha256:
            return evidence.model_copy(
                update={
                    "amd_usage_claim": (
                        "AMD training evidence exists, but the deployed FRA model artifact "
                        "is missing or does not match the recorded SHA-256."
                    ),
                    "status": "incomplete",
                }
            )

    return evidence
