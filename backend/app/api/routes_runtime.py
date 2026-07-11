"""Runtime and AMD platform evidence API routes."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.schemas.runtime import AMDEvidenceResponse
from app.services.runtime_evidence import (
    AMDEvidenceError,
    load_amd_evidence,
)

router = APIRouter(prefix="/api/runtime", tags=["runtime"])


async def get_amd_evidence_path() -> Path:
    """Return the AMD notebook evidence path."""

    return get_settings().amd_evidence_path


async def get_fra_model_path() -> Path:
    """Return the FRA model path used by the runtime."""

    return get_settings().fra_model_path


@router.get("/amd-evidence", response_model=AMDEvidenceResponse)
async def amd_evidence(
    evidence_path: Annotated[Path, Depends(get_amd_evidence_path)],
    model_path: Annotated[Path, Depends(get_fra_model_path)],
) -> AMDEvidenceResponse:
    """Return AMD training evidence without requiring a live GPU."""

    try:
        return load_amd_evidence(evidence_path, model_path=model_path)
    except AMDEvidenceError as exc:
        raise HTTPException(
            status_code=500,
            detail="AMD training evidence is invalid.",
        ) from exc
