"""Persisted FRA and DCRM diagnosis routes."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
import pandas as pd

from app.agents.orchestrator import DiagnosisOrchestrator, DiagnosisReferenceData
from app.config import Settings, get_settings
from app.api.routes_upload import get_upload_store
from app.models.fra_model_loader import create_fra_artifact_predictor
from app.schemas.diagnosis import DiagnosisRequest, DiagnosisResponse, DiagnosisResult
from app.storage.json_store import JsonUploadStore
from app.storage.record_store import JsonRecordStore


router = APIRouter(prefix="/api", tags=["diagnostics"])
_DIAGNOSIS_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gridops-diagnosis")

_CONTEXT_UPLOAD_FIELDS = {
    "scada": "scada_upload_id",
    "maintenance": "maintenance_upload_id",
    "assets": "assets_upload_id",
}


class _ReferenceDataUnavailable(RuntimeError):
    """Raised when canonical diagnostic reference data cannot be loaded."""


async def get_diagnosis_settings() -> Settings:
    """Return settings through an async dependency for ASGI test compatibility."""

    return get_settings()


async def get_diagnosis_store(
    settings: Annotated[Settings, Depends(get_diagnosis_settings)],
) -> JsonRecordStore:
    """Return the diagnosis record store configured for this runtime."""

    return JsonRecordStore(settings.diagnosis_directory)


def _load_context_uploads(
    request: DiagnosisRequest,
    upload_store: JsonUploadStore,
) -> tuple[dict[str, pd.DataFrame], tuple[str, ...]]:
    frames: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []

    for expected_type, field_name in _CONTEXT_UPLOAD_FIELDS.items():
        upload_id = getattr(request, field_name)
        if upload_id is None:
            continue

        metadata = upload_store.load_metadata(upload_id)
        uploaded_type = str(metadata.get("file_type", "")).lower()
        if uploaded_type != expected_type:
            raise ValueError(
                f"Upload {upload_id} contains {uploaded_type.upper()} data, "
                f"not {expected_type.upper()} data"
            )
        selected_asset = metadata.get("asset_id")
        if selected_asset is not None and str(selected_asset) != request.asset_id:
            raise ValueError(
                f"Upload {upload_id} was selected for asset {selected_asset}, "
                f"not {request.asset_id}"
            )

        frames[expected_type] = upload_store.load_frame(upload_id)
        warnings.extend(
            f"{expected_type.capitalize()} context: {warning}"
            for warning in metadata.get("warnings", [])
        )

    return frames, tuple(warnings)


def _run_diagnosis_workflow(
    *,
    settings: Settings,
    request: DiagnosisRequest,
    upload_store: JsonUploadStore,
) -> DiagnosisResult:
    metadata = upload_store.load_metadata(request.upload_id)
    current = upload_store.load_frame(request.upload_id)

    uploaded_type = str(metadata.get("file_type", "")).lower()
    if uploaded_type != request.diagnostic_type:
        raise ValueError(
            f"Upload {request.upload_id} contains {uploaded_type.upper()} data, "
            f"not {request.diagnostic_type.upper()} data"
        )

    selected_asset = metadata.get("asset_id")
    if selected_asset is not None and str(selected_asset) != request.asset_id:
        raise ValueError(
            f"Upload {request.upload_id} was selected for asset {selected_asset}, "
            f"not {request.asset_id}"
        )

    warnings = tuple(str(item) for item in metadata.get("warnings", []))
    context_frames, context_warnings = _load_context_uploads(request, upload_store)
    warnings = (*warnings, *context_warnings)

    try:
        reference_data = DiagnosisReferenceData.from_directory(settings.data_directory)
    except (ValueError, FileNotFoundError) as exc:
        raise _ReferenceDataUnavailable from exc

    if context_frames:
        reference_data = replace(reference_data, **context_frames)

    fra_predictor = None
    if request.diagnostic_type == "fra":
        fra_predictor = create_fra_artifact_predictor(
            settings.fra_model_path,
            settings.fra_label_map_path,
        )
    return DiagnosisOrchestrator(
        reference_data,
        fra_artifact_predictor=fra_predictor,
    ).diagnose(
        asset_id=request.asset_id,
        diagnostic_type=request.diagnostic_type,
        current=current,
        data_quality_warnings=warnings,
    )


@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose_upload(
    request: DiagnosisRequest,
    upload_store: Annotated[JsonUploadStore, Depends(get_upload_store)],
    diagnosis_store: Annotated[JsonRecordStore, Depends(get_diagnosis_store)],
    settings: Annotated[Settings, Depends(get_diagnosis_settings)],
) -> DiagnosisResponse:
    """Run the deterministic agent workflow for one validated upload."""

    diagnosis_future = _DIAGNOSIS_EXECUTOR.submit(
        _run_diagnosis_workflow,
        settings=settings,
        request=request,
        upload_store=upload_store,
    )
    while not diagnosis_future.done():
        await asyncio.sleep(0.01)
    try:
        result = diagnosis_future.result()
    except _ReferenceDataUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="Diagnostic reference data is unavailable.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    diagnosis_id = f"diag_{uuid4().hex[:12]}"
    response = DiagnosisResponse(diagnosis_id=diagnosis_id, **result.model_dump())
    diagnosis_store.save(diagnosis_id, response.model_dump(mode="json"))
    return response
