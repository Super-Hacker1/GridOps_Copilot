"""Engineer-facing diagnostic report routes."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.schemas.report import ReportRequest, ReportResponse
from app.services.report_generation import generate_report
from app.storage.record_store import JsonRecordStore


router = APIRouter(prefix="/api/reports", tags=["reports"])
_REPORT_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gridops-report")


async def get_diagnosis_store() -> JsonRecordStore:
    settings = get_settings()
    return JsonRecordStore(settings.diagnosis_directory)


async def get_report_store() -> JsonRecordStore:
    settings = get_settings()
    return JsonRecordStore(settings.generated_report_directory)


async def get_report_settings() -> Settings:
    return get_settings()


@router.post("/generate", response_model=ReportResponse)
async def generate_diagnostic_report(
    request: ReportRequest,
    diagnosis_store: Annotated[JsonRecordStore, Depends(get_diagnosis_store)],
    report_store: Annotated[JsonRecordStore, Depends(get_report_store)],
    settings: Annotated[Settings, Depends(get_report_settings)],
) -> ReportResponse:
    """Generate and persist a conservative report for a diagnosis."""

    try:
        diagnosis = diagnosis_store.load(request.diagnosis_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    generation = _REPORT_EXECUTOR.submit(generate_report, diagnosis, settings)
    while not generation.done():
        await asyncio.sleep(0.01)
    content, generation_mode = generation.result()
    report_id = f"report_{uuid4().hex[:12]}"
    payload = {
        "report_id": report_id,
        "diagnosis_id": request.diagnosis_id,
        "format": "markdown",
        "content": content,
        "generation_mode": generation_mode,
    }
    report_store.save(report_id, payload)
    Path(report_store.root_directory).mkdir(parents=True, exist_ok=True)
    (report_store.root_directory / f"{report_id}.md").write_text(content, encoding="utf-8")

    return ReportResponse(**payload)
