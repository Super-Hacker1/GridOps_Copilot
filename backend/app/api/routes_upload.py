"""Diagnostic-file upload API routes."""

from typing import Annotated
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.agents.ingestion_agent import ingest_csv
from app.config import get_settings
from app.schemas.upload import UploadResponse
from app.storage.json_store import JsonUploadStore

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024

router = APIRouter(
    prefix="/api",
    tags=["uploads"],
)


async def get_upload_store() -> JsonUploadStore:
    """Return the application's upload store."""

    return JsonUploadStore(get_settings().upload_directory)


async def _read_limited_upload(file: UploadFile) -> bytes:
    if file.size is not None and file.size > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the 10 MB limit")

    content = bytearray()
    while chunk := await file.read(UPLOAD_READ_CHUNK_BYTES):
        content.extend(chunk)
        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Uploaded file exceeds the 10 MB limit",
            )
    return bytes(content)


@router.post(
    "/upload",
    response_model=UploadResponse,
)
async def upload_diagnostic_file(
    file: Annotated[UploadFile, File(...)],
    file_type: Annotated[str, Form(...)],
    store: Annotated[JsonUploadStore, Depends(get_upload_store)],
    asset_id: Annotated[str | None, Form()] = None,
) -> UploadResponse:
    """Validate, normalize, and persist a diagnostic CSV."""

    content = await _read_limited_upload(file)
    try:
        result = ingest_csv(
            content,
            declared_type=file_type,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    selected_asset_id = asset_id.strip() if asset_id else None

    if selected_asset_id and selected_asset_id not in result.asset_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Asset '{selected_asset_id}' is not present in "
                f"the uploaded {result.file_type} data"
            ),
        )

    if selected_asset_id is None and len(result.asset_ids) == 1:
        selected_asset_id = result.asset_ids[0]

    stored = store.save(
        result=result,
        original_filename=file.filename or "upload.csv",
        asset_id=selected_asset_id,
    )

    return UploadResponse(
        upload_id=stored.upload_id,
        file_type=stored.file_type,
        asset_id=stored.asset_id,
        validation_status="valid",
        rows=stored.rows,
        warnings=list(stored.warnings),
    )
