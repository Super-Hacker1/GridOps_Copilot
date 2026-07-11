"""Asset registry API routes."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.schemas.asset import AssetSummary
from app.services.asset_registry import (
    AssetRegistryError,
    load_asset_summaries,
)

router = APIRouter(prefix="/api", tags=["assets"])


async def get_asset_registry_path() -> Path:
    """Return the canonical asset registry path."""

    return get_settings().data_directory / "assets.csv"


async def get_asset_diagnosis_directory() -> Path:
    """Return the persisted digital-twin diagnosis directory."""

    return get_settings().diagnosis_directory


@router.get("/assets", response_model=list[AssetSummary])
async def list_assets(
    registry_path: Annotated[Path, Depends(get_asset_registry_path)],
    diagnosis_directory: Annotated[Path, Depends(get_asset_diagnosis_directory)],
) -> list[AssetSummary]:
    """List normalized summaries for all registered demo assets."""

    try:
        return load_asset_summaries(
            registry_path,
            diagnosis_directory=diagnosis_directory,
        )
    except AssetRegistryError as exc:
        raise HTTPException(
            status_code=503,
            detail="Asset registry is unavailable.",
        ) from exc
