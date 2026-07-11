"""Public asset API schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class AssetSummary(BaseModel):
    """Dashboard-ready summary of an asset registry entry."""

    asset_id: str
    asset_type: str
    voltage_level: str
    criticality: int = Field(ge=1, le=4)
    health_score: int = Field(ge=0, le=100)
    risk_level: Literal["Low", "Medium", "High", "Critical"]
