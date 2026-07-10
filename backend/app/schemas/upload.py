"""Upload API data models."""

from typing import Literal
from pydantic import BaseModel, Field
from app.agents.ingestion_agent import FileType


class UploadResponse(BaseModel):
    """Response returned after validating and storing an upload"""

    upload_id: str
    file_type: FileType
    asset_id: str | None
    validation_status: Literal["valid"] = "valid"
    rows: int = Field(ge=1)
    warnings: list[str]
