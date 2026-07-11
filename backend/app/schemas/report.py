"""Diagnostic report API models."""

from typing import Literal

from pydantic import BaseModel


class ReportRequest(BaseModel):
    """Request an engineer-facing report for a stored diagnosis."""

    diagnosis_id: str


class ReportResponse(BaseModel):
    """Generated diagnostic report."""

    report_id: str
    format: Literal["markdown"] = "markdown"
    content: str
    generation_mode: Literal["template", "fireworks"]
