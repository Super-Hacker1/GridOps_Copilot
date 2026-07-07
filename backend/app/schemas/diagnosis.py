"""Diagnostic workflow data models."""

from pydantic import BaseModel, Field


class DCRMAnalysisResult(BaseModel):
    """Structured result returned by the DCRM analyzer"""

    is_anomalous: bool
    anomaly_types: list[str]
    likely_faults: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    metrics: dict[str, float]
    evidence: list[str]
