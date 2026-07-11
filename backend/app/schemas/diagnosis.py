"""Diagnostic workflow data models."""

from typing import Literal

from pydantic import BaseModel, Field

from app.safety import SAFETY_STATEMENT


DiagnosticType = Literal["fra", "dcrm"]
AnalysisMethod = Literal[
    "dcrm_rule_based",
    "fra_rule_fallback",
    "fra_model_artifact",
]
RiskLevel = Literal["Low", "Medium", "High", "Critical"]


class DCRMAnalysisResult(BaseModel):
    """Structured result returned by the DCRM analyzer"""

    is_anomalous: bool
    anomaly_types: list[str]
    likely_faults: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    metrics: dict[str, float]
    evidence: list[str]


class DiagnosticAnalysisResult(BaseModel):
    """Normalized diagnostic output consumed by the orchestrator."""

    fault_class: str
    is_anomalous: bool
    confidence: float = Field(ge=0.0, le=1.0)
    anomaly_score: float = Field(ge=0.0, le=1.0)
    evidence: list[str]
    metrics: dict[str, float]
    analysis_method: AnalysisMethod
    requires_human_review: bool
    anomaly_types: list[str] = Field(default_factory=list)
    likely_faults: list[str] = Field(default_factory=list)


class SCADAContext(BaseModel):
    """Operational evidence associated with one asset."""

    has_recent_alarm: bool
    alarm_codes: list[str]
    max_temperature_c: float | None
    latest_current_a: float | None
    baseline_current_a: float | None
    current_deviation_pct: float | None
    status_changed: bool
    evidence: list[str]


class MaintenanceContext(BaseModel):
    """Maintenance recurrence and due-date evidence for one asset."""

    is_overdue: bool
    overdue_days: int = Field(ge=0)
    has_recurrence: bool
    matching_issues: list[str]
    evidence: list[str]


class RiskAssessment(BaseModel):
    """Weighted risk score and transparent contributing factors."""

    score: int = Field(ge=0, le=100)
    level: RiskLevel
    factors: dict[str, int]


class DiagnosisResult(BaseModel):
    """Complete deterministic diagnosis returned by the core workflow."""

    asset_id: str
    asset_type: str
    diagnostic_type: DiagnosticType
    fault_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    anomaly_score: float = Field(ge=0.0, le=1.0)
    risk_score: int = Field(ge=0, le=100)
    risk_level: RiskLevel
    evidence: list[str]
    recommended_action: str
    requires_human_review: bool
    analysis_method: AnalysisMethod
    metrics: dict[str, float]
    risk_factors: dict[str, int]
    data_quality_warnings: list[str] = Field(default_factory=list)
    safety_statement: str = SAFETY_STATEMENT


class DiagnosisRequest(BaseModel):
    """Request linking a validated upload to an asset diagnosis."""

    asset_id: str = Field(min_length=1, max_length=100)
    diagnostic_type: DiagnosticType
    upload_id: str = Field(pattern=r"^upload_[0-9a-f]{12}$")
    scada_upload_id: str | None = Field(
        default=None,
        pattern=r"^upload_[0-9a-f]{12}$",
    )
    maintenance_upload_id: str | None = Field(
        default=None,
        pattern=r"^upload_[0-9a-f]{12}$",
    )
    assets_upload_id: str | None = Field(
        default=None,
        pattern=r"^upload_[0-9a-f]{12}$",
    )


class DiagnosisResponse(DiagnosisResult):
    """Persisted diagnosis returned by the HTTP API."""

    diagnosis_id: str = Field(pattern=r"^diag_[0-9a-f]{12}$")
