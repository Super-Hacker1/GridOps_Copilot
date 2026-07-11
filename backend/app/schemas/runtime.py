"""AMD training and runtime evidence schemas."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class AMDTrainingMetrics(BaseModel):
    """Core evaluation metrics exported by the FRA notebook."""

    model_config = ConfigDict(extra="allow")

    accuracy: float = Field(ge=0.0, le=1.0)
    f1_macro: float = Field(ge=0.0, le=1.0)


class AMDBenchmarks(BaseModel):
    """CPU and AMD GPU batch inference measurements."""

    model_config = ConfigDict(extra="allow")

    cpu_batch_ms: float = Field(gt=0.0)
    amd_gpu_batch_ms: float | None = Field(default=None, gt=0.0)
    speedup: float | None = Field(default=None, gt=0.0)


class AMDEvidenceReady(BaseModel):
    """Validated evidence exported by the AMD ROCm training notebook."""

    model_config = ConfigDict(extra="allow")

    amd_usage_claim: str
    training_platform: str
    framework: str
    gpu_available: StrictBool
    device_name: str
    torch_version: str
    hip_version: str | None
    model_artifact: str
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    metrics: AMDTrainingMetrics
    benchmarks: AMDBenchmarks
    status: Literal["complete", "incomplete"] = "complete"

    @model_validator(mode="after")
    def require_rocm_details_for_gpu_claim(self) -> Self:
        """Reject GPU evidence that cannot prove a ROCm/HIP runtime."""

        if self.gpu_available:
            if not self.hip_version or not self.hip_version.strip():
                raise ValueError("AMD GPU evidence requires a captured HIP version")
            if self.artifact_sha256 is None:
                raise ValueError("AMD GPU evidence requires an artifact SHA-256")
            if self.benchmarks.amd_gpu_batch_ms is None or self.benchmarks.speedup is None:
                raise ValueError("AMD GPU evidence requires GPU benchmark values")
        return self


class AMDEvidencePending(BaseModel):
    """Response used before an AMD notebook evidence file is available."""

    amd_usage_claim: str
    status: Literal["pending"]


AMDEvidenceResponse = AMDEvidenceReady | AMDEvidencePending
