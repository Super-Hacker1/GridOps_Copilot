"""Application configuration and environment settings."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or `.env`."""

    service_name: str = "gridops-copilot-backend"
    environment: str = "development"
    cors_origins: list[str] = ["http://localhost:3000"]

    data_directory: Path = PROJECT_ROOT / "data" / "synthetic"
    upload_directory: Path = PROJECT_ROOT / "data" / "runtime" / "uploads"
    diagnosis_directory: Path = PROJECT_ROOT / "data" / "runtime" / "diagnoses"
    generated_report_directory: Path = PROJECT_ROOT / "reports" / "generated"
    amd_evidence_path: Path = PROJECT_ROOT / "reports" / "amd_training_evidence.json"
    fra_model_path: Path = PROJECT_ROOT / "models" / "fra_cnn_rocm.pt"
    fra_label_map_path: Path = PROJECT_ROOT / "models" / "fra_label_map.json"

    use_fireworks: bool = False
    fireworks_api_key: str | None = None
    fireworks_model: str | None = None
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"
    fireworks_timeout_seconds: float = 20.0

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", PROJECT_ROOT / "backend" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()
