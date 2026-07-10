"""Filesystem storage for validated diagnostic uploads."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from uuid import uuid4

import pandas as pd

from app.agents.ingestion_agent import FileType, IngestionResult


UPLOAD_ID_PATTERN = re.compile(r"upload_[0-9a-f]{12}")


@dataclass(frozen=True)
class StoredUpload:
    """Metadata describing a persisted upload."""

    upload_id: str
    file_type: FileType
    asset_id: str | None
    original_filename: str
    rows: int
    warnings: tuple[str, ...]


class JsonUploadStore:
    """Store normalized CSV files alongside JSON metadata."""

    def __init__(self, root_directory: Path) -> None:
        self.root_directory = Path(root_directory)

    def save(
        self,
        *,
        result: IngestionResult,
        original_filename: str,
        asset_id: str | None,
    ) -> StoredUpload:
        self.root_directory.mkdir(parents=True, exist_ok=True)

        upload_id = f"upload_{uuid4().hex[:12]}"
        csv_path = self.root_directory / f"{upload_id}.csv"
        metadata_path = self.root_directory / f"{upload_id}.json"

        result.frame.to_csv(
            csv_path,
            index=False,
            lineterminator="\n",
        )

        metadata = {
            "upload_id": upload_id,
            "file_type": result.file_type,
            "asset_id": asset_id,
            "original_filename": original_filename,
            "validation_status": "valid",
            "rows": result.row_count,
            "warnings": list(result.warnings),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "csv_path": csv_path.name,
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        return StoredUpload(
            upload_id=upload_id,
            file_type=result.file_type,
            asset_id=asset_id,
            original_filename=original_filename,
            rows=result.row_count,
            warnings=result.warnings,
        )

    def load_metadata(self, upload_id: str) -> dict[str, object]:
        self._validate_upload_id(upload_id)

        metadata_path = self.root_directory / f"{upload_id}.json"

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Upload metadata not found: {upload_id}"
            )

        return json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

    def load_frame(self, upload_id: str) -> pd.DataFrame:
        self._validate_upload_id(upload_id)

        csv_path = self.root_directory / f"{upload_id}.csv"

        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Upload data not found: {upload_id}"
            )

        return pd.read_csv(csv_path)

    @staticmethod
    def _validate_upload_id(upload_id: str) -> None:
        if UPLOAD_ID_PATTERN.fullmatch(upload_id) is None:
            raise ValueError(f"Invalid upload ID: {upload_id}")


PROJECT_ROOT = Path(__file__).resolve().parents[3]

default_upload_store = JsonUploadStore(
    PROJECT_ROOT / "data" / "runtime" / "uploads"
)