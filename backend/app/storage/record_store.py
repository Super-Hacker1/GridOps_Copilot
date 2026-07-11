"""Small JSON record store used by diagnosis and report routes."""

import json
from pathlib import Path
import re
from typing import Any


RECORD_ID_PATTERN = re.compile(r"[a-z]+_[0-9a-f]{12}")


class JsonRecordStore:
    """Persist JSON-compatible dictionaries by generated record ID."""

    def __init__(self, root_directory: Path) -> None:
        self.root_directory = Path(root_directory)

    def save(self, record_id: str, payload: dict[str, Any]) -> None:
        self._validate_record_id(record_id)
        self.root_directory.mkdir(parents=True, exist_ok=True)
        path = self.root_directory / f"{record_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, record_id: str) -> dict[str, Any]:
        self._validate_record_id(record_id)
        path = self.root_directory / f"{record_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Record not found: {record_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _validate_record_id(record_id: str) -> None:
        if RECORD_ID_PATTERN.fullmatch(record_id) is None:
            raise ValueError(f"Invalid record ID: {record_id}")
