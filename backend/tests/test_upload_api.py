"""Tests for the diagnostic-file upload API"""

from pathlib import Path
import pytest
from httpx2 import ASGITransport, AsyncClient
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.api import routes_upload
from app.api.routes_upload import get_upload_store
from app.config import Settings
from app.main import app
from app.services import synthetic_data
from app.storage.json_store import JsonUploadStore


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def generated_files(tmp_path: Path) -> dict[str, Path]:
    return synthetic_data.generate_synthetic_data(
        tmp_path / "generated",
        seed=42,
    )


@pytest.fixture
def upload_store(tmp_path: Path) -> JsonUploadStore:
    return JsonUploadStore(tmp_path / "uploads")


@pytest.fixture(autouse=True)
def override_upload_store(upload_store: JsonUploadStore):
    async def override() -> JsonUploadStore:
        return upload_store

    app.dependency_overrides[get_upload_store] = override
    yield
    app.dependency_overrides.pop(get_upload_store, None)


async def post_upload(
    *,
    filename: str,
    content: bytes,
    file_type: str,
    asset_id: str | None = None,
):
    transport = ASGITransport(app=app)
    form_data = {"file_type": file_type}

    if asset_id is not None:
        form_data["asset_id"] = asset_id

    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.post(
            "/api/upload",
            data=form_data,
            files={
                "file": (
                    filename,
                    content,
                    "text/csv",
                )
            },
        )


@pytest.mark.anyio
async def test_upload_accepts_and_stores_fra_file(
    generated_files: dict[str, Path],
    upload_store: JsonUploadStore,
) -> None:
    response = await post_upload(
        filename="fra_healthy.csv",
        content=generated_files["fra_healthy.csv"].read_bytes(),
        file_type="fra",
    )
    body = response.json()
    assert response.status_code == 200
    assert body["upload_id"].startswith("upload_")
    assert body["file_type"] == "fra"
    assert body["asset_id"] == "TX-1"
    assert body["validation_status"] == "valid"
    assert body["rows"] == 512
    assert body["warnings"] == []

    metadata = upload_store.load_metadata(body["upload_id"])
    stored_frame = upload_store.load_frame(body["upload_id"])

    assert metadata["original_filename"] == "fra_healthy.csv"
    assert metadata["file_type"] == "fra"
    assert len(stored_frame) == 512


@pytest.mark.anyio
async def test_upload_normalizes_and_stores_dcrm_file(
    generated_files: dict[str, Path],
    upload_store: JsonUploadStore,
) -> None:
    response = await post_upload(
        filename="dcrm_healthy.csv",
        content=generated_files["dcrm_healthy.csv"].read_bytes(),
        file_type="dcrm",
    )

    assert response.status_code == 200

    body = response.json()
    stored_frame = upload_store.load_frame(body["upload_id"])

    assert body["asset_id"] == "CB-402"
    assert body["rows"] == 201
    assert "coil_current_a" in stored_frame.columns
    assert "coil_current_A" not in stored_frame.columns
    assert body["warnings"] == ["Normalized column 'coil_current_A' to 'coil_current_a'."]


@pytest.mark.anyio
async def test_upload_accepts_explicit_asset_from_multi_asset_file(
    generated_files: dict[str, Path],
) -> None:
    response = await post_upload(
        filename="scada_events.csv",
        content=generated_files["scada_events.csv"].read_bytes(),
        file_type="scada",
        asset_id="CB-401",
    )

    assert response.status_code == 200
    assert response.json()["asset_id"] == "CB-401"


@pytest.mark.anyio
async def test_upload_returns_null_asset_file_without_selection(
    generated_files: dict[str, Path],
) -> None:
    response = await post_upload(
        filename="assets.csv",
        content=generated_files["assets.csv"].read_bytes(),
        file_type="assets",
    )

    assert response.status_code == 200
    assert response.json()["asset_id"] is None


@pytest.mark.anyio
async def test_upload_rejects_asset_not_present_in_file(
    generated_files: dict[str, Path],
) -> None:
    response = await post_upload(
        filename="fra_healthy.csv",
        content=generated_files["fra_healthy.csv"].read_bytes(),
        file_type="fra",
        asset_id="TX-2",
    )

    assert response.status_code == 400
    assert "TX-2" in response.json()["detail"]
    assert "not present" in response.json()["detail"]


@pytest.mark.anyio
async def test_upload_rejects_invalid_csv_schema() -> None:
    response = await post_upload(
        filename="invalid.csv", content=b"unknown, value\nexample,1\n", file_type="fra"
    )

    assert response.status_code == 400
    assert "Missing required fra columns" in response.json()["detail"]


@pytest.mark.anyio
async def test_upload_rejects_unsupported_file_type(
    generated_files: dict[str, Path],
) -> None:
    response = await post_upload(
        filename="assets.csv",
        content=generated_files["assets.csv"].read_bytes(),
        file_type="unknown",
    )
    assert response.status_code == 400
    assert "Unsupported declared_type" in response.json()["detail"]


@pytest.mark.anyio
async def test_upload_store_uses_configured_runtime_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_directory = tmp_path / "configured-uploads"
    monkeypatch.setattr(
        routes_upload,
        "get_settings",
        lambda: Settings(upload_directory=configured_directory),
        raising=False,
    )

    store = await get_upload_store()

    assert store.root_directory == configured_directory


@pytest.mark.anyio
async def test_upload_reads_file_in_bounded_chunks(
    generated_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_sizes: list[int] = []
    original_read = StarletteUploadFile.read

    async def tracked_read(upload, size: int = -1):
        read_sizes.append(size)
        return await original_read(upload, size)

    monkeypatch.setattr(StarletteUploadFile, "read", tracked_read)
    response = await post_upload(
        filename="fra_healthy.csv",
        content=generated_files["fra_healthy.csv"].read_bytes(),
        file_type="fra",
    )

    assert response.status_code == 200
    assert read_sizes
    assert -1 not in read_sizes
