"""Tests for the asset registry API."""

from inspect import signature
from pathlib import Path

import pytest
from httpx2 import ASGITransport, AsyncClient

from app.api.routes_assets import get_asset_registry_path
from app.main import app
from app.services.asset_registry import load_asset_summaries
from app.storage.record_store import JsonRecordStore


DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "synthetic"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_assets_endpoint_lists_demo_registry() -> None:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get("/api/assets")

    assert response.status_code == 200
    assert response.json()[0] == {
        "asset_id": "TX-1",
        "asset_type": "Transformer",
        "voltage_level": "400kV",
        "criticality": 4,
        "health_score": 68,
        "risk_level": "High",
    }
    assert {asset["asset_id"] for asset in response.json()} == {
        "TX-1",
        "CB-401",
        "CB-402",
        "TX-2",
        "CB-221",
    }


@pytest.mark.anyio
async def test_assets_endpoint_returns_service_unavailable_for_missing_registry(
    tmp_path: Path,
) -> None:
    async def override_registry_path() -> Path:
        return tmp_path / "missing-assets.csv"

    app.dependency_overrides[get_asset_registry_path] = override_registry_path
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/assets")
    finally:
        app.dependency_overrides.pop(get_asset_registry_path, None)

    assert response.status_code == 503
    assert response.json()["detail"] == "Asset registry is unavailable."


@pytest.mark.anyio
async def test_assets_endpoint_rejects_invalid_criticality(tmp_path: Path) -> None:
    registry_path = tmp_path / "assets.csv"
    registry_path.write_text(
        "asset_id,asset_type,voltage_level,criticality\n"
        "TX-9,transformer,400 kV,mission_impossible\n",
        encoding="utf-8",
    )

    async def override_registry_path() -> Path:
        return registry_path

    app.dependency_overrides[get_asset_registry_path] = override_registry_path
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/assets")
    finally:
        app.dependency_overrides.pop(get_asset_registry_path, None)

    assert response.status_code == 503


def test_asset_summaries_overlay_latest_persisted_diagnosis(tmp_path: Path) -> None:
    if "diagnosis_directory" not in signature(load_asset_summaries).parameters:
        pytest.fail("Asset summaries do not yet accept persisted diagnosis state")

    diagnosis_store = JsonRecordStore(tmp_path / "diagnoses")
    diagnosis_store.save(
        "diag_a1b2c3d4e5f6",
        {
            "asset_id": "TX-1",
            "risk_score": 82,
            "risk_level": "Critical",
        },
    )

    summaries = load_asset_summaries(
        DATA_ROOT / "assets.csv",
        diagnosis_directory=diagnosis_store.root_directory,
    )
    tx_1 = next(summary for summary in summaries if summary.asset_id == "TX-1")

    assert tx_1.health_score == 18
    assert tx_1.risk_level == "Critical"
