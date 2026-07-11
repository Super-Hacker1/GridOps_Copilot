"""Tests for diagnostic report routes."""

import asyncio
from pathlib import Path
import threading

from fastapi import FastAPI
import pytest
from httpx2 import ASGITransport, AsyncClient

from app.api.routes_reports import (
    get_diagnosis_store,
    get_report_settings,
    get_report_store,
    router,
)
from app.api import routes_reports
from app.config import Settings
from app.services.report_generation import SAFETY_STATEMENT
from app.storage.record_store import JsonRecordStore


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def stores(tmp_path: Path) -> tuple[JsonRecordStore, JsonRecordStore]:
    return (
        JsonRecordStore(tmp_path / "diagnoses"),
        JsonRecordStore(tmp_path / "reports"),
    )


@pytest.fixture
def report_app(stores: tuple[JsonRecordStore, JsonRecordStore]) -> FastAPI:
    diagnosis_store, report_store = stores

    async def override_diagnosis_store() -> JsonRecordStore:
        return diagnosis_store

    async def override_report_store() -> JsonRecordStore:
        return report_store

    async def override_settings() -> Settings:
        return Settings(
            use_fireworks=False,
            fireworks_api_key=None,
            fireworks_model=None,
        )

    test_app = FastAPI()
    test_app.include_router(router)

    @test_app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    test_app.dependency_overrides[get_diagnosis_store] = override_diagnosis_store
    test_app.dependency_overrides[get_report_store] = override_report_store
    test_app.dependency_overrides[get_report_settings] = override_settings
    return test_app


@pytest.mark.anyio
async def test_generate_report_uses_template_fallback(
    report_app: FastAPI,
    stores: tuple[JsonRecordStore, JsonRecordStore],
) -> None:
    diagnosis_store, report_store = stores
    diagnosis_store.save(
        "diag_a1b2c3d4e5f6",
        {
            "asset_id": "CB-402",
            "diagnostic_type": "dcrm",
            "fault_class": "contact_wear_suspected",
            "confidence": 0.84,
            "risk_score": 76,
            "risk_level": "High",
            "evidence": ["Resistance peak exceeded baseline tolerance."],
            "recommended_action": "Schedule inspection with a qualified engineer.",
        },
    )
    transport = ASGITransport(app=report_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/reports/generate",
            json={"diagnosis_id": "diag_a1b2c3d4e5f6"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["generation_mode"] == "template"
    assert SAFETY_STATEMENT in body["content"]
    assert report_store.load(body["report_id"])["diagnosis_id"] == "diag_a1b2c3d4e5f6"
    assert (report_store.root_directory / f"{body['report_id']}.md").is_file()


@pytest.mark.anyio
async def test_generate_report_rejects_unknown_diagnosis(report_app: FastAPI) -> None:
    transport = ASGITransport(app=report_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/reports/generate",
            json={"diagnosis_id": "diag_000000000000"},
        )

    assert response.status_code == 404


@pytest.mark.anyio
async def test_report_generation_does_not_block_the_event_loop(
    report_app: FastAPI,
    stores: tuple[JsonRecordStore, JsonRecordStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnosis_store, _ = stores
    diagnosis_store.save("diag_a1b2c3d4e5f6", {"asset_id": "CB-402"})

    generation_started = threading.Event()
    release_generation = threading.Event()

    def slow_report(diagnosis, settings):
        generation_started.set()
        release_generation.wait(timeout=1.0)
        return "# Report", "template"

    monkeypatch.setattr(routes_reports, "generate_report", slow_report)
    report_transport = ASGITransport(app=report_app)
    ping_transport = ASGITransport(app=report_app)
    async with (
        AsyncClient(transport=report_transport, base_url="http://testserver") as report_client,
        AsyncClient(transport=ping_transport, base_url="http://testserver") as ping_client,
    ):
        report_task = asyncio.create_task(
            report_client.post(
                "/api/reports/generate",
                json={"diagnosis_id": "diag_a1b2c3d4e5f6"},
            )
        )
        for _ in range(100):
            if generation_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert generation_started.is_set()
        ping_response = await ping_client.get("/ping")
        report_was_running_during_ping = not report_task.done()
        release_generation.set()
        report_response = await asyncio.wait_for(report_task, timeout=2.0)

    assert ping_response.status_code == 200
    assert report_was_running_during_ping is True
    assert report_response.status_code == 200
