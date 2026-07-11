"""End-to-end API tests for persisted diagnostic workflows."""

import asyncio
from pathlib import Path
import threading

import pytest
from httpx2 import ASGITransport, AsyncClient

from app.api import routes_diagnostics
from app.api.routes_assets import get_asset_diagnosis_directory
from app.api.routes_reports import get_diagnosis_store as get_report_diagnosis_store
from app.api.routes_reports import get_report_settings, get_report_store
from app.api.routes_diagnostics import get_diagnosis_settings, get_diagnosis_store
from app.api.routes_upload import get_upload_store
from app.config import Settings, get_settings
from app.main import app
from app.services.report_generation import SAFETY_STATEMENT
from app.schemas.diagnosis import DiagnosticAnalysisResult
from app.storage.json_store import JsonUploadStore
from app.storage.record_store import JsonRecordStore


DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "synthetic"
REQUIRED_SAFETY_STATEMENT = (
    "This system is a decision-support prototype for hackathon demonstration. "
    "It does not control grid equipment, does not replace certified engineering analysis, "
    "and does not provide final fault certification. All high-risk, critical, or "
    "low-confidence findings require confirmation by a qualified human engineer before action."
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def api_stores(tmp_path: Path) -> tuple[JsonUploadStore, JsonRecordStore, JsonRecordStore]:
    return (
        JsonUploadStore(tmp_path / "uploads"),
        JsonRecordStore(tmp_path / "diagnoses"),
        JsonRecordStore(tmp_path / "reports"),
    )


@pytest.fixture(autouse=True)
def override_dependencies(
    tmp_path: Path,
    api_stores: tuple[JsonUploadStore, JsonRecordStore, JsonRecordStore],
):
    upload_store, diagnosis_store, report_store = api_stores
    settings = Settings(
        data_directory=DATA_ROOT,
        upload_directory=upload_store.root_directory,
        diagnosis_directory=diagnosis_store.root_directory,
        generated_report_directory=report_store.root_directory,
        fra_model_path=tmp_path / "missing-fra-model.pt",
        fra_label_map_path=tmp_path / "missing-label-map.json",
        use_fireworks=False,
        fireworks_api_key=None,
        fireworks_model=None,
    )

    async def settings_override() -> Settings:
        return settings

    async def upload_store_override() -> JsonUploadStore:
        return upload_store

    async def diagnosis_store_override() -> JsonRecordStore:
        return diagnosis_store

    async def report_store_override() -> JsonRecordStore:
        return report_store

    async def asset_diagnosis_directory_override() -> Path:
        return diagnosis_store.root_directory

    app.dependency_overrides[get_settings] = settings_override
    app.dependency_overrides[get_diagnosis_settings] = settings_override
    app.dependency_overrides[get_diagnosis_store] = diagnosis_store_override
    app.dependency_overrides[get_upload_store] = upload_store_override
    app.dependency_overrides[get_report_diagnosis_store] = diagnosis_store_override
    app.dependency_overrides[get_report_store] = report_store_override
    app.dependency_overrides[get_report_settings] = settings_override
    app.dependency_overrides[get_asset_diagnosis_directory] = asset_diagnosis_directory_override
    yield
    app.dependency_overrides.clear()


async def upload_file(
    client: AsyncClient,
    *,
    filename: str,
    file_type: str,
    asset_id: str | None = None,
) -> dict[str, object]:
    data = {"file_type": file_type}
    if asset_id is not None:
        data["asset_id"] = asset_id
    response = await client.post(
        "/api/upload",
        data=data,
        files={"file": (filename, (DATA_ROOT / filename).read_bytes(), "text/csv")},
    )
    assert response.status_code == 200
    return response.json()


@pytest.mark.anyio
async def test_diagnose_persists_result_and_drives_report_generation(
    api_stores: tuple[JsonUploadStore, JsonRecordStore, JsonRecordStore],
) -> None:
    _, diagnosis_store, report_store = api_stores
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        upload = await upload_file(
            client,
            filename="dcrm_fault_contact_wear.csv",
            file_type="dcrm",
        )
        diagnosis_response = await client.post(
            "/api/diagnose",
            json={
                "asset_id": "CB-402",
                "diagnostic_type": "dcrm",
                "upload_id": upload["upload_id"],
            },
        )

        assert diagnosis_response.status_code == 200
        diagnosis = diagnosis_response.json()
        assert diagnosis["diagnosis_id"].startswith("diag_")
        assert diagnosis["fault_class"] == "possible_contact_wear"
        assert diagnosis["risk_level"] == "High"
        assert diagnosis["requires_human_review"] is True
        assert "qualified engineer" in diagnosis["recommended_action"]
        assert diagnosis["safety_statement"] == REQUIRED_SAFETY_STATEMENT
        assert diagnosis_store.load(diagnosis["diagnosis_id"])["asset_id"] == "CB-402"

        report_response = await client.post(
            "/api/reports/generate",
            json={"diagnosis_id": diagnosis["diagnosis_id"]},
        )
        assets_response = await client.get("/api/assets")

    assert report_response.status_code == 200
    report = report_response.json()
    assert report["generation_mode"] == "template"
    assert SAFETY_STATEMENT in report["content"]
    assert report_store.load(report["report_id"])["diagnosis_id"] == diagnosis["diagnosis_id"]
    cb_402 = next(asset for asset in assets_response.json() if asset["asset_id"] == "CB-402")
    assert cb_402["risk_level"] == diagnosis["risk_level"]
    assert cb_402["health_score"] == 100 - diagnosis["risk_score"]


@pytest.mark.anyio
async def test_diagnose_rejects_upload_type_mismatch() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        upload = await upload_file(
            client,
            filename="fra_fault_winding_shift.csv",
            file_type="fra",
        )
        response = await client.post(
            "/api/diagnose",
            json={
                "asset_id": "TX-1",
                "diagnostic_type": "dcrm",
                "upload_id": upload["upload_id"],
            },
        )

    assert response.status_code == 400
    assert "contains FRA data" in response.json()["detail"]


@pytest.mark.anyio
async def test_diagnose_rejects_unknown_upload() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/diagnose",
            json={
                "asset_id": "TX-1",
                "diagnostic_type": "fra",
                "upload_id": "upload_000000000000",
            },
        )

    assert response.status_code == 404


@pytest.mark.anyio
async def test_diagnose_rejects_asset_selection_mismatch() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        upload = await upload_file(
            client,
            filename="dcrm_fault_contact_wear.csv",
            file_type="dcrm",
        )
        response = await client.post(
            "/api/diagnose",
            json={
                "asset_id": "CB-401",
                "diagnostic_type": "dcrm",
                "upload_id": upload["upload_id"],
            },
        )

    assert response.status_code == 400
    assert "was selected for asset CB-402" in response.json()["detail"]


@pytest.mark.anyio
async def test_fra_diagnosis_uses_configured_artifact_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_columns: list[str] = []

    def predictor(frame):
        observed_columns.extend(frame.columns)
        return DiagnosticAnalysisResult(
            fault_class="winding_deformation_suspected",
            is_anomalous=True,
            confidence=0.91,
            anomaly_score=0.78,
            evidence=["AMD artifact inference."],
            metrics={},
            analysis_method="fra_model_artifact",
            requires_human_review=True,
        )

    monkeypatch.setattr(
        routes_diagnostics,
        "create_fra_artifact_predictor",
        lambda model_path, label_map_path: predictor,
        raising=False,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        upload = await upload_file(
            client,
            filename="fra_fault_winding_shift.csv",
            file_type="fra",
        )
        response = await client.post(
            "/api/diagnose",
            json={
                "asset_id": "TX-1",
                "diagnostic_type": "fra",
                "upload_id": upload["upload_id"],
            },
        )

    assert response.status_code == 200
    assert response.json()["analysis_method"] == "fra_model_artifact"
    assert observed_columns == ["frequency_hz", "magnitude_db", "phase_deg"]


@pytest.mark.anyio
async def test_dcrm_diagnosis_does_not_load_fra_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        routes_diagnostics,
        "create_fra_artifact_predictor",
        lambda model_path, label_map_path: load_calls.append((model_path, label_map_path)),
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        upload = await upload_file(
            client,
            filename="dcrm_fault_contact_wear.csv",
            file_type="dcrm",
        )
        response = await client.post(
            "/api/diagnose",
            json={
                "asset_id": "CB-402",
                "diagnostic_type": "dcrm",
                "upload_id": upload["upload_id"],
            },
        )

    assert response.status_code == 200
    assert load_calls == []


@pytest.mark.anyio
async def test_diagnose_reports_unavailable_reference_data_as_service_error(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_directory=tmp_path / "missing-reference-data",
        diagnosis_directory=tmp_path / "diagnoses",
        use_fireworks=False,
    )

    async def settings_override() -> Settings:
        return settings

    app.dependency_overrides[get_diagnosis_settings] = settings_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        upload = await upload_file(
            client,
            filename="dcrm_fault_contact_wear.csv",
            file_type="dcrm",
        )
        response = await client.post(
            "/api/diagnose",
            json={
                "asset_id": "CB-402",
                "diagnostic_type": "dcrm",
                "upload_id": upload["upload_id"],
            },
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Diagnostic reference data is unavailable."


@pytest.mark.anyio
async def test_diagnose_uses_optional_uploaded_scada_context() -> None:
    custom_scada = (
        b"asset_id,timestamp,voltage_kv,current_a,temperature_c,status,alarm_code\n"
        b"CB-402,2026-07-08T09:30:00Z,400,510,55,closed,CUSTOM_SCADA_ALARM\n"
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        diagnostic_upload = await upload_file(
            client,
            filename="dcrm_fault_contact_wear.csv",
            file_type="dcrm",
        )
        scada_response = await client.post(
            "/api/upload",
            data={"file_type": "scada", "asset_id": "CB-402"},
            files={"file": ("custom_scada.csv", custom_scada, "text/csv")},
        )
        assert scada_response.status_code == 200

        response = await client.post(
            "/api/diagnose",
            json={
                "asset_id": "CB-402",
                "diagnostic_type": "dcrm",
                "upload_id": diagnostic_upload["upload_id"],
                "scada_upload_id": scada_response.json()["upload_id"],
            },
        )

    assert response.status_code == 200
    evidence = response.json()["evidence"]
    assert any("CUSTOM_SCADA_ALARM" in item for item in evidence)
    assert not any("CB_TIMING_DEVIATION" in item for item in evidence)


@pytest.mark.anyio
async def test_diagnosis_workflow_does_not_block_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_started = threading.Event()
    release_workflow = threading.Event()
    original_loader = JsonUploadStore.load_frame

    def slow_frame_loader(store, upload_id):
        workflow_started.set()
        release_workflow.wait()
        return original_loader(store, upload_id)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as setup_client:
        upload = await upload_file(
            setup_client,
            filename="dcrm_fault_contact_wear.csv",
            file_type="dcrm",
        )

    monkeypatch.setattr(
        JsonUploadStore,
        "load_frame",
        slow_frame_loader,
    )
    diagnosis_transport = ASGITransport(app=app)
    health_transport = ASGITransport(app=app)
    watchdog = threading.Timer(1.0, release_workflow.set)
    watchdog.start()
    async with (
        AsyncClient(
            transport=diagnosis_transport,
            base_url="http://testserver",
        ) as diagnosis_client,
        AsyncClient(transport=health_transport, base_url="http://testserver") as health_client,
    ):
        diagnosis_task = asyncio.create_task(
            diagnosis_client.post(
                "/api/diagnose",
                json={
                    "asset_id": "CB-402",
                    "diagnostic_type": "dcrm",
                    "upload_id": upload["upload_id"],
                },
            )
        )
        for _ in range(100):
            if workflow_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert workflow_started.is_set()
        health_response = await health_client.get("/api/health")
        frame_load_was_still_waiting = not release_workflow.is_set()
        release_workflow.set()
        diagnosis_response = await asyncio.wait_for(diagnosis_task, timeout=2.0)
    watchdog.cancel()

    assert health_response.status_code == 200
    assert frame_load_was_still_waiting is True
    assert diagnosis_response.status_code == 200
