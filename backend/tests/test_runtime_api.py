"""Tests for the AMD runtime evidence API."""

import hashlib
from inspect import signature
import json
from pathlib import Path

import pytest
from httpx2 import ASGITransport, AsyncClient

from app.api.routes_runtime import get_amd_evidence_path, get_fra_model_path
from app.main import app
from app.services.runtime_evidence import load_amd_evidence


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_amd_evidence_endpoint_reports_pending_when_file_is_missing(
    tmp_path: Path,
) -> None:
    async def override_evidence_path() -> Path:
        return tmp_path / "missing-amd-evidence.json"

    app.dependency_overrides[get_amd_evidence_path] = override_evidence_path
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get("/api/runtime/amd-evidence")
    finally:
        app.dependency_overrides.pop(get_amd_evidence_path, None)

    assert response.status_code == 200
    assert response.json() == {
        "amd_usage_claim": "No AMD training evidence file found yet.",
        "status": "pending",
    }


@pytest.mark.anyio
async def test_amd_evidence_endpoint_returns_notebook_evidence(tmp_path: Path) -> None:
    model_path = tmp_path / "fra_cnn_rocm.pt"
    model_path.write_bytes(b"verified-amd-model")
    artifact_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    evidence_path = tmp_path / "amd_training_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-07-11T00:00:00+00:00",
                "training_platform": "AMD AI Notebooks / AMD Developer Cloud",
                "framework": "PyTorch ROCm",
                "torch_version": "2.7.1+rocm6.3",
                "hip_version": "6.3",
                "gpu_available": True,
                "device_name": "AMD Instinct MI300X",
                "model_artifact": "models/fra_cnn_rocm.pt",
                "artifact_sha256": artifact_sha256,
                "metrics": {"accuracy": 0.91, "f1_macro": 0.89},
                "benchmarks": {
                    "cpu_batch_ms": 120.0,
                    "amd_gpu_batch_ms": 18.0,
                    "speedup": 6.67,
                },
            }
        ),
        encoding="utf-8",
    )

    async def override_evidence_path() -> Path:
        return evidence_path

    async def override_model_path() -> Path:
        return model_path

    app.dependency_overrides[get_amd_evidence_path] = override_evidence_path
    app.dependency_overrides[get_fra_model_path] = override_model_path
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get("/api/runtime/amd-evidence")
    finally:
        app.dependency_overrides.pop(get_amd_evidence_path, None)
        app.dependency_overrides.pop(get_fra_model_path, None)

    assert response.status_code == 200
    assert response.json() == {
        "amd_usage_claim": "FRA model trained and benchmarked on AMD GPU using ROCm PyTorch.",
        "generated_at": "2026-07-11T00:00:00+00:00",
        "training_platform": "AMD AI Notebooks / AMD Developer Cloud",
        "framework": "PyTorch ROCm",
        "torch_version": "2.7.1+rocm6.3",
        "hip_version": "6.3",
        "gpu_available": True,
        "device_name": "AMD Instinct MI300X",
        "model_artifact": "models/fra_cnn_rocm.pt",
        "artifact_sha256": artifact_sha256,
        "metrics": {"accuracy": 0.91, "f1_macro": 0.89},
        "benchmarks": {
            "cpu_batch_ms": 120.0,
            "amd_gpu_batch_ms": 18.0,
            "speedup": 6.67,
        },
        "status": "complete",
    }


@pytest.mark.anyio
async def test_cpu_only_notebook_does_not_claim_amd_gpu_training(tmp_path: Path) -> None:
    evidence_path = tmp_path / "amd_training_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "training_platform": "Local CPU smoke test",
                "framework": "PyTorch",
                "torch_version": "2.7.1",
                "hip_version": None,
                "gpu_available": False,
                "device_name": "CPU only / not captured",
                "model_artifact": "models/fra_cnn_rocm.pt",
                "metrics": {"accuracy": 0.5, "f1_macro": 0.4},
                "benchmarks": {
                    "cpu_batch_ms": 120.0,
                    "amd_gpu_batch_ms": None,
                    "speedup": None,
                },
            }
        ),
        encoding="utf-8",
    )

    async def override_evidence_path() -> Path:
        return evidence_path

    app.dependency_overrides[get_amd_evidence_path] = override_evidence_path
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/runtime/amd-evidence")
    finally:
        app.dependency_overrides.pop(get_amd_evidence_path, None)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "incomplete"
    assert body["gpu_available"] is False
    assert body["amd_usage_claim"] == (
        "AMD GPU training evidence is incomplete because the notebook did not detect a GPU."
    )


@pytest.mark.anyio
async def test_amd_evidence_endpoint_rejects_malformed_json(tmp_path: Path) -> None:
    evidence_path = tmp_path / "amd_training_evidence.json"
    evidence_path.write_text("{not valid JSON", encoding="utf-8")

    async def override_evidence_path() -> Path:
        return evidence_path

    app.dependency_overrides[get_amd_evidence_path] = override_evidence_path
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/runtime/amd-evidence")
    finally:
        app.dependency_overrides.pop(get_amd_evidence_path, None)

    assert response.status_code == 500
    assert response.json()["detail"] == "AMD training evidence is invalid."


@pytest.mark.anyio
async def test_amd_evidence_rejects_string_gpu_flag_instead_of_claiming_training(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "amd_training_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "training_platform": "Unknown",
                "framework": "PyTorch",
                "torch_version": "2.7.1",
                "hip_version": None,
                "gpu_available": "false",
                "device_name": "CPU",
                "model_artifact": "models/fra_cnn_rocm.pt",
                "metrics": {"accuracy": 0.5, "f1_macro": 0.4},
                "benchmarks": {
                    "cpu_batch_ms": 120.0,
                    "amd_gpu_batch_ms": None,
                    "speedup": None,
                },
            }
        ),
        encoding="utf-8",
    )

    async def override_evidence_path() -> Path:
        return evidence_path

    app.dependency_overrides[get_amd_evidence_path] = override_evidence_path
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/runtime/amd-evidence")
    finally:
        app.dependency_overrides.pop(get_amd_evidence_path, None)

    assert response.status_code == 500


@pytest.mark.anyio
async def test_amd_evidence_rejects_gpu_claim_without_rocm_hip_version(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "amd_training_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "training_platform": "Unverified GPU",
                "framework": "PyTorch",
                "torch_version": "2.7.1",
                "hip_version": None,
                "gpu_available": True,
                "device_name": "Unknown GPU",
                "model_artifact": "models/fra_cnn_rocm.pt",
                "metrics": {"accuracy": 0.9, "f1_macro": 0.8},
                "benchmarks": {
                    "cpu_batch_ms": 120.0,
                    "amd_gpu_batch_ms": 18.0,
                    "speedup": 6.67,
                },
            }
        ),
        encoding="utf-8",
    )

    async def override_evidence_path() -> Path:
        return evidence_path

    app.dependency_overrides[get_amd_evidence_path] = override_evidence_path
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/runtime/amd-evidence")
    finally:
        app.dependency_overrides.pop(get_amd_evidence_path, None)

    assert response.status_code == 500


def test_amd_evidence_marks_deployed_artifact_hash_mismatch_incomplete(
    tmp_path: Path,
) -> None:
    if "model_path" not in signature(load_amd_evidence).parameters:
        pytest.fail("AMD evidence loader does not verify the deployed model artifact")

    recorded_artifact = b"trained-on-amd"
    deployed_model = tmp_path / "fra_cnn_rocm.pt"
    deployed_model.write_bytes(b"different-runtime-artifact")
    evidence_path = tmp_path / "amd_training_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "training_platform": "AMD AI Notebooks / AMD Developer Cloud",
                "framework": "PyTorch ROCm",
                "torch_version": "2.7.1+rocm",
                "hip_version": "6.3",
                "gpu_available": True,
                "device_name": "AMD Instinct",
                "model_artifact": "models/fra_cnn_rocm.pt",
                "artifact_sha256": hashlib.sha256(recorded_artifact).hexdigest(),
                "metrics": {"accuracy": 0.9, "f1_macro": 0.8},
                "benchmarks": {
                    "cpu_batch_ms": 120.0,
                    "amd_gpu_batch_ms": 18.0,
                    "speedup": 6.67,
                },
            }
        ),
        encoding="utf-8",
    )

    evidence = load_amd_evidence(evidence_path, model_path=deployed_model)

    assert evidence.status == "incomplete"
    assert "does not match" in evidence.amd_usage_claim
