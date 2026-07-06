from fastapi.testclient import TestClient
from app import main


def test_health_endpoint_reports_service_status() -> None:
    assert hasattr(main, "app"), "FastAPI application has not been created"

    response = TestClient(main.app).get('/health')

    assert response.status_code == 200
    assert response.json() == {
        "status":"ok",
        "service":"gridops_copilot",
    }