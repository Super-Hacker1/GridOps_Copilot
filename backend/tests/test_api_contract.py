"""Public API contract smoke tests."""

import pytest
from httpx2 import ASGITransport, AsyncClient

from app.main import app


def test_required_backend_routes_are_registered() -> None:
    paths = set(app.openapi()["paths"])

    assert {
        "/api/health",
        "/api/assets",
        "/api/upload",
        "/api/diagnose",
        "/api/reports/generate",
        "/api/runtime/amd-evidence",
    } <= paths


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_frontend_origin_can_call_the_api() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.options(
            "/api/assets",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
