"""System health API router"""

from fastapi import APIRouter

router = APIRouter(
    prefix="/api",
    tags=["system"],
)


@router.get("/health")
async def health() -> dict[str, str]:
    """Report whether the backend is running"""

    return {
        "status": "ok",
        "service": "gridops-copilot-backend",
    }
