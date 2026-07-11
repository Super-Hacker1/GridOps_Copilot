"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes_assets import router as assets_router
from app.api.routes_diagnostics import router as diagnostics_router
from app.api.routes_health import router as health_router
from app.api.routes_reports import router as reports_router
from app.api.routes_runtime import router as runtime_router
from app.api.routes_upload import router as upload_router
from app.config import get_settings

app = FastAPI(
    title="GridOps Copilot App",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(assets_router)
app.include_router(diagnostics_router)
app.include_router(health_router)
app.include_router(reports_router)
app.include_router(runtime_router)
app.include_router(upload_router)
