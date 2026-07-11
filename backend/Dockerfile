FROM python:3.12-slim

ARG INSTALL_ML=false
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY backend /workspace/backend
COPY data /workspace/data
COPY models /workspace/models
COPY reports /workspace/reports

RUN python -m pip install --upgrade pip && \
    if [ "$INSTALL_ML" = "true" ]; then \
        python -m pip install --index-url "$TORCH_INDEX_URL" "torch>=2.5,<3.0"; \
    fi && \
    python -m pip install /workspace/backend

WORKDIR /workspace/backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
