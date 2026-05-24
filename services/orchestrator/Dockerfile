# orchestrator service Dockerfile. Same layered pattern as data_ingest/Dockerfile.

FROM marketplus/python-base:0.1

COPY pyproject.toml uv.lock /app/
COPY shared/pyproject.toml /app/shared/
COPY services/orchestrator/pyproject.toml /app/services/orchestrator/

RUN uv sync --frozen --package marketplus-orchestrator --no-install-project

COPY shared /app/shared
COPY services/orchestrator /app/services/orchestrator

RUN uv sync --frozen --package marketplus-orchestrator

EXPOSE 8000

CMD ["uv", "run", "--package", "marketplus-orchestrator", \
     "uvicorn", "orchestrator.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
