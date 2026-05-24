# api_gateway service Dockerfile. Same layered pattern as data_ingest/Dockerfile.

FROM marketplus/python-base:0.1

COPY pyproject.toml uv.lock /app/
COPY shared/pyproject.toml /app/shared/
COPY services/api_gateway/pyproject.toml /app/services/api_gateway/

RUN uv sync --frozen --package marketplus-api-gateway --no-install-project

COPY shared /app/shared
COPY services/api_gateway /app/services/api_gateway

RUN uv sync --frozen --package marketplus-api-gateway

EXPOSE 8000

CMD ["uv", "run", "--package", "marketplus-api-gateway", \
     "uvicorn", "api_gateway.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
