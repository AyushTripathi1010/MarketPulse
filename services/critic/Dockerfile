# critic service Dockerfile. Same layered pattern as data_ingest/Dockerfile.

FROM marketplus/python-base:0.1

COPY pyproject.toml uv.lock /app/
COPY shared/pyproject.toml /app/shared/
COPY services/critic/pyproject.toml /app/services/critic/

RUN uv sync --frozen --package marketplus-critic --no-install-project

COPY shared /app/shared
COPY services/critic /app/services/critic

RUN uv sync --frozen --package marketplus-critic

EXPOSE 8000

CMD ["uv", "run", "--package", "marketplus-critic", \
     "uvicorn", "critic.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
