# report service Dockerfile. Same layered pattern as data_ingest/Dockerfile.

FROM marketplus/python-base:0.1

COPY pyproject.toml uv.lock /app/
COPY shared/pyproject.toml /app/shared/
COPY services/report/pyproject.toml /app/services/report/

RUN uv sync --frozen --package marketplus-report --no-install-project

COPY shared /app/shared
COPY services/report /app/services/report

RUN uv sync --frozen --package marketplus-report

EXPOSE 8000

CMD ["uv", "run", "--package", "marketplus-report", \
     "uvicorn", "report.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
