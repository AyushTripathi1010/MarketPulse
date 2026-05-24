# forecast service Dockerfile. See services/data_ingest/Dockerfile for the
# layer-caching rationale (it's the same pattern across every service).

FROM marketplus/python-base:0.1

COPY pyproject.toml uv.lock /app/
COPY shared/pyproject.toml /app/shared/
COPY services/forecast/pyproject.toml /app/services/forecast/

RUN uv sync --frozen --package marketplus-forecast --no-install-project

COPY shared /app/shared
COPY services/forecast /app/services/forecast

RUN uv sync --frozen --package marketplus-forecast

EXPOSE 8000

CMD ["uv", "run", "--package", "marketplus-forecast", \
     "uvicorn", "forecast.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
