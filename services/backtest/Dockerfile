# backtest service Dockerfile. Same layered pattern as data_ingest/Dockerfile.

FROM marketplus/python-base:0.1

COPY pyproject.toml uv.lock /app/
COPY shared/pyproject.toml /app/shared/
COPY services/backtest/pyproject.toml /app/services/backtest/

RUN uv sync --frozen --package marketplus-backtest --no-install-project

COPY shared /app/shared
COPY services/backtest /app/services/backtest

RUN uv sync --frozen --package marketplus-backtest

EXPOSE 8000

CMD ["uv", "run", "--package", "marketplus-backtest", \
     "uvicorn", "backtest.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
