# ============================================================================
# data_ingest service Dockerfile
#
# Build context is the REPO ROOT (set in docker-compose.yml) — we need root
# pyproject.toml + uv.lock for workspace resolution. uv installs ONLY this
# service's dependency slice via `--package marketplus-data-ingest`.
# ============================================================================

FROM marketplus/python-base:0.1

# Step 1: copy workspace metadata first (separate layer so dep installs are cached
# until pyproject/lock change). Code copy comes AFTER deps so a code-only edit
# doesn't bust the slow `uv sync` cache layer.
COPY pyproject.toml uv.lock /app/
COPY shared/pyproject.toml /app/shared/
COPY services/data_ingest/pyproject.toml /app/services/data_ingest/

# Step 2: install only this service's slice from the workspace lock.
# --frozen fails if the lockfile doesn't match pyproject.toml — guarantees
# reproducible builds.
RUN uv sync --frozen --package marketplus-data-ingest --no-install-project

# Step 3: now copy actual source code. If you only changed Python code, ONLY
# this layer and below rebuild — install layer stays cached.
COPY shared /app/shared
COPY services/data_ingest /app/services/data_ingest

# Step 4: install the project itself (the workspace members).
RUN uv sync --frozen --package marketplus-data-ingest

EXPOSE 8000

# `uv run` automatically activates the venv. --package picks the right
# workspace member's entrypoint resolution.
CMD ["uv", "run", "--package", "marketplus-data-ingest", \
     "uvicorn", "data_ingest.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
