# ============================================================================
# python-base.Dockerfile — shared base image for every MarketPulse service.
#
# Building this once and FROM-ing it in each service Dockerfile means:
#   1. uv only gets downloaded once across all images
#   2. python:3.12-slim only gets pulled once
#   3. service images build faster
#
# Build it locally with:
#   docker build -f infra/docker/python-base.Dockerfile -t marketplus/python-base:0.1 .
#
# Then each service's Dockerfile starts with:
#   FROM marketplus/python-base:0.1
# ============================================================================

FROM python:3.12-slim

# Install system-level tools that almost every service needs.
# - curl: health checks, debugging
# - ca-certificates: TLS for any HTTPS API call (yFinance, Alpaca, etc.)
# - git: some pip/uv packages still install from git refs
# Then clean apt cache to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install uv by copying the pre-built binary from Astral's official image.
# This is the recommended pattern from Astral — much faster than `pip install uv`.
# Version pin matches what you installed locally so dev = prod.
COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /usr/local/bin/uv

# uv-specific environment knobs for container builds.
# - UV_COMPILE_BYTECODE=1: pre-compile .pyc files at install (faster startup)
# - UV_LINK_MODE=copy:    don't hardlink (which can break across Docker volumes)
# - UV_PROJECT_ENVIRONMENT=/app/.venv: explicit venv path
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app
