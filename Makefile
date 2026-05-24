# ============================================================================
# MarketPulse — Makefile
#
# Just shortcuts for common commands. Type `make help` to see them all.
#
# Make targets are NOT shell scripts — each target line runs in its own
# subshell, so use `&&` to chain commands. Lines must be indented with a TAB
# (not spaces) — that's a make quirk from 1976.
# ============================================================================

.PHONY: help install up down restart logs test lint format typecheck health clean

# Self-documenting help — grep this Makefile for targets with ## comments.
help:
	@grep -E '^[a-zA-Z_-]+:.*?##' Makefile | awk -F':.*?##' '{printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install all Python deps (one venv at the root via uv workspaces)
	uv sync

up:  ## Start every service + Qdrant + Redis locally
	docker compose up -d
	@echo "Services starting. Run 'make health' in ~10s to verify."

down:  ## Stop and remove all containers
	docker compose down

restart:  ## Restart everything (down + up)
	docker compose down && docker compose up -d

logs:  ## Tail logs from all services
	docker compose logs -f

health:  ## Check /health endpoint of every service
	@for port in 8001 8002 8003 8004 8005 8006 8007; do \
		echo -n "Port $$port: "; \
		curl -s -m 2 http://localhost:$$port/health || echo "(no response)"; \
		echo; \
	done

test:  ## Run pytest across the workspace
	uv run pytest -q

lint:  ## Lint with ruff
	uv run ruff check .

format:  ## Auto-format with ruff
	uv run ruff format .
	uv run ruff check --fix .

typecheck:  ## Run mypy
	uv run mypy services shared

clean:  ## Remove caches and build artifacts (NOT containers/volumes)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
