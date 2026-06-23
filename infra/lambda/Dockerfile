# ============================================================================
# infra/lambda/Dockerfile — container image for the orchestrator Lambda.
#
# Why a SEPARATE Dockerfile from services/orchestrator/Dockerfile?
#   The dev Dockerfile uses python:3.12-slim — a Debian image you'd run
#   with `docker run`. Lambda wants the AWS Lambda Python base image
#   (public.ecr.aws/lambda/python:3.12). It includes the AWS Lambda Runtime
#   Interface Client (RIC) — the C glue that bridges Lambda's invoke API
#   to a Python function. Without the RIC the container starts and the
#   Lambda invoke pipe just hangs.
#
# Why ARM64?
#   Lambda Graviton (arm64) is ~20% cheaper and runs Python equally well.
#   For free-tier this doesn't matter on price, but it does matter for
#   image size — arm64 Lambda images are smaller. We pin to linux/arm64.
#
# Why one image for both api_handler AND cron_handler?
#   Same code, same dependencies. The SAM template overrides CMD per
#   function (CMD ["handler.api_handler"] vs CMD ["handler.cron_handler"]).
#   This is the AWS-idiomatic pattern for "many Lambdas, one container."
#
# Build context is the REPO ROOT — set in deploy.sh.
# ============================================================================

# Multi-arch base — arm64 is the default for cost; the build script can
# override with --platform=linux/amd64 if needed for a x86_64 dev iteration.
FROM --platform=$TARGETPLATFORM public.ecr.aws/lambda/python:3.12

# uv copied from Astral's official image (same trick we use in
# infra/docker/python-base.Dockerfile). We use uv to install deps into
# the Lambda task root from the workspace lockfile — much faster than
# pip and gives us the same reproducibility guarantees as dev.
COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /usr/local/bin/uv

# Lambda's base image sets $LAMBDA_TASK_ROOT to /var/task. That's where
# the runtime expects your code. We point uv at it.
ENV UV_PROJECT_ENVIRONMENT=$LAMBDA_TASK_ROOT/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR ${LAMBDA_TASK_ROOT}

# --- Layer 1: workspace metadata + lockfile (changes rarely) -----------
COPY pyproject.toml uv.lock ./
COPY shared/pyproject.toml shared/
COPY services/orchestrator/pyproject.toml services/orchestrator/

# Install ONLY the orchestrator's slice — keep the image lean. --frozen
# fails loudly if uv.lock doesn't match pyproject.toml (reproducibility).
RUN uv sync --frozen --package marketplus-orchestrator --no-install-project

# --- Layer 2: source code (changes often) ------------------------------
COPY shared/ shared/
COPY services/orchestrator/ services/orchestrator/

# Install our package into the lambda venv.
RUN uv sync --frozen --package marketplus-orchestrator

# --- Layer 3: Lambda handler -------------------------------------------
# Copied LAST because it's the most-edited file during deploy iterations.
COPY infra/lambda/handler.py ./

# Make our deps + handler discoverable to the Lambda runtime.
# The Lambda Python base image expects to import the handler module from
# /var/task — we add our venv's site-packages and the orchestrator src dir.
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}:${LAMBDA_TASK_ROOT}/.venv/lib/python3.12/site-packages:${LAMBDA_TASK_ROOT}/services/orchestrator/src:${LAMBDA_TASK_ROOT}/shared/src

# Default CMD — overridden in template.yaml per Lambda function.
# Format is `<module>.<callable>`. Module path is relative to LAMBDA_TASK_ROOT.
CMD ["handler.cron_handler"]
