#!/usr/bin/env bash
# ============================================================================
# deploy.sh — convenience wrapper around `sam build && sam deploy`.
#
# Why not just `sam deploy` directly? Two reasons:
#  1. SAM's working dir convention is the dir containing template.yaml,
#     but our Dockerfile needs the repo ROOT as build context. This
#     wrapper cd's to the repo root before invoking sam so paths resolve.
#  2. We bake in --use-container by default so the build happens in a
#     Lambda-shaped Docker container, not on the dev host. That catches
#     Apple Silicon (arm64) vs Lambda (arm64) match-ups locally.
#
# Pre-reqs:
#   - AWS CLI configured (`aws configure` or env vars)
#   - SAM CLI installed (`brew install aws-sam-cli`)
#   - Docker Desktop running (sam uses it to build the image)
#   - An ECR repo (sam creates one on first deploy if missing)
#
# Usage:
#   bash infra/lambda/deploy.sh                    # standard deploy
#   bash infra/lambda/deploy.sh --guided           # first-time setup
# ============================================================================

set -euo pipefail

# Resolve repo root regardless of where this script is invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${REPO_ROOT}"

# Sanity-check toolchain before doing anything destructive.
command -v sam >/dev/null 2>&1 || {
    echo "ERROR: 'sam' CLI not found. Install with: brew install aws-sam-cli"
    exit 1
}
command -v aws >/dev/null 2>&1 || {
    echo "ERROR: 'aws' CLI not found. Install with: brew install awscli"
    exit 1
}
docker info >/dev/null 2>&1 || {
    echo "ERROR: Docker daemon not running. Start Docker Desktop."
    exit 1
}

TEMPLATE="infra/lambda/template.yaml"
STACK_NAME="${STACK_NAME:-marketplus-orchestrator}"
REGION="${AWS_REGION:-us-east-1}"

echo "==> Building (sam build --use-container)..."
sam build \
    --template "${TEMPLATE}" \
    --use-container \
    --region "${REGION}"

echo "==> Deploying ${STACK_NAME} to ${REGION}..."
# Pass through extra args (e.g. --guided on first deploy).
sam deploy \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
    --capabilities CAPABILITY_IAM \
    --resolve-image-repos \
    "$@"

echo ""
echo "==> Deploy complete. Useful next steps:"
echo "   aws lambda invoke --function-name marketplus-orchestrator-cron-prod /tmp/out.json && cat /tmp/out.json"
echo "   aws logs tail /aws/lambda/marketplus-orchestrator-cron-prod --follow"
