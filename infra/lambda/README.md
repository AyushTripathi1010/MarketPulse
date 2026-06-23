# MarketPulse — Lambda + EventBridge deployment

This directory contains the artifacts to deploy the orchestrator service
to **AWS Lambda** behind an **EventBridge** hourly schedule. The same
container image hosts both the cron-driven Lambda and a manually-invokable
HTTP API Lambda fronted by API Gateway.

## What gets deployed

Two Lambda functions sharing one Docker image:

| Function | Trigger | Purpose |
|---|---|---|
| `marketplus-orchestrator-cron-prod` | EventBridge (default: hourly `cron(0 * * * ? *)`) | Loops every supported ticker through the LangGraph pipeline and writes the markdown report to S3 |
| `marketplus-orchestrator-api-prod` | API Gateway (HTTP API) | Lets you POST `/run` for ad-hoc runs |

Both run from `public.ecr.aws/lambda/python:3.12` on **arm64 (Graviton)**.

## Pre-requisites

| Tool | How to get it |
|---|---|
| `aws` CLI ≥ 2.15 | `brew install awscli` then `aws configure` |
| `sam` CLI ≥ 1.140 | `brew install aws-sam-cli` |
| Docker | Docker Desktop running |
| AWS account | Free tier covers everything; pre-existing S3 bucket recommended |

The downstream services (`data_ingest`, `forecast`, `critic`, `report`)
must already be reachable at the URLs you'll pass as deploy parameters.
**For the portfolio-project case** you can either:

1. Deploy each as its own Lambda using the same SAM pattern as the
   orchestrator (cheapest, but ~5 lambdas to manage), OR
2. Run them on a single EC2 t2.micro behind nginx (single host, single
   port-mapped service, easier to reason about), OR
3. Skip live deployment and demo the orchestrator locally via
   `docker compose up` — the SAM template is then a *receipt* of the
   production architecture for interview discussion.

I went with option 3 during development. Option 2 is the cleanest cheap
path for a public-facing demo.

## First-time deploy

```bash
# 1. Make sure Docker Desktop is running.
docker info >/dev/null && echo "Docker OK"

# 2. Configure AWS credentials (you should already have a profile from
#    Phase 0 setup; see learning/12 — no, that one's QLoRA; see
#    learning/18 for AWS basics).
aws sts get-caller-identity   # sanity check

# 3. Run the guided deploy. SAM will prompt for stack name, region,
#    parameter values, and write samconfig.toml on success.
bash infra/lambda/deploy.sh --guided
```

SAM will walk you through:

- **Stack Name**: `marketplus-orchestrator` (default fine)
- **Region**: `us-east-1` is cheapest; `ap-south-1` if you want lower
  latency from India
- **Parameters**: paste the URLs of your already-deployed downstream
  services + the secrets (Groq, Langfuse, HuggingFace tokens)
- **Confirm changes**: `Y` (you'll see the resources sam is about to create)
- **Allow IAM role creation**: `Y` (sam creates least-privilege roles
  for each function)
- **Save args to samconfig.toml**: `Y` (so re-deploys are one command)

The first build takes ~5 min (Docker pulls the Lambda base image and uv).
Subsequent builds use the layer cache and finish in ~30 seconds.

## Day-to-day deploys

After the guided deploy created `samconfig.toml`:

```bash
bash infra/lambda/deploy.sh         # build + deploy with saved args
```

## Manually triggering the cron (without waiting for the hour)

```bash
aws lambda invoke \
    --function-name marketplus-orchestrator-cron-prod \
    --payload '{}' \
    /tmp/cron-out.json
cat /tmp/cron-out.json | jq .
```

You should see a summary dict with one entry per supported ticker:

```json
{
  "fired_at": "2026-05-24T09:00:00+00:00",
  "tickers_processed": 7,
  "tickers_failed": 0,
  "results": [
    {"ticker": "AAPL", "trace_id": "abc...", "drift_detected": false,
     "report_uri": "s3://marketplus-reports/.../AAPL/abc.md"},
    ...
  ]
}
```

## Tail the live logs

```bash
aws logs tail /aws/lambda/marketplus-orchestrator-cron-prod --follow
```

## Save the free-tier budget during development

The default cron is hourly (`cron(0 * * * ? *)`). That's 24 invocations
per day = ~720 per month. Well under Lambda's free 1M req/mo.

The bigger budget item is the downstream LLM and HF Inference API calls.
For development:

```bash
# Re-deploy with a daily schedule:
bash infra/lambda/deploy.sh \
    --parameter-overrides CronSchedule="cron(0 14 * * ? *)"
```

Or disable the schedule entirely:

```bash
aws events disable-rule \
    --name marketplus-orchestrator-OrchestratorCronFn-HourlyTick
```

## Tear-down

```bash
sam delete --stack-name marketplus-orchestrator --region us-east-1
```

This removes both Lambdas, the EventBridge rule, the API Gateway, and
the IAM roles. The S3 bucket persists (you Ref'd a pre-existing one,
not created it via the template) so reports survive teardown.

## What the SAM template intentionally does NOT do

- **Create the S3 reports bucket.** You pre-create it once; the
  template only Refs the name. Saves the headache of "stack delete also
  wipes my report history."
- **Provision Qdrant or Langfuse.** Those are infra you host separately
  (free Qdrant Cloud + self-hosted Langfuse on an EC2 t2.micro). The
  template just plumbs URLs/keys.
- **Manage secrets in Secrets Manager.** For portfolio scope I pass
  secrets as NoEcho parameters. Production would swap this for
  `AWS::SecretsManager::Secret` references.

See [`learning/18-aws-lambda-eventbridge.md`](../../learning/18-aws-lambda-eventbridge.md)
for the deeper story on Lambda + EventBridge + why we picked this
architecture.
