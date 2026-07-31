#!/usr/bin/env bash
# One-time AWS bootstrap: ECR repos + S3 buckets for DVC remote and MLflow artifacts.
# Usage: ./infra/bootstrap-aws.sh <aws-account-id> [region]
set -euo pipefail

ACCOUNT_ID="${1:?Usage: bootstrap-aws.sh <aws-account-id> [region]}"
REGION="${2:-us-east-1}"
SUFFIX="${ACCOUNT_ID: -6}"   # keep bucket names globally unique

for repo in mlops-cv/serving mlops-cv/drift; do
  aws ecr describe-repositories --repository-names "$repo" --region "$REGION" >/dev/null 2>&1 \
    || aws ecr create-repository --repository-name "$repo" --region "$REGION" >/dev/null
  echo "ECR repo ready: $repo"
done

for bucket in "mlops-cv-dvc-$SUFFIX" "mlops-cv-mlflow-$SUFFIX"; do
  aws s3api head-bucket --bucket "$bucket" 2>/dev/null \
    || aws s3 mb "s3://$bucket" --region "$REGION"
  echo "S3 bucket ready: $bucket"
done

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

cat <<EOF

Export these before running make targets:
  export AWS_ACCOUNT_ID=$ACCOUNT_ID
  export AWS_REGION=$REGION
  export ECR=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
  export DVC_BUCKET=mlops-cv-dvc-$SUFFIX
  export MLFLOW_BUCKET=mlops-cv-mlflow-$SUFFIX
EOF
