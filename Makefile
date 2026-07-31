# MLOps CV pipeline — every runbook step is a make target.
#
# Config comes from .env (gitignored). Tracked manifests keep ${VAR} placeholders
# and are rendered into build/ at apply time, so your account values never enter
# version control and no tracked file is ever rewritten.
#
#   cp .env.example .env && $EDITOR .env
#   make help

SHELL := /bin/bash

ifneq (,$(wildcard .env))
include .env
export
endif

IMAGE_TAG ?= latest
BUILD := build
K8S_SRC := $(wildcard k8s/*.yaml)
K8S_OUT := $(patsubst k8s/%.yaml,$(BUILD)/%.yaml,$(K8S_SRC))

.PHONY: help check-env cluster storageclass render deploy-core mlflow-tunnel train \
        fetch-model build push monitoring deploy smoke-test drift-run promote \
        clean destroy

help: ## Show available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

check-env: ## Fail early if .env is missing or incomplete
	@test -f .env || { echo "ERROR: no .env — run: cp .env.example .env"; exit 1; }
	@test -n "$(ECR)" || { echo "ERROR: ECR unset in .env"; exit 1; }
	@test -n "$(AWS_REGION)" || { echo "ERROR: AWS_REGION unset in .env"; exit 1; }
	@echo "env ok: region=$(AWS_REGION) registry=$(ECR)"

# ---------------------------------------------------------------- infrastructure
cluster: ## Create the EKS cluster (~20 min, unattended — start this first)
	eksctl create cluster -f infra/eksctl-cluster.yaml

storageclass: ## Install a default gp3 StorageClass (EKS ships none — PVCs hang without it)
	kubectl apply -f infra/storageclass-gp3.yaml

# ---------------------------------------------------------------- render + deploy
$(BUILD)/%.yaml: k8s/%.yaml | $(BUILD)
	@envsubst '$$ECR $$MLFLOW_BUCKET $$IMAGE_TAG' < $< > $@

$(BUILD):
	@mkdir -p $(BUILD)

render: check-env $(K8S_OUT) ## Resolve ${VARS} from .env into build/ (gitignored)
	@echo "rendered $(words $(K8S_OUT)) manifests into $(BUILD)/"

deploy-core: render storageclass ## Namespace + MLflow tracking server & registry
	kubectl apply -f $(BUILD)/namespace.yaml
	kubectl apply -f $(BUILD)/mlflow.yaml
	kubectl -n mlops rollout status deploy/mlflow --timeout=300s

monitoring: ## Install kube-prometheus-stack + Pushgateway (MUST run before `deploy`)
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
	helm repo update
	helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
	  -n monitoring --create-namespace \
	  -f monitoring/kube-prometheus-values.yaml \
	  --set grafana.adminPassword="$(GRAFANA_ADMIN_PASSWORD)"
	helm upgrade --install pushgateway prometheus-community/prometheus-pushgateway \
	  -n monitoring --set serviceMonitor.enabled=true

deploy: render ## Deploy serving + drift CronJob + alerts + dashboard (run `monitoring` first)
	@kubectl get crd servicemonitors.monitoring.coreos.com >/dev/null 2>&1 || \
	  { echo "ERROR: Prometheus CRDs missing — run 'make monitoring' first"; exit 1; }
	kubectl apply -f $(BUILD)/serving.yaml
	kubectl apply -f $(BUILD)/drift-cronjob.yaml
	kubectl apply -f $(BUILD)/alerts.yaml
	kubectl apply -f monitoring/grafana-dashboard-configmap.yaml
	kubectl -n mlops rollout status deploy/cv-serving --timeout=300s

# ---------------------------------------------------------------- model lifecycle
mlflow-tunnel: ## Port-forward MLflow to localhost:5000 (keep running in its own terminal)
	kubectl -n mlops port-forward svc/mlflow 5000:5000

train: ## Fine-tune, gate on mAP, export ONNX, register (needs mlflow-tunnel)
	MLFLOW_TRACKING_URI=http://localhost:5000 python training/train.py

fetch-model: ## Download the ONNX behind an alias into serving/ (ALIAS=candidate|production)
	MLFLOW_TRACKING_URI=http://localhost:5000 python training/fetch_model.py $(or $(ALIAS),candidate)

promote: ## Point the production alias at a version: make promote VERSION=2
	@test -n "$(VERSION)" || { echo "Usage: make promote VERSION=<n>"; exit 1; }
	@MLFLOW_TRACKING_URI=http://localhost:5000 python -c "import mlflow; \
c = mlflow.MlflowClient(); \
c.set_registered_model_alias('cv-detector', 'production', '$(VERSION)'); \
print('cv-detector v$(VERSION) -> production')"

# ---------------------------------------------------------------- images
build: check-env ## Build serving + drift images
	docker build -t $(ECR)/mlops-cv/serving:$(IMAGE_TAG) -f serving/Dockerfile .
	docker build -t $(ECR)/mlops-cv/drift:$(IMAGE_TAG) -f monitoring/drift/Dockerfile .

push: check-env ## Push both images to ECR
	aws ecr get-login-password --region $(AWS_REGION) | \
	  docker login --username AWS --password-stdin $(ECR)
	docker push $(ECR)/mlops-cv/serving:$(IMAGE_TAG)
	docker push $(ECR)/mlops-cv/drift:$(IMAGE_TAG)

# ---------------------------------------------------------------- verification
smoke-test: ## Send one image through the deployed API: make smoke-test IMG=test.jpg
	@test -f "$(or $(IMG),test.jpg)" || { echo "No image. Try: curl -sLo test.jpg https://ultralytics.com/images/bus.jpg"; exit 1; }
	@kubectl -n mlops port-forward svc/cv-serving 8080:80 >/dev/null 2>&1 & \
	 PF=$$!; sleep 4; \
	 curl -sS -X POST -F "file=@$(or $(IMG),test.jpg)" http://localhost:8080/predict | python3 -m json.tool; \
	 kill $$PF

drift-run: ## Trigger the drift detector immediately instead of waiting for the hourly schedule
	-kubectl -n mlops delete job drift-manual --ignore-not-found
	kubectl -n mlops create job --from=cronjob/drift-detector drift-manual
	kubectl -n mlops wait --for=condition=complete job/drift-manual --timeout=180s || true
	kubectl -n mlops logs job/drift-manual

# ---------------------------------------------------------------- housekeeping
clean: ## Remove rendered manifests
	rm -rf $(BUILD)

destroy: ## Delete the cluster — nodes bill hourly, do not skip this
	eksctl delete cluster -f infra/eksctl-cluster.yaml
