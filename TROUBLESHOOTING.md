# Troubleshooting

Every failure below was hit while building this pipeline for real. The fixes are already applied in this repository — this document exists so you recognize the symptoms when a variation shows up in your own cluster, and so the reasoning behind some non-obvious settings is on the record.

A useful pattern runs through all of them: **the model was the easy part.** Training took five minutes; storage classes, version skew, memory limits, and resource ordering took the afternoon.

---

## 1. Pods stuck `Pending` forever, PVC shows `<unset>` storage class

**Symptom**

```
pod/mlflow-...   0/1   Pending
persistentvolumeclaim/mlflow-data   Pending   ...   <unset>
Warning  FailedScheduling  pod has unbound immediate PersistentVolumeClaims
```

**Cause.** Recent EKS versions ship a `gp2` StorageClass that is *not* marked default. A PVC that doesn't name a class gets nothing, binds to nothing, and its pod can never schedule.

**Fix.** Create a default class — `infra/storageclass-gp3.yaml`, applied by `make storageclass` (and automatically by `make deploy-core`):

```bash
kubectl get storageclass          # look for "(default)" next to a name
make storageclass
```

**Gotcha.** A PVC's storage class is immutable. Existing Pending claims will not pick up the new default — delete and recreate them.

**Lesson.** Storage is the first thing to verify on any new cluster.

---

## 2. MLflow model registration fails — HTTP 500, then 404

**Symptom A** — artifact upload fails with repeated 500s:

```
failed to log artifacts: ... too many 500 error responses
```

**Symptom B** — after that is fixed, registration fails with:

```
API request to endpoint /api/2.0/mlflow/logged-models/search failed with error code 404
```

**Cause A.** The server was configured to proxy artifacts to S3, but the pod had no AWS credentials (pods do not inherit your laptop's `aws configure`) and the stock image lacks the AWS libraries. Every upload failed.

**Cause B.** Client/server version skew. An MLflow **3.x client** calls registry endpoints that a **2.x server** does not implement.

**Fix.** Artifacts are written to the pod's own volume (`--artifacts-destination=/data/artifacts`), and the server image major version is kept aligned with `training/requirements.txt` (`mlflow>=3.0,<4.0` ↔ `ghcr.io/mlflow/mlflow:v3.14.0`).

**Lesson.** Pin client and server as a *pair*. Version skew across an API boundary is a production incident waiting to happen. For real S3-backed artifacts, give the pod an identity with IRSA rather than working around it.

---

## 3. MLflow pod `CrashLoopBackOff` — but the logs look perfectly healthy

**Symptom.** Pod restarts repeatedly; `kubectl logs` shows a clean startup ending in `Uvicorn running on http://0.0.0.0:5000` with no traceback.

**Diagnosis.** The kill reason is in pod *status*, not application logs:

```bash
kubectl -n mlops describe pod -l app=mlflow | grep -A5 "Last State"
#   Reason: OOMKilled     Exit Code: 137
```

**Cause.** MLflow 3.x starts multiple server workers by default; the pod exceeded its memory limit at 1 GiB and again at 2 GiB.

**Fix.** `--workers=1` plus a 2 GiB limit (both in `k8s/mlflow.yaml`). A single worker is correct here anyway — the backing store is SQLite on a ReadWriteOnce volume.

**Lesson.** When a container dies without saying why, read `Last State` and the exit code. 137 = OOMKilled, 143 = SIGTERM.

---

## 4. `kubectl apply` fails: no matches for kind "ServiceMonitor"

**Symptom**

```
error: resource mapping not found for kind "ServiceMonitor" in version "monitoring.coreos.com/v1"
ensure CRDs are installed first
```

**Cause.** `ServiceMonitor` and `PrometheusRule` are Custom Resource Definitions installed by kube-prometheus-stack. Applying manifests that reference them before the operator exists fails.

**Fix.** Install monitoring first. `make deploy` now hard-checks for the CRD and tells you what to run instead of failing cryptically:

```bash
make monitoring     # then
make deploy
```

**Lesson.** Deployment ordering is part of the architecture, not an implementation detail.

---

## 5. `port-forward` dies immediately after a rollout

**Symptom**

```
error forwarding port 5000 to pod ...: connection refused
error: lost connection to pod
```

**Cause.** `kubectl port-forward` attaches to a *specific pod*. Any rollout replaces that pod and the tunnel dies. Restarting it too quickly then hits a container that is running but whose server hasn't finished booting.

**Fix.** Restart the tunnel after the rollout completes. `k8s/mlflow.yaml` now defines a readiness probe, so `rollout status` means "actually serving" rather than merely "started."

**Lesson.** Without a readiness probe, "successfully rolled out" is a weaker statement than it appears.

---

## 6. `pip install -r requirements.txt` fails to resolve

**Symptom.** `No matching distribution found` on a dependency (in our case during the DVC install) on a very new Python version.

**Cause.** Exact pins from one era stop resolving as Python moves on — prebuilt wheels don't exist yet for the newest interpreter.

**Fix.** `training/requirements.txt` uses lower bounds with one deliberate ceiling on the MLflow major version (see #2). Serving and drift containers keep exact behavior via their own pinned images on Python 3.11.

**Lesson.** Pin what must not move (API-boundary versions); leave room where the ecosystem moves fast.

---

## 7. The model called a bus an "airplane"

**Symptom.** The gate passed at mAP@50 = 0.703, then the first live request on a normal street photo returned `"airplane"` at 0.68 confidence for the bus.

**Cause.** Fine-tuning on a 128-image dataset partially eroded classes the pretrained backbone already knew — catastrophic forgetting. The validation split was too small and too similar to catch it.

**Fix.** None applied — it is left visible on purpose. The real fixes are structural: larger and better-sliced evaluation sets, per-class gates rather than a single aggregate, and qualitative smoke tests on representative images before promotion.

**Lesson.** A passing metric is not a working model. Automated gates plus human eyes on real inputs — always both.

---

## 8. Drift reported as not detected

**Symptom.** After sending visibly different images, the drift job still reports `detected=False`.

**Causes and fixes.**

- **Too few samples.** Statistical tests are unreliable on tiny windows. Send 30+ requests per window; more is better.
- **Reference window is stale or wrong.** The first run bootstraps `reference.jsonl` from whatever traffic exists. If that traffic was already drifted, everything after looks normal. Reset it:
  ```bash
  kubectl -n mlops exec deploy/cv-serving -- rm -f /data/reference.jsonl /data/predictions/*.jsonl
  ```
- **After a retrain.** Re-baseline deliberately — a new model changes prediction-side features even when inputs are unchanged.

**Lesson.** Drift detection is only as meaningful as the reference window it compares against. Treat re-baselining as an explicit step in the promotion process.

---

## Quick diagnostic commands

```bash
kubectl -n mlops get pods,pvc                      # first look
kubectl -n mlops describe pod -l app=<name>        # events + Last State (OOM, scheduling)
kubectl -n mlops logs deploy/<name> --tail=50      # application view
kubectl get storageclass                           # is one marked (default)?
kubectl get crd | grep monitoring.coreos.com       # are the Prometheus CRDs installed?
kubectl -n mlops get servicemonitor                # will Prometheus scrape the model?
```
