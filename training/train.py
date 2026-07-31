"""Fine-tune a pretrained YOLO detector, log to MLflow, gate on mAP, export ONNX,
and register the result in the MLflow Model Registry.

This is the "develop + verify" stage. The pretrained backbone is bought; the
fine-tune, the quality gate, and the packaging are ours.

    python training/train.py                # reads params.yaml
    MLFLOW_TRACKING_URI=http://localhost:5000 python training/train.py

Exit code 1 means the quality gate failed — CI treats that as a red build and
nothing is registered or deployed.
"""
import os
import sys
from pathlib import Path

import mlflow
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "cv-detector"


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)["train"]


def register(client: "mlflow.MlflowClient", model_uri: str, run_id: str):
    """Create a model version from a plain artifact path.

    Note: mlflow.register_model() expects an artifact logged via a model flavor.
    We log a raw .onnx file, so we go through the registry API directly — this
    works identically on MLflow 2.x and 3.x.
    """
    try:
        client.create_registered_model(MODEL_NAME)
    except Exception:
        pass  # already exists
    return client.create_model_version(MODEL_NAME, source=model_uri, run_id=run_id)


def main() -> int:
    p = load_params()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("cv-object-detection")

    with mlflow.start_run() as run:
        mlflow.log_params(p)

        # --- Develop: transfer-learn from pretrained weights ---
        model = YOLO(p["base_weights"])
        results = model.train(
            data=p["dataset"],
            epochs=p["epochs"],
            imgsz=p["imgsz"],
            batch=p["batch"],
            project=str(ROOT / "training" / "runs"),
            name="finetune",
            exist_ok=True,
            verbose=False,
        )

        # --- Verify: validation metrics + hard quality gate ---
        metrics = {
            "mAP50": float(results.box.map50),
            "mAP50_95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }
        mlflow.log_metrics(metrics)
        print(f"Validation: {metrics}")

        gate = p["min_map50"]
        if metrics["mAP50"] < gate:
            print(f"QUALITY GATE FAILED: mAP50 {metrics['mAP50']:.3f} < {gate}")
            mlflow.set_tag("quality_gate", "failed")
            return 1
        mlflow.set_tag("quality_gate", "passed")

        # --- Package: export ONNX, log artifacts ---
        onnx_path = model.export(format="onnx", imgsz=p["imgsz"], opset=12)
        mlflow.log_artifact(str(onnx_path), artifact_path="onnx")
        best_pt = Path(model.trainer.best) if model.trainer else None
        if best_pt and best_pt.exists():
            mlflow.log_artifact(str(best_pt), artifact_path="weights")

        # --- Register and mark as the current candidate ---
        client = mlflow.MlflowClient()
        mv = register(client, f"runs:/{run.info.run_id}/onnx", run.info.run_id)
        client.set_registered_model_alias(MODEL_NAME, "candidate", mv.version)
        print(f"Registered {MODEL_NAME} v{mv.version} with alias 'candidate'.")
        print(f"Review it, then promote:  make promote VERSION={mv.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
