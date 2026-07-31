"""Download the ONNX artifact for a registered model version from MLflow
and place it at serving/model.onnx for the Docker build.

Usage: python training/fetch_model.py [alias-or-version]   (default: candidate)
"""
import os
import shutil
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ref = sys.argv[1] if len(sys.argv) > 1 else "candidate"
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = mlflow.MlflowClient()

    if ref.isdigit():
        mv = client.get_model_version("cv-detector", ref)
    else:
        mv = client.get_model_version_by_alias("cv-detector", ref)

    local_dir = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="onnx")
    onnx_files = list(Path(local_dir).glob("*.onnx"))
    if not onnx_files:
        print("No .onnx artifact found for that version.")
        return 1
    dest = ROOT / "serving" / "model.onnx"
    shutil.copy(onnx_files[0], dest)
    print(f"cv-detector v{mv.version} -> {dest}")
    print(f"Set MODEL_VERSION={mv.version} in k8s/serving.yaml (or via make deploy).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
