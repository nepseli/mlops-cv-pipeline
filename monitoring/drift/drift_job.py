"""Evidently drift detection over serving prediction logs.

Compares the last CURRENT_HOURS of traffic against reference.jsonl (captured
during a known-good window) and pushes drift metrics to Prometheus Pushgateway.
Runs as a Kubernetes CronJob (see cronjob.yaml).

Maintain stage: alerts fire in Alertmanager when drift_share crosses threshold;
retraining is then a `make retrain` / GitHub Actions dispatch away.
"""
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

PRED_LOG_DIR = Path(os.environ.get("PRED_LOG_DIR", "/data/predictions"))
REFERENCE_PATH = Path(os.environ.get("REFERENCE_PATH", "/data/reference.jsonl"))
PUSHGATEWAY = os.environ.get("PUSHGATEWAY", "pushgateway.monitoring:9091")
CURRENT_HOURS = int(os.environ.get("CURRENT_HOURS", "24"))

NUMERIC_FEATURES = ["brightness", "contrast", "n_detections", "mean_confidence"]
CATEGORICAL_FEATURES = ["top_class"]


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def load_current() -> pd.DataFrame:
    cutoff = datetime.now(UTC) - timedelta(hours=CURRENT_HOURS)
    frames = []
    for f in sorted(PRED_LOG_DIR.glob("*.jsonl")):
        df = load_jsonl(f)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    return df[df["ts"] >= cutoff]


def main() -> int:
    if not REFERENCE_PATH.exists():
        # First run bootstraps the reference window from current traffic.
        current = load_current()
        if current.empty:
            print("No traffic yet; nothing to do.")
            return 0
        current.to_json(REFERENCE_PATH, orient="records", lines=True, date_format="iso")
        print(f"Bootstrapped reference window with {len(current)} records.")
        return 0

    reference = load_jsonl(REFERENCE_PATH)
    current = load_current()
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if current.empty or len(current) < 20:
        print(f"Only {len(current)} current records; skipping (need >= 20).")
        return 0

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[cols], current_data=current[cols])
    result = report.as_dict()

    summary = next(m for m in result["metrics"]
                   if m["metric"] == "DatasetDriftMetric")["result"]
    per_col = next(m for m in result["metrics"]
                   if m["metric"] == "DataDriftTable")["result"]["drift_by_columns"]

    registry = CollectorRegistry()
    g_share = Gauge("drift_share", "Share of drifting features", registry=registry)
    g_detected = Gauge("dataset_drift_detected", "1 if dataset drift detected",
                       registry=registry)
    g_col = Gauge("feature_drift_score", "Per-feature drift score (stattest-specific)",
                  ["feature"], registry=registry)
    g_rows = Gauge("drift_current_rows", "Rows in current window", registry=registry)

    g_share.set(summary["share_of_drifted_columns"])
    g_detected.set(1 if summary["dataset_drift"] else 0)
    g_rows.set(len(current))
    for col, info in per_col.items():
        g_col.labels(feature=col).set(info["drift_score"])

    push_to_gateway(PUSHGATEWAY, job="drift-detector", registry=registry)
    print(f"Pushed drift metrics: share={summary['share_of_drifted_columns']:.2f} "
          f"detected={summary['dataset_drift']} rows={len(current)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
