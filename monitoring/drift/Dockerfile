# Drift detection job (Evidently -> Prometheus Pushgateway).
# Build from the repo root:  docker build -f monitoring/drift/Dockerfile .
FROM python:3.14-slim

RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY monitoring/drift/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY monitoring/drift/drift_job.py .

RUN chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1

CMD ["python", "drift_job.py"]
