# Serving image: FastAPI + ONNX Runtime.
# Build from the repo root:  docker build -f serving/Dockerfile .
FROM python:3.14-slim

# Never run application code as root in a cluster.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/app ./app

# The ONNX model is baked in for immutability: an image tag identifies exactly
# one model version. `make fetch-model` places it here from the MLflow registry.
# At larger scale, pull from the registry at startup via an initContainer.
COPY serving/model.onnx /models/model.onnx

# Prediction logs go to a mounted volume; make it writable by the non-root user.
RUN mkdir -p /data/predictions && chown -R appuser:appuser /data /models /app

USER appuser

ENV MODEL_PATH=/models/model.onnx \
    PYTHONUNBUFFERED=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/healthz')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
