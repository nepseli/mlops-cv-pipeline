FROM python:3.11-slim

WORKDIR /app
COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/app ./app
# Bake the ONNX model into the image (fetched from MLflow by `make fetch-model`).
# In a larger setup you'd pull from the registry at startup via an initContainer instead.
COPY serving/model.onnx /models/model.onnx

ENV MODEL_PATH=/models/model.onnx
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
