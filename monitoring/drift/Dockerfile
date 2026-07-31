FROM python:3.11-slim
WORKDIR /app
COPY monitoring/drift/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY monitoring/drift/drift_job.py .
CMD ["python", "drift_job.py"]
