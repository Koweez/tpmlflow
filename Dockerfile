FROM ghcr.io/mlflow/mlflow

WORKDIR /app

RUN pip install --no-cache-dir mlflow fastapi uvicorn

COPY server.py /app
COPY utils.py /app

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
