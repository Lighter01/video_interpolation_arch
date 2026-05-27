# Stage 1 MLflow Infrastructure

This directory contains the local MLflow/PostgreSQL/MinIO stack for Stage 1 experiment tracking.

## Setup

1. Copy `.env.example` to `.env` at the repository root if needed.
2. Replace placeholder values such as `MLFLOW_POSTGRES_PASSWORD`, `MINIO_ROOT_PASSWORD`, and `AWS_SECRET_ACCESS_KEY`.
3. Keep `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` aligned with the MinIO credentials used by the MLflow server.

Start the stack:

```bash
# Первый запуск или после изменения Dockerfile
docker compose --env-file .env -f infra/mlflow/docker-compose.yml up -d --build

# Обычный запуск
docker compose --env-file .env -f infra/mlflow/docker-compose.yml up -d
```

Open:

- MLflow UI: `http://localhost:5000`
- MinIO console: `http://localhost:9001`

Stop the stack:

```bash
# Остановить без удаления данных
docker compose --env-file .env -f infra/mlflow/docker-compose.yml down

# Полный сброс dev-данных
docker compose --env-file .env -f infra/mlflow/docker-compose.yml down -v
```

## Smoke Check

After the stack is running and `MLFLOW_TRACKING_URI=http://localhost:5000` is set:

```bash
uv run python -m video_interpolation.cli mlflow smoke-log
```

The command logs one param, one metric, and one tiny artifact to the configured tracking server.

## Notes

- Dataset files remain local in Stage 1. MinIO is used for MLflow artifacts.
- Baseline and future training commands use `MLFLOW_TRACKING_URI` from `.env`.
- Tiny local smoke commands may pass `--disable-mlflow`; real evaluation/training runs should log to MLflow.
- If the MLflow container cannot connect to PostgreSQL or MinIO, check the required variables in `.env` and restart the stack.
