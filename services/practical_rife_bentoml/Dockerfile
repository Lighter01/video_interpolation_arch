FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    PYTHONPATH=/app/src \
    MODEL_WEIGHTS_ROOT=/app/model_weights \
    MODEL_REPOS_ROOT=/app/model_repos \
    DATASET_ROOT=/app/datasets \
    RIFE_SHARED_DIR=/shared/rife \
    RIFE_DEVICE=cuda \
    RIFE_CODEC=h264_nvenc \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    BENTOML_HOST=0.0.0.0 \
    BENTOML_PORT=3000 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir uv

COPY README.md pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY configs ./configs
COPY services/practical_rife_bentoml ./services/practical_rife_bentoml

RUN mkdir -p /shared/rife /app/model_weights /app/model_repos /app/datasets

EXPOSE 3000

CMD ["bentoml", "serve", "services/practical_rife_bentoml/service.py:PracticalRIFEInterpolationService", "--host", "0.0.0.0", "--port", "3000"]
