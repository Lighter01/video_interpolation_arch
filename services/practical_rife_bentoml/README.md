# Practical-RIFE BentoML MVP Service

This directory contains the worker-facing MVP BentoML service for Practical-RIFE v4.26.

The service exposes one endpoint:

```http
POST /interpolate_video
```

It reads an input video from a shared filesystem path, runs Practical-RIFE v4.26 with the PyTorch CUDA runtime, writes the output video to the requested path, verifies that the output exists and is non-empty, then returns a small JSON response.

Detailed run/debug documentation lives in:

```text
docs/bentoml_practical_rife_mvp_service.md
```

## Contract

Required request:

```json
{
  "request": {
    "input_path": "/shared/rife/job-123/input.mp4",
    "output_path": "/shared/rife/job-123/output.mp4",
    "interpolation_factor": 2
  }
}
```

Optional fields:

```json
{
  "request": {
    "input_path": "/shared/rife/job-123/input.mp4",
    "output_path": "/shared/rife/job-123/output.mp4",
    "interpolation_factor": 2,
    "scale": 1.0,
    "output_playback_mode": "real_time",
    "quality_evaluation_enabled": true,
    "quality_sample_count": 16
  }
}
```

Success response:

```json
{
  "status": "completed",
  "output_path": "/shared/rife/job-123/output.mp4",
  "interpolation_factor": 2,
  "duration_seconds": 12.34,
  "psnr_mean": 31.42,
  "ssim_mean": 0.948
}
```

`psnr_mean` and `ssim_mean` are included only when quality evaluation produces them.

## Runtime Defaults

- Model: Practical-RIFE v4.26
- Backend: PyTorch
- Device: `cuda`
- Execution mode: `sequential`
- Interpolation mode: `arbitrary_nx`
- Allowed service factors: `2`, `3`, `4`
- Codec: `h264_nvenc` by default, override with `RIFE_CODEC`
- Shared path: `/shared/rife`

This service does not implement MinIO, Postgres, Redis, queues, frontend/backend APIs, ONNX, EMA, AMT, model selection, or internal multi-GPU routing.

## Local Serve

From the repository root:

```bash
MODEL_WEIGHTS_ROOT=model_weights \
MODEL_REPOS_ROOT=model_repos \
RIFE_SHARED_DIR=/shared/rife \
PYTHONPATH=src \
uv run bentoml serve services/practical_rife_bentoml/service.py:PracticalRIFEInterpolationService \
  --host 0.0.0.0 \
  --port 3000
```

The default service config uses CUDA and `h264_nvenc`, so this command requires a CUDA/NVENC-capable host for real requests.

Use CPU FFmpeg encoding as a compatibility fallback while keeping CUDA model inference:

```bash
RIFE_CODEC=libx264 \
MODEL_WEIGHTS_ROOT=model_weights \
MODEL_REPOS_ROOT=model_repos \
RIFE_SHARED_DIR=/shared/rife \
PYTHONPATH=src \
uv run bentoml serve services/practical_rife_bentoml/service.py:PracticalRIFEInterpolationService \
  --host 0.0.0.0 \
  --port 3000
```

## Docker Compose

The GPU example runs one service instance for one visible GPU:

```bash
docker compose -f services/practical_rife_bentoml/docker-compose.gpu.example.yml build rife-gpu0
docker compose -f services/practical_rife_bentoml/docker-compose.gpu.example.yml up rife-gpu0
```

The Compose file mounts:

- `${RIFE_SHARED_DIR:-/shared/rife}` to `/shared/rife`;
- `../../model_weights` to `/app/model_weights:ro`;
- `../../model_repos` to `/app/model_repos:ro`;
- `../../configs` to `/app/configs:ro`.

It also sets `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`, which is required for NVENC access inside the container. If NVENC is unavailable on the host, run the same service with `RIFE_CODEC=libx264` to keep CUDA inference and use CPU H.264 encoding.

```bash
RIFE_CODEC=libx264 \
docker compose -f services/practical_rife_bentoml/docker-compose.gpu.example.yml up --build rife-gpu0
```

From another Compose service on the same network, use:

```env
RIFE_SERVICE_URL=http://rife-gpu0:3000
```

## Smoke Client

After the service is running and the input video exists at a path visible to both the worker and service:

```bash
python services/practical_rife_bentoml/smoke_test.py \
  --url http://localhost:3000/interpolate_video \
  --input-path /shared/rife/test/input.mp4 \
  --output-path /shared/rife/test/output.mp4 \
  --interpolation-factor 2
```

The smoke client checks the HTTP response and verifies that the output path exists and has nonzero size.
