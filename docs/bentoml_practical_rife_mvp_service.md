# BentoML Practical-RIFE MVP Service

This guide describes the Stage 3 MVP BentoML service for integrating Practical-RIFE v4.26 with the external `pirsii_interpolator` worker.

The service is intentionally small. It is a path-based inference worker, not an application backend.

## What It Does

- Exposes `POST /interpolate_video`.
- Reads an already-downloaded input video from a shared filesystem path.
- Runs Practical-RIFE v4.26 through the project serving facade.
- Writes the processed video to the requested output path.
- Verifies that the output file exists and has nonzero size.
- Returns a small JSON response after the output is ready.

Runtime defaults:

- model: Practical-RIFE v4.26;
- backend: PyTorch;
- device: `cuda`;
- execution mode: `sequential`;
- interpolation mode: `arbitrary_nx`;
- service-level `interpolation_factor`: `2`, `3`, or `4`;
- `scale`: request parameter, default `1.0`;
- `output_playback_mode`: `real_time`, default;
- codec: `h264_nvenc` by default, service-level override through `RIFE_CODEC`;
- quality evaluation: enabled by default with `quality_sample_count=16`.

## What It Does Not Do

The BentoML service does not implement:

- upload/download APIs;
- MinIO access;
- Postgres or job state;
- Redis/RQ/Celery queues;
- frontend or FastAPI backend behavior;
- ONNX, EMA-VFI, or AMT-S routes;
- batched serving;
- per-request model/backend switching;
- internal multi-GPU scheduling;
- Kubernetes, autoscaling, auth, monitoring, or CI/CD.

The external `pirsii_interpolator` worker owns object storage, database status, queue handling, and retries. BentoML only interpolates the file paths it receives.

## Request Contract

Endpoint:

```http
POST /interpolate_video
Content-Type: application/json
```

Required JSON:

```json
{
  "request": {
    "input_path": "/shared/rife/job-123/input.mp4",
    "output_path": "/shared/rife/job-123/output.mp4",
    "interpolation_factor": 2
  }
}
```

Required fields:

- `input_path`: absolute path to an existing input video visible inside the BentoML container.
- `output_path`: absolute path where BentoML should write the output video.
- `interpolation_factor`: integer `2`, `3`, or `4`.

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

Optional field behavior:

- `scale`: Practical-RIFE runtime scale. Allowed values come from the serving facade: `0.25`, `0.5`, `1.0`, `2.0`, `4.0`.
- `output_playback_mode`: `real_time` writes at `input_fps * interpolation_factor`; `slow_motion` writes the same frame sequence at input FPS and omits audio.
- `quality_evaluation_enabled`: when true, the project computes PSNR/SSIM from first overlapping source triplets after the main video output is generated.
- `quality_sample_count`: non-negative integer controlling the maximum first-triplet quality sample count.

Validation rejects missing, relative, or nonexistent `input_path`, invalid `output_path`, invalid factor, invalid scale, invalid playback mode, negative quality sample count, and output directories that cannot be created.

## Response Contract

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

Required response fields:

- `status`;
- `output_path`.

Allowed extra fields:

- `interpolation_factor`;
- `duration_seconds`;
- `psnr_mean`;
- `ssim_mean`.

Quality metrics are omitted when quality evaluation is disabled or when it does not produce metrics.

The response intentionally does not include backend/device internals, video bytes, MinIO URLs, quality triplet paths, or model implementation details.

## Shared Path Contract

The worker and BentoML container must see the same files at the same absolute paths.

Recommended shared root:

```text
/shared/rife
```

Expected flow:

1. Worker downloads the source object from MinIO to `/shared/rife/<job-id>/input.mp4`.
2. Worker sends `input_path` and `output_path` to BentoML.
3. BentoML reads `input_path`.
4. BentoML writes `output_path`.
5. BentoML verifies the output exists and is non-empty.
6. BentoML returns `status="completed"`.
7. Worker uploads `output_path` back to MinIO and updates job state.

Do not use ordinary `/tmp` across containers unless it is explicitly mounted at the same path in both containers.

## Environment And Mounts

Current service/Docker environment names:

```env
MODEL_WEIGHTS_ROOT=/app/model_weights
MODEL_REPOS_ROOT=/app/model_repos
DATASET_ROOT=/app/datasets
RIFE_SHARED_DIR=/shared/rife
RIFE_DEVICE=cuda
RIFE_CODEC=h264_nvenc
NVIDIA_VISIBLE_DEVICES=0
NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
BENTOML_HOST=0.0.0.0
BENTOML_PORT=3000
CUDA_VISIBLE_DEVICES=0
```

`MODEL_WEIGHTS_ROOT` and `MODEL_REPOS_ROOT` are read by the project settings object. The Docker image sets them to `/app/model_weights` and `/app/model_repos`; the Compose example mounts the repository's `model_weights/` and `model_repos/` there.

Required model weight path inside the container:

```text
/app/model_weights/Practical-RIFE/RIFEv4.26/train_log/flownet.pkl
```

Configs are copied into the image and also mounted read-only by the Compose example:

```text
/app/configs
```

## Local BentoML Serve

From the repository root, on a CUDA/NVENC-capable host:

```bash
MODEL_WEIGHTS_ROOT=model_weights \
MODEL_REPOS_ROOT=model_repos \
RIFE_SHARED_DIR=/shared/rife \
PYTHONPATH=src \
uv run bentoml serve services/practical_rife_bentoml/service.py:PracticalRIFEInterpolationService \
  --host 0.0.0.0 \
  --port 3000
```

Make sure the input file path exists before calling the service:

```bash
mkdir -p /shared/rife/test
cp raw_data/tmp_test/DORA_cut.mp4 /shared/rife/test/input.mp4
```

This local command uses the default MVP service config: CUDA device and `h264_nvenc`. For a CUDA host without NVENC support, set `RIFE_CODEC=libx264` to keep CUDA model inference and use CPU FFmpeg encoding:

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

For a CPU-only development machine, use tests/import checks or project CLI CPU smokes rather than this MVP service, unless `RIFE_DEVICE` is intentionally overridden for debugging.

## Docker Build And Run

Build the image from the repository root:

```bash
docker build \
  -f services/practical_rife_bentoml/Dockerfile \
  -t video-interpolation/practical-rife-bentoml:mvp \
  .
```

Run one GPU-bound service container:

```bash
docker run --rm \
  --gpus '"device=0"' \
  -p 3000:3000 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -e MODEL_WEIGHTS_ROOT=/app/model_weights \
  -e MODEL_REPOS_ROOT=/app/model_repos \
  -e RIFE_SHARED_DIR=/shared/rife \
  -e RIFE_DEVICE=cuda \
  -e RIFE_CODEC=h264_nvenc \
  -v /shared/rife:/shared/rife \
  -v "$PWD/model_weights:/app/model_weights:ro" \
  -v "$PWD/model_repos:/app/model_repos:ro" \
  -v "$PWD/configs:/app/configs:ro" \
  video-interpolation/practical-rife-bentoml:mvp
```

The image does not copy datasets, raw videos, outputs, model exports, or model weights. Those must be mounted when needed.

To use CPU FFmpeg encoding instead of NVENC, add or override:

```env
RIFE_CODEC=libx264
```

Do not pass codec in the HTTP request; codec is service-level configuration.

## Docker Compose GPU Example

Use the provided example:

```bash
docker compose -f services/practical_rife_bentoml/docker-compose.gpu.example.yml build rife-gpu0
docker compose -f services/practical_rife_bentoml/docker-compose.gpu.example.yml up rife-gpu0
```

The example defines:

- service name: `rife-gpu0`;
- visible GPU: `CUDA_VISIBLE_DEVICES=0`;
- NVIDIA runtime device selection: `NVIDIA_VISIBLE_DEVICES=0`;
- NVIDIA driver capabilities: `compute,utility,video`;
- NVIDIA device reservation: `device_ids: ["0"]`;
- host port: `3000`;
- shared volume: `${RIFE_SHARED_DIR:-/shared/rife}:/shared/rife`;
- read-only model/config mounts.

From another Compose service on the same network, configure the worker with:

```env
RIFE_SERVICE_URL=http://rife-gpu0:3000
```

For multiple GPUs, run multiple service instances, one per GPU, with distinct service names, host ports, and `CUDA_VISIBLE_DEVICES` values. Example names:

```text
rife-gpu0 -> CUDA_VISIBLE_DEVICES=0 -> host port 3000
rife-gpu1 -> CUDA_VISIBLE_DEVICES=1 -> host port 3001
```

The external worker or deployment layer should choose which service URL to call. This MVP does not route across GPUs inside one BentoML process.

To run the Compose service with CPU H.264 encoding fallback while keeping CUDA inference:

```bash
RIFE_CODEC=libx264 \
RIFE_SHARED_DIR="$(pwd)/shared/rife" \
docker compose -f services/practical_rife_bentoml/docker-compose.gpu.example.yml up --build rife-gpu0
```

## Curl Smoke

With the service running:

```bash
curl -X POST http://localhost:3000/interpolate_video \
  -H "Content-Type: application/json" \
  -d '{
    "request": {
      "input_path": "/shared/rife/test/input.mp4",
      "output_path": "/shared/rife/test/output_2x.mp4",
      "interpolation_factor": 2,
      "scale": 1.0,
      "output_playback_mode": "real_time",
      "quality_evaluation_enabled": true,
      "quality_sample_count": 16
    }
  }'
```

Then verify:

```bash
test -s /shared/rife/test/output_2x.mp4
```

## Smoke Client

The repository also includes a small standard-library HTTP smoke client:

```bash
python services/practical_rife_bentoml/smoke_test.py \
  --url http://localhost:3000/interpolate_video \
  --input-path /shared/rife/test/input.mp4 \
  --output-path /shared/rife/test/output_2x.mp4 \
  --interpolation-factor 2
```

Disable quality evaluation for a faster smoke:

```bash
python services/practical_rife_bentoml/smoke_test.py \
  --url http://localhost:3000/interpolate_video \
  --input-path /shared/rife/test/input.mp4 \
  --output-path /shared/rife/test/output_2x_no_quality.mp4 \
  --interpolation-factor 2 \
  --disable-quality-evaluation
```

The client fails if the service is unreachable, the response JSON is not valid, the response status is not `completed`, the response output path does not match the requested path, or the output file is missing/empty.

## Worker Integration Notes

For `pirsii_interpolator`, the expected request lifecycle is:

1. Backend receives upload and stores it in MinIO.
2. Backend creates a database job record and enqueues work.
3. Worker downloads input to a mounted shared path, usually `/shared/rife/<job-id>/input.mp4`.
4. Worker calls `POST ${RIFE_SERVICE_URL}/interpolate_video`.
5. Worker waits for the HTTP response.
6. Worker verifies or trusts the service-verified output path.
7. Worker uploads the output file to MinIO.
8. Worker updates the job status.

Worker environment:

```env
RIFE_SERVICE_URL=http://rife-gpu0:3000
RIFE_SHARED_DIR=/shared/rife
```

The worker should send absolute paths from `RIFE_SHARED_DIR`. Relative paths are rejected.

## Common Troubleshooting

CUDA unavailable:

- Symptom: model startup fails, `torch.cuda.is_available()` is false, or container cannot see a GPU.
- Check `nvidia-smi` on the host and inside a CUDA-capable container.
- Check that Docker has NVIDIA runtime support and the Compose service has a GPU reservation.
- Confirm `CUDA_VISIBLE_DEVICES` matches the intended GPU.

Missing weights:

- Symptom: startup or first request fails while loading `flownet.pkl`.
- Check `MODEL_WEIGHTS_ROOT`.
- Confirm this file exists inside the service container:

```bash
ls /app/model_weights/Practical-RIFE/RIFEv4.26/train_log/flownet.pkl
```

Missing shared mount:

- Symptom: request fails with `input_path must point to an existing video file`.
- Confirm the worker and service containers mount the same host directory to the same absolute path.
- Avoid writing input to a worker-only `/tmp`.

Invalid interpolation factor:

- Symptom: request validation fails for `interpolation_factor`.
- Use only `2`, `3`, or `4` for this MVP service.

`h264_nvenc` unavailable:

- Symptom: encoding fails even though model inference starts.
- Confirm the host has NVIDIA encoder support and the FFmpeg build exposes `h264_nvenc`:

```bash
ffmpeg -hide_banner -encoders | grep nvenc
```

- Confirm the container can see NVIDIA video devices:

```bash
ls -l /dev/nvidia* /dev/nvidia-caps 2>/dev/null
```

- Run a direct one-frame encoder smoke inside the container:

```bash
ffmpeg -hide_banner \
  -f lavfi -i testsrc=size=128x128:rate=1 \
  -frames:v 1 \
  -c:v h264_nvenc \
  -f null -
```

- Ensure `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` is set for the service container.
- If NVENC remains unavailable, run the service with `RIFE_CODEC=libx264`. This keeps Practical-RIFE model inference on CUDA unless `RIFE_DEVICE` is also changed, but video encoding runs on CPU.

Output file missing or empty:

- Symptom: service returns an internal error after inference.
- Check that the output parent directory is writable by the BentoML process.
- Check codec errors in service logs.
- Check available disk space on the shared volume.

## Current Validation Status

Automated tests and static checks passed in the implementation environment:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests services
```

Safe command checks passed:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python services/practical_rife_bentoml/smoke_test.py --help
docker compose -f services/practical_rife_bentoml/docker-compose.gpu.example.yml config
```

Docker CLI was available, but real Docker/GPU request validation was deferred because `nvidia-smi` reported GPU access blocked by the operating system in the implementation environment. Run the Docker/Compose commands above on the target GPU server before treating deployment as production-ready.
