# TASK — MVP BentoML Practical-RIFE Service for `pirsii_interpolator`

## Purpose

Implement a **ready-to-run MVP BentoML inference service** for integration with the existing `pirsii_interpolator` application.

This task is DevOps/integration-focused. The inference logic already exists in `src/video_interpolation/`; the goal is to wrap it in a BentoML service that the external worker can call over HTTP.

The implementation must prioritize a fast, stable MVP over a broad production architecture. Do not let the plan grow into a long 9–10 milestone project.

## Integration target

The service must be compatible with the current worker-side contract of `pirsii_interpolator`.

External application responsibilities:

- frontend talks only to backend;
- backend stores uploaded videos in MinIO;
- backend creates DB records and enqueues jobs;
- worker downloads input video from MinIO into a shared local/container path;
- worker calls BentoML;
- worker waits for the output file;
- worker uploads output back to MinIO;
- worker updates status.

BentoML responsibilities:

- read `input_path`;
- run Practical-RIFE inference;
- write `output_path`;
- return only after the output file is written;
- not know about MinIO/Postgres/Redis/frontend/backend internals.

Do not add MinIO, Postgres, Redis/RQ, frontend, FastAPI backend, queueing, or application DB logic to the BentoML service.

---

## Service location

Create the MVP service under:

```text
services/practical_rife_bentoml/
```

Expected files:

```text
services/practical_rife_bentoml/
  service.py
  Dockerfile
  README.md
  smoke_test.py
  docker-compose.gpu.example.yml
```

A `bentofile.yaml` may be added if useful, but do not block the MVP on BentoML Model Store or a complex Bento build flow.

The service must import the project package from:

```text
src/video_interpolation/
```

It must work from the repository root with:

```bash
PYTHONPATH=src
```

---

## Model/backend decisions

Use only:

- model: `Practical-RIFE v4.26`;
- backend: PyTorch runtime backend;
- device: `cuda`;
- execution mode: `sequential`;
- interpolation mode: `arbitrary_nx`;
- interpolation factor: only `2`, `3`, or `4`;
- default codec: `h264_nvenc`.

Do not use ONNX in this MVP service.

Do not use EMA-VFI.

Do not use AMT-S.

Do not use batch inference.

Do not dynamically switch backend/model per request.

The service should load the Practical-RIFE model once at service startup / worker initialization and reuse it for requests. It must not reload weights per frame or per request if the current runner can keep the model resident.

If the current `PracticalRIFEVideoInferenceRunner` cannot keep the model resident, add the smallest serving-facing fix needed.

---

## Required HTTP API

Implement exactly this endpoint:

```http
POST /interpolate_video
Content-Type: application/json
```

Required JSON body:

```json
{
  "input_path": "/shared/rife/job-123/input.mp4",
  "output_path": "/shared/rife/job-123/output.mp4",
  "interpolation_factor": 2
}
```

Required request fields:

- `input_path`: absolute path to an existing input video file;
- `output_path`: absolute path where the service must write the processed video;
- `interpolation_factor`: integer, exactly one of `2`, `3`, `4`.

Allowed optional fields:

```json
{
  "scale": 1.0,
  "output_playback_mode": "real_time",
  "quality_evaluation_enabled": true,
  "quality_sample_count": 16
}
```

Default optional values:

- `scale = 1.0`;
- `output_playback_mode = "real_time"`;
- `quality_evaluation_enabled = true`;
- `quality_sample_count = 16`.

The endpoint must reject invalid input clearly:

- missing `input_path`;
- non-absolute or nonexistent `input_path`;
- invalid or missing `output_path`;
- invalid `interpolation_factor`;
- invalid `scale`;
- invalid `output_playback_mode`;
- output directory cannot be created.

Validation errors should produce a 4xx-style response/exception where practical.

Inference errors should produce a 5xx-style response/exception where practical.

Do not fake success.

---

## Required response

Successful response must be small and worker-friendly.

Required fields:

```json
{
  "status": "completed",
  "output_path": "/shared/rife/job-123/output.mp4"
}
```

Allowed additional fields:

```json
{
  "interpolation_factor": 2,
  "duration_seconds": 12.34,
  "psnr_mean": 31.42,
  "ssim_mean": 0.948
}
```

Do not include in response:

- `device`;
- `backend`;
- `runtime_backend`;
- `quality_triplets_written`;
- `quality_triplet_output_dir`;
- MinIO URLs;
- video bytes;
- internal model/runtime details.

After inference, verify:

- `output_path` exists;
- `output_path` file size is greater than zero.

If output is missing or empty, treat it as an error.

---

## Path/shared-volume contract

The worker and BentoML container must see the same shared directory at the same absolute path.

Recommended path:

```text
/shared/rife
```

Expected runtime flow:

1. Worker downloads input from MinIO to `/shared/rife/.../input.mp4`.
2. Worker sends `input_path` and `output_path` to BentoML.
3. BentoML reads `input_path`.
4. BentoML writes `output_path`.
5. BentoML returns after `output_path` exists and is non-empty.
6. Worker uploads `output_path` to MinIO.

Support env variable:

```env
RIFE_SHARED_DIR=/shared/rife
```

The endpoint still receives concrete `input_path` and `output_path`; `RIFE_SHARED_DIR` is for documentation, validation helpers, smoke tests, and future deployment clarity.

Do not use ordinary `/tmp` across containers unless it is explicitly mounted as a shared volume.

---

## Environment variables

Support or document the actual env variables used by the current codebase.

Recommended MVP variables:

```env
MODEL_WEIGHTS_DIR=/app/model_weights
MODEL_REPOS_DIR=/app/model_repos
CONFIGS_DIR=/app/configs
RIFE_MODEL_CONFIG=/app/configs/models/practical_rife_v4_26.yaml
RIFE_SHARED_DIR=/shared/rife
RIFE_DEVICE=cuda
RIFE_CODEC=h264_nvenc
BENTOML_HOST=0.0.0.0
BENTOML_PORT=3000
CUDA_VISIBLE_DEVICES=0
```

If the project already uses different variable names through `Settings`, reuse existing names and document the exact mapping.

Weights are file-based, not BentoML Model Store-based.

Do not move model weights into BentoML Model Store in this task.

---

## Docker/MVP packaging

Implement a Docker path that is likely to work quickly.

Required:

- Dockerfile for the BentoML service;
- imports `src/video_interpolation` correctly;
- installs project dependencies;
- starts BentoML on `0.0.0.0:3000`;
- supports GPU runtime;
- expects model weights and shared job directory to be mounted if they are not part of the image;
- documents exactly what must be mounted.

Preferred MVP image behavior:

- copy project source and configs into the image;
- copy lightweight code/model repo files if required;
- do not require datasets/raw_data/outputs/notebooks inside the image;
- do not copy huge model weights into image unless Codex confirms they are intentionally present and this is the fastest reliable path.

Mount model weights by default if they are not tracked in git.

---

## GPU deployment model

MVP deployment model:

```text
one worker container + one BentoML container + one shared volume per GPU
```

Example mapping:

```text
worker-gpu0 -> http://rife-gpu0:3000
worker-gpu1 -> http://rife-gpu1:3000
worker-gpu2 -> http://rife-gpu2:3000
worker-gpu3 -> http://rife-gpu3:3000
```

Each BentoML container should see exactly one GPU where possible.

Use:

- Docker Compose GPU reservations;
- `CUDA_VISIBLE_DEVICES`;
- or both, if needed.

Do not implement internal multi-GPU load balancing inside one BentoML service.

Do not make `RIFE_SERVICE_URL` a list in this project.

Prepare a compact `docker-compose.gpu.example.yml` or documented overlay showing how to run one service instance bound to one GPU and mounted to one shared volume.

---

## BentoML service shape

Use a class-based BentoML service.

Recommended properties:

- one BentoML worker process;
- one GPU resource;
- long timeout, at least 600 seconds;
- synchronous API endpoint.

Add health/model-info endpoint if quick:

```text
GET/POST /health or /health_check
```

It should report that the service is loaded/ready without running a full video.

Keep health endpoint simple.

Do not implement async endpoint unless there is a strong reason.

Do not implement an internal task queue.

---

## Service behavior details

The endpoint must call existing project inference logic, preferably through `PracticalRIFEVideoInferenceRunner`.

Expected fixed runtime settings:

- `backend="torch"`;
- `device="cuda"`;
- `execution_mode="sequential"`;
- `interpolation_mode="arbitrary_nx"`;
- `codec="h264_nvenc"` by default;
- `quality_evaluation_enabled=True` by default.

Pass request/runtime values through:

- `input_path`;
- `output_path`;
- `interpolation_factor`;
- `scale`;
- `output_playback_mode`;
- `quality_evaluation_enabled`;
- `quality_sample_count`.

The service must not duplicate model-specific preprocessing, tensor formatting, checkpoint loading, or postprocessing logic.

---

## Logging

Add clear logs for:

- service startup;
- model/config path;
- selected device;
- endpoint request received;
- input path;
- output path;
- interpolation factor;
- scale;
- output playback mode;
- quality evaluation enabled/disabled;
- inference duration;
- output file check result;
- success/failure.

Do not log large payloads or video bytes.

---

## Smoke test client

Add `smoke_test.py` under `services/practical_rife_bentoml/`.

It should:

- send a POST request to `/interpolate_video`;
- use a local/shared input path;
- check HTTP status;
- check response JSON;
- check that output file exists and is non-empty;
- print concise success/failure messages.

It should be usable without `pirsii_interpolator`.

Example request:

```bash
python services/practical_rife_bentoml/smoke_test.py   --url http://localhost:3000/interpolate_video   --input-path /shared/rife/test/input.mp4   --output-path /shared/rife/test/output.mp4   --interpolation-factor 2
```

---

## Documentation requirement

Add detailed human-facing documentation.

Suggested doc location:

```text
docs/bentoml_practical_rife_mvp_service.md
```

The documentation must be practical and step-by-step, similar in spirit to `pirsii_interpolator/docs/local_debugging.md`.

It must explain:

1. What this BentoML service does.
2. What it does not do.
3. Required files/directories.
4. Required env variables.
5. Where Practical-RIFE weights must live.
6. How to start the service locally with `bentoml serve`.
7. How to build/run the Docker container.
8. How to mount `/shared/rife`.
9. How to run the smoke client.
10. How to call with `curl`.
11. Expected response.
12. How to verify the output file.
13. How to connect from `pirsii_interpolator` using:
    - `RIFE_SERVICE_URL=http://localhost:3000` outside Docker;
    - `RIFE_SERVICE_URL=http://rife-gpu0:3000` inside Docker Compose.
14. How one-worker/one-BentoML/one-GPU deployment works.
15. How to extend to multiple GPUs using multiple service instances.
16. Known MVP limitations.
17. Common troubleshooting:
    - CUDA unavailable;
    - model weights missing;
    - `h264_nvenc` unavailable;
    - shared path not mounted;
    - invalid interpolation factor;
    - output file missing/empty.

Update:

- `.agent/docs/PROJECT_MAP.md`;
- relevant Stage 2 docs / handoff docs;
- active ExecPlan.

---

## Tests and validation

Add focused tests where practical:

- request validation;
- invalid interpolation factors;
- service module import;
- service construction without running full GPU inference if CUDA is unavailable;
- response formatting;
- output file existence check helper;
- health endpoint if added.

Run:

```bash
uv run pytest
uv run ruff check src tests services
```

If the uv cache is not writable:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests services
```

Run only safe smoke tests locally.

If CUDA is unavailable, record GPU inference smoke as deferred.

---

## Manual verification commands

Documentation must include commands for at least:

### Local serve

```bash
PYTHONPATH=src bentoml serve services/practical_rife_bentoml/service.py:PracticalRIFEInterpolationService   --host 0.0.0.0   --port 3000
```

### Curl smoke

```bash
curl -X POST http://localhost:3000/interpolate_video   -H "Content-Type: application/json"   -d '{
    "input_path": "/shared/rife/test/input.mp4",
    "output_path": "/shared/rife/test/output.mp4",
    "interpolation_factor": 2
  }'
```

### Output verification

```bash
ls -lh /shared/rife/test/output.mp4
ffprobe /shared/rife/test/output.mp4
```

### Docker build/run

Provide exact commands chosen by the implementation.

### Multi-GPU compose example

Provide a minimal example showing:

- `rife-gpu0`;
- `CUDA_VISIBLE_DEVICES=0`;
- shared volume mounted at `/shared/rife`;
- port `3000`.

---

## Non-goals

Do not implement:

- MinIO access inside BentoML;
- Postgres access inside BentoML;
- Redis/RQ access inside BentoML;
- frontend/backend code;
- internal queues;
- Kubernetes;
- autoscaling;
- CI/CD or registry publishing;
- ONNX backend;
- EMA/AMT serving;
- model store release flow;
- multi-GPU scheduling inside one BentoML process;
- asynchronous job management inside BentoML.

---

## Acceptance criteria

The task is complete when:

- `services/practical_rife_bentoml/service.py` defines a BentoML service.
- `POST /interpolate_video` accepts JSON with `input_path`, `output_path`, `interpolation_factor`.
- Valid interpolation factors `2`, `3`, `4` pass.
- Invalid factors fail.
- The service uses Practical-RIFE v4.26 PyTorch CUDA sequential inference.
- The model is loaded once per service worker/instance.
- The service writes output exactly to `output_path`.
- The service checks that `output_path` exists and is non-empty.
- The service does not require MinIO/Redis/Postgres.
- Docker/service run docs exist.
- Shared-volume integration is documented.
- One-container-per-GPU deployment is documented.
- Smoke test client exists.
- Human-facing docs contain exact local run/debug commands.
- Tests and safe validation commands pass or CUDA-only checks are clearly deferred.

## End-of-task summary

At the end, summarize:

- files added/changed;
- service location;
- endpoint implemented;
- request/response fields;
- runner/model loading behavior;
- Docker/GPU/shared-volume support added;
- docs added;
- validation commands and results;
- CUDA smoke status;
- remaining manual integration steps with `pirsii_interpolator`.
