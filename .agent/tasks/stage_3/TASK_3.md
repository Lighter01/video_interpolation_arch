# Codex Prompt — Start Stage 3 Milestone 3 Implementation

Continue Stage 3 / MVP BentoML Practical-RIFE service integration.

Use `.agent/docs/exec-plans/active/03_practical_rife_bentoml_mvp.execplan.md` as the main working document.

Briefly re-read:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/tasks/stage_3/TASK_0.md`
- `.agent/tasks/stage_3/bentoml_mvp_service_task.md`
- `.agent/docs/exec-plans/active/03_practical_rife_bentoml_mvp.execplan.md`

Implement only **Milestone 3 — Docker, GPU, and Smoke Client**.

Do not start Milestone 4.

## Task

Add the practical container/run layer for the MVP Practical-RIFE BentoML service.

The goal is to make the service runnable as a GPU-enabled container with a shared `/shared/rife` volume and provide a small smoke client that verifies the worker-facing endpoint.

## Scope

Add or update files under:

```text
services/practical_rife_bentoml/
```

Expected files for this milestone:

```text
Dockerfile
docker-compose.gpu.example.yml
smoke_test.py
```

Optional if useful:

```text
bentofile.yaml
```

The implementation must support the current MVP contract:

- Practical-RIFE v4.26;
- PyTorch backend;
- CUDA device;
- sequential execution;
- `POST /interpolate_video`;
- shared path `/shared/rife`;
- one BentoML service container per GPU;
- no MinIO/Postgres/Redis/frontend/backend logic inside BentoML.

## Docker requirements

The Docker setup should:

- build from the repository root context;
- install the project/runtime dependencies;
- make `src/video_interpolation` importable;
- start BentoML on `0.0.0.0:3000`;
- support GPU runtime;
- document or implement required mounts for:
  - `/shared/rife`;
  - model weights;
  - model repos if required;
  - configs if not copied into the image.

Do not copy datasets, raw videos, outputs, notebooks, or other large/generated artifacts into the image.

Do not move model weights into BentoML Model Store.

## Compose example requirements

Add a compact GPU Compose example showing one service instance bound to one GPU.

It should demonstrate:

- `CUDA_VISIBLE_DEVICES=0`;
- port mapping to the host;
- `/shared/rife` shared volume;
- mounts or env vars required for model weights/configs/repos;
- service name suitable for worker usage, e.g. `rife-gpu0`;
- expected URL such as `http://rife-gpu0:3000` from another Compose service.

Do not implement internal multi-GPU balancing in one BentoML process.

Do not implement worker containers in this repository unless needed only as a documented example.

## Smoke client requirements

Add `smoke_test.py`.

It should:

- send a POST request to `/interpolate_video`;
- accept CLI arguments:
  - `--url`;
  - `--input-path`;
  - `--output-path`;
  - `--interpolation-factor`;
  - optional `--scale`;
  - optional `--output-playback-mode`;
  - optional `--disable-quality-evaluation`;
- check HTTP status;
- check response JSON;
- verify that `output_path` exists and is non-empty;
- print concise success/failure messages.

It must not require MinIO, Redis, Postgres, or the external worker.

## Constraints

Do not implement:

- Stage 3 Milestone 4 documentation closeout;
- changes to the worker/backend/frontend;
- MinIO/Postgres/Redis integration;
- ONNX/EMA/AMT paths;
- batch inference;
- new inference features;
- Kubernetes/autoscaling/CI/CD.

Keep this milestone focused on containerization, GPU run example, and smoke testing.

## Validation

Add tests where practical for:

- smoke client argument parsing;
- smoke client response validation helpers;
- service Docker/Compose files existing and containing expected key settings;
- service import remains valid.

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

If Docker/GPU is unavailable locally, do not fake container validation. Record Docker/GPU smoke as deferred and provide exact manual commands.

## Updates

Update:

- `.agent/docs/exec-plans/active/03_practical_rife_bentoml_mvp.execplan.md`
- `.agent/docs/PROJECT_MAP.md` if repository structure changes
- service README or minimal notes only as needed for the new files

Record in the ExecPlan:

- Dockerfile/build strategy;
- Compose GPU/shared-volume strategy;
- smoke client usage;
- validation results;
- deferred Docker/GPU checks, if any;
- what remains for Milestone 4.

## End-of-task summary

Summarize:

- Docker/service files added;
- how the container is expected to run;
- how GPU binding is configured;
- how `/shared/rife` is mounted;
- how the smoke client is used;
- what tests were added;
- validation command results;
- Docker/GPU smoke status;
- project-map/ExecPlan/doc updates;
- what remains for Milestone 4.

