# 03 Practical-RIFE BentoML MVP Service ExecPlan

## 1. Title and Metadata

- Stage: Stage 3 kickoff / MVP BentoML Practical-RIFE service integration
- Status: completed, pending user acceptance
- Created: 2026-06-03
- Updated: 2026-06-03
- Stage plan: no dedicated Stage 3 stage plan is present in `.agent/stage_plans/`; scope is controlled by `.agent/tasks/stage_3/TASK_0.md` and `.agent/tasks/stage_3/bentoml_mvp_service_task.md`
- Global plan: `.agent/docs/general_plan.md` is referenced by `.agent/AGENTS.md` and `.agent/docs/PLANS.md`, but is not present in the current repository tree
- Prior ExecPlan: `.agent/docs/exec-plans/completed/02_inference_runtime_refactor.execplan.md`; `.agent/docs/exec-plans/completed/02_5_inference_runtime_stabilization.execplan.md`
- Scope authority: direct user requests on 2026-06-03 plus `.agent/tasks/stage_3/TASK_0.md`; implementation is proceeding milestone-by-milestone from this active ExecPlan

## 2. Stage Goal

Build a fast, ready-to-run MVP BentoML service for Practical-RIFE v4.26 that matches the `pirsii_interpolator` worker contract.

The service must expose path-based HTTP inference:

```http
POST /interpolate_video
```

It reads an already-downloaded input video from a shared filesystem path, runs Practical-RIFE v4.26 through the existing project inference stack, writes the output video to the requested path, verifies the file exists and is non-empty, and returns only after that output is ready.

This task is not a broad backend stage. BentoML must not know about MinIO, Postgres, Redis, job queues, frontend upload flows, or worker state. The external `pirsii_interpolator` worker owns those responsibilities and calls this service over HTTP.

## 3. Source Documents and Authority

Read for this plan:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/tasks/stage_3/TASK_0.md`
- `.agent/tasks/stage_3/bentoml_mvp_service_task.md`
- `.agent/docs/exec-plans/completed/02_inference_runtime_refactor.execplan.md`
- `.agent/docs/exec-plans/completed/02_5_inference_runtime_stabilization.execplan.md`
- `docs/stage2_inference_runtime_refactor.md`
- `docs/stage2_5_inference_runtime_stabilization.md`
- `src/video_interpolation/serving.py`
- `src/video_interpolation/inference.py`
- `src/video_interpolation/settings.py`
- `configs/models/practical_rife_v4_26.yaml`
- `configs/inference/practical_rife_v4_26_2x.yaml`
- `examples/bentoml/practical_rife_torch_service/service.py`
- `examples/bentoml/practical_rife_onnx_service/service.py`
- `pyproject.toml`

Direct task files control scope. The completed Stage 2 and Stage 2.5 ExecPlans control accepted runtime decisions: Practical-RIFE v4.26, PyTorch first, CUDA device, sequential serving, arbitrary-Nx mode, request-time factor `2..4`, runtime `scale`, and no serving recommendation for current batched inference.

## 4. Context and Current Repository State

The core inference implementation already exists under `src/video_interpolation/`:

- `src/video_interpolation/serving.py` provides `PracticalRIFEServingConfig`, `PracticalRIFEVideoInferenceRunner`, and `run_practical_rife_video_inference(...)`.
- `src/video_interpolation/inference.py` owns video decode, preprocessing, frame interleaving, model calls, postprocessing, PyAV/FFmpeg encoding, real-time/slow-motion output FPS, audio remuxing, and optional Practical-RIFE quality evaluation.
- `src/video_interpolation/inference_runtime/rife.py` and `src/video_interpolation/inference_runtime/rife_upstream/` own the Practical-RIFE PyTorch runtime boundary and project-owned RIFE source.
- `src/video_interpolation/adapters/rife.py` loads Practical-RIFE v4.26 weights and exposes adapter methods used by the serving runner.
- `src/video_interpolation/video_quality/` owns optional first-triplet quality evaluation with quality-only downscaling and triplet writing disabled by default in serving config.

Current Practical-RIFE model/config state:

- `configs/models/practical_rife_v4_26.yaml` is the active model config and defaults to `device: cuda`, `scale: 1.0`, `divisor: 128`, and mode support for `fixed_2x` plus `arbitrary_nx`.
- `configs/inference/practical_rife_v4_26_2x.yaml` uses `interpolation_mode: arbitrary_nx`, `execution_mode: sequential`, `inference_batch_size: 1`, `output_playback_mode: real_time`, and `codec: h264_nvenc` for GPU-oriented local inference.
- `model_weights/Practical-RIFE/RIFEv4.26/train_log/flownet.pkl` is the expected checkpoint through the configured weight root.

Current BentoML example state:

- `examples/bentoml/practical_rife_torch_service/service.py` is a developer-facing compatibility example. It is not the requested production-ish MVP service directory.
- `examples/bentoml/practical_rife_onnx_service/service.py` exists as an alternate ONNX example, but ONNX is out of scope for this MVP.
- `services/practical_rife_bentoml/service.py` now exists after Milestones 1-2 with the worker-compatible service contract, persistent runner startup, request-value pass-through, and output verification. Docker, smoke client, and production docs are still pending.

Dependency state:

- `pyproject.toml` already includes `bentoml>=1.4.39`, `torch`, `av`, `opencv-python-headless`, `pydantic`, and related inference dependencies.
- No additional dependency is planned for the MVP unless implementation proves that a small HTTP client dependency is missing for `smoke_test.py`; the standard library or already-installed stack should be preferred.

Repository state notes:

- There are active misc ExecPlans for video quality and slow-motion output playback. They are not part of this MVP service plan.
- Existing untracked generated artifacts under `model_exports/` and `outputs/` are not part of this task and must not be cleaned by this plan.

## 5. Stage Requirements Restated

Implement an MVP service under:

```text
services/practical_rife_bentoml/
  service.py
  Dockerfile
  README.md
  smoke_test.py
  docker-compose.gpu.example.yml
```

Optional if useful:

```text
services/practical_rife_bentoml/bentofile.yaml
```

Runtime decisions:

- model: Practical-RIFE v4.26 only;
- backend: PyTorch only;
- device: `cuda`;
- execution mode: `sequential`;
- interpolation mode: `arbitrary_nx`;
- service-level `interpolation_factor`: exactly `2`, `3`, or `4`;
- default codec: `h264_nvenc`;
- default `scale`: `1.0`;
- default `output_playback_mode`: `real_time`;
- default `quality_evaluation_enabled`: `true`;
- default `quality_sample_count`: `16`;
- no ONNX, EMA-VFI, AMT-S, or batched inference in this MVP.

Required endpoint:

```http
POST /interpolate_video
Content-Type: application/json
```

Required JSON request:

```json
{
  "input_path": "/shared/rife/job-123/input.mp4",
  "output_path": "/shared/rife/job-123/output.mp4",
  "interpolation_factor": 2
}
```

Allowed optional request fields:

```json
{
  "scale": 1.0,
  "output_playback_mode": "real_time",
  "quality_evaluation_enabled": true,
  "quality_sample_count": 16
}
```

Required success response:

```json
{
  "status": "completed",
  "output_path": "/shared/rife/job-123/output.mp4"
}
```

Allowed extra response fields:

- `interpolation_factor`
- `duration_seconds`
- `psnr_mean`
- `ssim_mean`

Response must not include device/backend/runtime internals, video bytes, MinIO URLs, quality triplet paths, or model implementation details.

Validation requirements:

- reject missing, relative, or nonexistent `input_path`;
- reject missing or invalid `output_path`;
- create the output directory if possible, otherwise return a clear validation error;
- reject interpolation factors outside `2`, `3`, `4`;
- reject invalid `scale` and `output_playback_mode`;
- treat inference failures as service errors;
- after inference, verify `output_path` exists and has size greater than zero;
- do not fake success.

Shared-volume contract:

- worker and BentoML container must see the same path, recommended `/shared/rife`;
- document and support `RIFE_SHARED_DIR=/shared/rife` for smoke tests and deployment clarity;
- endpoint still receives concrete absolute paths.

## 6. Non-Goals and Deferred Work

Do not implement:

- MinIO, Postgres, Redis/RQ, frontend, FastAPI backend, queueing, or application DB logic;
- ONNX service path;
- EMA-VFI or AMT-S service path;
- batched inference for serving;
- Kubernetes, autoscaling, CI/CD, monitoring, auth, model registry, or BentoML Model Store migration;
- internal multi-GPU load balancing in one BentoML process;
- `RIFE_SERVICE_URL` list handling;
- video upload/download APIs or byte payload APIs;
- audio stretching for slow-motion output;
- model-weight relocation into the image unless the user explicitly approves it.

Deferred but documented:

- CUDA and `h264_nvenc` smoke validation if the current environment lacks GPU/NVENC support;
- multi-GPU production rollout by running one service instance per GPU;
- worker-side integration changes in `pirsii_interpolator`.

## 7. Architecture and Implementation Strategy

Use the existing project serving runner as the only inference boundary:

```text
BentoML request JSON
→ request validation and absolute path checks
→ PracticalRIFEVideoInferenceRunner.load() once per service instance
→ runner.run(input_path, output_path, factor, scale, playback mode, quality flags)
→ output file existence/size check
→ small worker-compatible response
```

The new service must not duplicate model-specific preprocessing, padding, checkpoint loading, frame ordering, quality evaluation, or encoding. Those stay in `src/video_interpolation/serving.py` and `src/video_interpolation/inference.py`.

Service implementation:

- `services/practical_rife_bentoml/service.py` should define a class-based BentoML service using BentoML `1.4.39` style.
- Configure one worker process where practical, one GPU resource, and a long timeout, at least 600 seconds.
- Initialize one `PracticalRIFEVideoInferenceRunner` in service construction and load/reuse it for requests.
- Use fixed config: `backend="torch"`, `device` from `RIFE_DEVICE` defaulting to `cuda`, `codec` from `RIFE_CODEC` defaulting to `h264_nvenc`.
- Add a simple health/model-info endpoint if it can be done without broadening the task. It should report readiness/config only and must not run a video.

Validation and response helpers:

- Add small project-local helpers in the service directory for request parsing, path validation, output-directory creation, output-file verification, and response shaping.
- Keep helper tests fast by using fakes/mocks for runner calls where possible.

Docker and shared-volume integration:

- `Dockerfile` should build from repository root context, install dependencies, include `src/`, `configs/`, the service directory, and required project files, and start BentoML on `0.0.0.0:3000`.
- Do not copy datasets, raw data, outputs, or huge generated artifacts into the image.
- Prefer mounted model weights and model repos unless implementation verifies a small, intentional copy is faster and reliable.
- `docker-compose.gpu.example.yml` should show one BentoML service instance bound to one GPU with `/shared/rife`, model weights, model repos, and configs mounted.

Documentation:

- Add `docs/bentoml_practical_rife_mvp_service.md` for detailed local run/debug instructions in the same practical spirit as `pirsii_interpolator/docs/local_debugging.md`.
- Update `docs/stage2_inference_runtime_refactor.md` only as needed to point from the previous examples to the new MVP service.
- Update `.agent/docs/PROJECT_MAP.md` after implementation adds `services/` and the new docs.

## 8. Milestones and Work Breakdown

### Milestone 1 - Service Contract and Validation

Objective: Implement the worker-compatible BentoML endpoint with strict request validation and small response shaping.

Likely files:

- `services/practical_rife_bentoml/service.py`
- `tests/` focused service tests

Expected output:

- `POST /interpolate_video` accepts required fields and allowed optional fields.
- Invalid requests fail clearly before model inference.
- Successful fake-run path returns `{"status": "completed", "output_path": ...}` plus allowed metric fields.
- Health/model-info endpoint added only if quick and low-risk.

Validation checkpoint:

- focused tests for request validation, response formatting, output-file verification, and service import without CUDA inference.

### Milestone 2 - Persistent Runner Startup and Path-Based Inference Wiring

Objective: Wire the service to `PracticalRIFEVideoInferenceRunner` without per-request model reload and enforce the shared-volume path contract.

Likely files:

- `services/practical_rife_bentoml/service.py`
- `src/video_interpolation/serving.py` only if a minimal runner fix is required
- `tests/` focused runner/service tests

Expected output:

- Service creates a persistent Practical-RIFE runner once.
- Runner uses Practical-RIFE v4.26, PyTorch, CUDA, sequential, arbitrary-Nx, `interpolation_factor` `2..4`.
- Request values pass through: `input_path`, `output_path`, `interpolation_factor`, `scale`, `output_playback_mode`, `quality_evaluation_enabled`, `quality_sample_count`.
- Output directory creation and output existence/size checks happen around the runner call.

Validation checkpoint:

- fake-run test proves the runner object is reused and not reconstructed per request.
- CPU-safe import/construction tests pass without CUDA.

### Milestone 3 - Docker, GPU, and Smoke Client

Objective: Add a practical local/container path for one BentoML service instance per GPU.

Likely files:

- `services/practical_rife_bentoml/Dockerfile`
- `services/practical_rife_bentoml/docker-compose.gpu.example.yml`
- `services/practical_rife_bentoml/smoke_test.py`
- optional `services/practical_rife_bentoml/bentofile.yaml`

Expected output:

- Dockerfile installs/runs the service with `PYTHONPATH=src` or equivalent package install.
- Compose example documents one GPU, one shared volume, one service URL.
- Smoke client posts to `/interpolate_video`, checks JSON response, and verifies output file existence/nonzero size.

Validation checkpoint:

- service module import check.
- smoke client help/argument parsing check.
- Docker commands documented; actual GPU container smoke deferred if CUDA/NVENC is unavailable locally.

### Milestone 4 - Documentation, Project Map, and Handoff

Objective: Leave the MVP usable by backend/service developers.

Likely files:

- `services/practical_rife_bentoml/README.md`
- `docs/bentoml_practical_rife_mvp_service.md`
- `docs/stage2_inference_runtime_refactor.md`
- `.agent/docs/PROJECT_MAP.md`
- this ExecPlan

Expected output:

- Human-facing docs explain local serve, Docker run, shared mounts, curl smoke, smoke client, expected response, `pirsii_interpolator` URL values, multi-GPU-by-multiple-instances model, and troubleshooting.
- ExecPlan records validation results, deferred GPU checks, and handoff.

Validation checkpoint:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests services`
- safe import/smoke checks only; no long videos or full GPU jobs unless available and explicitly bounded.

## 9. Validation Strategy

Automated tests should cover behavior that affects the external worker contract:

- request validation for required fields;
- absolute/existing input path checks;
- output path and output-directory handling;
- interpolation factor validation for exactly `2`, `3`, `4`;
- scale and output playback mode validation;
- response shape excludes runtime internals;
- output file exists and is non-empty after fake inference;
- service module import and class construction without running full GPU inference;
- persistent runner reuse in the test path;
- optional health endpoint shape if implemented.

Validation commands:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests services
```

Safe manual checks to document and run when feasible:

```bash
PYTHONPATH=src bentoml serve services/practical_rife_bentoml/service.py:PracticalRIFEInterpolationService --host 0.0.0.0 --port 3000
```

```bash
python services/practical_rife_bentoml/smoke_test.py \
  --url http://localhost:3000/interpolate_video \
  --input-path /shared/rife/test/input.mp4 \
  --output-path /shared/rife/test/output.mp4 \
  --interpolation-factor 2
```

```bash
curl -X POST http://localhost:3000/interpolate_video \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "/shared/rife/test/input.mp4",
    "output_path": "/shared/rife/test/output.mp4",
    "interpolation_factor": 2
  }'
```

If CUDA or NVENC is unavailable, record the GPU smoke as deferred and verify only import, validation, fake-run tests, and documentation commands.

## 10. Expected Artifacts

Expected implementation artifacts:

- `services/practical_rife_bentoml/service.py`
- `services/practical_rife_bentoml/Dockerfile`
- `services/practical_rife_bentoml/README.md`
- `services/practical_rife_bentoml/smoke_test.py`
- `services/practical_rife_bentoml/docker-compose.gpu.example.yml`
- optional `services/practical_rife_bentoml/bentofile.yaml`
- focused tests under `tests/`
- `docs/bentoml_practical_rife_mvp_service.md`
- updates to `docs/stage2_inference_runtime_refactor.md`
- updates to `.agent/docs/PROJECT_MAP.md`
- this ExecPlan updated with progress, decisions, validation, and handoff

No model weights, datasets, ONNX artifacts, benchmark artifacts, or output videos are expected as source artifacts for this MVP.

## 11. Risks, Assumptions, and Recovery

Assumptions:

- `bentoml>=1.4.39` is installed through the project environment.
- Practical-RIFE v4.26 weights are available through `MODEL_WEIGHTS_ROOT` or mounted at the documented container path.
- The deployment machine has CUDA and NVENC when using default `device=cuda` and `codec=h264_nvenc`.
- Worker and service containers mount the same shared directory at the same absolute path.

Risks:

- CUDA may be unavailable in the local implementation environment.
- FFmpeg/PyAV may not expose `h264_nvenc` even when CUDA is present.
- BentoML API exception-to-HTTP mapping may require a small compatibility choice.
- Mounted path mistakes can look like nonexistent input files or missing output files.
- Loading the model in BentoML class initialization may be too eager for some local smoke commands; keep import tests separate from GPU-load smoke.

Recovery:

- Document `libx264` as a CPU fallback for local debugging only, while keeping MVP default `h264_nvenc`.
- If CUDA/NVENC smoke cannot run locally, validate with fake runner/import tests and record GPU smoke as deferred.
- If BentoML object construction loads CUDA too early for tests, isolate request/response helper tests and use monkeypatching/fakes.
- If `PracticalRIFEVideoInferenceRunner` cannot keep the model resident, make the smallest fix in `src/video_interpolation/serving.py` and cover it with a focused reuse test.

Dependency decision:

- No new runtime dependency is planned. If implementation needs an HTTP client for `smoke_test.py`, prefer standard-library `urllib.request`; otherwise, add a dependency only after confirming it is already implied or asking the user.

## 12. Progress

- 2026-06-03: Created planning-only ExecPlan. No production code, service files, tests, or docs were implemented yet.
- 2026-06-03: Implemented Milestone 1 service contract and validation. Added `services/practical_rife_bentoml/service.py` with the BentoML `PracticalRIFEInterpolationService`, `POST /interpolate_video` route, `InterpolateVideoRequest` model, validation helpers, fake-run-friendly `run_interpolate_video_request(...)`, output file existence/non-empty verification, and small worker response shaping. Added `tests/test_practical_rife_bentoml_mvp_service.py` with focused request-validation, response-formatting, fake-run, and endpoint exposure tests. Updated `.agent/docs/PROJECT_MAP.md` for the new `services/` directory and active Stage 3 ExecPlan.
- 2026-06-03: Implemented Milestone 2 persistent runner/path-based inference wiring. `PracticalRIFEInterpolationService.__init__()` now creates one runner and calls `load()` once during service instance construction. `run_interpolate_video_request(...)` calls the runner with validated `input_path`, `output_path`, `interpolation_factor`, `scale`, `output_playback_mode`, `quality_evaluation_enabled`, and `quality_sample_count`, then verifies the output file. `PracticalRIFEVideoInferenceRunner.run(...)` and `run_practical_rife_video_inference(...)` now accept request-level `quality_sample_count`, defaulting to the serving config when omitted, and pass it into `VideoQualityEvaluationConfig.sample_count`. Added focused tests for service runner loading/reuse, request-value pass-through, inference-error surfacing, and runner-to-`VideoInferenceConfig` quality sample count wiring. Updated `.agent/docs/PROJECT_MAP.md` for the new persistent runner behavior.
- 2026-06-03: Implemented Milestone 3 Docker/GPU/smoke-client layer. Added `services/practical_rife_bentoml/Dockerfile`, `services/practical_rife_bentoml/docker-compose.gpu.example.yml`, and `services/practical_rife_bentoml/smoke_test.py`. The Dockerfile builds from repository root, installs runtime dependencies with `uv sync --frozen --no-dev`, copies only `README.md`, dependency metadata, `src/`, `configs/`, and service files, exposes port `3000`, and starts BentoML on `0.0.0.0:3000`. The Compose example defines `rife-gpu0` with `CUDA_VISIBLE_DEVICES=0`, one NVIDIA GPU reservation, `/shared/rife`, model weight/repo/config mounts, and host port `3000`. The smoke client sends the worker-compatible request, validates JSON response fields, and checks that the output path exists and is non-empty. Added focused tests for smoke-client parsing/validation helpers and Docker/Compose key settings. Updated `.agent/docs/PROJECT_MAP.md`.
- 2026-06-03: Implemented Milestone 4 documentation and handoff. Added `services/practical_rife_bentoml/README.md` as the quick service reference and `docs/bentoml_practical_rife_mvp_service.md` as the human-facing run/debug guide for backend/service developers. Updated `docs/stage2_inference_runtime_refactor.md` with a pointer from the Stage 2 serving facade/examples to the Stage 3 worker-compatible MVP service. Updated `.agent/docs/PROJECT_MAP.md` for the new service docs and completed ExecPlan location. Reran full tests, ruff, smoke-client help, Compose static config, Docker CLI check, and CUDA availability check. Real Docker/GPU request validation remains deferred because local GPU access is blocked by the OS.
- 2026-06-03: Post-closeout service run fix. Updated the MVP service to read `RIFE_DEVICE` and `RIFE_CODEC` from environment with defaults `cuda` and `h264_nvenc`, rejecting empty values. Updated Docker/Compose to advertise NVIDIA video capability (`NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`) and documented `RIFE_CODEC=libx264` as the service-level CPU-encoding fallback when NVENC is unavailable while CUDA inference remains enabled.

## 13. Surprises & Discoveries

- `.agent/docs/general_plan.md` is referenced by the planning instructions but is absent from the current repository tree.
- No dedicated Stage 3 stage plan exists under `.agent/stage_plans/`; the current Stage 3 scope is defined by `.agent/tasks/stage_3/TASK_0.md` and `.agent/tasks/stage_3/bentoml_mvp_service_task.md`.
- `services/` does not exist yet. Current BentoML material is limited to developer examples under `examples/bentoml/`.
- `pyproject.toml` already includes BentoML `>=1.4.39`, so the MVP should not need a dependency change.
- `src/video_interpolation/serving.py` already has the persistent Practical-RIFE runner and PyTorch CUDA defaults, so the MVP should wrap that facade instead of creating new model-loading logic.
- The existing `PracticalRIFEVideoInferenceRunner.run(...)` originally exposed request-level `enable_quality_evaluation` but not request-level `quality_sample_count`. Milestone 2 added the narrow pass-through so non-default request sample counts reach `VideoQualityEvaluationConfig.sample_count`.
- Directly testing a service module loaded by file path can trip Pydantic/dataclass annotation resolution when the module is not registered in `sys.modules`. The Milestone 1 validated request container is intentionally a small plain class to keep tests and BentoML imports stable.
- `torch.cuda.is_available()` returned `False` in this environment during Milestone 2, so real Practical-RIFE CUDA/NVENC inference smoke remains deferred.
- Docker CLI is installed locally, and `docker compose -f services/practical_rife_bentoml/docker-compose.gpu.example.yml config` renders successfully. GPU access is blocked by the OS (`nvidia-smi` fails with NVML access blocked), so Docker build/run and real GPU/NVENC service smoke remain deferred.

## 14. Decision Log

- 2026-06-03: Limit this task to four milestones. Rationale: the user and task file explicitly require a compact MVP plan, not a broad 9-10 milestone backend project.
- 2026-06-03: Use `services/practical_rife_bentoml/` for the new MVP service and keep `examples/bentoml/` as developer examples. Rationale: the task requires a worker-compatible service under `services/`, while examples are not enough for the requested Docker/shared-volume handoff.
- 2026-06-03: Route inference through `PracticalRIFEVideoInferenceRunner`. Rationale: this is the accepted Stage 2 serving facade and avoids duplicating video/model logic in BentoML.
- 2026-06-03: Keep PyTorch/CUDA/sequential/arbitrary-Nx as the only MVP service runtime. Rationale: direct task constraints exclude ONNX, EMA, AMT, and batched inference.
- 2026-06-03: Prefer mounted weights and shared job directories in Docker. Rationale: model weights are file-based and may be large; the task forbids blocking the MVP on BentoML Model Store or broad image packaging.
- 2026-06-03: Implement Milestone 1 with helper functions around a replaceable runner rather than full model startup. Rationale: request validation and response behavior can be tested without CUDA or model loading, while Milestone 2 remains responsible for persistent runner/model startup behavior.
- 2026-06-03: Do not add human-facing service docs in Milestone 1. Rationale: the task explicitly excludes production docs for this milestone; the project map and ExecPlan now document the new location and contract, while full local/debug documentation remains a Milestone 4 deliverable.
- 2026-06-03: Load the service runner in `PracticalRIFEInterpolationService.__init__()` instead of waiting for the first request. Rationale: Milestone 2 requires model startup once per service instance/worker; tests avoid CUDA loading by monkeypatching `create_runner`.
- 2026-06-03: Add `quality_sample_count` to the shared serving runner instead of handling it only in the BentoML service wrapper. Rationale: the service should call the existing project inference stack, and quality-evaluation configuration belongs in `VideoInferenceConfig`, not in duplicated service-side logic.
- 2026-06-03: Use a tight Docker build context copy set instead of `COPY . .`. Rationale: the MVP image should include source/config/service code while excluding datasets, raw videos, outputs, notebooks, model exports, and other generated artifacts.
- 2026-06-03: Keep model weights mounted instead of copying them into the image or BentoML Model Store. Rationale: the task requires file-based weights and explicitly avoids Model Store complexity for this MVP.
- 2026-06-03: Implement the smoke client with `argparse` and `urllib.request` from the standard library. Rationale: no new dependency is needed for a simple worker-facing HTTP smoke.

## 15. Outcomes & Handoff

Stage 3 MVP BentoML service output:

- Added `services/practical_rife_bentoml/service.py`.
- Added `services/practical_rife_bentoml/Dockerfile`.
- Added `services/practical_rife_bentoml/docker-compose.gpu.example.yml`.
- Added `services/practical_rife_bentoml/smoke_test.py`.
- Added `services/practical_rife_bentoml/README.md`.
- Added `docs/bentoml_practical_rife_mvp_service.md`.
- Added `tests/test_practical_rife_bentoml_mvp_service.py`.
- Updated `src/video_interpolation/serving.py`.
- Updated `tests/test_serving.py`.
- Updated `docs/stage2_inference_runtime_refactor.md`.
- Updated `.agent/docs/PROJECT_MAP.md`.

Endpoint implemented:

```http
POST /interpolate_video
Content-Type: application/json
```

Required request fields:

```json
{
  "input_path": "/shared/rife/job-123/input.mp4",
  "output_path": "/shared/rife/job-123/output.mp4",
  "interpolation_factor": 2
}
```

Allowed optional request fields:

```json
{
  "scale": 1.0,
  "output_playback_mode": "real_time",
  "quality_evaluation_enabled": true,
  "quality_sample_count": 16
}
```

Success response fields:

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

Response policy:

- `status` and `output_path` are required on success.
- `interpolation_factor` and `duration_seconds` are included by the service helper.
- `psnr_mean` and `ssim_mean` are included only when quality evaluation produces metrics.
- Response intentionally excludes backend/device internals, model implementation details, video bytes, MinIO URLs, and quality triplet paths.

Runtime path:

- Practical-RIFE v4.26 only.
- PyTorch backend only.
- CUDA device by default.
- Sequential execution only.
- `arbitrary_nx` interpolation mode.
- Service-level `interpolation_factor` limited to `2`, `3`, and `4`.
- Runtime `scale` accepted per request and validated by the serving facade.
- `output_playback_mode` accepted per request as `real_time` or `slow_motion`.
- Optional quality evaluation enabled by default, with request-level `quality_sample_count`.
- No ONNX, EMA-VFI, AMT-S, batch serving, model switching, or internal multi-GPU routing in the MVP.

Runner behavior:

- `PracticalRIFEInterpolationService.__init__()` creates one `PracticalRIFEVideoInferenceRunner` through `create_runner()` and calls `load()` once per service instance.
- Unit tests monkeypatch the factory with a fake runner to prove load/reuse behavior without CUDA/model loading.
- The service delegates actual video inference to the existing `src/video_interpolation/serving.py` facade and `src/video_interpolation/inference.py` pipeline.
- The service validates the request, creates the output parent directory if needed, calls the runner, then verifies that `output_path` exists and has size greater than zero.

Docker/GPU/shared-volume strategy:

- `Dockerfile` builds from repository root.
- Image copies only `README.md`, `pyproject.toml`, `uv.lock`, `src/`, `configs/`, and `services/practical_rife_bentoml/`.
- Image installs runtime dependencies with `uv sync --frozen --no-dev`.
- Image sets `PYTHONPATH=/app/src`, `MODEL_WEIGHTS_ROOT=/app/model_weights`, `MODEL_REPOS_ROOT=/app/model_repos`, `RIFE_SHARED_DIR=/shared/rife`, `RIFE_DEVICE=cuda`, and `RIFE_CODEC=h264_nvenc`.
- Image starts `bentoml serve services/practical_rife_bentoml/service.py:PracticalRIFEInterpolationService --host 0.0.0.0 --port 3000`.
- Model weights, model repos, configs, and `/shared/rife` are mounted paths, not copied model-store artifacts.
- Compose example defines one service `rife-gpu0` for one GPU with `CUDA_VISIBLE_DEVICES=0`, NVIDIA device reservation for GPU `0`, host port `3000`, `/shared/rife` bind mount, and read-only model/config mounts.
- From another Compose service on the same network, the worker should use `RIFE_SERVICE_URL=http://rife-gpu0:3000`.

Documentation handoff:

- `services/practical_rife_bentoml/README.md` gives the quick contract, runtime defaults, local serve, Compose, and smoke-client commands.
- `docs/bentoml_practical_rife_mvp_service.md` gives the detailed backend/service developer guide: service responsibilities, non-goals, request/response contract, shared path contract, environment variables, mounts, local BentoML serve, Docker run, Compose one-GPU service, curl smoke, smoke client, worker integration, multi-GPU-by-multiple-instances strategy, known limitations, and troubleshooting.
- `docs/stage2_inference_runtime_refactor.md` now points from the Stage 2 serving facade/examples to the Stage 3 worker-compatible MVP service.
- `.agent/docs/PROJECT_MAP.md` records the new service docs, service files, and completed ExecPlan location.

Validation performed:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

Result: `206 passed, 88 warnings`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests services
```

Result: `All checks passed!`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python services/practical_rife_bentoml/smoke_test.py --help
```

Result: help printed successfully.

```bash
docker compose -f services/practical_rife_bentoml/docker-compose.gpu.example.yml config
```

Result: Compose config rendered successfully with repository-root build context, GPU reservation, `rife-gpu0`, `CUDA_VISIBLE_DEVICES=0`, port `3000`, `/shared/rife`, and model/config mounts.

```bash
docker --version
```

Result: Docker `29.4.3` is available.

```bash
nvidia-smi
```

Result: failed with `GPU access blocked by the operating system`. Real Docker/GPU/NVENC service request validation is therefore deferred and must be run on the target GPU server.

Manual GPU-server validation commands:

```bash
docker compose -f services/practical_rife_bentoml/docker-compose.gpu.example.yml build rife-gpu0
docker compose -f services/practical_rife_bentoml/docker-compose.gpu.example.yml up rife-gpu0
```

With an input visible to both worker and service at `/shared/rife/test/input.mp4`:

```bash
python services/practical_rife_bentoml/smoke_test.py \
  --url http://localhost:3000/interpolate_video \
  --input-path /shared/rife/test/input.mp4 \
  --output-path /shared/rife/test/output.mp4 \
  --interpolation-factor 2
```

Known limitations:

- Real CUDA/NVENC service smoke was not run in this environment.
- The service assumes the deployment machine provides CUDA and an FFmpeg/PyAV path that can use `h264_nvenc`.
- The service is path-based only and requires worker/service shared mounts at the same absolute path.
- The service does not handle MinIO, Postgres, Redis, queues, auth, monitoring, upload APIs, or frontend/backend orchestration.
- The service does not provide internal multi-GPU load balancing; run one service instance per GPU and route externally.
- The service does not expose ONNX, EMA-VFI, AMT-S, or batched inference.
- Slow-motion output omits audio because audio time-stretching is not implemented.
- Optional quality evaluation is a lightweight online metric side path and may be disabled per request for lower latency.

Next steps for `pirsii_interpolator` integration:

1. Mount a shared host directory into both worker and BentoML containers at `/shared/rife`.
2. Configure the worker with `RIFE_SERVICE_URL=http://rife-gpu0:3000` when both services share the Compose network, or `http://<host>:3000` for host-routed calls.
3. Configure `RIFE_SHARED_DIR=/shared/rife` in the worker and write job input/output files under that root.
4. Worker downloads the input video from MinIO to `/shared/rife/<job-id>/input.mp4`.
5. Worker sends `POST ${RIFE_SERVICE_URL}/interpolate_video` with absolute `input_path`, absolute `output_path`, and `interpolation_factor`.
6. Worker waits for success, then uploads `output_path` back to MinIO and updates job state.
7. On a multi-GPU server, run one BentoML service instance per GPU, with distinct service names/ports and `CUDA_VISIBLE_DEVICES` values, then choose the target service URL in the worker/deployment layer.

ExecPlan status:

- Stage 3 MVP BentoML service plan is complete pending user acceptance.
- This file should be moved from `active/` to `completed/` as part of closeout, matching project convention for completed ExecPlans.
