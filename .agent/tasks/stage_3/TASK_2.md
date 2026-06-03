# Codex Prompt — Start Stage 3 Milestone 2 Implementation

Continue Stage 3 / MVP BentoML Practical-RIFE service integration.

Use `.agent/docs/exec-plans/active/03_practical_rife_bentoml_mvp.execplan.md` as the main working document.

Briefly re-read:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/tasks/stage_3/TASK_0.md`
- `.agent/tasks/stage_3/bentoml_mvp_service_task.md`
- `.agent/docs/exec-plans/active/03_practical_rife_bentoml_mvp.execplan.md`

Implement only **Milestone 2 — Persistent Runner Startup and Path-Based Inference Wiring**.

Do not start Milestone 3.

## Task

Wire the BentoML service contract from Milestone 1 to the real Practical-RIFE serving runner.

The service must use the existing project inference stack through `PracticalRIFEVideoInferenceRunner` and must not duplicate model preprocessing, checkpoint loading, frame ordering, encoding, or quality-evaluation logic.

## Scope

Implement real path-based inference wiring for:

- Practical-RIFE v4.26;
- PyTorch backend only;
- device `cuda`;
- sequential execution only;
- `arbitrary_nx` interpolation mode;
- allowed service factors `2`, `3`, `4`;
- runtime `scale`;
- runtime `output_playback_mode`;
- runtime `quality_evaluation_enabled`;
- runtime `quality_sample_count`;
- default codec `h264_nvenc`.

The service should initialize and load the Practical-RIFE runner once per service instance/worker and reuse it across requests.

Do not reload model weights per request if the current runner can keep the model resident.

## Required behavior

The endpoint must:

- validate request paths and parameters;
- create the output directory if possible;
- call the persistent runner with request values;
- write output exactly to `output_path`;
- verify that `output_path` exists and is non-empty after inference;
- return the small worker-compatible response.

Keep the response free of internal runtime details.

## Constraints

Do not implement:

- Dockerfile;
- docker-compose GPU overlay;
- smoke client;
- production docs;
- MinIO/Postgres/Redis/frontend/backend logic;
- ONNX/EMA/AMT paths;
- batch inference;
- multi-GPU routing.

Those belong to later milestones or are out of scope.

## Validation

Add focused tests for:

- service uses/reuses a persistent runner object;
- request values are passed to the runner correctly;
- output file verification after runner call;
- inference errors are surfaced clearly;
- response contains only allowed worker-facing fields;
- CUDA/model-heavy paths can be faked/mocked in unit tests.

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

If CUDA is unavailable, do not run real GPU inference. Record CUDA smoke as deferred.

## Updates

Update:

- `.agent/docs/exec-plans/active/03_practical_rife_bentoml_mvp.execplan.md`
- `.agent/docs/PROJECT_MAP.md` if repository structure changes
- human-facing docs only if service usage or behavior changed in a way developers need to know now

Record in the ExecPlan:

- runner wiring details;
- startup/loading behavior;
- request values passed to inference;
- validation results;
- any CUDA smoke deferral;
- what remains for Milestone 3.

## End-of-task summary

Summarize:

- service files changed;
- how persistent runner startup works;
- how path-based inference is wired;
- how request fields map to runner/inference parameters;
- what tests were added;
- validation command results;
- CUDA smoke status;
- project-map/ExecPlan/doc updates;
- what remains for Milestone 3.

