# Codex Prompt — Start Stage 3 Milestone 1 Implementation

Start implementation of Stage 3 / MVP BentoML Practical-RIFE service integration.

Use `.agent/docs/exec-plans/active/03_practical_rife_bentoml_mvp.execplan.md` as the main working document.

Briefly re-read:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/tasks/stage_3/TASK_0.md`
- `.agent/tasks/stage_3/bentoml_mvp_service_task.md`
- `.agent/docs/exec-plans/active/03_practical_rife_bentoml_mvp.execplan.md`

Implement only **Milestone 1 — Service Contract and Validation**.

Do not start Milestone 2.

## Task

Create the MVP BentoML service contract for the Practical-RIFE inference service.

The service must expose the worker-compatible endpoint:

```http
POST /interpolate_video
Content-Type: application/json
```

Required request body:

```json
{
  "input_path": "/shared/rife/job-123/input.mp4",
  "output_path": "/shared/rife/job-123/output.mp4",
  "interpolation_factor": 2
}
```

Allowed optional fields:

```json
{
  "scale": 1.0,
  "output_playback_mode": "real_time",
  "quality_evaluation_enabled": true,
  "quality_sample_count": 16
}
```

Successful response must be small and worker-friendly:

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

Do not include device/backend/runtime internals, video bytes, MinIO URLs, model internals, or quality triplet paths in the response.

## Scope

Create the initial service directory and service contract code under:

```text
services/practical_rife_bentoml/
```

Expected initial files for this milestone:

```text
services/practical_rife_bentoml/service.py
```

Add tests for request validation and response formatting.

In this milestone, it is acceptable to use a fake/stubbed runner path for tests. Full Practical-RIFE model loading and real inference wiring belongs to Milestone 2.

## Validation behavior

Implement validation for:

- missing `input_path`;
- non-absolute `input_path`;
- nonexistent `input_path`;
- missing `output_path`;
- non-absolute `output_path`;
- output directory creation when possible;
- invalid `interpolation_factor`;
- allowed factors exactly `2`, `3`, `4`;
- invalid `scale`;
- invalid `output_playback_mode`;
- output file existence and non-empty check after a successful runner call.

Validation errors should fail clearly before inference.

Do not fake success if output file is missing or empty.

## Constraints

Do not implement:

- full Docker setup;
- GPU container deployment;
- smoke client;
- production docs;
- runner/model persistent loading;
- real GPU inference;
- MinIO/Postgres/Redis/frontend/backend logic;
- ONNX/EMA/AMT paths;
- batch inference.

Keep changes focused on the service contract and validation.

## Validation commands

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

If `services` linting is not yet configured, use the closest scoped command and record why.

## Updates

Update:

- `.agent/docs/exec-plans/active/03_practical_rife_bentoml_mvp.execplan.md`
- `.agent/docs/PROJECT_MAP.md` if `services/` is added
- human-facing docs only if needed for the new service location or endpoint contract

Record in the ExecPlan:

- files added;
- endpoint contract implemented;
- validation rules implemented;
- tests added;
- validation results;
- what remains for Milestone 2.

## End-of-task summary

Summarize:

- service files added;
- endpoint/request/response contract;
- validation behavior;
- tests added;
- validation command results;
- project-map/ExecPlan/doc updates;
- what remains for Milestone 2.
