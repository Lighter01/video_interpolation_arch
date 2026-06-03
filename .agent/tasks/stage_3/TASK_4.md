# Codex Prompt — Start Stage 3 Milestone 4 Implementation

Continue Stage 3 / MVP BentoML Practical-RIFE service integration.

Use `.agent/docs/exec-plans/active/03_practical_rife_bentoml_mvp.execplan.md` as the main working document.

Briefly re-read:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/tasks/stage_3/TASK_0.md`
- `.agent/tasks/stage_3/bentoml_mvp_service_task.md`
- `.agent/docs/exec-plans/active/03_practical_rife_bentoml_mvp.execplan.md`

Implement only **Milestone 4 — Documentation, Project Map, and Handoff**.

This is the final closeout milestone for the MVP BentoML Practical-RIFE service. Do not implement new runtime or service features unless a tiny fix is required to make documented commands accurate.

## Task

Finalize documentation, project map updates, ExecPlan handoff, and practical run/debug instructions for the MVP BentoML service.

The goal is to make the service usable by backend/service developers and ready for integration with `pirsii_interpolator`.

## Documentation scope

Add or finalize detailed human-facing documentation, preferably:

```text
docs/bentoml_practical_rife_mvp_service.md
```

The documentation must be step-by-step and practical, similar in spirit to `pirsii_interpolator/docs/local_debugging.md`.

It must explain:

- what the BentoML service does;
- what it does not do;
- request/response contract for `POST /interpolate_video`;
- required shared path contract using `/shared/rife`;
- required env variables and mounts;
- where Practical-RIFE weights/configs/model repos must be available;
- how to run the service locally with `bentoml serve`;
- how to run the Docker container;
- how to use `docker-compose.gpu.example.yml`;
- how one worker + one BentoML service + one GPU is expected to work;
- how to extend to multiple GPUs by running multiple service instances;
- how to call the service with `curl`;
- how to run `smoke_test.py`;
- how to verify output file existence and nonzero size;
- how `pirsii_interpolator` should set `RIFE_SERVICE_URL`;
- known MVP limitations;
- common troubleshooting:
  - CUDA unavailable;
  - missing weights;
  - missing shared mount;
  - invalid interpolation factor;
  - `h264_nvenc` unavailable;
  - output file missing or empty.

Keep documentation concise enough to be useful under time pressure, but detailed enough that a teammate can follow it without reading source code.

## Required handoff updates

Update:

- `.agent/docs/exec-plans/active/03_practical_rife_bentoml_mvp.execplan.md`
- `.agent/docs/PROJECT_MAP.md`
- `services/practical_rife_bentoml/README.md`
- `docs/bentoml_practical_rife_mvp_service.md`
- Stage 2/serving docs only if they need a pointer to the new MVP service

The ExecPlan must include an `Outcomes & Handoff` section summarizing:

- service files added;
- endpoint implemented;
- request/response contract;
- Docker/GPU/shared-volume strategy;
- validation performed;
- CUDA/Docker smoke status;
- known limitations;
- exact next steps for integrating with `pirsii_interpolator`.

If project convention requires moving completed ExecPlans from `active/` to `completed/`, do so only after the handoff section is complete and all validation results are recorded.

## Validation

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

Run safe smoke/import checks needed to verify documented commands where practical.

Do not run long videos or full directories.

If Docker or CUDA is unavailable locally, do not fake validation. Record the check as deferred and provide exact manual commands.

## Constraints

Do not implement:

- new inference features;
- new endpoint behavior;
- ONNX/EMA/AMT paths;
- MinIO/Postgres/Redis/frontend/backend logic;
- internal multi-GPU routing;
- Kubernetes/autoscaling/CI/CD;
- Model Store migration.

Do not modify model weights or datasets.

## End-of-task summary

Summarize:

- what docs were added/updated;
- what project-map entries were updated;
- what ExecPlan handoff content was added;
- whether the ExecPlan was completed/moved;
- validation command results;
- Docker/CUDA smoke status;
- whether the MVP BentoML service is ready for user acceptance;
- exact next manual steps for server integration.
