# Codex Prompt — Plan MVP BentoML Practical-RIFE Service

Prepare a compact ExecPlan for implementing the MVP BentoML Practical-RIFE service.

This is a planning-only task.

Do not implement code yet.

Read first:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `docs/` files related to Stage 2, serving, BentoML, local debugging
- `services/` and `examples/bentoml/` if present
- `src/video_interpolation/serving.py`
- `src/video_interpolation/inference.py`
- `configs/models/practical_rife_v4_26.yaml`
- `configs/inference/practical_rife_v4_26_2x.yaml`
- `pyproject.toml`
- `bentoml_mvp_service_task.md` or the task file provided with this prompt

Main goal:

Implement a fast MVP BentoML service compatible with the `pirsii_interpolator` worker contract.

Hard constraints:

- keep the plan small;
- no 9–10 milestone plan;
- use 3–4 milestones maximum;
- focus on a working MVP that can be run on a multi-GPU server soon;
- do not plan MinIO/Postgres/Redis/frontend/backend work inside BentoML;
- do not plan ONNX, EMA, AMT, Kubernetes, autoscaling, Model Store, or CI/CD;
- use Practical-RIFE v4.26, PyTorch, CUDA, sequential mode only.

The ExecPlan should cover:

1. BentoML service endpoint and request/response contract.
2. Runner/model startup behavior and path-based inference.
3. Docker/GPU/shared-volume integration.
4. Documentation, smoke tests, and handoff.

Required endpoint:

```http
POST /interpolate_video
```

Required JSON fields:

```json
{
  "input_path": "/shared/rife/job-123/input.mp4",
  "output_path": "/shared/rife/job-123/output.mp4",
  "interpolation_factor": 2
}
```

Required response:

```json
{
  "status": "completed",
  "output_path": "/shared/rife/job-123/output.mp4"
}
```

The ExecPlan must include a documentation requirement:

Add or update human-facing documentation with detailed local run/debug steps and manual verification commands. The documentation should be practical and similar in spirit to `pirsii_interpolator/docs/local_debugging.md`.

Create the active ExecPlan at a suitable path, for example:

```text
.agent/docs/exec-plans/active/03_practical_rife_bentoml_mvp.execplan.md
```

or another name consistent with the project.

After creating the ExecPlan, summarize:

- proposed milestones;
- key files to implement;
- integration contract with `pirsii_interpolator`;
- unresolved questions, if any;
- first implementation step.
