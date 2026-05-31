# TASK — Start Stage 2 Planning: Inference Runtime Refactor and Serving Readiness

## Purpose

This task starts the new simplified Stage 2. It is a planning task, not an implementation task.

The goal is to analyze the current project state and prepare for a refactor that makes the inference subsystem easier to use with future ONNX Runtime and BentoML services.

Do not implement code yet.

---

## Required Reading

First read:

```text
.agent/AGENTS.md
.agent/docs/PLANS.md
.agent/docs/PROJECT_MAP.md
.agent/general_plan.md
.agent/stage_plans/01_ml_core_stage_plan.md
.agent/docs/exec-plans/active/01_ml_core_selected.execplan.md
.agent/stage_plans/02_inference_runtime_refactor_stage_plan.md
```

Then inspect the current repository state, with special attention to:

```text
src/video_interpolation/adapters/base.py
src/video_interpolation/adapters/ema_vfi.py
src/video_interpolation/adapters/rife.py
src/video_interpolation/inference.py
src/video_interpolation/batch_inference.py
src/video_interpolation/image_io.py
src/video_interpolation/training.py
src/video_interpolation/validation.py
src/video_interpolation/cli.py
configs/models/
configs/inference/
model_repos/EMA-VFI/
model_repos/Practical-RIFE/
model_weights/EMA-VFI/
model_weights/Practical-RIFE/
```

AMT-S may be inspected only to understand existing dependencies, but AMT-S is not in scope for new Stage 2 work.

---

## Stage 1 ExecPlan Closeout

Before creating a new Stage 2 ExecPlan:

1. Inspect `.agent/docs/exec-plans/active/01_ml_core_selected.execplan.md`.
2. Ensure its `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Handoff` sections are current enough to serve as Stage 1 handoff.
3. If no further Stage 1 work is required by the user, move it from:

```text
.agent/docs/exec-plans/active/01_ml_core_selected.execplan.md
```

into:

```text
.agent/docs/exec-plans/completed/01_ml_core_selected.execplan.md
```

4. Do not rewrite Stage 1 history. Only add a brief closeout/handoff note if necessary.

If the repository state suggests Stage 1 is not ready to close, ask the user before moving the ExecPlan.

---

## Planning Task

After reading the files and inspecting the code, produce a concise analysis report for the user.

The report must include:

1. Current inference architecture summary.
2. Current adapter architecture summary.
3. Current coupling problems that block ONNX/BentoML readiness.
4. EMA-VFI-specific inference/export concerns.
5. Practical-RIFE-specific inference/export concerns.
6. Suggested module boundary changes.
7. Suggested runtime backend abstraction.
8. Suggested ONNX export boundary for each model.
9. Suggested Nx interpolation strategy for EMA-VFI and Practical-RIFE.
10. Risks and likely blockers.
11. Concrete clarifying questions for the user.

Do not create the Stage 2 ExecPlan yet unless the user explicitly says the analysis is accepted and all clarification questions are answered.

---

## Questions to Ask the User

Ask only questions that are needed for implementation planning. Avoid vague questions.

At minimum, consider asking about:

- whether AMT-S should remain in the codebase but be excluded from active Stage 2 work;
- where ONNX artifacts should be stored;
- whether initial ONNX export may use fixed input shapes;
- whether dynamic height/width is required for ONNX in this stage;
- whether BentoML service skeletons are in scope or deferred;
- whether Nx support should be implemented after PyTorch runtime refactor or after ONNX export;
- whether Practical-RIFE v4.25 must remain default or v4.26 may be used if technically easier;
- acceptable PyTorch-vs-ONNX tolerance thresholds;
- how aggressive module reorganization may be.

---

## ExecPlan Creation Rule

Only after the user answers the clarification questions, create the new Stage 2 ExecPlan at:

```text
.agent/docs/exec-plans/active/02_inference_runtime_refactor.execplan.md
```

The ExecPlan must follow `.agent/docs/PLANS.md`.

The ExecPlan must be based on:

- user answers;
- `.agent/stage_plans/02_inference_runtime_refactor_stage_plan.md`;
- current repository inspection;
- completed Stage 1 handoff.

---

## Scope Guardrails

Do not implement during this task:

- code refactor;
- ONNX export scripts;
- Nx interpolation;
- BentoML services;
- FastAPI;
- Celery/Redis;
- PostgreSQL application schema;
- frontend;
- retraining triggers;
- monitoring.

Do not modify model repositories during planning.

Do not delete AMT-S code.

Do not add dependencies during planning.

---

## Expected Output of This Task

The response should include:

1. Files read and inspected.
2. Current architecture findings.
3. Proposed Stage 2 implementation direction.
4. Clarifying questions.
5. Whether Stage 1 ExecPlan was moved to `completed/` or why it was not moved.

No production code should be changed.
