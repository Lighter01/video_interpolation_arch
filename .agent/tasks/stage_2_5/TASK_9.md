# Start Stage 2.5 Milestone 9 Implementation

Continue Stage 2.5 — ONNX Stabilization, Real-Image Equivalence, and Batched Inference.

Use `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md` as the main working document.

Briefly re-read:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/stage_plans/02_5_inference_runtime_stabilization_stage_plan.md`
- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`
- `.agent/docs/exec-plans/active/02_inference_runtime_refactor.execplan.md`, if it still exists

Implement only **Milestone 9 — Documentation, Project Map, and Stage 2 Handoff**.

This is a closeout and handoff milestone. Do not implement new features.

## Task

Finalize Stage 2.5 documentation, project map updates, ExecPlan handoff, and the pointer back to the paused Stage 2 ExecPlan.

The goal is to make Stage 2.5 restartable, understandable, and ready for user review before resuming Stage 2 Milestone 8.

## Scope

Review and update human-facing documentation for:

- why Stage 2.5 exists;
- EMA ONNX dynamic/constrained-dynamic status;
- Practical-RIFE ONNX equivalence status;
- synthetic and real-image equivalence checks;
- true model batch inference;
- video-level chunked batch inference;
- ONNX batch support status where relevant;
- mini-benchmarks and MLflow logging;
- sequential fallback;
- batch size / VRAM / resolution considerations;
- recommended backend/model path for the next BentoML proof;
- known limitations and unresolved blockers.

Update `.agent/docs/PROJECT_MAP.md` so it reflects important Stage 2.5 files, modules, docs, commands, and output locations.

Update the active Stage 2.5 ExecPlan outcomes/handoff section.

Update the paused Stage 2 ExecPlan with a concise pointer to the completed Stage 2.5 handoff and any key decisions that affect resuming Stage 2.

## Constraints

Do not implement new runtime, ONNX, batch, benchmark, BentoML, or backend features.

Do not run long video jobs or full datasets.

Do not modify model weights or datasets.

Keep documentation practical and concise. Do not duplicate the entire ExecPlan.

## Validation

Run:

```bash
uv run pytest
uv run ruff check src tests
```

If the uv cache is not writable:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
```

If some validation is skipped because this milestone only changes docs/agent files, record why.

Run only safe smoke commands if needed to verify documented commands still exist. Do not run long GPU jobs.

## Handoff requirements

The Stage 2.5 ExecPlan must summarize:

- EMA ONNX final decision;
- Practical-RIFE ONNX final decision;
- PyTorch vs ONNX serving recommendation;
- batch inference status;
- benchmark status and key results;
- recommended first model/backend for BentoML proof;
- unresolved blockers;
- exact next step for resuming Stage 2 Milestone 8.

The Stage 2 ExecPlan should clearly state that Stage 2 was paused, Stage 2.5 was completed, and Stage 2 may now resume from the BentoML compatibility proof using the Stage 2.5 handoff decisions.

## Updates

Update:

- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`
- `.agent/docs/exec-plans/active/02_inference_runtime_refactor.execplan.md`, if still active
- `.agent/docs/PROJECT_MAP.md`
- Stage 2.5 human-facing docs
- related config README files only if needed

## End-of-task summary

Summarize:

- what docs were updated;
- what project-map entries were updated;
- what Stage 2.5 outcomes were recorded;
- what Stage 2 pointer/handoff was added;
- validation command results;
- whether Stage 2.5 is ready for user acceptance;
- the recommended next prompt/action for resuming Stage 2 Milestone 8.
