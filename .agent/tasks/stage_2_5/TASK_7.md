Continue Stage 2.5 — ONNX Stabilization, Real-Image Equivalence, and Batched Inference.

Use `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md` as the main working document.

Briefly re-read:

* `.agent/AGENTS.md`
* `.agent/docs/PLANS.md`
* `.agent/docs/PROJECT_MAP.md`
* `.agent/stage_plans/02_5_inference_runtime_stabilization_stage_plan.md`
* `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`

Implement only **Milestone 7 — ONNX Batch Inference Where Viable**.

Do not start Milestone 8.

## Task

Add ONNX batch inference only for model/backend combinations that are currently classified as viable in the active ExecPlan.

Do not force ONNX batch support for models whose ONNX path is blocked, deferred, or not trusted after prior milestones.

## Scope

Use the true model batch API introduced in Milestone 4 and the batch execution semantics implemented in Milestones 5–6.

Implement ONNX batch inference for:

* EMA-VFI only if its constrained-dynamic ONNX path is still classified as viable;
* Practical-RIFE only if real-image equivalence from Milestone 3 supports keeping its ONNX path under investigation or use.

For each implemented model:

* support fixed 2x;
* support arbitrary Nx;
* preserve pair×timestep flattening semantics;
* reconstruct outputs as `outputs[pair_index][timestep_index]`;
* preserve sequential ONNX inference fallback where available;
* preserve PyTorch batch inference behavior.

If a model/backend is skipped, record the reason clearly in the ExecPlan.

## Constraints

Do not implement:

* new ONNX export work beyond small compatibility fixes;
* BentoML proof;
* benchmark workflow;
* FastAPI/Celery/Redis/frontend/monitoring;
* AMT-S ONNX batch support;
* training/fine-tuning changes.

Do not run long videos or full directories.

Do not change model weights.

## Validation

Add focused tests for:

* ONNX batch request handling with fake/lightweight backend where useful;
* output ordering for fixed 2x and Nx;
* pair×timestep flatten/reconstruction through ONNX backend;
* sequential fallback compatibility;
* skipped/deferred model behavior when ONNX is not viable;
* provider/artifact metadata in results or reports.

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

If ONNX artifacts and runtime providers are available, run small tensor-level ONNX batch smoke checks for the viable models only.

Do not run long video jobs.

## Updates

Update:

* `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`
* `.agent/docs/PROJECT_MAP.md` if repository structure changes
* human-facing docs if new ONNX batch commands, configs, behavior, or limitations are added

Record in the ExecPlan:

* which ONNX batch paths were implemented;
* which were skipped/deferred and why;
* provider/artifact used in smoke checks;
* whether original or simplified ONNX artifacts were used;
* validation results.

## End-of-task summary

Summarize:

* which models received ONNX batch inference support;
* which models/backends were skipped or deferred;
* how fixed 2x and Nx ONNX batching work;
* how output ordering is reconstructed;
* what tests were added;
* what smoke checks were run or deferred;
* validation command results;
* docs/project-map/ExecPlan updates;
* what remains for Milestone 8.
