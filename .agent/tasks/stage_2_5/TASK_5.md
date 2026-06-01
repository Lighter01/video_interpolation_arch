# Start Stage 2.5 Milestone 5 Implementation

Continue Stage 2.5 — ONNX Stabilization, Real-Image Equivalence, and Batched Inference.

Use `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md` as the main working document.

Briefly re-read:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/stage_plans/02_5_inference_runtime_stabilization_stage_plan.md`
- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`

Implement only **Milestone 5 — PyTorch Batch Inference for EMA-VFI and Practical-RIFE**.

Do not start Milestone 6.

## Task

Wire the model-level batch API from Milestone 4 into the active PyTorch runtimes:

- EMA-VFI;
- Practical-RIFE.

This milestone is about runtime/model batch execution only. Do not refactor video-level chunked inference yet.

## Scope

Implement true batched PyTorch inference for both active models.

The implementation must support:

- fixed 2x;
- arbitrary Nx;
- pair-by-timestep flattening for Nx;
- correct output reconstruction as `outputs[pair_index][timestep_index]`;
- configurable effective batch size where relevant;
- sequential fallback compatibility;
- existing `predict_pair`, `predict_intermediate_frames`, and Stage 2 runtime APIs.

Keep Stage 1 EMA training/fine-tuning behavior unchanged.

Do not implement ONNX batch inference in this milestone.

## Validation

Add focused unit tests with fake/lightweight runtimes for:

- fixed 2x batch output ordering;
- Nx pair×timestep flatten/reconstruction;
- output reconstruction as `outputs[pair_index][timestep_index]`;
- sequential fallback compatibility;
- adapter wrapper compatibility where touched;
- invalid batch shapes and factor validation.

Also add or run integration smoke checks with the real EMA and Practical-RIFE runtimes when practical:

- prefer CUDA if available;
- if CUDA is unavailable, try CPU smoke checks on very small inputs;
- keep inputs tiny, for example 32x32, 64x64, or the smallest shape supported by the model;
- test factors 2, 4, and 8 where feasible;
- record if a real-model CPU smoke is too slow or blocked by model assumptions.

Fake/lightweight tests are required for correctness of batching logic.
Real-model smoke checks are required only when the environment makes them practical.

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

Do not run long video jobs.

## Updates

Update:

- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`
- `.agent/docs/PROJECT_MAP.md` if repository structure changes
- human-facing docs if new developer-facing batch API behavior needs explanation

Record in the ExecPlan:

- batch API integration details;
- whether EMA and RIFE batch paths use one model call for flattened Nx batches;
- any memory/OOM limitations found;
- validation results.

## End-of-task summary

Summarize:

- what EMA/RIFE batch runtime code was added or changed;
- how fixed 2x and Nx batch inference work;
- how output ordering is reconstructed;
- whether sequential compatibility was preserved;
- what tests were added;
- validation command results;
- docs/project-map/ExecPlan updates;
- what remains for Milestone 6.
