# Start Stage 2.5 Milestone 4 Implementation

Continue Stage 2.5 — ONNX Stabilization, Real-Image Equivalence, and Batched Inference.

Use `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md` as the main working document.

Briefly re-read:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/stage_plans/02_5_inference_runtime_stabilization_stage_plan.md`
- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`

Implement only **Milestone 4 — Model Batch API Contract**.

Do not start Milestone 5.

## Task

Design and implement the true model batch inference API contract in the inference runtime layer.

This milestone is about API/data structures and validation only. It should not yet wire the batch API into EMA-VFI, Practical-RIFE, local video inference, or ONNX runtime execution.

## Scope

Implement clear request/result structures for model-level batched inference, distinct from directory-wide `batch_inference.py`.

The API must support:

- BCHW left/right frame tensors;
- fixed 2x interpolation;
- arbitrary Nx interpolation;
- `interpolation_factor` in range `2..8`;
- pair-by-timestep flattening semantics for Nx;
- output reconstruction contract: `outputs[pair_index][timestep_index]`;
- metadata needed by later runtime/video/benchmark milestones.

Preserve existing sequential APIs and behavior.

Avoid naming that confuses true model batch inference with directory-wide all-target inference.

## Validation

Add focused tests for:

- valid/invalid BCHW inputs;
- batch size agreement between left/right tensors;
- interpolation factor validation;
- timestep generation for factors `2`, `4`, and `8`;
- flattened pair×timestep ordering;
- reconstruction shape/order contract;
- fixed 2x compatibility.

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

## Updates

Update:

- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`
- `.agent/docs/PROJECT_MAP.md` if repository structure changes
- human-facing docs only if a developer-facing API or concept needs explanation now

## End-of-task summary

Summarize:

- what batch API structures were added;
- how Nx pair×timestep flattening is represented;
- how output ordering is documented/enforced;
- what tests were added;
- validation command results;
- docs/project-map/ExecPlan updates;
- what remains for Milestone 5.
