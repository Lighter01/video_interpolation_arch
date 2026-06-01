# Start Stage 2.5 Milestone 2 Implementation

Continue Stage 2.5 — ONNX Stabilization, Real-Image Equivalence, and Batched Inference.

Use `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md` as the main working document.

Before implementation, briefly re-read:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/stage_plans/02_5_inference_runtime_stabilization_stage_plan.md`
- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`

Then inspect only the files that are directly relevant to **Milestone 2 — EMA Dynamic or Constrained-Dynamic ONNX Investigation**.

Implement only Milestone 2.

Do not start Milestone 3.

## Scope

Investigate and, if reasonably possible, fix EMA-VFI ONNX dynamic/constrained-dynamic H/W support.

The goal is to determine whether EMA-VFI ONNX can support:

- true dynamic H/W; or
- constrained dynamic H/W through project-owned external padding, likely to a multiple of 56.

If neither is feasible without broad or fragile rewriting, classify EMA ONNX as blocked/deferred and keep EMA serving as PyTorch-only.

Do not implement static ONNX bucket fallback.

Do not implement real-image equivalence checks in this milestone; that belongs to Milestone 3.

Do not implement batch inference; that belongs to later milestones.

## Key requirements

- Reproduce or confirm the current EMA ONNX dynamic-shape failure.
- Test the external padding hypothesis, especially padding to valid multiples such as 56.
- Try a modern export route only where useful, such as `dynamo=True` with `dynamic_shapes`, but do not treat export flags alone as the fix.
- Inspect and minimally patch EMA export blockers only when needed:
  - shape-dependent Python logic;
  - hardcoded CUDA/device assumptions;
  - forward-time cache or state mutation;
  - attention mask / window padding logic;
  - coordinate/grid cache behavior.
- Keep PyTorch EMA inference working.
- Preserve Stage 1 EMA training/fine-tuning behavior.
- Keep all changes narrow and documented.

## Expected outcome

At the end of this milestone, EMA ONNX must be classified as one of:

- dynamic H/W ONNX works;
- constrained-dynamic ONNX works with documented external padding rules;
- ONNX works only for export-size input and is not suitable for dynamic serving;
- EMA ONNX remains blocked/deferred, and EMA should use PyTorch backend for serving.

## Validation

Run the normal project checks:

```bash
uv run pytest
uv run ruff check src tests
```

If the uv cache is not writable, use:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
```

Run small EMA ONNX export/runtime smoke checks only. Use short synthetic inputs and at least two H/W sizes for any claimed dynamic or constrained-dynamic path.

Do not run long video jobs.

## Updates required

Update:

- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`
- `.agent/docs/PROJECT_MAP.md` if repository structure changes
- human-facing docs if new commands, configs, outputs, or limitations are added

Record in the ExecPlan:

- what was tried;
- which export/runtime path was used;
- what shapes were tested;
- whether external padding helped;
- exact blockers if dynamic ONNX still fails;
- whether EMA ONNX is usable or should be deferred.

## End-of-task summary

Summarize:

- what files/modules were changed;
- what EMA ONNX strategy was tested;
- whether true dynamic or constrained-dynamic H/W works;
- what validation commands were run;
- whether PyTorch EMA inference/training compatibility was preserved;
- what documentation/project-map/ExecPlan updates were made;
- what remains for Milestone 3.
