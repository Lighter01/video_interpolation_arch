# EMA-VFI Deep ONNX Dynamic H/W Refactor Attempt

Continue Stage 2.5 — ONNX Stabilization, Real-Image Equivalence, and Batched Inference.

This task is a focused EMA-VFI investigation/refactor task before moving to Stage 2.5 Milestone 3.

Use `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md` as the main working document.

Before making implementation decisions, read:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/stage_plans/02_5_inference_runtime_stabilization_stage_plan.md`
- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`
- the Stage 2 active ExecPlan if needed for prior ONNX/runtime decisions.

Then inspect the current EMA-VFI implementation and export/runtime code relevant to this task.

Focus especially on:

- `model_repos/EMA-VFI/model/feature_extractor.py`
- `model_repos/EMA-VFI/model/flow_estimation.py`
- `model_repos/EMA-VFI/model/warplayer.py`
- `src/video_interpolation/inference_runtime/ema.py`
- `src/video_interpolation/inference_runtime/onnx_export.py`
- `src/video_interpolation/inference_runtime/onnx_validation.py`
- existing Stage 2.5 Milestone 2 outputs under:
  - `outputs/onnx_validation/stage2_5_m2/`
  - `model_exports/onnx/stage2_5_m2_export112/`
  - `model_exports/onnx/stage2_5_m2_dynamo_after_cache_patch/`

## Context

Stage 2.5 Milestone 2 classified EMA-VFI ONNX dynamic H/W as blocked/deferred:

```text
EMA ONNX works only for export-size/static-like input and is not suitable for dynamic serving.
EMA ONNX dynamic H/W is blocked/deferred.
EMA serving should remain PyTorch-only for now.
```

However, before moving on to Stage 2.5 Milestone 3, attempt one deeper, bounded refactor of EMA-VFI to see whether the model can be made ONNX-exportable with dynamic or constrained-dynamic input H/W.

The goal is not to change the model mathematically. The goal is to make the internal inference/export code more correct and export-friendly while preserving functional identity.

## Main goal

Refactor EMA-VFI inference/export-relevant internals so that:

- PyTorch EMA-VFI inference still produces functionally equivalent results before and after the refactor;
- EMA-VFI can be exported to ONNX;
- the exported ONNX model can run on different input H/W sizes without requiring separate static bucket models;
- project-owned preprocessing/padding/unpadding may be used outside ONNX;
- ONNX inference can accept arbitrary input images through dynamic or constrained-dynamic shape support.

The preferred outcome is:

```text
arbitrary input H/W
  -> project-owned external padding if required
  -> one EMA ONNX artifact
  -> ONNX Runtime execution
  -> project-owned unpadding
  -> output at original H/W
```

## Important non-goal: no static bucket fallback

Do not implement static bucket fallback for EMA-VFI ONNX.

Do not design the final solution around multiple exported EMA ONNX models for different fixed resolutions.

Examples of what NOT to implement as the final strategy:

```text
ema_512x512.onnx
ema_720p.onnx
ema_1080p.onnx
```

If a single dynamic or constrained-dynamic ONNX artifact cannot be achieved, keep EMA-VFI as PyTorch-only for serving.

## Allowed upstream changes

You may freely modify or add upstream EMA-VFI code under `model_repos/EMA-VFI/` if needed, as long as the functional behavior of the model is preserved.

Allowed:

- refactor shape-dependent Python logic into export-friendlier tensor or helper logic;
- remove or replace forward-time state mutation;
- replace cache logic that breaks export;
- make coordinate/grid/mask generation device-, dtype-, and shape-aware;
- add export-specific helper paths if they preserve PyTorch inference correctness;
- add project-owned wrappers if cleaner than patching all upstream code;
- add tests or smoke scripts to compare pre/post-refactor PyTorch outputs.

Not allowed:

- intentionally changing model architecture or learned behavior;
- changing weights;
- silently changing the numerical meaning of inference;
- breaking existing PyTorch inference;
- breaking Stage 1 EMA training/fine-tuning compatibility;
- implementing static ONNX bucket fallback as the production solution.

All upstream modifications must be documented in the active Stage 2.5 ExecPlan.

## Rollback / safety requirement

The project is currently on a dev branch, and a commit exists after Stage 2.5 Milestone 2.

Before making invasive changes, choose a safe rollback strategy.

Acceptable options:

- create an experiment branch;
- save a patch/diff before starting;
- rely on the current clean commit and keep a clear `git diff` that can be reverted.

Prefer the simplest reliable option.

If the deep refactor does not achieve dynamic or constrained-dynamic EMA ONNX support, revert implementation code to the state at the end of Stage 2.5 Milestone 2.

Even if code is reverted, keep useful findings in the active ExecPlan and documentation.

## Functional identity requirement

The refactor must preserve EMA-VFI PyTorch inference behavior.

Before accepting the refactor, compare PyTorch outputs before and after the refactor on small representative inputs.

The comparison should include:

- same checkpoint;
- same device where practical;
- same input tensors;
- same timestep;
- same preprocessing/padding path.

Record:

- MAE;
- max absolute error;
- MSE if easy;
- whether outputs are effectively identical or acceptably close.

A perfect bitwise match is not required if the refactor changes only safe internal construction details, but differences must be small and explained.

If output differences are large, do not accept the refactor without investigation.

## ONNX acceptance criterion for this task

At this stage, the main criterion is whether the exported EMA ONNX model can run on multiple input H/W sizes.

PyTorch-vs-ONNX numerical equivalence is important but is not the final rejection criterion yet, because the real-image equivalence pipeline will be implemented in the next Stage 2.5 milestone.

Therefore:

- do record PyTorch-vs-ONNX MAE/max error where possible;
- do not reject a dynamic ONNX export solely because tensor-level equivalence is imperfect;
- do reject or defer the ONNX path if the model only runs at the export shape or requires static buckets.

Minimum ONNX runtime success criterion:

- one exported EMA ONNX artifact;
- runs on at least three different input H/W shapes;
- at least one non-square shape;
- at least one shape different from the export sample shape;
- output shape matches expected original/unpadded H/W after project-owned postprocessing.

Example test shapes may include:

```text
64x64
112x168
320x512 or pair_test-derived padded equivalent
```

You may choose safer shapes if required by model constraints, but explain the choice.

## Investigation targets

Use the Milestone 2 evidence as starting point.

Likely problematic areas include:

### `feature_extractor.py`

Inspect and refactor where needed:

- `pad_if_needed(...)`
- `depad_if_needed(...)`
- `window_partition(...)`
- `window_reverse(...)`
- `MotionFormerBlock.forward(...)`
- attention mask generation
- `self.HW`
- `attn_mask`
- `feature_bone.cor`
- Python `math.ceil(...)` based on input shapes
- Python `int(...)` based on tensor-derived shapes
- `.item()` based shape checks
- `if` branches based on tensor-derived shape values
- forward-time `register_buffer(...)`
- caches that depend on shape/device/dtype but are not safely keyed

### Device and dtype behavior

Ensure export/inference paths do not depend on hardcoded CUDA.

Prefer:

```python
device = input_tensor.device
dtype = input_tensor.dtype
```

and tensor creation on the input device/dtype.

## Export approach

Try modern export where useful:

- `torch.onnx.export(..., dynamo=True, dynamic_shapes=...)`
- avoid using only legacy `dynamic_axes` as proof of true dynamic behavior

If legacy export remains necessary for some reason, document why.

If ONNX simplification is part of the current export pipeline:

- run simplification by default if configured;
- keep original artifact available;
- report whether original and simplified artifacts behave differently.

## Expected workflow

1. Establish or reference the current Stage 2.5 Milestone 2 baseline.
2. Choose and record rollback strategy.
3. Inspect and plan minimal invasive changes.
4. Implement targeted refactor.
5. Compare PyTorch outputs before/after refactor.
6. Export EMA ONNX.
7. Run ONNX Runtime on multiple H/W shapes.
8. Classify result.
9. If successful, keep changes and update docs/ExecPlan.
10. If unsuccessful, revert implementation code and update docs/ExecPlan with findings.

## Required classification at the end

At the end, classify EMA-VFI ONNX as exactly one of:

```text
1. EMA dynamic ONNX works.
2. EMA constrained-dynamic ONNX works with documented external padding constraints.
3. EMA ONNX still works only for export-size/static-like input.
4. EMA ONNX remains blocked/deferred; EMA serving should be PyTorch-only.
```

If the result is 3 or 4, revert invasive implementation changes unless they are harmless cleanup already validated and explicitly useful for PyTorch inference.

## Scope boundaries

Do not implement:

- real-image ONNX-vs-PyTorch equivalence pipeline;
- Practical-RIFE equivalence work;
- batch inference;
- benchmarks;
- BentoML proof;
- FastAPI/Celery/Redis/frontend/monitoring;
- AMT-S work;
- static bucket fallback.

Do not modify model weights.

Do not run long video jobs.

## Validation

Run normal checks after any code change:

```bash
uv run pytest
uv run ruff check src tests
```

If uv cache permissions fail:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
```

Run focused EMA tests:

- EMA adapter tests;
- ONNX export tests;
- ONNX runtime tests;
- any new pre/post-refactor PyTorch identity tests.

Run small EMA ONNX Runtime checks on multiple shapes if export succeeds.

Do not run long videos.

## Documentation and ExecPlan updates

Update:

- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`
- `docs/stage2_5_inference_runtime_stabilization.md`
- `.agent/docs/PROJECT_MAP.md` if repository structure changes

Record in the ExecPlan:

- rollback strategy used;
- files changed;
- upstream patches made;
- PyTorch pre/post-refactor identity results;
- export commands/options;
- tested shapes;
- ONNX runtime results;
- whether simplification helped or hurt;
- final EMA ONNX classification;
- whether code was kept or reverted;
- what remains for Stage 2.5 Milestone 3.

## End-of-task summary

At the end, summarize:

- whether deep EMA refactor was attempted;
- rollback strategy used;
- what code was changed;
- whether changes were kept or reverted;
- whether PyTorch EMA output identity was preserved;
- whether EMA ONNX runs on multiple input H/W sizes;
- final EMA ONNX classification;
- validation commands and results;
- documentation/ExecPlan/project-map updates;
- whether Stage 2.5 can proceed to Milestone 3.
