# Start Stage 2.5 Milestone 1 Implementation

Continue with Stage 2.5 — ONNX Stabilization, Real-Image Equivalence, and Batched Inference.

Before working, read:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/stage_plans/02_5_inference_runtime_stabilization_stage_plan.md`
- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`
- `.agent/docs/exec-plans/active/02_inference_runtime_refactor.execplan.md`, if it still exists

Use `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md` as the main working document.

Implement only **Milestone 1 — Baseline Audit and Reproducible Status Capture** from the active Stage 2.5 ExecPlan.

Do not start Milestone 2.

This milestone is primarily an audit/status-capture milestone. It should freeze the current technical facts before changing EMA ONNX export, Practical-RIFE equivalence logic, or batch inference code.

## Main goal

Produce a clear, reproducible baseline status for the current inference runtime and ONNX artifacts before any Stage 2.5 refactor work begins.

The audit should answer:

1. What ONNX artifacts currently exist?
2. What validation reports currently exist?
3. Which model/backend/provider/artifact/shape each report corresponds to?
4. What is the current known EMA-VFI ONNX dynamic H/W status?
5. What is the current known Practical-RIFE ONNX-vs-PyTorch equivalence status?
6. What real image pairs are available under `raw_data/pair_test/`?
7. What short videos are available under `raw_data/tmp_test/` for future batch/benchmark smoke runs?
8. What is the current meaning of `batch_inference.py` versus true model batch inference?

## Scope

Allowed:

- inspect files and directories;
- inspect existing ONNX artifacts and validation JSON/CSV files;
- inspect configs and docs;
- run short, safe metadata commands if useful;
- run short, safe validation commands only if needed to clarify provenance;
- update the active Stage 2.5 ExecPlan with findings;
- create or update a concise human-facing Stage 2.5 notes file if useful, for example:
  - `docs/stage2_5_inference_runtime_stabilization.md`

Not allowed:

- do not refactor EMA-VFI;
- do not change ONNX export code;
- do not change Practical-RIFE runtime code;
- do not implement real-image equivalence checks yet;
- do not implement batch inference;
- do not implement benchmarks;
- do not modify model repositories;
- do not modify model weights;
- do not add dependencies;
- do not run long video jobs;
- do not run full-dataset jobs;
- do not start Milestone 2.

If a very small helper script/function is clearly needed to extract report metadata reproducibly, ask first unless it is purely local/documentation-oriented and does not affect production code. Prefer inspection and documentation in this milestone.

## Required audit outputs

Update the active ExecPlan with a Milestone 1 progress entry and findings covering the items below.

### 1. ONNX artifact inventory

Inspect `model_exports/onnx/`.

Record for each artifact:

- model target;
- artifact path;
- original vs simplified artifact;
- file size if useful;
- apparent opset/version if readily available;
- whether the artifact is expected to be dynamic H/W or fixed/static-like;
- whether it is currently preferred or fallback.

At minimum, inspect:

- EMA-VFI ONNX artifacts;
- Practical-RIFE v4.26 ONNX artifacts.

### 2. ONNX validation report inventory

Inspect `outputs/onnx_validation/`.

For each relevant report, record:

- model;
- artifact path used;
- original vs simplified artifact if recorded;
- provider used, for example CPUExecutionProvider or CUDAExecutionProvider;
- torch device used, if recorded;
- input shape(s);
- interpolation mode/factor/timestep if recorded;
- MAE;
- max absolute error;
- MSE if available;
- allclose result if available;
- whether the report used synthetic input or real images;
- whether visual artifacts were written.

If existing reports lack enough provenance, record this as a limitation and recommend which metadata must be added in later milestones.

### 3. EMA-VFI current ONNX status

Record the current EMA status based on existing artifacts, logs, and reports:

- export/checker/simplification status;
- whether ONNX Runtime works at the original export-like shape;
- whether changed H/W fails;
- exact failure class if visible, for example LayerNormalization shape mismatch;
- why the artifact is considered superficially dynamic rather than genuinely dynamic;
- known suspicious code areas:
  - `feature_extractor.py`;
  - `pad_if_needed`;
  - `depad_if_needed`;
  - `window_partition`;
  - `window_reverse`;
  - `self.HW`;
  - `attn_mask`;
  - `feature_bone.cor`;
  - Python `math.ceil`, `int`, `.item()`, and shape-based `if` logic.

Do not fix EMA in this milestone.

### 4. Practical-RIFE current ONNX status

Record the current Practical-RIFE status:

- whether ONNX Runtime can execute on more than one input size;
- current provider-specific equivalence status;
- current known CPU vs CUDA differences;
- whether simplified or original artifact appears preferred;
- whether existing reports are sufficient or need controlled reruns in Milestone 3.

Do not investigate the root cause in this milestone beyond current evidence capture.

### 5. `raw_data/pair_test` inventory

Inspect `raw_data/pair_test/`.

Record:

- available pair directories;
- expected file names;
- image dimensions;
- image mode / channel format if easy to inspect;
- whether the structure is suitable for real-image ONNX-vs-PyTorch equivalence checks.

Do not implement the real-image equivalence pipeline yet.

### 6. `raw_data/tmp_test` inventory

Inspect `raw_data/tmp_test/`.

Record:

- available short videos;
- candidate videos for future safe smoke tests;
- duration/resolution/FPS if easy and cheap to inspect;
- any video that should be preferred for future benchmarks.

Do not run long video inference.

### 7. Batch terminology clarification

Record the current status:

- `src/video_interpolation/batch_inference.py` currently means directory-wide/all-target inference orchestration;
- it is not true model batch inference;
- Stage 2.5 must introduce a separate true model-batch API with clearer naming.

Do not implement that API in this milestone.

## Documentation requirements

If `docs/stage2_5_inference_runtime_stabilization.md` does not exist, create a concise initial version.

It should explain:

- why Stage 2.5 exists;
- current baseline status at a high level;
- where ONNX artifacts live;
- where ONNX validation outputs live;
- where real image pair inputs live;
- that true model batch inference is not the same as directory-wide `batch_inference.py`;
- that detailed implementation is still planned in later Stage 2.5 milestones.

Keep this documentation concise and practical. Do not duplicate the entire ExecPlan.

## Validation

Because this milestone is audit-focused and should not change production code, full validation may be unnecessary if only docs/ExecPlan are updated.

Still, if any source code, tests, configs, or Python modules are changed, run at minimum:

```bash
uv run pytest
uv run ruff check src tests
```

If sandbox `uv` cache permissions fail, rerun with:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
```

If only documentation and ExecPlan files are updated, record that code validation was not required.

Do not run long GPU jobs.

## ExecPlan and project map updates

Update:

- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`

Record:

- progress;
- discoveries;
- decisions, if any;
- validation performed or skipped;
- next step for Milestone 2.

Update `.agent/docs/PROJECT_MAP.md` only if a new documentation file or important repository artifact is added.

## End-of-task summary

At the end, summarize:

- which files/directories were inspected;
- what ONNX artifacts currently exist;
- what validation reports currently exist;
- what EMA-VFI ONNX status is currently known;
- what Practical-RIFE ONNX status is currently known;
- what `raw_data/pair_test` contains;
- what `raw_data/tmp_test` contains;
- whether Stage 2.5 documentation was created or updated;
- whether `PROJECT_MAP.md` was updated;
- whether code validation was run or skipped;
- what remains for Milestone 2.
