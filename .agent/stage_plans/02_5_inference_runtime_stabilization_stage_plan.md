# Stage 2.5 Plan — ONNX Stabilization, Real-Image Equivalence, and Batched Inference

## 1. Stage Goal

Stage 2.5 is an intermediate refactor and optimization stage between:

- **Stage 2 — Inference Runtime Refactor and Serving Readiness**;
- **Stage 2 Milestone 8 — Minimal BentoML Compatibility Proof**.

Stage 2 remains active but is paused before the BentoML proof milestone.

The purpose of Stage 2.5 is to stabilize and benchmark the inference runtime before wrapping it in BentoML. The stage focuses on three problems discovered at the end of Stage 2 Milestone 7:

1. **EMA-VFI ONNX dynamic input resolution is not working correctly.**
2. **Practical-RIFE ONNX and PyTorch outputs show non-trivial differences.**
3. **Local video inference is still sequential and does not use true model batch inference.**

The stage must resolve, classify, or explicitly defer these problems before Stage 2 continues to the BentoML compatibility proof.

Stage 2.5 is not a backend stage. It does not implement FastAPI, Celery, Redis, application PostgreSQL, frontend, monitoring, or production deployment.

---

## 2. Relationship to Existing Stages and ExecPlans

### 2.1 Stage 1

Stage 1 built the local ML core:

- preprocessing;
- dataset manifests;
- metrics and baselines;
- adapters;
- validation;
- local inference;
- MLflow helpers;
- model integration.

Stage 2.5 may reuse Stage 1 utilities, especially:

- `src/video_interpolation/mlflow.py`;
- metrics helpers;
- image I/O helpers;
- inference measurement conventions;
- human-facing documentation style.

Stage 2.5 must not break Stage 1 training/fine-tuning behavior.

### 2.2 Stage 2

Stage 2 introduced the new inference runtime refactor:

- request/result inference API;
- PyTorch runtime boundaries;
- ONNX export/runtime work;
- EMA-VFI and Practical-RIFE runtime separation;
- Nx interpolation support;
- local video inference through the refactored API.

Stage 2.5 continues from that point.

The active Stage 2 ExecPlan remains important and should be read for prior decisions. However, Stage 2.5 must have its own ExecPlan:

```text
.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md
```

After Stage 2.5 is complete, the Stage 2 ExecPlan should either:

- record a concise handoff entry pointing to the completed Stage 2.5 ExecPlan; or
- be updated with a short summary of Stage 2.5 changes that affect the remaining Stage 2 milestones.

Do not duplicate the full Stage 2.5 history inside the Stage 2 ExecPlan.

---

## 3. Scope

Stage 2.5 includes:

1. EMA-VFI ONNX dynamic-resolution investigation and refactor attempt.
2. ONNX-vs-PyTorch equivalence checks on both synthetic tensors and real image pairs.
3. True batched inference for EMA-VFI and Practical-RIFE, starting with PyTorch backend.
4. Optional ONNX batched inference only if ONNX support remains viable for the relevant model.
5. Mini-benchmarks comparing inference backends and execution modes.
6. MLflow logging for benchmark runs.
7. Human-facing documentation for the new diagnostic, equivalence, batching, and benchmark workflows.

---

## 4. Non-Goals

Do not implement during Stage 2.5:

- BentoML compatibility proof;
- production BentoML services;
- FastAPI backend;
- Celery/Redis job system;
- PostgreSQL application schema;
- MinIO upload/output orchestration;
- frontend;
- monitoring/alerts;
- retraining triggers;
- AMT-S refactor;
- training/fine-tuning changes, except where needed to preserve compatibility;
- static bucket fallback for EMA-VFI ONNX;
- long full-dataset or full-video experiments by default.

---

## 5. Key Decisions

### 5.1 EMA-VFI ONNX Dynamic H/W Policy

EMA-VFI PyTorch inference supports varying input resolutions through project-owned padding and runtime logic.

The desired ONNX behavior is:

```text
arbitrary input H/W
→ project-owned padding if needed
→ ONNX model inference
→ project-owned unpadding
→ output at original resolution
```

Stage 2.5 should attempt to make EMA-VFI ONNX support dynamic or constrained-dynamic input resolution.

A promising hypothesis is that EMA-VFI may work if the project pads input frames externally to sizes compatible with the model's internal window logic. Based on the current architecture and logs, a likely candidate is padding to a multiple of:

```text
8 * 7 = 56
```

because the feature extractor downsamples spatial dimensions by 8 and then applies window-based operations with window size 7.

This hypothesis must be verified. Do not assume that 56 is sufficient without testing.

### 5.2 No Static Bucket Fallback for EMA-VFI

If EMA-VFI ONNX cannot be made to support dynamic or constrained-dynamic input resolution, do not implement static shape bucket fallback as the production strategy.

Do not create a deployment strategy based on multiple fixed EMA ONNX artifacts such as:

```text
512x512
720p
1080p
portrait bucket
...
```

Do not design a service that dynamically loads/unloads different EMA ONNX sessions for different image-size buckets.

If dynamic/constrained-dynamic EMA ONNX is not feasible, the expected outcome is:

```text
EMA-VFI serving remains PyTorch-only.
EMA-VFI ONNX dynamic H/W is marked blocked/deferred.
```

### 5.3 Practical-RIFE ONNX Equivalence Policy

Practical-RIFE ONNX can execute on dynamic input sizes, but previous synthetic-tensor checks showed non-trivial differences between PyTorch and ONNX outputs.

Stage 2.5 must not treat this as either safe or broken without additional evidence.

The stage must add real-image equivalence checks using image pairs from:

```text
raw_data/pair_test/
```

These checks must be added in addition to synthetic input checks, not instead of them.

### 5.4 Batch Inference Policy

True batch inference means processing multiple neighboring frame pairs in one model call.

For a sequence of frames:

```text
[0, 1, 2, 3, 4]
```

the model should process pairs:

```text
[[0, 1], [1, 2], [2, 3], [3, 4]]
```

and produce intermediate frames:

```text
[0.5, 1.5, 2.5, 3.5]
```

The next chunk must begin with the last source frame from the previous chunk:

```text
previous chunk: [0, 1, 2, 3, 4]
next chunk:     [4, 5, 6, 7, 8]
```

Batch inference must support both:

- fixed 2x interpolation;
- arbitrary Nx interpolation.

For Nx, Stage 2.5 should implement the more direct and efficient strategy:

```text
batch dimension = pair_index × timestep_index
```

For example, if a chunk contains `B` neighboring pairs and `interpolation_factor = N`, the model batch may contain:

```text
B * (N - 1)
```

inference items, one per pair/timestep combination.

This avoids a Python loop over timesteps.

Sequential Nx inference must remain available as fallback.

### 5.5 Batch Inference Defaults

The new batched path may become the default inference path with:

```text
inference_batch_size = 1
```

by default.

This preserves sequential-like behavior while routing through the same batched implementation.

A fully sequential interface should still remain available for debugging and for low-VRAM cases.

### 5.6 VRAM and Resolution Policy

Batch inference increases VRAM usage.

Stage 2.5 must account for VRAM safety:

- configurable `inference_batch_size`;
- clear errors on CUDA OOM;
- optional auto-retry with smaller batch size if implemented cleanly;
- optional maximum input resolution limits if needed;
- documentation explaining how batch size and resolution affect memory use.

Do not silently downscale user inputs unless explicitly configured.

If a maximum resolution guard is added, it must fail clearly or route to a documented fallback.

### 5.7 Benchmarking Policy

Stage 2.5 must add small, repeatable inference benchmarks.

Benchmarks should compare:

- PyTorch sequential inference;
- PyTorch batch inference;
- ONNX sequential inference if ONNX remains viable;
- ONNX batch inference if implemented;
- fixed 2x;
- arbitrary Nx where relevant.

Benchmarks must break down time into useful components where practical:

- decode time;
- preprocessing / tensor conversion time;
- model inference time;
- postprocessing time;
- video encoding time;
- total time.

Benchmark runs should log results to MLflow using the existing Stage 1 MLflow utilities and infrastructure.

---

## 6. EMA-VFI Dynamic ONNX Problem

### 6.1 Observed Problem

EMA-VFI ONNX export may succeed and ONNX checker/simplification may pass, but the exported model fails when run on a different input size than the one used during export.

This means the model is only superficially dynamic: H/W axes may be marked dynamic, but internal graph logic still contains shape-specialized assumptions.

### 6.2 Evidence from Logs

When exporting with `dynamo=True`, PyTorch warned:

```text
'dynamic_axes' is not recommended when dynamo=True
Supply the 'dynamic_shapes' argument instead
```

The export also warned that a tensor attribute was assigned during export:

```text
self.net.feature_bone.cor[...] was assigned during export.
Such attributes must be registered as buffers...
```

This indicates forward-time model state mutation during export.

The exported graph also showed shape-dependent operations and tensors created directly on CUDA:

```text
torch.linspace(... device = device(type='cuda', index=0))
```

When exporting without `dynamo`, PyTorch emitted warnings from EMA-VFI `feature_extractor.py`:

```text
math.ceil(h / window_size...)
math.ceil(w / window_size...)
if pad_h > 0 or pad_w > 0
int(...)
self.HW.item() == H_p * W_p
```

These warnings mean Python-side shape computations were traced as constants. Therefore, the traced ONNX graph may not generalize to other H/W sizes.

### 6.3 Likely Root Causes

The likely root causes are inside:

```text
model_repos/EMA-VFI/model/feature_extractor.py
```

and related EMA-VFI model code.

Potentially problematic logic includes:

- `pad_if_needed`;
- `depad_if_needed`;
- `window_partition`;
- `window_reverse`;
- attention mask generation;
- cached `self.HW`;
- cached `attn_mask`;
- `feature_bone.cor`;
- Python `math.ceil`, `int`, `.item()`, and `if` logic based on input shapes;
- hardcoded CUDA tensor creation;
- module-level or forward-time mutable state.

### 6.4 Required Investigation Route

Codex must approach EMA-VFI ONNX dynamic H/W as an investigation/refactor, not as a one-line export flag change.

Expected steps:

1. Inspect EMA-VFI feature extractor and runtime wrapper.
2. Identify every shape-dependent Python branch used by the export path.
3. Identify forward-time state mutation and caches.
4. Check whether external padding to a valid multiple, likely 56, can avoid problematic internal padding branches.
5. Try `torch.onnx.export(..., dynamo=True, dynamic_shapes=...)` only after obvious blockers are addressed.
6. Validate ONNX Runtime on at least two different input sizes.
7. Classify the result.

### 6.5 Possible Outcomes

Allowed outcomes:

```text
EMA dynamic ONNX works.
EMA constrained-dynamic ONNX works with external padding to documented multiples.
EMA ONNX works only for original export size and is not suitable.
EMA ONNX remains blocked; EMA serving should use PyTorch backend.
```

Not allowed as a final Stage 2.5 production strategy:

```text
EMA static bucket ONNX fallback.
```

---

## 7. Practical-RIFE ONNX Equivalence Problem

### 7.1 Observed Problem

Practical-RIFE ONNX Runtime can execute on more than one input shape, but CUDA ONNX output may differ noticeably from PyTorch output.

Previous synthetic tests showed low average error but large local maximum differences. This may or may not be visually important.

### 7.2 Required Validation Expansion

Stage 2.5 must extend equivalence checks to real image pairs.

Use:

```text
raw_data/pair_test/
```

Expected directory style:

```text
raw_data/pair_test/
  001/
    frame1.png
    frame2.png
  002/
    frame1.png
    frame2.png
  ...
```

Codex must verify the actual structure before implementation.

For every pair, compare:

```text
PyTorch output
vs
ONNX output
```

for the same:

- model;
- checkpoint;
- backend/provider;
- input tensors;
- padding/preprocessing;
- timestep;
- interpolation factor.

### 7.3 Required Equivalence Metrics

For each sample:

- MAE;
- max absolute error;
- MSE if useful;
- PSNR between PyTorch output and ONNX output;
- SSIM between PyTorch output and ONNX output if available;
- optional LPIPS if cheap and already available, but do not make LPIPS mandatory for smoke runs.

Output visual artifacts should include:

```text
left input frame
right input frame
pytorch generated frame
onnx generated frame
absolute difference visualization
```

### 7.4 Synthetic Tests Remain Required

Real-image tests are added to the pipeline. They do not replace synthetic tests.

Synthetic tests remain useful because they stress shape/operator behavior.

Real-image tests are useful because they answer the practical service question:

```text
Do ONNX and PyTorch outputs look materially different on realistic inputs?
```

---

## 8. Batch Inference

### 8.1 Problem

Current video inference is sequential. The model processes one neighboring frame pair at a time.

This is slow and underuses GPU parallelism.

A one-second video can take much longer than one second to process, especially with high resolution and Nx interpolation.

### 8.2 Required Batch Inference Semantics

Implement true batch inference for neighboring frame pairs.

For fixed 2x:

```text
input frames: [0, 1, 2, 3, 4]
pairs:        [[0,1], [1,2], [2,3], [3,4]]
outputs:      [0.5, 1.5, 2.5, 3.5]
```

For Nx:

```text
for each pair:
  generate t = 1/N, 2/N, ..., (N-1)/N
```

Preferred Nx batching strategy:

```text
batch items = all pair/timestep combinations in the chunk
```

This means the runtime may internally flatten:

```text
pair index × timestep index
```

into one batch dimension.

Output ordering must be reconstructed exactly.

### 8.3 Chunking

Video-level batch inference should process chunks.

If a chunk contains `K + 1` source frames, it contains `K` neighboring pairs.

The next chunk must overlap by one source frame:

```text
chunk 1: [0, 1, 2, ..., K]
chunk 2: [K, K+1, K+2, ...]
```

This preserves the pair between the final frame of the previous chunk and the next frame.

### 8.4 Runtime API

The runtime API should support a batched pair call, conceptually:

```python
predict_pairs_batch(...)
```

or an equivalent request/result API.

The public API should make clear whether it is:

- sequential pair inference;
- batched pair inference;
- video-level chunked inference.

Existing `predict_batch` naming must be reviewed carefully because Stage 1 used some "batch inference" terms for directory-wide all-target inference.

Avoid naming confusion.

### 8.5 OOM and Memory Handling

Batch inference must be configurable.

Expected parameters:

```text
inference_batch_size
max_resolution or max_pixels guard if implemented
fallback_on_oom
```

OOM handling should be clear.

Acceptable behavior:

- fail with a clear message recommending a smaller batch size;
- optionally retry with smaller batch size if simple and safe;
- log OOM events in benchmark output.

Do not silently reduce image resolution unless explicitly configured.

---

## 9. Mini-Benchmarks

### 9.1 Purpose

The project needs objective measurements before and after batch inference.

Benchmarks should establish baseline performance and evaluate whether batch inference improves practical throughput.

### 9.2 Benchmark Targets

Benchmarks should support:

- EMA-VFI PyTorch sequential;
- EMA-VFI PyTorch batch;
- Practical-RIFE PyTorch sequential;
- Practical-RIFE PyTorch batch;
- ONNX variants only if ONNX remains viable.

### 9.3 Benchmark Inputs

Use small and safe inputs:

- synthetic tensors for pure runtime microbenchmarks;
- real image pairs from `raw_data/pair_test`;
- very short video clips from `raw_data/tmp_test` if video-level benchmark is needed.

Do not run full videos from `raw_data/anime` by default.

### 9.4 Benchmark Metrics

Record:

- model name;
- backend;
- inference mode;
- interpolation factor;
- batch size;
- input shape;
- number of source frames;
- number of pairs;
- number of generated frames;
- decode time;
- preprocessing time;
- model inference time;
- postprocessing time;
- encode time;
- total time;
- pairs per second;
- generated frames per second;
- peak VRAM if available;
- output paths if visual samples are generated.

### 9.5 MLflow Logging

Benchmarks must integrate with existing MLflow utilities.

Use:

```text
src/video_interpolation/mlflow.py
```

and existing MLflow setup conventions from Stage 1.

If MLflow service is unavailable, allow explicit `--disable-mlflow` for local smoke tests, but benchmark code should be capable of logging to MLflow.

Log:

- params;
- timing metrics;
- output CSVs;
- visual artifacts if generated;
- benchmark config.

---

## 10. Expected Stage Deliverables

Stage 2.5 is complete when the repository contains:

1. EMA-VFI ONNX dynamic/constrained-dynamic investigation and result.
2. Clear decision: EMA ONNX usable or EMA PyTorch-only.
3. Real-image ONNX-vs-PyTorch equivalence pipeline.
4. Equivalence reports for EMA where ONNX is usable and Practical-RIFE.
5. True batched inference API for EMA and Practical-RIFE.
6. Video-level batched inference support, at least for PyTorch backend.
7. Sequential fallback still available.
8. Configurable `inference_batch_size`.
9. Mini-benchmark commands or APIs.
10. MLflow logging for benchmark results.
11. Updated documentation.
12. Updated `PROJECT_MAP.md`.
13. Completed Stage 2.5 ExecPlan handoff.
14. Short pointer/update in the Stage 2 ExecPlan explaining that Stage 2.5 was completed and where to find its handoff.

---

## 11. Suggested Milestones

The Stage 2.5 ExecPlan should decide final milestones, but a recommended structure is:

### Milestone 1 — EMA Dynamic ONNX Investigation

- inspect EMA feature extractor;
- test `dynamo=True + dynamic_shapes`;
- remove or bypass export blockers;
- test external padding to valid multiples;
- validate multiple H/W sizes;
- classify EMA ONNX support.

### Milestone 2 — Real-Image ONNX/PyTorch Equivalence

- add pair-test input discovery;
- run synthetic and real-image checks;
- write equivalence reports and visual artifacts;
- support EMA if ONNX usable;
- support Practical-RIFE;
- document whether ONNX output is acceptable.

### Milestone 3 — Runtime Batch Inference API

- design clear batch API;
- avoid naming conflict with directory-wide batch inference;
- implement batch pair requests/results;
- support pair×timestep flattening;
- keep sequential fallback.

### Milestone 4 — PyTorch Batch Inference for EMA and RIFE

- implement batch inference for PyTorch EMA and RIFE;
- support fixed 2x and Nx;
- add chunking for video inference;
- preserve output order;
- add OOM-safe behavior.

### Milestone 5 — ONNX Batch Inference Where Viable

- implement only if ONNX support remains viable;
- otherwise mark as skipped/deferred with reason.

### Milestone 6 — Benchmarks and MLflow Logging

- add benchmark CLI/API;
- log to MLflow;
- compare sequential vs batch;
- compare backend variants if available;
- write CSV/JSON summaries.

### Milestone 7 — Documentation and Handoff

- update human-facing docs;
- update project map;
- update Stage 2 ExecPlan with pointer to Stage 2.5;
- complete Stage 2.5 ExecPlan.

---

## 12. Validation Strategy

Run normal project validation:

```bash
uv run pytest
uv run ruff check src tests
```

If cache permissions require it:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
```

Add focused tests for:

- real pair discovery;
- equivalence metrics;
- report writing;
- batch output ordering;
- chunk overlap correctness;
- factor/timestep ordering;
- OOM fallback behavior if implemented;
- benchmark metric aggregation.

Add CUDA smoke checks where available, but keep them safe and small.

Do not require long GPU jobs for automated tests.

---

## 13. Documentation Requirements

Create or update a human-facing documentation file, for example:

```text
docs/stage2_5_inference_runtime_stabilization.md
```

Document:

- why Stage 2.5 exists;
- EMA ONNX dynamic H/W outcome;
- Practical-RIFE ONNX equivalence outcome;
- how to run real-image equivalence checks;
- how to interpret MAE/max error/PSNR/SSIM;
- how batch inference works;
- difference between sequential and batch inference;
- difference between directory-wide all-target inference and true model batch inference;
- batch size configuration;
- VRAM and resolution considerations;
- benchmark commands;
- MLflow logging behavior;
- known limitations;
- what Stage 2 can resume doing after Stage 2.5.

---

## 14. Final Handoff Requirements

At the end of Stage 2.5, the ExecPlan must summarize:

- EMA ONNX decision;
- Practical-RIFE ONNX decision;
- whether ONNX should be used for future BentoML serving;
- whether PyTorch backend should be the default serving path;
- batch inference status;
- benchmark results;
- recommended backend/model for first BentoML proof;
- unresolved blockers;
- exact next step for resuming Stage 2 Milestone 8.
