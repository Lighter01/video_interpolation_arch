# Title and Metadata

- Stage: Stage 2.5 - ONNX Stabilization, Real-Image Equivalence, and Batched Inference
- Status: Completed ExecPlan - accepted, Stage 2 resumed at Milestone 8
- Created: 2026-06-01
- Updated: 2026-06-01
- Stage plan: `.agent/stage_plans/02_5_inference_runtime_stabilization_stage_plan.md`
- Global plan: `.agent/general_plan.md`
- Prior ExecPlan: `.agent/docs/exec-plans/active/02_inference_runtime_refactor.execplan.md` and `.agent/docs/exec-plans/completed/01_ml_core_selected.execplan.md`
- Scope authority: current user task plus `.agent/stage_plans/02_5_inference_runtime_stabilization_stage_plan.md`

## Stage Goal

Stage 2.5 is an intermediate stabilization stage before Stage 2 resumes with the minimal BentoML compatibility proof.

The stage must resolve, classify, or explicitly defer the technical blockers found by the end of Stage 2 Milestone 7:

- EMA-VFI ONNX exports are only superficially dynamic and fail on changed H/W.
- Practical-RIFE ONNX can execute at dynamic sizes, but ONNX-vs-PyTorch equivalence is not yet trusted.
- Local video inference is still model-sequential, and `batch_inference.py` means directory-wide orchestration rather than true model batching.

The stage must produce a clear runtime path for PyTorch batch inference, expanded ONNX equivalence evidence on synthetic and real images, video-pipeline benchmarks with MLflow logging, user-facing documentation, and a handoff pointer back to Stage 2.

This is not a backend stage. It must not implement BentoML proof work, production BentoML services, FastAPI, Celery, Redis, PostgreSQL application schema, MinIO upload/output orchestration, frontend, monitoring, retraining triggers, AMT-S refactor, or training/fine-tuning changes.

## Source Documents and Authority

Read before creating this ExecPlan:

- `.agent/TASK.md`
- `.agent/tasks/TASK_2_5.md`
- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/general_plan.md`
- `.agent/stage_plans/02_inference_runtime_refactor_stage_plan.md`
- `.agent/stage_plans/02_5_inference_runtime_stabilization_stage_plan.md`
- `.agent/docs/exec-plans/active/02_inference_runtime_refactor.execplan.md`
- `.agent/docs/exec-plans/completed/01_ml_core_selected.execplan.md`
- `docs/stage1_ml_core.md`
- `docs/stage2_inference_runtime_refactor.md`
- `src/video_interpolation/inference_runtime/api.py`
- `src/video_interpolation/inference_runtime/ema.py`
- `src/video_interpolation/inference_runtime/rife.py`
- `src/video_interpolation/inference_runtime/onnx_export.py`
- `src/video_interpolation/inference_runtime/onnx_validation.py`
- `src/video_interpolation/inference_runtime/backends/torch.py`
- `src/video_interpolation/inference_runtime/backends/onnx.py`
- `src/video_interpolation/inference_runtime/rife_upstream/IFNet_HDv3.py`
- `src/video_interpolation/inference_runtime/rife_upstream/RIFE_HDv3.py`
- `src/video_interpolation/inference_runtime/rife_upstream/warplayer.py`
- `src/video_interpolation/adapters/base.py`
- `src/video_interpolation/adapters/ema_vfi.py`
- `src/video_interpolation/adapters/rife.py`
- `src/video_interpolation/inference.py`
- `src/video_interpolation/batch_inference.py`
- `src/video_interpolation/cli.py`
- `src/video_interpolation/mlflow.py`
- `src/video_interpolation/metrics.py`
- `src/video_interpolation/image_io.py`
- `configs/models/ema_vfi_small.yaml`
- `configs/models/practical_rife_v4_26.yaml`
- `configs/models/practical_rife_v4_25.yaml`
- `configs/inference/ema_vfi_small_2x.yaml`
- `configs/inference/practical_rife_v4_26_2x.yaml`
- `configs/inference/practical_rife_v4_25_2x.yaml`
- `tests/test_inference_runtime_api.py`
- `tests/test_ema_adapter.py`
- `tests/test_rife_adapter.py`
- `tests/test_inference.py`
- `tests/test_onnx_export.py`
- `tests/test_onnx_runtime.py`
- `model_repos/EMA-VFI/model/feature_extractor.py`
- `model_repos/EMA-VFI/model/flow_estimation.py`
- `model_repos/EMA-VFI/model/warplayer.py`
- `model_repos/Practical-RIFE/`
- `model_weights/EMA-VFI/`
- `model_weights/Practical-RIFE/`
- `model_exports/onnx/`
- `outputs/onnx_validation/`
- `raw_data/pair_test/`
- `raw_data/tmp_test/`
- `pyproject.toml`

The current user task and the Stage 2.5 stage plan control this stage. The active Stage 2 ExecPlan controls prior implementation facts and may resume from Milestone 8 after user acceptance of this Stage 2.5 closeout.

## Context and Current Repository State

Repository root: `/home/lighter_01/projects/itmo/ai_architecture/video_interpolation`.

Stage 1 is complete for handoff. Stage 2.5 has completed Milestones 1 through 9 and is ready for user acceptance before Stage 2 resumes.

Current runtime implementation:

- `src/video_interpolation/inference_runtime/api.py` defines `FramePairRequest`, `FramePairResult`, `ModelBatchRequest`, `ModelBatchResult`, `FlattenedFramePairBatch`, `PairTimestepIndex`, `InferenceMode`, `RuntimeBackendKind`, mode/factor validation, and timestep generation for fixed 2x and arbitrary Nx.
- `FramePairRequest` accepts CHW or NCHW tensors. `ModelBatchRequest` is the separate true model batch contract and requires BCHW `left`/`right` tensors with matching batch size and shape.
- `ModelBatchRequest.flatten_pair_timesteps()` defines the later runtime execution order as pair-major/timestep-minor. `ModelBatchResult.from_flattened_outputs(...)` reconstructs flattened BCHW outputs into `outputs[pair_index][timestep_index]`.
- `src/video_interpolation/inference_runtime/ema.py` has `EMAVFIPyTorchRuntime` and `EMAVFIOnnxRuntime`. Both PyTorch and viable constrained-dynamic ONNX paths now support `predict_batch(ModelBatchRequest)` using pair-major pair×timestep flattening; EMA ONNX uses true multi-row ORT calls when the artifact supports symbolic batch.
- `src/video_interpolation/inference_runtime/rife.py` has `PracticalRIFEPyTorchRuntime` and `PracticalRIFEOnnxRuntime`. PyTorch and accepted dynamic-batch ONNX paths support true model-batch execution; RIFE ONNX requires request scale to match the scale baked into the artifact and rejects fixed-batch artifacts.
- `src/video_interpolation/inference_runtime/onnx_export.py` exports neural-core-only wrappers: `left`, `right`, `timestep` to one generated frame. Dynamo export is the default and declares symbolic batch plus dynamic H/W; legacy export remains available explicitly.
- `src/video_interpolation/inference_runtime/onnx_validation.py` compares synthetic and real tensor pairs. It reports MAE, max absolute error, MSE, `torch.allclose`, graph I/O, padded/output shapes, and visual PyTorch/ONNX/diff images when relevant.
- `src/video_interpolation/inference.py` decodes and encodes videos through the local workflow, supports explicit `sequential` and default `batched` execution modes, and uses `ModelBatchRequest` for chunked EMA/RIFE video inference where adapters support it.
- `src/video_interpolation/batch_inference.py` discovers videos, resolves target names, builds output paths and run names, and writes measurement CSVs. It remains directory-wide workflow orchestration, separate from true model batch inference.
- `ModelAdapter.predict_batch(...)` in `src/video_interpolation/adapters/base.py` remains a sequential default wrapper over `predict_pair(...)`; EMA and Practical-RIFE override it for fixed 2x model-batch execution and expose `predict_frame_pairs_batch(ModelBatchRequest)` for fixed 2x and Nx.

Current model/config state:

- `configs/models/ema_vfi_small.yaml` uses `EMA-VFI/ours_small_t.pkl` as the Stage 2 inference checkpoint and records `EMA-VFI/ours_small.pkl` as the training checkpoint.
- `configs/models/practical_rife_v4_26.yaml` is the active Practical-RIFE default. `configs/models/practical_rife_v4_25.yaml` remains available as an alternative.
- EMA and Practical-RIFE configs declare `fixed_2x` and `arbitrary_nx` support with factor bounds `2..8` and optional `inference_batch_size` caps for flattened PyTorch model-batch rows.
- `pyproject.toml` already lists `onnx`, `onnxruntime`, `onnxruntime-gpu`, `onnx-simplifier`, `bentoml`, and `onnxscript`; Stage 2.5 should not add dependencies unless a specific gap is discovered.

Current ONNX artifacts and validation outputs:

- `model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx`
- `model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx.data`
- `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx`
- `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx.data`
- ONNX validation reports exist under `outputs/onnx_validation/`.
- Obsolete EMA Milestone 2 and Task 2.5 trial export/output directories were cleaned before Milestone 3. The accepted EMA artifact is now the promoted constrained-dynamic opset 18 artifact under `model_exports/onnx/ema_vfi_small/`.
- EMA ONNX requires project-owned external divisor `112` padding; divisor `32` still fails and should not be used for the constrained-dynamic artifact.
- Practical-RIFE now has a dynamo opset 18 artifact with symbolic batch plus symbolic `128*height_units` and `128*width_units`. It runs at multiple H/W shapes and true flattened batch sizes; strict `1e-3` allclose still fails on some larger synthetic shapes, while controlled real-image equivalence passes on the current pair fixtures.

Current EMA ONNX risk area:

- `model_repos/EMA-VFI/model/feature_extractor.py` still contains shape-dependent Python logic: `math.ceil`, `int(...)`, `.item()`, `if` branches on tensor-derived shapes, forward-time `register_buffer(...)`, cached `attn_mask` and `HW`, and `window_partition` / `window_reverse` shape math.
- `feature_extractor.get_cor(...)` caches coordinate tensors by shape and device, but not dtype.
- `model_repos/EMA-VFI/model/flow_estimation.py` has already replaced hardcoded CUDA timestep creation with input-device/dtype-aware tensor creation.
- `model_repos/EMA-VFI/model/warplayer.py` has already been patched to device/dtype/shape-aware grid creation and cache keys.
- The Stage 2.5 plan's 56-multiple hypothesis is plausible because EMA downscales spatial dimensions by 8 and then applies window size 7 logic, but it is unverified.

Current data for Stage 2.5:

- `raw_data/pair_test/` contains three real image-pair directories: `001`, `002`, and `003`.
- Each pair currently has `frame1.png` and `frame2.png`, and `file` reports `512 x 320`, 8-bit RGB PNGs.
- `raw_data/tmp_test/` contains short MP4 files `001.mp4` through `007.mp4` plus `DORA_cut.mp4`.

Current repository cleanliness:

- `git status --short` shows a modified `outputs/candidate_validation/ema_vfi_small/stage1_default/candidate_validation_report.json` and untracked `model_exports/` plus `outputs/onnx_validation/`.
- These appear to be existing generated artifacts, not Stage 2.5 implementation work. Do not revert or delete them.

## Stage Requirements Restated

Implementation-ready requirements:

- Keep Stage 2 paused before BentoML. Resume Stage 2 only after Stage 2.5 is complete.
- Investigate EMA-VFI ONNX dynamic H/W as a real export/runtime/refactor problem, not as a one-flag export change.
- Verify whether external padding to a constrained dynamic multiple, likely 56, makes EMA ONNX usable across more than one input H/W.
- Do not implement static bucket fallback for EMA-VFI ONNX.
- If EMA dynamic or constrained-dynamic ONNX cannot be made usable, mark EMA ONNX blocked/deferred and keep EMA serving PyTorch-only.
- Add real-image ONNX-vs-PyTorch equivalence checks using `raw_data/pair_test/`.
- Keep synthetic ONNX equivalence checks. Real-image checks are additive, not a replacement.
- Expand equivalence metrics to include MAE, max absolute error, MSE, PSNR, SSIM, and optional LPIPS where cheap and already available.
- Write visual equivalence artifacts for real-image checks: left input, right input, PyTorch output, ONNX output, and absolute-difference visualization.
- Design and implement a true model batch inference API distinct from directory-wide `batch_inference.py`.
- Support batch inference for fixed 2x and arbitrary Nx.
- Use pair-by-timestep flattening for Nx batches instead of a Python loop over timesteps.
- Keep sequential inference fallback available for debugging and low-VRAM runs.
- Make batch inference VRAM-aware and configurable with an explicit batch-size control and clear OOM behavior.
- Implement PyTorch batch inference for EMA-VFI and Practical-RIFE.
- Implement ONNX batch inference only if the relevant ONNX path remains viable.
- Add video-pipeline benchmarks comparing backend, execution mode, interpolation mode, factor, and batch size where applicable.
- Benchmark timing should include decode, preprocessing/tensor conversion, model inference, postprocessing, encode, and total time where practical.
- Log benchmark params, metrics, CSV/JSON summaries, and artifacts to MLflow using existing `src/video_interpolation/mlflow.py` conventions when MLflow is enabled.
- Add human-facing documentation for diagnostics, equivalence, batch inference, benchmark usage, limitations, and fallback recommendations.
- Update `.agent/docs/PROJECT_MAP.md` when implementation changes important files or commands.
- At Stage 2.5 completion, update the Stage 2 ExecPlan with a concise pointer to the completed Stage 2.5 handoff.

Mandatory preserved decisions:

- No EMA static ONNX bucket fallback.
- EMA PyTorch-only is acceptable if dynamic/constrained-dynamic ONNX remains blocked.
- Real-image equivalence checks must supplement synthetic checks.
- Batch inference must support fixed 2x and Nx.
- Preferred Nx batching is pair-by-timestep flattening.
- Sequential fallback must remain available.
- Batch inference must be configurable and VRAM-aware.
- Benchmarks must log to MLflow where available.
- Stage 2 remains paused until Stage 2.5 is complete.

## Non-Goals and Deferred Work

Do not implement in Stage 2.5:

- Minimal BentoML compatibility proof.
- Production BentoML services.
- FastAPI backend.
- Celery/Redis jobs.
- PostgreSQL application schema.
- MinIO upload/output orchestration.
- Frontend.
- Monitoring, alerts, retraining triggers, or online quality history.
- AMT-S runtime refactor, AMT-S ONNX export, or AMT-S Nx support.
- Training/fine-tuning changes, except minimal compatibility preservation if touched code paths require it.
- EMA static ONNX shape bucket serving strategy.
- Full-dataset benchmarks or long full-video jobs by default.
- New dependencies unless a specific missing dependency is identified, recorded, and justified.

Safely deferred within or after Stage 2.5:

- ONNX batch inference for EMA if EMA ONNX is blocked.
- ONNX batch inference for Practical-RIFE if real-image equivalence indicates unacceptable divergence.
- CUDA-provider ONNX acceptance if the local environment cannot produce stable CUDA evidence; record CPU and CUDA separately.
- Automatic OOM retry if a clear first implementation would be risky. A clear error recommending a smaller batch size is acceptable first.

## Architecture and Implementation Strategy

### EMA-VFI ONNX stabilization

Start with evidence, not rewrites:

- Reproduce controlled EMA ONNX validation from the existing artifact and newly exported artifacts.
- Record exact provider, artifact path, simplified/original choice, export sample shape, runtime input shape, external padding policy, and command.
- Test dynamic shapes at normal divisors and constrained 56-multiple shapes. Candidate synthetic shapes include `56x56`, `112x112`, `112x168`, and a pair-test-derived externally padded shape such as `336x560`.
- If 56-multiple constrained dynamic execution works, encode that as a documented external padding policy: original H/W to padded multiple, ONNX inference, then unpad.
- If constrained dynamic still fails, inspect and minimally refactor the EMA feature extractor export path.

Likely EMA source areas:

- `pad_if_needed(...)` and `depad_if_needed(...)` in `model_repos/EMA-VFI/model/feature_extractor.py`
- `MotionFormerBlock.forward(...)` branch using `self.HW.item() == H_p * W_p`
- forward-time `register_buffer("attn_mask", ...)` and `register_buffer("HW", ...)`
- `window_reverse(...)` using Python `int(...)`
- `feature_extractor.get_cor(...)` cache key missing dtype

Do not introduce a serving design with multiple fixed EMA ONNX artifacts. If the refactor would become broad, classify EMA ONNX as blocked/deferred and keep EMA PyTorch-only.

### Real-image equivalence

Extend `src/video_interpolation/inference_runtime/onnx_validation.py` or add a small adjacent module to support real image pairs.

Planned behavior:

- Discover pairs under `raw_data/pair_test/*/frame1.png` and `frame2.png`.
- Load images through project image I/O helpers and convert to CHW float tensors.
- Reuse the same PyTorch and ONNX runtime predictors used by synthetic checks.
- Compare output frames at the same model, checkpoint, provider, artifact, scale, mode, factor, timestep, and padding policy.
- Write per-sample metrics and aggregate summary.
- Write left/right/PyTorch/ONNX/absdiff PNGs.
- Include synthetic and real records in one report format or clearly separated sibling reports under the same run directory.
- Reuse `metrics.compute_psnr(...)` and `metrics.compute_ssim(...)` for PyTorch-vs-ONNX output image similarity.

### True model batch API

Avoid overloading the existing directory-wide `batch_inference.py` meaning. Add a model-runtime batch API with names that make the level explicit, for example:

- `FramePairsBatchRequest`
- `FramePairsBatchResult`
- `predict_frame_pairs_batch(...)`

The batch request should describe a batch of neighboring frame pairs:

- `left`: BCHW tensor
- `right`: BCHW tensor
- `mode`: fixed 2x or arbitrary Nx
- `interpolation_factor`: `2..8`
- `timesteps`: optional validated tuple
- `backend_kind`: torch or onnx
- `backend_options`: runtime options such as Practical-RIFE scale

The batch result should preserve structure:

- one generated frame per pair for fixed 2x;
- `N - 1` generated frames per pair for Nx;
- output ordering should be `pair_index`, then `timestep_index`;
- metadata should include model name, backend, original shapes, padded shape, elapsed time, batch size, and effective flattened batch size.

For arbitrary Nx, build one flattened model batch:

```text
left_flat  = repeat each left pair for every timestep
right_flat = repeat each right pair for every timestep
t_flat     = tile timesteps across pairs
```

After one model call, reconstruct outputs as:

```text
outputs[pair_index][timestep_index]
```

Sequential wrappers should remain:

- existing `predict_pair(...)`
- existing `predict_intermediate_frames(...)`
- existing video inference sequential mode

Only after the batch API is proven should adapter `predict_batch(...)` be considered for optimized fixed-2x compatibility. If changed, preserve the Stage 1 contract of returning `list[torch.Tensor]`.

### PyTorch batch runtimes

EMA:

- Use existing EMA padding logic on BCHW tensors.
- Ensure `EMAVFIPyTorchRuntime._run_timestep(...)` can receive a tensor timestep shaped for each flattened batch item.
- Call the EMA neural core once for the flattened pair/timestep batch.
- Unpad and reshape the output back to pair/timestep structure.

Practical-RIFE:

- Use existing `pad_to_divisor(...)` on BCHW tensors.
- Build a per-item timestep tensor for the flattened batch.
- Keep `scale` per request; do not mix different scales in one model batch unless a future API explicitly supports grouping by scale.
- Call `model.inference(...)` or the lower `flownet` boundary once for the flattened pair/timestep batch.
- Unpad and reshape outputs back to pair/timestep structure.

### Video-level batched inference

Add a chunked video path in `src/video_interpolation/inference.py` after the model batch API is stable.

Expected strategy:

- Decode source frames in chunks of `inference_batch_size + 1` frames.
- For a chunk with `K + 1` source frames, build `K` neighboring pairs.
- The next chunk starts with the previous chunk's final source frame.
- Call `predict_frame_pairs_batch(...)` once per chunk when enabled.
- Interleave outputs as `source_i`, generated frames for pair `i`, then continue to the next source frame.
- Keep the current sequential path available through an explicit execution mode or by setting `inference_batch_size=1`.

Likely config/CLI fields:

- `inference_batch_size`: default `1`
- `execution_mode`: `sequential` or `batched`, if useful
- `fallback_on_oom`: optional, default false unless a simple safe retry is implemented
- optional `max_pixels` or `max_resolution` guard only if implemented with clear errors and no silent downscale

### ONNX batch inference

Only implement ONNX batch inference for a model after its ONNX path is classified viable:

- EMA ONNX batch is skipped if EMA dynamic/constrained-dynamic ONNX remains blocked.
- Practical-RIFE ONNX batch is skipped if real-image equivalence shows unacceptable divergence or provider instability.
- If ONNX batch is implemented, it should use the same batch request/result contract and provider/artifact metadata.

### Benchmarks and MLflow

Milestone 8 implements a video-pipeline benchmark workflow in `src/video_interpolation/inference_benchmark.py`, with CLI wiring in `src/video_interpolation/cli.py`.

Benchmark targets:

- one input video through `--input`, defaulting to `raw_data/tmp_test/DORA_cut.mp4`;
- directory inputs through `--input-dir` and optional `--limit-videos`;
- existing local video inference paths through `run_video_inference(...)`, including decode, preprocessing, model interpolation, postprocessing, encode/flush, and audio remux timing;
- no separate video interpolation implementation.

Benchmark dimensions:

- model: EMA-VFI and Practical-RIFE;
- backend: PyTorch, ONNX where viable;
- execution mode: sequential, batch;
- interpolation mode: fixed 2x, arbitrary Nx;
- interpolation factor;
- batch size;
- input shape or sample id;
- provider and RIFE scale.

Timing should capture, where practical:

- decode time;
- preprocessing/tensor conversion time;
- model inference time;
- postprocessing time;
- encode time;
- total time.

Use `src/video_interpolation/mlflow.py` rather than direct ad hoc MLflow setup. It is acceptable to add a general `log_benchmark_run(...)` helper if `log_stage1_run(...)` is too Stage-1-named for clean benchmark semantics. Keep `--disable-mlflow` available for local smoke runs when the MLflow server is not running.

## Milestones and Work Breakdown

### Milestone 1 - Baseline Audit and Reproducible Status Capture

Objective: Freeze the current technical facts before changing ONNX or runtime code.

Likely files/modules:

- `outputs/onnx_validation/`
- `model_exports/onnx/`
- `src/video_interpolation/inference_runtime/onnx_validation.py`
- this ExecPlan

Expected output:

- Concise baseline notes for existing EMA and RIFE ONNX artifacts.
- Reconciled provider/artifact provenance for current validation reports.
- Confirmation of `raw_data/pair_test` structure and image sizes.
- No production code changes unless a tiny report/provenance helper is clearly needed.

Validation checkpoint:

- Existing tests are not required for this audit-only milestone.
- Any rerun commands must be short and recorded.

### Milestone 2 - EMA Dynamic or Constrained-Dynamic ONNX Investigation

Objective: Determine whether EMA ONNX can support dynamic or constrained-dynamic H/W without static buckets.

Likely files/modules:

- `model_repos/EMA-VFI/model/feature_extractor.py`
- `model_repos/EMA-VFI/model/flow_estimation.py`
- `model_repos/EMA-VFI/model/warplayer.py`
- `src/video_interpolation/inference_runtime/onnx_export.py`
- `src/video_interpolation/inference_runtime/ema.py`
- `src/video_interpolation/inference_runtime/onnx_validation.py`
- tests under `tests/`

Expected output:

- Tested and documented 56-multiple external-padding hypothesis.
- If feasible, an EMA constrained-dynamic ONNX export/runtime policy.
- If not feasible, a documented EMA ONNX blocked/deferred classification and PyTorch-only recommendation.
- No static bucket fallback.

Validation checkpoint:

- Synthetic ONNX validation on at least two H/W sizes for any proposed EMA ONNX policy.
- If external 56-multiple padding works, validate at two distinct constrained shapes.
- Focused tests for any new padding policy or report fields.

### Milestone 3 - Real-Image ONNX-vs-PyTorch Equivalence

Objective: Add real-image equivalence checks without removing synthetic checks.

Likely files/modules:

- `src/video_interpolation/inference_runtime/onnx_validation.py`
- possible new `src/video_interpolation/inference_runtime/real_pairs.py` or equivalent
- `src/video_interpolation/metrics.py`
- `src/video_interpolation/image_io.py`
- `src/video_interpolation/cli.py`
- `tests/test_onnx_runtime.py` or a new focused test file
- `raw_data/pair_test/`

Expected output:

- Pair-test discovery and loading.
- Synthetic and real-image equivalence report support.
- Metrics: MAE, max absolute error, MSE, PSNR, SSIM, optional LPIPS.
- Visual artifacts: left, right, PyTorch output, ONNX output, absdiff.
- Practical-RIFE real-image evidence.
- EMA real-image evidence if EMA ONNX remains usable.

Validation checkpoint:

- Focused tests for pair discovery, image loading, metric computation, and report writing.
- Short smoke over the three `raw_data/pair_test` pairs where runtime availability permits.

### Milestone 4 - Model Batch API Contract

Objective: Add a clear batch pair request/result API before wiring model runtimes.

Likely files/modules:

- `src/video_interpolation/inference_runtime/api.py`
- `src/video_interpolation/adapters/base.py`
- focused tests under `tests/test_inference_runtime_api.py`

Expected output:

- Batch request/result dataclasses or equivalent.
- Validated BCHW pair inputs.
- Fixed 2x and arbitrary Nx batch timestep semantics.
- Pair-by-timestep output-order contract.
- Sequential API untouched.

Validation checkpoint:

- Tests for invalid shapes, factor validation, timestep count, pair/timestep ordering, and result shape/order reconstruction.

Milestone 4 implementation status:

- Added `ModelBatchRequest`, `FlattenedFramePairBatch`, `PairTimestepIndex`, and `ModelBatchResult` to `src/video_interpolation/inference_runtime/api.py`.
- The request contract accepts only BCHW `left`/`right` frame batches, validates batch-size and full-shape agreement, reuses existing factor/mode/timestep validation, and preserves sequential `FramePairRequest` behavior.
- Fixed 2x produces one flattened row per pair with repeated timestep `0.5`.
- Nx uses pair-by-timestep flattening: for `B` pairs and factor `N`, flattened batch size is `B * (N - 1)` and row order is `(pair 0, timestep 0..N-2)`, then `(pair 1, timestep 0..N-2)`, and so on.
- `ModelBatchResult` documents/enforces reconstructed output indexing as `outputs[pair_index][timestep_index]` and can rebuild this structure from flattened BCHW model outputs.
- No EMA-VFI, Practical-RIFE, local video inference, or ONNX Runtime batch execution was wired in this milestone.

Milestone 4 validation:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_inference_runtime_api.py` passed: `20 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` passed: `106 passed, 14 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` passed.
- `git diff --check` passed.

### Milestone 5 - PyTorch Batch Inference for EMA-VFI and Practical-RIFE

Objective: Implement true model batch calls for both active PyTorch runtimes.

Likely files/modules:

- `src/video_interpolation/inference_runtime/ema.py`
- `src/video_interpolation/inference_runtime/rife.py`
- `src/video_interpolation/adapters/ema_vfi.py`
- `src/video_interpolation/adapters/rife.py`
- `tests/test_ema_adapter.py`
- `tests/test_rife_adapter.py`

Expected output:

- EMA batch runtime path for fixed 2x and Nx.
- Practical-RIFE batch runtime path for fixed 2x and Nx.
- Pair-by-timestep flattening for Nx.
- Adapter-level batch method with clear naming, such as `predict_frame_pairs_batch(...)`.
- Existing `predict_pair(...)`, `predict_intermediate_frames(...)`, and sequential fallback preserved.

Validation checkpoint:

- Unit tests with fake EMA/RIFE models proving one model call for flattened Nx batches.
- Output order tests for multiple pairs and multiple timesteps.
- Focused CPU-safe tests where possible.
- CUDA smoke commands recorded if available.

Milestone 5 implementation status:

- Added `predict_batch(ModelBatchRequest) -> ModelBatchResult` to `EMAVFIPyTorchRuntime` and `PracticalRIFEPyTorchRuntime`.
- Both PyTorch runtimes use the Milestone 4 pair-major pair×timestep flattening contract. Fixed 2x produces `B` flattened rows. Nx produces `B * (N - 1)` flattened rows and reconstructs `outputs[pair_index][timestep_index]`.
- Both runtimes default to one model call for the full flattened batch. Optional `inference_batch_size` in model configs or `ModelBatchRequest.backend_options` caps flattened rows per PyTorch model call for memory control.
- EMA `fast_tta` forces effective batch size `1`, because the upstream fast-TTA inference implementation indexes only the first two augmented predictions and is not correct for larger model batches.
- Added adapter-level `predict_frame_pairs_batch(ModelBatchRequest)` for EMA/RIFE and changed EMA/RIFE `predict_batch([(left, right), ...])` to use the true fixed-2x model-batch path. Existing `predict_pair`, `predict_intermediate_frames`, `predict_frame_pair`, `predict`, and `__call__` remain available.
- Stage 1 EMA training/fine-tuning code paths remain unchanged; `train_step` and `eval_step` still use the upstream training/update path and do not instantiate the inference runtime.
- ONNX batch inference and local video chunked batching were not started.

Milestone 5 validation:

- Fake/lightweight tests cover fixed 2x batch output ordering, Nx pair×timestep flattening/reconstruction, `outputs[pair_index][timestep_index]`, effective batch-size chunking, adapter wrappers, sequential compatibility, invalid batch shapes via `ModelBatchRequest`, and factor validation.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_ema_adapter.py tests/test_rife_adapter.py` passed: `27 passed`.
- CUDA was unavailable: `torch.cuda.is_available()` returned `False`.
- Real EMA-VFI CPU batch smokes passed on two `32x32` synthetic pairs at factors `2`, `4`, and `8`; each factor used one flattened model call and returned `3x32x32` outputs.
- Real Practical-RIFE CPU batch smokes passed on two `32x32` synthetic pairs at factors `2`, `4`, and `8`; each factor used one flattened model call and returned `3x32x32` outputs.

### Milestone 6 - Video-Level Chunked Batch Inference

Objective: Make local video inference able to use model batching while preserving sequential fallback.

Likely files/modules:

- `src/video_interpolation/inference.py`
- `src/video_interpolation/batch_inference.py`
- `src/video_interpolation/cli.py`
- `configs/inference/*.yaml`
- `tests/test_inference.py`

Expected output:

- Configurable `inference_batch_size`.
- Chunk overlap by one source frame.
- Correct frame interleaving and output FPS for fixed 2x and Nx.
- Sequential fallback with equivalent output order.
- Measurement CSVs include execution mode, inference batch size, and timing breakdown fields where available.
- Clear CUDA OOM errors or safe retry behavior if implemented.

Validation checkpoint:

- Tests for chunk overlap, output ordering, factor 2/4/8 frame counts, fallback behavior, and measurement metadata.
- Tiny video smoke only, no long jobs.

Milestone 6 implementation status:

- Added `VideoInferenceExecutionMode` and top-level `execution_mode` / `inference_batch_size` fields to `VideoInferenceConfig`.
- Local video inference defaults to `batched`, with explicit `sequential` available through config or CLI.
- The batched path builds overlapping source-frame chunks, stacks neighboring pairs into BCHW tensors, calls adapter `predict_frame_pairs_batch(ModelBatchRequest)`, and writes `outputs[pair_index][timestep_index]` in video order.
- Chunk boundaries overlap by one source frame: the final source frame of one chunk is carried into the next chunk as its first source frame.
- `inference_batch_size` is treated as the flattened model-row cap used by Milestone 5. Fixed 2x consumes one row per source pair; Nx consumes `interpolation_factor - 1` rows per source pair, so video pair chunk size is derived from factor and cap.
- Adapters without `predict_frame_pairs_batch(...)` automatically use `sequential_fallback`; explicit `sequential` mode bypasses the batch adapter path.
- Measurement metadata now records requested and actual execution mode, inference batch size, batch chunk count, model batch request count, interpolation factor, existing timing fields, FPS, pair/frame counts, runtime backend, runtime options, and audio preservation counts.
- CLI support was added for `--execution-mode` and `--inference-batch-size` on EMA/RIFE `infer-video` and the directory-wide `infer-all-videos` wrapper.
- Configs for EMA and Practical-RIFE video inference now declare batched mode fields. ONNX batch inference, benchmark workflows, and MLflow benchmark logging were not started.

Milestone 6 validation:

- Focused fake-runtime video tests cover chunk overlap, output frame counts for factors `2`, `4`, and `8`, generated-frame ordering, output FPS multiplication, `inference_batch_size` validation, explicit sequential mode, sequential fallback, measurement CSV fields, and sequential/batched order equivalence.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_inference.py` passed: `27 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` passed: `124 passed, 14 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` passed.
- `git diff --check` passed.
- CUDA was unavailable: `torch.cuda.is_available()` returned `False`.
- Real Practical-RIFE CPU video smokes passed on `raw_data/tmp_test/DORA_cut.mp4` with `limit_pairs=1`, `execution_mode=batched`, and `codec=libx264` for fixed `2x` and Nx factor `4`.
- Real EMA-VFI CPU video smokes passed on `raw_data/tmp_test/DORA_cut.mp4` with `limit_pairs=1`, timestep-capable checkpoint `EMA-VFI/ours_small_t.pkl`, `execution_mode=batched`, and `codec=libx264` for fixed `2x` and Nx factor `4`.

### Milestone 7 - ONNX Batch Inference Where Viable

Objective: Add ONNX batch inference only for models whose ONNX paths remain acceptable.

Likely files/modules:

- `src/video_interpolation/inference_runtime/ema.py`
- `src/video_interpolation/inference_runtime/rife.py`
- `src/video_interpolation/inference_runtime/backends/onnx.py`
- `tests/test_onnx_runtime.py`

Expected output:

- Practical-RIFE ONNX batch support if equivalence is accepted.
- EMA ONNX batch support only if EMA ONNX is dynamic/constrained-dynamic usable.
- Explicit skipped/deferred notes for any model/backend combination not implemented.

Validation checkpoint:

- Synthetic batch equivalence smoke for implemented ONNX batch paths.
- Real-image batch equivalence smoke where feasible.
- Provider and artifact metadata recorded.

Milestone 7 implementation status:

- Implemented `predict_batch(ModelBatchRequest) -> ModelBatchResult` for `EMAVFIOnnxRuntime`.
- EMA ONNX batch uses the same pair-major pair×timestep flattening contract as PyTorch. The accepted EMA dynamo artifact has symbolic batch and constrained dynamic H/W, so fixed 2x and Nx flattened rows can execute in true multi-row ONNX Runtime calls. Optional `inference_batch_size` on the ONNX runtime config or request backend options caps flattened rows per ORT call.
- Implemented `predict_batch(ModelBatchRequest) -> ModelBatchResult` for `PracticalRIFEOnnxRuntime`.
- Practical-RIFE ONNX real-image CPU evidence from Milestone 3 supports retaining the ONNX path. The initial Milestone 7 implementation preserved batch-request semantics through static-batch-1 chunks because the then-current RIFE dynamo artifact had fixed batch `1`; Milestone 7.5 supersedes that runtime path with a dynamic-batch artifact and fixed-batch artifact rejection.
- Sequential ONNX `predict(FramePairRequest)` remains available for both models.
- PyTorch batch inference behavior was not changed.
- AMT-S ONNX batch support was skipped because AMT-S ONNX is out of Stage 2.5 scope.
- True multi-row Practical-RIFE ONNX batching moved to the Milestone 7.5 follow-up before Milestone 8.
- CUDA-provider ONNX batch validation was deferred because CUDA is unavailable in this environment.

Milestone 7 validation:

- Focused ONNX runtime tests cover EMA fixed 2x and Nx batch reconstruction, EMA sequential-vs-batch compatibility, invalid backend handling, RIFE fixed-batch artifact rejection, RIFE dynamic-batch reconstruction, RIFE scale mismatch validation, and provider/artifact metadata.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_onnx_runtime.py` passed: `20 passed, 19 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` passed: `130 passed, 24 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` passed.
- `git diff --check` passed.
- Accepted EMA artifact smoke: `model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx`, original unsimplified plus `.onnx.data`, CPU provider. Fixed 2x on two `112x112` pairs used one ORT call; Nx factor 4 on two `112x112` pairs used one ORT call for six flattened rows.
- Pre-7.5 Practical-RIFE artifact smoke: `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamo_dynamic_hw_opset18_h384w512.onnx`, original unsimplified plus `.onnx.data`, CPU provider. Fixed 2x on two `128x128` pairs used two ORT calls; Nx factor 4 on two `128x128` pairs used six ORT calls because the artifact batch dimension is static `1`.

### Milestone 7.5 - Dynamic-Batch ONNX Export Follow-Up

Objective: Replace Practical-RIFE static-batch ONNX behavior with a dynamic-batch dynamo artifact before benchmark work.

Implementation status:

- Added shared symbolic batch to the dynamo `dynamic_shapes` export policy for `left`, `right`, and `timestep`, preserving constrained dynamic H/W multiples.
- Exposed ONNX Runtime graph input/output shapes from the backend so model runtimes can inspect fixed versus symbolic batch dimensions.
- Updated Practical-RIFE ONNX defaults to `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx`, default `inference_batch_size=None`, and true multi-row flattened batch execution.
- Practical-RIFE ONNX runtime now rejects fixed-batch artifacts at load time; no static-batch-1 fallback support was added.
- Default dynamo artifact resolution now prefers `dynamic_batch_hw` artifacts when present, while explicit `--onnx-path` remains available.

Milestone 7.5 validation:

- Exported `practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx` plus `.onnx.data` using dynamo/opset 18, external H/W multiple `128`, sample shape `1x3x384x512`, and scale `1.0`.
- Graph I/O: `left[batch,3,128*height_units,128*width_units]`, `right[batch,3,128*height_units,128*width_units]`, `timestep[batch,1,1,1]`, output batch `batch`.
- CPU RIFE fixed-2x synthetic validation ran on `128x128`, `128x256`, and `320x512` and wrote `outputs/onnx_validation/stage2_5_m7_5_rife_dynamic_batch_fixed2x/`. Runtime shape behavior passes; strict synthetic allclose retains the known larger-shape RIFE drift with mean MAE `4.6398597e-05` and max abs `0.0087888837`.
- CPU RIFE batch smoke report `outputs/onnx_validation/stage2_5_m7_5_rife_dynamic_batch_batch_smoke/batch_dynamic_report.json` covers fixed 2x flattened batches `1`, `2`, and `4`, non-square `128x256`, sample-different `320x512`, and Nx factor 4 flattened batches `3` and `6`; each scenario used one ORT call.
- CPU RIFE real-pair validation was rerun against the dynamic-batch artifact under `outputs/onnx_validation/stage2_5_m7_5_rife_dynamic_batch_real_pairs/`; all three `raw_data/pair_test` pairs pass strict `1e-3` allclose with MAE `6.4718541e-07`, max abs `9.4920397e-05`, PSNR `116.6313`, and SSIM `1.0`.
- The old fixed-batch RIFE dynamo artifact was explicitly load-tested and rejected with a dynamic-batch artifact error.
- EMA constrained-dynamic artifact was re-verified on CPU. `outputs/onnx_validation/stage2_5_m7_5_ema_dynamic_batch_fixed2x/` passed fixed 2x on `64x64`, `112x168`, and `320x512`; `outputs/onnx_validation/stage2_5_m7_5_ema_dynamic_batch_batch_smoke/batch_dynamic_report.json` passed fixed 2x flattened batches `1`, `2`, and `4` plus Nx factor 4 flattened batches `3` and `6`, all in one ORT call and all strict allclose.
- CUDA-provider ONNX batch validation remains deferred because CUDA is unavailable in this environment.

### Milestone 8 - Video Benchmarks and MLflow Logging

Objective: Add small repeatable benchmark workflows and log them to MLflow where available.

Implementation status:

- Reworked `src/video_interpolation/inference_benchmark.py` for video-pipeline runtime benchmarks over the existing `run_video_inference(...)` path.
- Added `benchmark runtime` CLI command in `src/video_interpolation/cli.py`.
- Added `log_benchmark_run(...)` in `src/video_interpolation/mlflow.py`, delegating to the existing bounded MLflow helper behavior.
- Added focused tests in `tests/test_inference_benchmark.py`.
- Benchmark reports write to `outputs/benchmarks/video/` by default and include `benchmark_report.json`, `benchmark_metrics.csv`, and real generated benchmark videos under `videos/<profile>/`.
- Video inference results now expose `VideoInferenceTiming` with decode, preprocessing, model inference, postprocessing, video encode/flush, audio remux, and total timings. Directory-wide inference measurement CSVs include the same breakdown.
- ONNX benchmark runs use lightweight benchmark adapters around `EMAVFIOnnxRuntime` and `PracticalRIFEOnnxRuntime`, so the existing video inference path can call `predict_frame_pair(...)` and `predict_frame_pairs_batch(...)`.

Expected output:

- CLI/API for single-video and directory video-pipeline runtime benchmarks.
- CSV/JSON benchmark summaries.
- Timing breakdowns for decode, preprocessing/tensor conversion, model inference, postprocessing/frame conversion, video encode/flush, audio remux, total time, throughput, batch chunks/model batch requests, and peak PyTorch CUDA VRAM when available.
- MLflow params, aggregate metrics, and report artifacts when enabled. Generated videos are logged only when `--log-output-videos` is set.
- `--disable-mlflow` for local smoke runs.
- No separate video interpolation implementation; benchmarks use the same local inference workflow as production-style smoke commands.

Validation checkpoint:

- Tests for benchmark config validation, directory discovery, output path layout, aggregation, CSV/JSON writing, MLflow-disabled behavior, fake-runtime sequential execution, fake-runtime batched Nx execution, failed-video reporting, and video timing fields passed.
- Focused command passed: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_inference_benchmark.py tests/test_inference.py`.
- Focused lint passed: `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/video_interpolation/inference.py src/video_interpolation/inference_benchmark.py src/video_interpolation/cli.py tests/test_inference_benchmark.py tests/test_inference.py`.

Milestone 8 smoke command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli benchmark runtime \
  --model practical_rife_v4_26 \
  --backend onnx \
  --execution-mode batched \
  --input raw_data/tmp_test/DORA_cut.mp4 \
  --limit-pairs 2 \
  --provider cpu \
  --codec libx264 \
  --output-dir outputs/benchmarks/stage2_5_m8_video_rife_onnx_batch_smoke \
  --disable-mlflow
```

Smoke result:

- output: `outputs/benchmarks/stage2_5_m8_video_rife_onnx_batch_smoke/benchmark_report.json`
- CSV: `outputs/benchmarks/stage2_5_m8_video_rife_onnx_batch_smoke/benchmark_metrics.csv`
- model/backend/mode: Practical-RIFE v4.26 ONNX batched fixed 2x, CPU provider (`CPUExecutionProvider`)
- input: `raw_data/tmp_test/DORA_cut.mp4`, capped to `--limit-pairs 2`
- output video: `outputs/benchmarks/stage2_5_m8_video_rife_onnx_batch_smoke/videos/practical_rife_v4_26_onnx_cpu_batched_fixed_2x_2x_bauto/DORA_cut_repeat00_2x.mp4`
- source frames/pairs/generated/frames written: `3` / `2` / `2` / `5`
- batch chunks/model batch requests: `2` / `2`
- timing: decode `0.11354650s`, preprocessing `0.09657892s`, model `5.62656432s`, postprocessing `0.04529195s`, encode `0.10067545s`, audio remux `0.00034524s`, total `6.38277020s`
- MLflow: disabled for smoke; MLflow server availability was not validated in this environment.

### Milestone 9 - Documentation, Project Map, and Stage 2 Handoff

Objective: Make Stage 2.5 restartable and prepare Stage 2 to resume.

Likely files/modules:

- `docs/stage2_inference_runtime_refactor.md`
- possible new `docs/stage2_5_inference_runtime_stabilization.md`
- `configs/README.md`
- `configs/inference/README.md`
- `configs/models/README.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/docs/exec-plans/active/02_inference_runtime_refactor.execplan.md`
- this ExecPlan

Expected output:

- Human-facing docs for EMA ONNX status, RIFE equivalence, real-image checks, batch inference, batch-size/VRAM guidance, benchmarks, MLflow logging, and fallback behavior.
- Project map updates.
- Stage 2 ExecPlan pointer to completed Stage 2.5 handoff.
- Stage 2.5 outcomes recorded and ready for user acceptance.

Validation checkpoint:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` passed after the documentation closeout: `141 passed, 30 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` passed after the documentation closeout: `All checks passed`.
- `git diff --check` passed after the documentation closeout.
- Smoke commands and deferred CUDA checks are recorded in this ExecPlan and the public Stage 2.5 documentation.

## Validation Strategy

Use proportional validation. Do not run long jobs by default.

Expected automated tests:

- Batch request/result validation.
- Pair-by-timestep flattening and reconstruction.
- Batch output ordering for fixed 2x and Nx.
- Chunk overlap correctness in video inference.
- Existing sequential inference compatibility.
- Real-pair discovery and validation.
- Equivalence metrics and report writing.
- Benchmark summary aggregation.
- MLflow-disabled benchmark behavior.
- OOM fallback or clear OOM error behavior if implemented.

Expected smoke checks:

- EMA ONNX synthetic multi-shape validation for attempted dynamic/constrained-dynamic policy.
- Practical-RIFE synthetic ONNX validation on original and dynamic shapes.
- Practical-RIFE real-image ONNX-vs-PyTorch check over `raw_data/pair_test`.
- EMA real-image ONNX-vs-PyTorch check only if EMA ONNX remains usable.
- PyTorch batch tensor smoke for EMA and RIFE, factors 2, 4, and 8.
- Tiny video batch smoke on `raw_data/tmp_test/DORA_cut.mp4` or one of `001.mp4` through `007.mp4`, with `--limit-pairs` or a tiny chunk count.
- Mini-benchmark smoke with MLflow disabled.
- MLflow benchmark logging smoke only when the MLflow server is available.

Expected final checks:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
```

If unprefixed `uv run ...` still fails due the read-only default uv cache under `/home/lighter_01/.cache/uv`, use `UV_CACHE_DIR=/tmp/uv-cache` and record the fallback.

If CUDA is unavailable, record CUDA-dependent EMA/RIFE/ONNX provider smoke checks as deferred to the user's CUDA environment. CPU synthetic and report-generation tests should still run where possible.

## Expected Artifacts

Expected completion artifacts:

- Updated EMA ONNX investigation evidence and classification.
- Real-image ONNX-vs-PyTorch equivalence reports and sample images under `outputs/onnx_validation/` or a clearly documented output root.
- True model batch API in `src/video_interpolation/inference_runtime/`.
- PyTorch batch inference support for EMA-VFI and Practical-RIFE.
- Video-level chunked batch inference support with sequential fallback.
- Optional ONNX batch support for viable model/backend combinations.
- Benchmark CLI/API and output summaries under `outputs/benchmarks/`.
- MLflow benchmark runs when enabled.
- Updated inference configs or README docs for `inference_batch_size`, execution mode, fallback behavior, and benchmark settings.
- Updated `docs/` for Stage 2.5 workflows.
- Updated `.agent/docs/PROJECT_MAP.md`.
- Updated active Stage 2 ExecPlan pointer after Stage 2.5 completion.
- Completed Stage 2.5 handoff in this ExecPlan.

## Risks, Assumptions, and Recovery

Assumptions:

- Stage 2 runtime API can be extended without breaking existing adapter compatibility wrappers.
- EMA and RIFE can share a batch request/result contract even though padding and model calls differ.
- Real-image pair inputs under `raw_data/pair_test/` are small enough for safe smoke checks.
- MLflow helper reuse is preferable to direct MLflow calls.
- `inference_batch_size=1` can preserve sequential-like behavior while allowing one implementation path.

Risks:

- EMA feature-extractor shape logic may not be exportable as dynamic ONNX without broad upstream refactoring.
- The 56-multiple hypothesis may fail or may work only for a narrow subset of sizes.
- EMA ONNX may remain viable only at export shape; the required recovery is PyTorch-only EMA serving.
- Practical-RIFE CPU and CUDA ONNX provider behavior may diverge.
- Practical-RIFE real-image differences may be visually significant even when MAE is low.
- Batch inference may increase VRAM sharply at high resolution and high interpolation factor.
- OOM recovery can be fragile if implemented too early; clear failure may be safer first.
- Batched video chunking can easily duplicate or drop boundary frames if overlap is wrong.
- MLflow may be unavailable locally during smoke runs.
- Existing generated artifact directories are untracked; implementation must not rely on git to preserve them.

Recovery:

- Keep sequential inference fallback for every model.
- If EMA ONNX cannot be stabilized without broad refactor, stop ONNX work for EMA and document PyTorch-only serving.
- If RIFE ONNX equivalence is not acceptable, keep RIFE PyTorch batch path and mark ONNX deferred.
- If batch OOM occurs, fail with a clear message recommending smaller `inference_batch_size` and record the resolution/model/factor.
- If MLflow is unavailable, rerun benchmarks with `--disable-mlflow` and record that MLflow logging remains unvalidated.
- If CUDA is unavailable, complete CPU-safe tests and leave exact CUDA smoke commands in the handoff.

## Progress

- 2026-06-01: Created this Stage 2.5 ExecPlan from the Stage 2.5 plan, Stage 2 active handoff, Stage 1 completed handoff, docs, current source/config/tests, model repository inspection, ONNX artifacts, validation reports, and available real-image/video smoke inputs. No source code, configs, model repositories, dependencies, or model weights were changed. Implementation has not started.
- 2026-06-01: Started and completed Milestone 1 baseline audit only. Inspected current ONNX artifacts, ONNX validation JSON/CSV outputs, pair-test images, temporary smoke videos, Stage 2 docs/ExecPlan handoff notes, EMA feature-extractor risk areas, current ONNX validation code, and existing batch terminology in `batch_inference.py` / adapter APIs. Added `docs/stage2_5_inference_runtime_stabilization.md` and updated `.agent/docs/PROJECT_MAP.md`. No source code, configs, model repositories, dependencies, model weights, ONNX artifacts, or validation reports were changed. Code validation was skipped because this milestone changed only documentation and the ExecPlan.
- 2026-06-01: Started and completed Milestone 2 EMA ONNX dynamic/constrained-dynamic investigation. Reproduced the existing legacy EMA ONNX dynamic-H/W failure, tested the 56-multiple external-padding hypothesis, tried a larger legacy trace shape, tried the modern `torch.onnx.export(..., dynamo=True, dynamic_shapes=...)` route, and applied narrow EMA cache/export-safety patches in `model_repos/EMA-VFI/model/feature_extractor.py` and `model_repos/EMA-VFI/model/warplayer.py`. Classification: EMA ONNX currently works only for export-size/static-like inputs and is not suitable for dynamic serving; EMA serving should remain PyTorch-only unless a later broad upstream refactor is approved. Milestone 3 was not started.
- 2026-06-01: Completed the explicit Task 2.5 broader EMA refactor trial requested in `.agent/tasks/TASK_2_5.md`. Refactored EMA transformer window reversal, frame-pair swapping, export-time padding/depadding, shift-mask construction, coordinate grids, and warp grids to support the modern dynamo exporter. Added selectable `legacy`/`dynamo` ONNX exporter support, constrained `--dynamic-hw-multiple` export shapes, custom `--artifact-stem`, EMA validation `--divisor`, and richer ONNX validation reports. Exported one bounded constrained-dynamic EMA artifact at opset 18 with symbolic `112*height_units` and `112*width_units`; validation passed on original `64x64`, `112x168`, and `320x512` inputs with external divisor `112`. Classification is now `2. EMA constrained-dynamic ONNX works with documented external padding constraints`; divisor `32` still fails, so this is not fully dynamic.
- 2026-06-01: Prepared the repository for Milestone 3 by promoting the accepted EMA constrained-dynamic ONNX artifact into `model_exports/onnx/ema_vfi_small/`, removing obsolete EMA Milestone 2 and Task 2.5 export directories plus EMA ONNX validation output directories, updating EMA ONNX runtime defaults to the promoted artifact with divisor `112`, and preserving Practical-RIFE artifacts/outputs for Milestone 3 real-image validation.
- 2026-06-01: Completed a pre-Milestone-3 Practical-RIFE dynamo export alignment. Made dynamo/opset 18/no-simplify the default ONNX export mode, kept legacy export and simplification available explicitly, added RIFE `--exporter`, `--dynamic-hw-multiple`, and `--artifact-stem`, patched project-owned RIFE warp grids for symbolic export, exported `practical_rife_v4_26_dynamo_dynamic_hw_opset18_h384w512.onnx` plus `.onnx.data`, and validated that one artifact runs at `128x128`, `128x256`, and `320x512`. Strict allclose still fails on the larger shapes, so Practical-RIFE equivalence remains a Milestone 3 evidence item.
- 2026-06-01: Started and completed Milestone 3 real-image ONNX-vs-PyTorch equivalence. Added real pair discovery/loading for `raw_data/pair_test/*/frame1.png` and `frame2.png`, real-pair validation commands `ema validate-onnx-real` and `rife validate-onnx-real`, per-pair visual artifacts, PSNR/SSIM metrics, and richer provenance fields. CPU real-pair checks passed for EMA constrained-dynamic divisor `112` and Practical-RIFE v4.26 dynamo artifacts on pairs `001`, `002`, and `003`. Batch inference, benchmarks, and BentoML work were not started.
- 2026-06-01: Started and completed Milestone 4 model batch API contract. Added BCHW-only `ModelBatchRequest`, pair-major `FlattenedFramePairBatch`/`PairTimestepIndex`, and `ModelBatchResult` reconstruction as `outputs[pair_index][timestep_index]`. Added focused tests for invalid batch inputs, batch-size agreement, factor/timestep validation, pair×timestep flattened ordering, reconstruction order, and fixed 2x compatibility. No model runtime, adapter, local-video, ONNX-batch, benchmark, or BentoML execution wiring was started.
- 2026-06-01: Started and completed Milestone 5 PyTorch batch inference. Wired `ModelBatchRequest` into EMA-VFI and Practical-RIFE PyTorch runtimes, added adapter `predict_frame_pairs_batch(...)`, changed EMA/RIFE adapter `predict_batch(...)` to use fixed-2x true model batching, added optional `inference_batch_size` memory caps, and preserved sequential pair APIs. Fake/lightweight tests and real CPU tiny smokes passed. No video chunking, ONNX batch inference, benchmarks, MLflow benchmark logging, or BentoML work was started.
- 2026-06-01: Started and completed Milestone 6 video-level chunked batch inference. Added local video execution-mode and flattened-row batch-size config, chunked overlapping video batching through `ModelBatchRequest`, sequential fallback, CLI flags, measurement CSV fields, focused fake-runtime tests, and documentation updates. Full pytest, ruff, diff check, and short CPU real-runtime video smokes for EMA/RIFE 2x and 4x passed. ONNX batch inference, benchmark workflows, MLflow benchmark logging, and BentoML work were not started.
- 2026-06-01: Started and completed Milestone 7 ONNX batch inference where viable. Added `predict_batch(ModelBatchRequest)` to EMA and Practical-RIFE ONNX runtimes. EMA uses true multi-row ORT batching with the constrained-dynamic dynamo artifact. Practical-RIFE initially preserved batch-request semantics through static-batch-1 chunks because the then-current RIFE artifact had fixed batch `1`; this was superseded by Milestone 7.5. Focused tests and CPU tensor-level artifact smokes passed. AMT-S ONNX batch, true multi-row RIFE ONNX batch, CUDA-provider validation, benchmarks, MLflow benchmark logging, and BentoML work were not started.
- 2026-06-01: Started and completed Milestone 7.5 dynamic-batch ONNX follow-up. Added shared symbolic batch to dynamo export shapes, exported `practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx` plus `.onnx.data`, updated Practical-RIFE ONNX runtime defaults to the new artifact, removed static-batch-1 chunking, and made fixed-batch RIFE artifacts fail fast. CPU RIFE runtime batch smokes proved one ORT call for flattened batches `1`, `2`, `3`, `4`, and `6`; EMA constrained-dynamic ONNX batch verification passed fixed 2x and Nx smokes with strict allclose. CUDA-provider validation, benchmarks, MLflow benchmark logging, and BentoML work were not started.
- 2026-06-01: Reworked Milestone 8 after user review from tensor/image-pair benchmarking to video-pipeline benchmarking. `benchmark runtime` now defaults to `raw_data/tmp_test/DORA_cut.mp4`, supports `--input` or `--input-dir` plus `--limit-videos`, calls the existing `run_video_inference(...)` path for every measured run, writes real benchmark videos, and records decode/preprocess/model/postprocess/encode/audio-remux/total timing in reports and measurement CSVs. No separate video interpolation loop was introduced.
- 2026-06-01: Completed Milestone 9 documentation and Stage 2 handoff. Updated this ExecPlan outcomes section, the paused Stage 2 ExecPlan pointer, `docs/stage2_5_inference_runtime_stabilization.md`, and `.agent/docs/PROJECT_MAP.md`. Recorded final EMA/RIFE ONNX decisions, PyTorch-vs-ONNX serving recommendations, batch and benchmark status, unresolved blockers, and the exact Stage 2 Milestone 8 restart path. No runtime, ONNX, batch, benchmark, BentoML, backend, model-weight, or dataset behavior was changed.

## Milestone 3 Real-Image Equivalence

Milestone 3 adds real-image ONNX-vs-PyTorch checks alongside the existing synthetic checks. The implementation extends `src/video_interpolation/inference_runtime/onnx_validation.py` with:

- `discover_real_image_pairs(...)` for validating pair directories under `raw_data/pair_test/`;
- `run_real_pair_equivalence_check(...)` for running PyTorch and ONNX predictors on the same loaded RGB tensors;
- report fields for `input_group`, `pair_id`, source frame paths, padded shapes, output shapes, artifact kind, padding policy, graph I/O, MAE, max absolute error, MSE, PSNR, SSIM, and visual output paths;
- visual artifacts under `<output>/<model>/<provider>/<pair_id>/left.png`, `right.png`, `pytorch_generated.png`, `onnx_generated.png`, and `absdiff.png`.

CLI commands:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli rife validate-onnx-real \
  --torch-device cpu \
  --provider cpu \
  --output-dir outputs/onnx_validation/stage2_5_m3_real_pairs

UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli ema validate-onnx-real \
  --torch-device cpu \
  --provider cpu \
  --divisor 112 \
  --output-dir outputs/onnx_validation/stage2_5_m3_real_pairs
```

Both commands default to evidence-gathering mode: they write reports even when `allclose` fails and exit non-zero only for setup/reporting failures. `--fail-on-mismatch` restores strict validation exit behavior. `--limit-pairs` bounds smoke runs.

Real-pair output root:

```text
outputs/onnx_validation/stage2_5_m3_real_pairs/
  ema_vfi_small/cpu/
  practical_rife_v4_26/cpu/
```

CPU results on `raw_data/pair_test/001`, `002`, and `003`:

| Model | Artifact / settings | Original shape | Padded shape | Result | Aggregate metrics |
| --- | --- | --- | --- | --- | --- |
| `ema_vfi_small` | `ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx`, external divisor `112`, CPU provider | `3x320x512` | `1x3x336x560` | all 3 pairs pass strict `1e-3` allclose | MAE `3.6674443e-07`, max abs `6.8575144e-05`, MSE `1.2800672e-12`, PSNR `125.4239`, SSIM `1.0` |
| `practical_rife_v4_26` | `practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx`, scale `1.0`, external divisor `128`, CPU provider | `3x320x512` | `1x3x384x512` | all 3 pairs pass strict `1e-3` allclose | MAE `6.4718541e-07`, max abs `9.4920397e-05`, MSE `4.2880933e-12`, PSNR `116.6313`, SSIM `1.0` |

Classification after Milestone 3:

- EMA ONNX remains constrained-dynamic, accepted for continued investigation with documented external divisor `112` padding. Real-pair CPU evidence is strong; no static bucket fallback was added.
- Practical-RIFE v4.26 ONNX real-pair CPU evidence is strong with the new dynamo artifact. The earlier synthetic larger-shape mismatch remains documented, but realistic `512x320` pair fixtures pass strict allclose after project-owned padding/unpadding.
- CUDA-provider real-pair validation is deferred because this environment does not provide a usable CUDA PyTorch/ORT runtime.

## Milestone 1 Baseline Audit

Milestone 1 froze current status before EMA ONNX export changes, Practical-RIFE real-image equivalence work, or true model batching work. The audit used only short metadata/inspection commands. `uv run` was used only to import `onnx` for graph metadata, with `UV_CACHE_DIR=/tmp/uv-cache` because the system Python did not have `onnx` installed.

### ONNX artifact inventory

Current artifacts under `model_exports/onnx/`:

| Model target | Artifact | Kind | Size | ONNX graph metadata | Dynamic/static status | Current role |
| --- | --- | --- | --- | --- | --- | --- |
| `ema_vfi_small` | `model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx` plus `.onnx.data` | original with external data | `1.8 MiB` ONNX + `56 MiB` external data | IR/opset from Task 2.5 dynamo export; inputs/outputs use symbolic `112*height_units` and `112*width_units` | Constrained-dynamic H/W with external divisor `112`; divisor `32` fails | Current accepted EMA artifact for Milestone 3 |
| `practical_rife_v4_26` | `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx` plus `.onnx.data` | original with external data | `935,650` bytes ONNX + `23,019,520` bytes external data | Inputs use symbolic `batch`, `128*height_units`, and `128*width_units`; output batch is symbolic | Dynamic batch plus constrained dynamic H/W with external divisor `128`; fixed 2x and Nx flattened batches run in one ORT call | Current Practical-RIFE artifact after Milestone 7.5 |
| `practical_rife_v4_26` | `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamic_hw_opset17.onnx` | original | `22,875,748` bytes | IR 8, opset `ai.onnx:17`, producer `pytorch 2.11.0`; inputs `left[batch,3,height,width]`, `right[batch,3,height,width]`, `timestep[batch,1,1,1]`; output `intermediate_frame[batch,Addintermediate_frame_dim_1,height,width]`; `1577` nodes | Can execute at multiple H/W based on current reports | Fallback/debug artifact; CPU report suggests it may have better equivalence than current simplified CUDA reports, but controlled rerun is needed |
| `practical_rife_v4_26` | `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamic_hw_opset17.simplified.onnx` | simplified | `22,848,842` bytes | IR 8, opset `ai.onnx:17`, producer `pytorch 2.11.0`; inputs `left[batch,3,height,width]`, `right[batch,3,height,width]`, `timestep[unk__123,1,1,1]`; output `intermediate_frame[batch,3,height,width]`; `768` nodes | Can execute at multiple H/W based on current reports | Legacy fallback/comparison artifact |

The old EMA legacy opset 17 artifacts and Stage 2.5 investigation export directories were removed after the Task 2.5 trial because they are superseded by the promoted constrained-dynamic artifact.

Milestone 1 did not run ONNX checker or simplifier again. Stage 2 handoff recorded that both original exports passed ONNX checker and simplification succeeded.

### ONNX validation report inventory

Historical report inventory from Milestone 1. The EMA rows below refer to outputs that were later cleaned before Milestone 3; the Practical-RIFE rows still describe retained comparison artifacts.

| Report | Model / artifact | Provider request and observed session | Shapes and status | Metrics | Input source / visuals |
| --- | --- | --- | --- | --- | --- |
| `outputs/onnx_validation/ema_vfi_small/equivalence_report.json` | EMA simplified artifact | requested `CUDAExecutionProvider`; session `CUDAExecutionProvider, CPUExecutionProvider` | `3x32x32`, fixed 2x, `t=0.5`, pass | MAE `1.5586907102260739e-04`, max abs `9.738802909851074e-04`, MSE `4.3931226656468425e-08`, allclose `True` | Synthetic tensor input only; no sample images |
| `outputs/onnx_validation/ema_dynamic_check/ema_vfi_small/equivalence_report.json` | EMA simplified artifact | requested `CUDAExecutionProvider`; session `CUDAExecutionProvider, CPUExecutionProvider` for passing shape | `3x32x32` passes; `3x64x64` fails | `32x32`: MAE `1.5586899826303124e-04`, max abs `9.738802909851074e-04`, MSE `4.393109875877599e-08`, allclose `True`; `64x64`: no metrics | Synthetic tensor input only; no sample images. Failure is ONNX Runtime `LayerNormalization` shape mismatch: `X.shape={2,64,98}`, `scale.shape={128}`, `bias.shape={128}`, `axis=2` |
| `outputs/onnx_validation/practical_rife_v4_26/equivalence_report.json` | Practical-RIFE simplified artifact | requested `CUDAExecutionProvider`; session `CUDAExecutionProvider, CPUExecutionProvider` | `3x128x128` and `3x128x256` both execute but fail strict allclose | `128x128`: MAE `0.002102541271597147`, max abs `0.16346415877342224`, MSE `5.0705635658232495e-05`; `128x256`: MAE `0.0031502440106123686`, max abs `0.3320295810699463`, MSE `0.00011819062638096511`; both allclose `False` | Synthetic tensor input only; PyTorch/ONNX/absdiff PNGs exist under `outputs/onnx_validation/sample_outputs/practical_rife_v4_26/` |
| `outputs/onnx_validation/rife_dynamic_check/practical_rife_v4_26/equivalence_report.json` | Practical-RIFE simplified artifact | requested `CUDAExecutionProvider`; session `CUDAExecutionProvider, CPUExecutionProvider` | `3x128x128` and `3x128x256` both execute but fail strict allclose | `128x128`: MAE `0.001979422988370061`, max abs `0.1629994511604309`, MSE `5.483828135766089e-05`; `128x256`: MAE `0.006543174851685762`, max abs `0.4248119592666626`, MSE `0.0003755491634365171`; both allclose `False` | Synthetic tensor input only; PyTorch/ONNX/absdiff PNGs exist under `outputs/onnx_validation/rife_dynamic_check/sample_outputs/practical_rife_v4_26/` |
| `outputs/onnx_validation/rife_256x128_check/practical_rife_v4_26/equivalence_report.json` | Practical-RIFE simplified artifact | requested/session `CPUExecutionProvider` | `3x256x128` executes but fails strict allclose | MAE `2.7893154765479267e-05`, max abs `0.0029497742652893066`, MSE `7.516998579149003e-09`, allclose `False` | Synthetic tensor input only; no sample images recorded |
| `outputs/onnx_validation/rife_dynamic_original_check/practical_rife_v4_26/equivalence_report.json` | Practical-RIFE original artifact | requested/session `CPUExecutionProvider` | `3x128x128` passes; `3x128x256` executes but fails strict allclose | `128x128`: MAE `8.475995855405927e-06`, max abs `9.101033210754395e-04`, MSE `6.186128831409121e-10`, allclose `True`; `128x256`: MAE `5.533109651878476e-05`, max abs `0.0037825703620910645`, MSE `2.5121643432157725e-08`, allclose `False` | Synthetic tensor input only; no sample images recorded |

Limitations in existing reports:

- Reports only store `metadata.seed`; they do not store the command, timestamp, Git revision, PyTorch device, export sample shape, artifact simplification status as a first-class field, RIFE scale, or whether the input was synthetic or real.
- Source input type is inferred as synthetic because `src/video_interpolation/inference_runtime/onnx_validation.py` currently creates random tensors and has no real-image path.
- Reports include requested providers and session providers, but provider-specific comparisons are not normalized across CPU and CUDA. Current report files mix Stage 2 CPU handoff facts with later CUDA-provider outputs.
- Reports do not include PSNR, SSIM, or LPIPS. Later real-image equivalence reports should add at least PSNR/SSIM when comparing image-like outputs.

### EMA-VFI current ONNX status

- The current accepted artifact is `model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx` plus `.onnx.data`.
- The graph exposes symbolic constrained H/W as `112*height_units` and `112*width_units`.
- EMA ONNX requires project-owned external divisor `112` padding and unpadding. Divisor `32` still fails.
- The old original/simplified legacy opset 17 EMA artifacts were removed before Milestone 3 because they were only superficially dynamic and failed changed H/W.
- Milestone 3 should use the accepted constrained-dynamic artifact for real-image equivalence and should not reopen the EMA dynamic-shape refactor.

### Practical-RIFE current ONNX status

- Export artifacts exist for original and simplified dynamic-H/W opset 17 graphs. Stage 2 handoff says ONNX checker and simplification succeeded.
- Practical-RIFE ONNX Runtime can execute on more than one input size in current evidence: `128x128`, `128x256`, and `256x128` all have report records.
- CPU original-artifact evidence is comparatively close: `128x128` passes strict allclose, while `128x256` fails only because max abs error is above `1e-3` despite low MAE.
- Current simplified CUDA-provider report files show much larger differences, with MAE around `0.002` to `0.0065` and max abs error around `0.16` to `0.42`.
- Existing reports are not sufficient to accept or reject ONNX serving because CPU/CUDA provider, original/simplified artifact, and shape coverage are inconsistent. Milestone 3 should run controlled synthetic and real-image checks with explicit provenance before deciding whether Practical-RIFE ONNX is acceptable.
- The code still prefers simplified artifacts by default when resolving ONNX paths; this is a default, not an acceptance decision.

### `raw_data/pair_test` inventory

Available real-image pair directories:

- `raw_data/pair_test/001/frame1.png`, `raw_data/pair_test/001/frame2.png`
- `raw_data/pair_test/002/frame1.png`, `raw_data/pair_test/002/frame2.png`
- `raw_data/pair_test/003/frame1.png`, `raw_data/pair_test/003/frame2.png`

All six files are `512 x 320`, 8-bit RGB, non-interlaced PNGs. The structure is suitable for Milestone 3 real-image ONNX-vs-PyTorch equivalence checks because every pair directory has the expected `frame1.png` and `frame2.png` names and consistent RGB dimensions.

### `raw_data/tmp_test` inventory

Available short videos:

| File | Size | Resolution | FPS | Duration | Frames | Future use |
| --- | --- | --- | --- | --- | --- | --- |
| `raw_data/tmp_test/001.mp4` | `40,924,233` bytes | `1920x1080` | `60/1` | `0.516667s` | `31` | Larger 1080p smoke; use only with pair limits or tiny batch settings |
| `raw_data/tmp_test/002.mp4` | `36,925,198` bytes | `1920x1080` | `60/1` | `0.516667s` | `31` | Larger 1080p smoke |
| `raw_data/tmp_test/003.mp4` | `24,438,571` bytes | `1920x1080` | `60/1` | `0.516667s` | `31` | Larger 1080p smoke |
| `raw_data/tmp_test/004.mp4` | `36,841,942` bytes | `1920x1080` | `60/1` | `0.516667s` | `31` | Larger 1080p smoke |
| `raw_data/tmp_test/005.mp4` | `33,949,149` bytes | `1920x1080` | `60/1` | `0.516667s` | `31` | Larger 1080p smoke |
| `raw_data/tmp_test/006.mp4` | `27,092,044` bytes | `1920x1080` | `60/1` | `0.516667s` | `31` | Larger 1080p smoke |
| `raw_data/tmp_test/007.mp4` | `22,421,227` bytes | `1920x1080` | `60/1` | `0.516667s` | `31` | Smaller of the numbered 60 FPS clips |
| `raw_data/tmp_test/DORA_cut.mp4` | `442,703` bytes | `1920x1080` | `24000/1001` | `1.753s` | `43` | Preferred future video smoke/benchmark candidate because it is by far the smallest file |

Milestone 1 did not run any video inference. Future benchmarks should prefer `DORA_cut.mp4` first and keep explicit pair/frame limits for 1080p videos.

### Batch terminology baseline

- `src/video_interpolation/batch_inference.py` currently means directory-wide or all-target workflow orchestration: video discovery, target selection, output path/run-name construction, and measurement CSV writing.
- It does not perform true model batch inference.
- `ModelAdapter.predict_batch(...)` is currently a sequential list wrapper over `predict_pair(...)`.
- Current EMA and Practical-RIFE runtimes loop over timesteps and append `prediction[0]`, so an NCHW `FramePairRequest` does not yet produce a true batch output contract.
- Stage 2.5 should add a separately named true model-batch API rather than overloading directory-wide `batch_inference.py` terminology.

### Milestone 1 validation status

- No production source code, tests, configs, or Python modules were modified.
- No long jobs, full-dataset jobs, ONNX reruns, or video inference jobs were run.
- Full `pytest` and `ruff` validation was skipped as permitted for documentation/ExecPlan-only audit work.
- Read-only metadata commands used: `find`, `file`, `ffprobe`, `rg`, and `UV_CACHE_DIR=/tmp/uv-cache uv run python` for ONNX graph metadata.
- Next step is Milestone 2: investigate EMA dynamic or constrained-dynamic ONNX behavior, starting from the captured LayerNormalization failure and the 56-multiple padding hypothesis.

## Milestone 2 EMA ONNX Investigation

Milestone 2 investigated whether EMA-VFI ONNX can support true dynamic H/W or a constrained dynamic external-padding policy. No real-image equivalence, batch inference, benchmarks, or BentoML work was started.

### Code changes

Narrow EMA upstream runtime/cache changes:

- `model_repos/EMA-VFI/model/warplayer.py`
  - Added export/compile detection around the warp-grid cache.
  - Eager PyTorch still caches grids for hashable concrete `(device, dtype, batch, H, W)` keys.
  - ONNX/dynamo export and symbolic-shape execution recompute grids instead of using symbolic `SymInt` values as dictionary keys.
- `model_repos/EMA-VFI/model/feature_extractor.py`
  - Added export/compile detection around coordinate and attention-mask caches.
  - `feature_bone.cor` cache now includes dtype and falls back to no-cache for symbolic/unhashable shapes.
  - Forward-time `register_buffer("attn_mask", ...)` and `register_buffer("HW", ...)` were replaced with non-buffer eager attributes outside export/compile mode. During export/compile, the shift mask is recomputed and not stored as module state.

Rationale: the modern `torch.export`/dynamo route initially failed on EMA shape-keyed caches with `TypeError: unhashable type: non-nested SymInt`. The patch removes that concrete blocker while preserving eager PyTorch behavior. It does not attempt a broad rewrite of EMA's shape-dependent window/padding logic.

### Historical legacy ONNX status after reproduction

These EMA legacy paths were Milestone 2 evidence and were removed before Milestone 3 after the Task 2.5 constrained-dynamic artifact was promoted.

Historical preferred legacy artifact:

```text
model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamic_hw_opset17.simplified.onnx
```

Reproduced command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli ema validate-onnx --config configs/models/ema_vfi_small.yaml --provider CPUExecutionProvider --shape 32x32 --shape 64x64 --output-dir outputs/onnx_validation/stage2_5_m2/baseline_default_div32
```

Result:

- `32x32` passes with MAE `3.139333273338707e-07`, max abs error `1.7881393432617188e-06`, allclose `True`.
- `64x64` fails in ONNX Runtime at `/net/feature_bone/block4.0/norm2/LayerNormalization` with `X.shape={2,64,98}`, `scale.shape={128}`, `bias.shape={128}`.

The same `32x32` pass / `64x64` failure remains after the cache patch:

```text
outputs/onnx_validation/stage2_5_m2/post_cache_patch_default_div32/ema_vfi_small/equivalence_report.json
```

### External padding hypothesis

Default runtime divisor `32`, tested shapes intended to probe 56-multiple behavior:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli ema validate-onnx --config configs/models/ema_vfi_small.yaml --provider CPUExecutionProvider --shape 56x56 --shape 112x112 --shape 112x168 --output-dir outputs/onnx_validation/stage2_5_m2/external_56_hypothesis_default_export
```

Result:

- `56x56` is padded to `64x64` by the default EMA padder and fails with the same LayerNormalization class of error.
- `112x112` and `112x168` also fail with ONNX Runtime shape/reshape errors.

Direct divisor `56` test using the same existing artifact:

```text
outputs/onnx_validation/stage2_5_m2/external_56_hypothesis_div56_abs/ema_vfi_small/equivalence_report.json
```

Result:

- `56x56` fails in PyTorch before ONNX with a multiscale feature-size mismatch: `Expected size 16 but got size 14`.
- `112x112` reaches ONNX but fails at `/net/feature_bone/block4.0/Reshape_26`.
- `112x168` fails in PyTorch with a multiscale feature-size mismatch: `Expected size 44 but got size 42`.
- `224x224` reaches ONNX but fails at LayerNormalization with `X.shape={2,784,8}`, `scale.shape={128}`, `bias.shape={128}`.

Conclusion: padding to multiples of `56` is not a viable constrained-dynamic policy for the current EMA ONNX path. Some 56-multiple input sizes are not even valid PyTorch EMA inference sizes with divisor `56`, and valid PyTorch sizes still fail in ONNX.

### Larger legacy trace-shape attempt

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli ema export-onnx --config configs/models/ema_vfi_small.yaml --device cpu --height 112 --width 112 --output-dir model_exports/onnx/stage2_5_m2_export112 --no-simplify
```

Artifact:

```text
model_exports/onnx/stage2_5_m2_export112/ema_vfi_small/ema_vfi_small_dynamic_hw_opset17.onnx
```

Validation with divisor `56`:

```text
outputs/onnx_validation/stage2_5_m2/export112_div56/ema_vfi_small/equivalence_report.json
```

Result:

- `112x112` executes but does not match PyTorch: MAE `0.1021115854382515`, max abs error `0.6074593663215637`.
- `224x224` executes but does not match PyTorch: MAE `0.10435428470373154`, max abs error `0.652391254901886`.

Conclusion: a larger legacy trace shape does not produce a usable dynamic or constrained-dynamic graph.

### Modern dynamo export attempt

Before the cache patch, the modern route failed before ONNX translation:

```text
outputs/onnx_validation/stage2_5_m2/dynamo_export_artifacts/onnx_export_2026-06-01_05-03-59-624272_pt_export.md
```

Blocker:

```text
TypeError: unhashable type: non-nested SymInt
```

The stack pointed to EMA shape-keyed grid caches, including `model/warplayer.py`.

After the cache patch, the modern route succeeded:

```text
model_exports/onnx/stage2_5_m2_dynamo_after_cache_patch/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset17.onnx
model_exports/onnx/stage2_5_m2_dynamo_after_cache_patch/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset17.onnx.data
outputs/onnx_validation/stage2_5_m2/dynamo_after_cache_patch_artifacts/onnx_export_2026-06-01_05-06-35-152382_success.md
```

Important caveat: despite requesting `dynamic_shapes`, the exported ONNX graph has static H/W inputs:

```text
left[batch, 3, 112, 112]
right[batch, 3, 112, 112]
timestep[batch, 1, 1, 1]
intermediate_frame[batch, 3, 112, 112]
```

The exporter also kept opset 18 after version-conversion to opset 17 failed for `Resize`.

Validation:

- With divisor `56`, `112x112` passes against PyTorch: MAE `1.7025370198098244e-06`, max abs error `2.849102020263672e-05`, allclose `True`.
- With divisor `56`, `224x224` is rejected by ONNX Runtime because the graph expects H/W `112`.
- With default divisor `32`, `32x32` and `64x64` are rejected because the graph expects H/W `112`; `112x112` is padded to `128x128` and is also rejected.

Reports:

```text
outputs/onnx_validation/stage2_5_m2/dynamo_div56/ema_vfi_small/equivalence_report.json
outputs/onnx_validation/stage2_5_m2/dynamo_div32/ema_vfi_small/equivalence_report.json
```

Conclusion: the modern export route can produce a usable static-like `112x112` artifact after cache fixes, but it does not solve EMA dynamic H/W or constrained-dynamic serving.

### EMA ONNX classification

EMA ONNX classification after Milestone 2:

```text
ONNX works only for export-size/static-like input and is not suitable for dynamic serving.
EMA ONNX dynamic H/W is blocked/deferred.
EMA serving should remain PyTorch-only for now.
```

Reasons:

- Existing legacy dynamic-axes artifact passes only at its export-like `32x32` shape and fails at changed H/W.
- External padding to likely 56-multiple shapes did not help.
- Re-exporting at `112x112` with the legacy route produced poor PyTorch-vs-ONNX equivalence.
- Modern dynamo export only became possible after cache fixes, but H/W remained static in the ONNX graph.
- A real dynamic solution would require a broader rewrite of EMA's feature extractor/window/coordinate/warp shape logic, including Python `math.ceil`, Python shape `if` branches, dynamic `linspace`/grid creation, `window_partition`, `window_reverse`, and attention-mask construction. That is broader and riskier than the allowed Milestone 2 patch.

### Milestone 2 validation status

Validation and smoke checks run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_ema_adapter.py tests/test_onnx_export.py tests/test_onnx_runtime.py
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
```

Results:

- Focused tests: `30 passed`.
- Full tests: `87 passed`.
- Ruff: `All checks passed`.

EMA PyTorch CPU smoke:

```text
32x32 frames 1 shape (3, 32, 32)
112x112 frames 1 shape (3, 112, 112)
```

The direct CLI smoke with `configs/models/ema_vfi_small.yaml` failed on this machine because that config requests CUDA and CUDA is unavailable. The CPU-configured adapter smoke passed.

Compatibility notes:

- Stage 1 EMA training/fine-tuning code was not modified.
- Adapter training/eval-path unit tests passed.
- EMA PyTorch inference remains usable on CPU.
- No dependencies were added.

### Milestone 2 next step

Superseded by the Task 2.5 addendum below. Milestone 3 may now include EMA real-image ONNX-vs-PyTorch equivalence using the constrained-dynamic divisor-112 artifact, in addition to the originally planned Practical-RIFE checks.

## Task 2.5 EMA Dynamic ONNX Refactor Trial

This addendum implements the broader bounded EMA refactor trial requested in `.agent/tasks/TASK_2_5.md`, after Milestone 2 and before Milestone 3.

### Code changes

- `model_repos/EMA-VFI/model/feature_extractor.py`
  - Replaced `window_reverse(...)` Python `int(...)` batch inference with an optional explicit `batch_size`.
  - Replaced dynamic frame-pair swapping slices with reshape/flip/reshape.
  - Added tensor-derived shift-mask construction with `torch.arange` and `window_partition`.
  - Added an export-only branchless padding/depadding path to avoid Python shape guards during `torch.export`.
  - Replaced dynamic-size `torch.linspace` coordinate grids with `torch.arange` normalization.
  - Replaced exporter-sensitive `-1` channel reshapes with explicit known channel dimensions.
- `model_repos/EMA-VFI/model/warplayer.py`
  - Replaced warp-grid `torch.linspace` construction with `torch.arange` normalization.
- `src/video_interpolation/inference_runtime/onnx_export.py`
  - Added `OnnxExporterKind` with `legacy` default and selectable `dynamo`.
  - Added dynamo `dynamic_shapes` export support.
  - Added optional constrained dynamic H/W shape declarations through `dynamic_hw_multiple`.
- `src/video_interpolation/cli.py`
  - Added `ema export-onnx --exporter`, `--artifact-stem`, and `--dynamic-hw-multiple`.
  - Added `ema validate-onnx --divisor` to validate external padding policies.
- `src/video_interpolation/inference_runtime/onnx_validation.py`
  - Added padded-shape and output-shape fields to per-shape records.
  - Added ONNX input/output dimension metadata to validation report metadata.

Existing legacy export behavior remains the default.

### Rollback and baseline

Rollback branch:

```text
ema-vfi-dynamic-onnx-trial
```

Rollback commit:

```text
df2f732c89405bd2ea654657f4c408168d1c41eb
```

Pre-trial tracked diff snapshot:

```text
outputs/onnx_validation/stage2_5_task_2_5/pre_trial_tracked.diff
```

Baseline PyTorch outputs before refactor:

```text
outputs/onnx_validation/stage2_5_task_2_5/baseline_torch_pre/
```

Final PyTorch identity report after refactor:

```text
outputs/onnx_validation/stage2_5_task_2_5/baseline_torch_post_final/identity_report.json
```

Final identity metrics against the pre-refactor baseline with divisor `112`:

| Original shape | Padded shape | MAE | Max abs |
| --- | --- | --- | --- |
| `64x64` | `1x3x112x112` | `7.345734047703445e-07` | `6.258487701416016e-06` |
| `112x168` | `1x3x112x224` | `1.5290747796825599e-06` | `5.59687614440918e-05` |
| `320x512` | `1x3x336x560` | `2.8164572540845256e-06` | `4.2319297790527344e-05` |

All are within the Task 2.5 acceptance threshold of MAE `<= 1e-5` and max abs `<= 1e-4`.

### ONNX artifact and validation

Accepted artifact:

```text
model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx
model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx.data
```

Export command:

```bash
uv run python -m video_interpolation.cli ema export-onnx --device cpu --output-dir model_exports/onnx --opset-version 18 --height 336 --width 560 --exporter dynamo --dynamic-hw-multiple 112 --artifact-stem ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560 --no-simplify
```

The graph exposes constrained symbolic H/W:

```text
left[batch, 3, 112*height_units, 112*width_units]
right[batch, 3, 112*height_units, 112*width_units]
timestep[batch, 1, 1, 1]
intermediate_frame[batch, 3, 112*height_units, 112*width_units]
```

Accepted divisor-112 validation:

```bash
uv run python -m video_interpolation.cli ema validate-onnx --torch-device cpu --onnx-path model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx --provider cpu --shape 64x64 --shape 112x168 --shape 320x512 --divisor 112 --output-dir outputs/onnx_validation/stage2_5_m3_smoke
```

Historical report:

```text
The Task 2.5 report directory was cleaned before Milestone 3; accepted metrics are preserved in this ExecPlan and `docs/stage2_5_inference_runtime_stabilization.md`.
```

Result:

- all three original shapes passed through one artifact;
- mean MAE `1.0100862e-06`;
- max abs error `4.1246414e-05`;
- unpadded output shapes matched original H/W.

Divisor-32 probe:

```text
The Task 2.5 divisor-32 probe directory was cleaned before Milestone 3.
```

Result: failed in the expected window reshape path because divisor `32` produces padded shapes that are not multiples of `112`.

### Classification

Task 2.5 final classification:

```text
2. EMA constrained-dynamic ONNX works with documented external padding constraints.
```

Serving implication:

- EMA ONNX is viable only when project-owned external padding uses divisor `112`.
- This is not fully dynamic H/W and should not be used with the existing default divisor `32`.
- No static bucket fallback was introduced.
- The existing PyTorch EMA path remains valid and keeps its current default divisor `32`.

## Surprises & Discoveries

- `raw_data/pair_test/` already has three real image pairs with `frame1.png` and `frame2.png`; all inspected images are `512 x 320` RGB PNGs.
- `raw_data/tmp_test/` contains multiple MP4 smoke inputs, including `DORA_cut.mp4`.
- `batch_inference.py` is directory-wide orchestration only. It does not perform model-level batching.
- `FramePairRequest` allows NCHW input, but current EMA/RIFE runtimes append only `prediction[0]` and therefore do not yet expose true batch outputs.
- Current ONNX validation artifacts are not all consistent with the Stage 2 handoff prose. Some current JSON files record `CUDAExecutionProvider` and larger Practical-RIFE differences, while the Stage 2 handoff described CPU-provider results with smaller errors. Stage 2.5 must rerun controlled checks and record provenance.
- `pyproject.toml` already includes `bentoml` and `onnxscript`, in addition to ONNX and ONNX Runtime packages. Stage 2.5 should not add dependencies unless a concrete missing package appears.
- `git status --short` shows existing generated/untracked `model_exports/` and `outputs/onnx_validation/`, plus a modified candidate validation report. These should be treated as user/workspace artifacts and left intact unless a Stage 2.5 workflow intentionally writes new outputs.
- The old EMA Milestone 2 and Task 2.5 generated export/output directories were removed before Milestone 3; Practical-RIFE ONNX artifacts and existing Practical-RIFE validation outputs were kept for comparison.
- Pre-Milestone-3 Practical-RIFE dynamo validation wrote `outputs/onnx_validation/stage2_5_pre_m3_rife_dynamo/practical_rife_v4_26/equivalence_report.json`: `128x128` passed strict allclose; `128x256` and `320x512` ran but failed strict allclose with max abs `0.0038332939` and `0.0087888837`.
- Milestone 1 report audit found that existing ONNX validation reports lack first-class provenance for command, timestamp, Git revision, PyTorch device, export sample shape, input source type, RIFE scale, and artifact simplification status. Later report schema updates should add these fields before accepting ONNX serving decisions.
- EMA's existing PyTorch inference path does not accept every `56`-multiple input when the external padder divisor is changed to `56`; `56x56` and `112x168` failed before ONNX with multiscale feature-size mismatches. The old divisor `32` behavior remains the safe PyTorch default.
- `torch.onnx.export(..., dynamo=True, dynamic_shapes=...)` initially failed on EMA's symbolic shape cache keys. After cache fixes, export succeeded but the graph still baked H/W `112x112`, so export flags alone are not a dynamic-H/W fix.
- The dynamo exporter produced an opset 18 ONNX model despite a requested opset 17 because conversion back to 17 failed for `Resize`. This is acceptable as an investigation artifact only, not a preferred serving artifact.
- The broader Task 2.5 refactor found that unconstrained dynamo dynamic shapes still specialize H/W to the export sample. Declaring H/W as derived dimensions (`112 * height_units`, `112 * width_units`) and exporting from the largest planned padded sample `336x560` produced a bounded constrained-dynamic ONNX artifact that ORT accepts at smaller divisor-112 padded shapes.
- EMA ONNX still fails with the existing divisor `32`; the accepted ONNX path requires explicit external divisor `112`.

## Decision Log

- 2026-06-01: Create a separate Stage 2.5 ExecPlan rather than continuing directly in the Stage 2 ExecPlan. Rationale: the Stage 2.5 plan is authoritative for this intermediate stabilization stage, and Stage 2 must remain paused before BentoML.
- 2026-06-01: Treat EMA ONNX as an investigation with allowed blocked/deferred outcome. Rationale: Stage 2.5 explicitly forbids static bucket fallback and permits EMA PyTorch-only serving if dynamic/constrained-dynamic ONNX is not usable.
- 2026-06-01: Plan real-image equivalence as additive to synthetic equivalence. Rationale: synthetic checks stress operators/shapes, while real pairs answer practical visual-service risk.
- 2026-06-01: Keep `batch_inference.py` as directory-wide orchestration and add separately named true model batch APIs. Rationale: current naming is overloaded; a separate request/result contract avoids confusing workflow batching with tensor/model batching.
- 2026-06-01: Prefer pair-by-timestep flattening for Nx batch inference. Rationale: the Stage 2.5 plan requires this over a Python loop over timesteps where feasible.
- 2026-06-01: Keep sequential fallback as a first-class path. Rationale: low-VRAM environments, debugging, and regression checks need behavior that is close to the current implementation.
- 2026-06-01: Integrate benchmarks with the existing MLflow helper layer. Rationale: Stage 1 already centralized MLflow setup, bounded connectivity behavior, params, metrics, and artifact logging.
- 2026-06-01: Rewrite Milestone 8 benchmarks from synthetic/image-pair runtime calls to full video-pipeline benchmarks over `run_video_inference(...)`. Rationale: backend and batch differences are more meaningful on short videos, and the benchmark must expose decode, preprocessing, model, postprocessing, encode/flush, audio remux, and total pipeline bottlenecks while still reusing the existing Stage 2.5 inference workflow rather than adding a separate interpolation loop.
- 2026-06-01: Classify EMA ONNX dynamic H/W as blocked/deferred after Milestone 2. Rationale: legacy dynamic-axes export fails at changed H/W, 56-multiple external padding fails, larger legacy trace-shape export is not equivalent, and modern dynamo export remains static-H/W. Superseded by the explicit Task 2.5 broader refactor trial below.
- 2026-06-01: Keep EMA serving on PyTorch backend after Milestone 2. Rationale: PyTorch EMA inference remains valid with the existing divisor `32`, while ONNX was only static-like at that point. Superseded for constrained ONNX only by the Task 2.5 divisor-112 artifact; PyTorch remains the default-safe EMA path.
- 2026-06-01: Accept EMA constrained-dynamic ONNX for the bounded divisor-112 policy after Task 2.5. Rationale: one dynamo opset-18 artifact with symbolic `112*height_units` and `112*width_units` ran in ORT on `64x64`, `112x168`, and `320x512` original inputs through project-owned padding/unpadding, with low PyTorch-vs-ONNX error. Divisor `32` still fails, so the result is constrained-dynamic, not fully dynamic.
- 2026-06-01: Promote the accepted EMA constrained-dynamic artifact to `model_exports/onnx/ema_vfi_small/` and clean obsolete EMA investigation artifacts before Milestone 3. Rationale: Milestone 3 should validate current serving candidates, not old Milestone 2 or Task 2.5 trial directories; Practical-RIFE artifacts remain because Milestone 3 still needs their real-image equivalence evidence.
- 2026-06-01: Make dynamo/opset 18/no-simplify the default ONNX export path for active models while retaining explicit legacy export and legacy simplification. Rationale: EMA's accepted artifact uses dynamo plus external data, and Practical-RIFE now has the same artifact format; legacy artifacts remain useful for comparison but should not drive the default pre-Milestone-3 path.
- 2026-06-01: Represent true model batch inference with a new BCHW-only `ModelBatchRequest` rather than expanding `FramePairRequest`. Rationale: the existing sequential request supports CHW and NCHW compatibility, while the batch contract needs explicit multiple-pair semantics and must not be confused with directory-wide `batch_inference.py`.
- 2026-06-01: Store Nx batch execution order as pair-major/timestep-minor and reconstruct outputs as `outputs[pair_index][timestep_index]`. Rationale: this directly matches the Stage 2.5 pair-by-timestep flattening requirement and gives video chunking/benchmark milestones deterministic ordering metadata.
- 2026-06-01: Default PyTorch model-batch runtimes to one model call for the full flattened pair×timestep batch, with optional `inference_batch_size` chunking. Rationale: this proves true batch execution for Milestone 5 while keeping a direct memory-control escape hatch for later video and benchmark work.
- 2026-06-01: Keep ONNX batch execution deferred after PyTorch batch runtime wiring. Rationale: Milestone 5 scope is PyTorch only; ONNX batch support depends on the provider/artifact acceptance work scheduled for a later milestone.
- 2026-06-01: Force EMA `fast_tta` batch execution to effective batch size `1`. Rationale: upstream EMA fast-TTA inference indexes only two augmented predictions and would be incorrect for larger flattened model batches.
- 2026-06-01: Recommend Practical-RIFE v4.26 PyTorch as the first Stage 2 BentoML compatibility-proof target. Rationale: it is the lowest-risk proof path after Stage 2.5 because it avoids EMA's ONNX divisor-112 constraint, avoids ONNX external-data/provider variables for the first service smoke, uses the active default model, and exercises the same request/result and batch-capable adapter surface that future service code should call.
- 2026-06-01: Keep ONNX serving as an explicit secondary proof path for Stage 2 Milestone 8. Rationale: Practical-RIFE dynamic-batch ONNX is viable on CPU with real-pair evidence, but `.onnx.data` adjacency, provider choice, and CUDA validation remain deployment concerns; EMA ONNX is acceptable only when the service enforces external divisor `112` padding/unpadding.

## Outcomes & Handoff

Stage 2.5 Milestones 1 through 9 are complete and accepted. The ExecPlan has moved to `completed/`, and Stage 2 has resumed from Milestone 8.

Implemented and validated:

- ONNX export defaults now use dynamo/opset 18/no simplification for active models, with explicit legacy export and legacy-only simplification still available.
- EMA-VFI-small has a promoted constrained-dynamic ONNX artifact at `model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx` plus adjacent `.onnx.data`.
- Practical-RIFE v4.26 has a promoted dynamic-batch constrained-dynamic ONNX artifact at `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx` plus adjacent `.onnx.data`.
- Real-image ONNX-vs-PyTorch validation exists for `raw_data/pair_test/001..003` with per-pair visuals and CSV/JSON reports.
- `ModelBatchRequest` / `ModelBatchResult` define true pair-major fixed 2x and Nx model batching, with EMA/RIFE PyTorch and viable ONNX runtimes supporting batched execution.
- Local video inference supports default chunked `batched` execution and explicit `sequential` fallback. `inference_batch_size` caps flattened model rows for VRAM control.
- `benchmark runtime` now benchmarks the complete local video pipeline using real videos, defaulting to `raw_data/tmp_test/DORA_cut.mp4`, and reports decode, preprocessing, model, postprocessing, encode/flush, audio remux, total time, throughput, counts, and generated output video paths.

Final ONNX decisions:

- EMA ONNX is accepted only as constrained-dynamic with project-owned external divisor `112` padding/unpadding. It is not fully dynamic; divisor `32` fails; no static artifact buckets were introduced. EMA PyTorch remains the general serving path when the service cannot enforce divisor `112`.
- Practical-RIFE ONNX is accepted for CPU real-pair and dynamic-batch runtime use with the new dynamic-batch artifact. The old fixed-batch dynamo artifact is obsolete and unsupported. Larger synthetic shapes still show strict-allclose drift, so real-image and provider-specific evidence should stay visible in serving decisions.
- Both accepted dynamo artifacts may require an adjacent `.onnx.data` file; service packaging must keep each external-data file next to its `.onnx` graph.
- CUDA-provider ONNX acceptance remains deferred. CPU evidence must not be treated as CUDA evidence.

PyTorch versus ONNX serving recommendation:

- Use PyTorch first for the minimal BentoML compatibility proof, specifically Practical-RIFE v4.26 through the existing adapter/runtime API. This proves the service can import, instantiate, load weights, validate request parameters, and call the request/result path without ONNX external-data/provider variables.
- Use Practical-RIFE dynamic-batch ONNX as the secondary BentoML proof if an ORT service path is included. Keep `.onnx.data` adjacent, request CPU provider first, and fail fast on fixed-batch artifacts.
- Use EMA PyTorch for general EMA serving. Use EMA ONNX only in a route that explicitly applies external divisor `112` padding/unpadding and documents the resolution constraint.

Batch inference status:

- Fixed 2x batching uses one flattened row per neighboring frame pair at timestep `0.5`.
- Nx batching uses pair-major `pair_index x timestep_index` flattening, then reconstructs `outputs[pair_index][timestep_index]`.
- `predict_batch([(left, right), ...])` on EMA/RIFE adapters is fixed-2x and now uses true model batching. Nx batching uses `predict_frame_pairs_batch(ModelBatchRequest)` or the video batched path.
- Sequential APIs remain available for debugging, low-memory operation, and models without batch support.

Benchmark status:

- Milestone 8 smoke used Practical-RIFE v4.26 ONNX, CPU provider, batched fixed 2x, `raw_data/tmp_test/DORA_cut.mp4`, and `--limit-pairs 2`.
- Reports: `outputs/benchmarks/stage2_5_m8_video_rife_onnx_batch_smoke/benchmark_report.json` and `outputs/benchmarks/stage2_5_m8_video_rife_onnx_batch_smoke/benchmark_metrics.csv`.
- Output video: `outputs/benchmarks/stage2_5_m8_video_rife_onnx_batch_smoke/videos/practical_rife_v4_26_onnx_cpu_batched_fixed_2x_2x_bauto/DORA_cut_repeat00_2x.mp4`.
- Counts: source frames `3`, pairs `2`, generated frames `2`, frames written `5`, batch chunks `2`, model batch requests `2`.
- Timings: decode `0.11354650s`, preprocessing `0.09657892s`, model `5.62656432s`, postprocessing `0.04529195s`, encode `0.10067545s`, audio remux `0.00034524s`, total `6.38277020s`.
- MLflow server logging was not validated in this environment; the smoke used `--disable-mlflow`. The benchmark code keeps MLflow enabled by default and logs generated videos only with `--log-output-videos`.

Documentation and project-map updates:

- `docs/stage2_5_inference_runtime_stabilization.md` now records the final Stage 2.5 state, ONNX decisions, batch/video benchmark behavior, limitations, and Stage 2 resume recommendation.
- `.agent/docs/PROJECT_MAP.md` now points to the Stage 2.5 closeout/handoff role and current artifact/report locations.
- `.agent/docs/exec-plans/active/02_inference_runtime_refactor.execplan.md` now states that Stage 2 was paused for Stage 2.5, Stage 2.5 is complete, and Stage 2 should resume from Milestone 8 using these decisions.

Unresolved blockers and deferred work:

- CUDA-provider validation remains deferred for EMA/RIFE ONNX and benchmark profiles.
- MLflow server connectivity/logging was not smoke-tested; report generation works with `--disable-mlflow`.
- Practical-RIFE ONNX still has known synthetic larger-shape strict-allclose drift, despite passing current real-pair fixtures.
- EMA ONNX remains constrained to divisor `112`; fully dynamic divisor-32 behavior is not available.
- AMT-S runtime refactor/ONNX/batch support remains out of scope.
- No BentoML compatibility proof, production service, FastAPI, Celery/Redis, PostgreSQL, MinIO orchestration, frontend, monitoring, or deployment work was implemented in Stage 2.5.

Closeout validation:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` passed after the Milestone 9 documentation closeout: `141 passed, 30 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` passed after the Milestone 9 documentation closeout: `All checks passed`.
- `git diff --check` passed after the Milestone 9 documentation closeout.

Stage 2 resume note:

Stage 2 resumed in `.agent/docs/exec-plans/active/02_inference_runtime_refactor.execplan.md` at Milestone 8. The accepted recommendation remains Practical-RIFE v4.26 PyTorch first, with Practical-RIFE dynamic-batch ONNX as a separate alternate proof path using the accepted artifact and adjacent `.onnx.data`.
