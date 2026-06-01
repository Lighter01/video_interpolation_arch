# Title and Metadata

- Stage: Stage 2.5 - ONNX Stabilization, Real-Image Equivalence, and Batched Inference
- Status: Active ExecPlan - Milestone 2 complete, Milestone 3 not started
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

The stage must produce a clear runtime path for PyTorch batch inference, expanded ONNX equivalence evidence on synthetic and real images, mini-benchmarks with MLflow logging, user-facing documentation, and a handoff pointer back to Stage 2.

This is not a backend stage. It must not implement BentoML proof work, production BentoML services, FastAPI, Celery, Redis, PostgreSQL application schema, MinIO upload/output orchestration, frontend, monitoring, retraining triggers, AMT-S refactor, or training/fine-tuning changes.

## Source Documents and Authority

Read before creating this ExecPlan:

- `.agent/TASK.md`
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

The current user task and the Stage 2.5 stage plan control this stage. The active Stage 2 ExecPlan controls prior implementation facts and must remain paused until Stage 2.5 is complete.

## Context and Current Repository State

Repository root: `/home/lighter_01/projects/itmo/ai_architecture/video_interpolation`.

Stage 1 is complete for handoff. Stage 2 has completed Milestones 1 through 7 and is paused before Milestone 8, the minimal BentoML compatibility proof.

Current runtime implementation:

- `src/video_interpolation/inference_runtime/api.py` defines `FramePairRequest`, `FramePairResult`, `InferenceMode`, `RuntimeBackendKind`, mode/factor validation, and timestep generation for fixed 2x and arbitrary Nx.
- `FramePairRequest` accepts CHW or NCHW tensors, but the current model runtimes treat the result as one logical pair result and append `prediction[0]`; there is not yet a public true batch result contract.
- `src/video_interpolation/inference_runtime/ema.py` has `EMAVFIPyTorchRuntime` and `EMAVFIOnnxRuntime`. Both loop over request timesteps in Python and return one unbatched intermediate frame per timestep.
- `src/video_interpolation/inference_runtime/rife.py` has `PracticalRIFEPyTorchRuntime` and `PracticalRIFEOnnxRuntime`. Both loop over request timesteps in Python. RIFE ONNX requires request scale to match the scale baked into the artifact.
- `src/video_interpolation/inference_runtime/onnx_export.py` exports neural-core-only wrappers: `left`, `right`, `timestep` to one generated frame. It currently uses legacy `torch.onnx.export(..., dynamo=False)` with dynamic axes for dynamic H/W mode.
- `src/video_interpolation/inference_runtime/onnx_validation.py` compares synthetic tensor pairs only. It reports MAE, max absolute error, MSE, and `torch.allclose`, and writes PyTorch/ONNX/diff images when allclose fails.
- `src/video_interpolation/inference.py` decodes and encodes videos and calls `_predict_video_pair(...)` once per neighboring frame pair. This path is sequential at the model-call level.
- `src/video_interpolation/batch_inference.py` discovers videos, resolves target names, builds output paths and run names, and writes measurement CSVs. It is directory-wide workflow orchestration, not model batch inference.
- `ModelAdapter.predict_batch(...)` in `src/video_interpolation/adapters/base.py` is explicitly a sequential default wrapper over `predict_pair(...)`.

Current model/config state:

- `configs/models/ema_vfi_small.yaml` uses `EMA-VFI/ours_small_t.pkl` as the Stage 2 inference checkpoint and records `EMA-VFI/ours_small.pkl` as the training checkpoint.
- `configs/models/practical_rife_v4_26.yaml` is the active Practical-RIFE default. `configs/models/practical_rife_v4_25.yaml` remains available as an alternative.
- EMA and Practical-RIFE configs declare `fixed_2x` and `arbitrary_nx` support with factor bounds `2..8`.
- `pyproject.toml` already lists `onnx`, `onnxruntime`, `onnxruntime-gpu`, `onnx-simplifier`, `bentoml`, and `onnxscript`; Stage 2.5 should not add dependencies unless a specific gap is discovered.

Current ONNX artifacts and validation outputs:

- `model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamic_hw_opset17.onnx`
- `model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamic_hw_opset17.simplified.onnx`
- `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamic_hw_opset17.onnx`
- `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamic_hw_opset17.simplified.onnx`
- ONNX validation reports exist under `outputs/onnx_validation/`.
- EMA validation at the original export-like size can pass strict tolerance, but changed H/W fails with a LayerNormalization shape error in the EMA feature extractor.
- Practical-RIFE validation reports show provider-sensitive differences. CPU original-artifact reports show low MAE with max absolute error slightly above `1e-3` at dynamic shapes. Some current JSON reports under `outputs/onnx_validation/` show CUDA provider runs with larger max differences. Stage 2.5 must reconcile report provenance by rerunning controlled validations and recording provider, artifact, shape, and command metadata.

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
- Add mini-benchmarks comparing backend, execution mode, interpolation mode, factor, and batch size where applicable.
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

Add a small benchmark workflow, likely in a new module such as `src/video_interpolation/inference_benchmark.py` or `src/video_interpolation/benchmarks.py`, with CLI wiring in `src/video_interpolation/cli.py`.

Benchmark targets:

- synthetic tensor pairs for pure runtime microbenchmarks;
- real image pairs from `raw_data/pair_test`;
- very short clips from `raw_data/tmp_test` for video-level benchmarks.

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

### Milestone 8 - Mini-Benchmarks and MLflow Logging

Objective: Add small repeatable benchmark workflows and log them to MLflow where available.

Likely files/modules:

- new benchmark module under `src/video_interpolation/`
- `src/video_interpolation/cli.py`
- `src/video_interpolation/mlflow.py`
- `tests/`
- output directory such as `outputs/benchmarks/`

Expected output:

- CLI/API for synthetic, real-pair, and tiny-video benchmarks.
- CSV/JSON benchmark summaries.
- Timing breakdowns.
- MLflow params, metrics, and artifacts when enabled.
- `--disable-mlflow` for local smoke runs.

Validation checkpoint:

- Tests for benchmark aggregation and MLflow-disabled behavior.
- Short benchmark smoke on synthetic inputs and/or `raw_data/pair_test`.

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

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests`
- Smoke commands and deferred CUDA checks recorded.

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
- Benchmark CLI/API and output summaries, likely under `outputs/benchmarks/`.
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

## Milestone 1 Baseline Audit

Milestone 1 froze current status before EMA ONNX export changes, Practical-RIFE real-image equivalence work, or true model batching work. The audit used only short metadata/inspection commands. `uv run` was used only to import `onnx` for graph metadata, with `UV_CACHE_DIR=/tmp/uv-cache` because the system Python did not have `onnx` installed.

### ONNX artifact inventory

Current artifacts under `model_exports/onnx/`:

| Model target | Artifact | Kind | Size | ONNX graph metadata | Dynamic/static status | Current role |
| --- | --- | --- | --- | --- | --- | --- |
| `ema_vfi_small` | `model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamic_hw_opset17.onnx` | original | `58,858,828` bytes | IR 8, opset `ai.onnx:17`, producer `pytorch 2.11.0`; inputs `left[batch,3,height,width]`, `right[batch,3,height,width]`, `timestep[batch,1,1,1]`; output `intermediate_frame[batch,3,height,width]`; `6122` nodes | Dynamic axes are present, but runtime evidence shows changed H/W is not truly safe | Fallback/debug artifact; resolver prefers simplified when present |
| `ema_vfi_small` | `model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamic_hw_opset17.simplified.onnx` | simplified | `58,534,582` bytes | IR 8, opset `ai.onnx:17`, producer `pytorch 2.11.0`; same nominal dynamic input/output axes; `2311` nodes | Dynamic axes are present, but runtime evidence shows changed H/W is not truly safe | Current default/preferred artifact by `resolve_preferred_onnx_artifact_path(..., prefer_simplified=True)` |
| `practical_rife_v4_26` | `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamic_hw_opset17.onnx` | original | `22,875,748` bytes | IR 8, opset `ai.onnx:17`, producer `pytorch 2.11.0`; inputs `left[batch,3,height,width]`, `right[batch,3,height,width]`, `timestep[batch,1,1,1]`; output `intermediate_frame[batch,Addintermediate_frame_dim_1,height,width]`; `1577` nodes | Can execute at multiple H/W based on current reports | Fallback/debug artifact; CPU report suggests it may have better equivalence than current simplified CUDA reports, but controlled rerun is needed |
| `practical_rife_v4_26` | `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamic_hw_opset17.simplified.onnx` | simplified | `22,848,842` bytes | IR 8, opset `ai.onnx:17`, producer `pytorch 2.11.0`; inputs `left[batch,3,height,width]`, `right[batch,3,height,width]`, `timestep[unk__123,1,1,1]`; output `intermediate_frame[batch,3,height,width]`; `768` nodes | Can execute at multiple H/W based on current reports | Current default/preferred artifact by resolver, but not yet accepted for serving equivalence |

Milestone 1 did not run ONNX checker or simplifier again. Stage 2 handoff recorded that both original exports passed ONNX checker and simplification succeeded.

### ONNX validation report inventory

Current reports under `outputs/onnx_validation/`:

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

- Export artifacts exist for original and simplified dynamic-H/W opset 17 graphs. Stage 2 handoff says ONNX checker and simplification succeeded.
- The exported graphs expose symbolic `batch`, `height`, and `width` axes, but current ONNX Runtime evidence shows the graph is only superficially dynamic.
- EMA runs at the original export-like `32x32` shape and passes strict tolerance in current reports.
- EMA fails at changed `64x64` H/W with an ONNX Runtime `LayerNormalization` shape mismatch in `/net/feature_bone/block4.0/norm2/LayerNormalization`.
- The failure is consistent with EMA feature-extractor Python shape logic and traced constants rather than a missing ONNX artifact.
- Known suspicious source areas remain in `model_repos/EMA-VFI/model/feature_extractor.py`: `pad_if_needed`, `depad_if_needed`, `window_partition`, `window_reverse`, `self.HW`, `attn_mask`, `feature_bone.cor`, `math.ceil`, `int(...)`, `.item()`, and shape-based `if` logic.
- Milestone 2 must investigate dynamic or constrained-dynamic behavior, including the Stage 2.5 plan's 56-multiple padding hypothesis. Milestone 1 made no EMA fixes.

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

### Legacy ONNX status after reproduction

Current preferred legacy artifact:

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

Milestone 3 should start real-image ONNX-vs-PyTorch equivalence for Practical-RIFE, and should treat EMA ONNX as deferred/PyTorch-only unless the user explicitly approves a broader EMA upstream dynamic-shape rewrite.

## Surprises & Discoveries

- `raw_data/pair_test/` already has three real image pairs with `frame1.png` and `frame2.png`; all inspected images are `512 x 320` RGB PNGs.
- `raw_data/tmp_test/` contains multiple MP4 smoke inputs, including `DORA_cut.mp4`.
- `batch_inference.py` is directory-wide orchestration only. It does not perform model-level batching.
- `FramePairRequest` allows NCHW input, but current EMA/RIFE runtimes append only `prediction[0]` and therefore do not yet expose true batch outputs.
- Current ONNX validation artifacts are not all consistent with the Stage 2 handoff prose. Some current JSON files record `CUDAExecutionProvider` and larger Practical-RIFE differences, while the Stage 2 handoff described CPU-provider results with smaller errors. Stage 2.5 must rerun controlled checks and record provenance.
- `pyproject.toml` already includes `bentoml` and `onnxscript`, in addition to ONNX and ONNX Runtime packages. Stage 2.5 should not add dependencies unless a concrete missing package appears.
- `git status --short` shows existing generated/untracked `model_exports/` and `outputs/onnx_validation/`, plus a modified candidate validation report. These should be treated as user/workspace artifacts and left intact unless a Stage 2.5 workflow intentionally writes new outputs.
- Milestone 1 report audit found that existing ONNX validation reports lack first-class provenance for command, timestamp, Git revision, PyTorch device, export sample shape, input source type, RIFE scale, and artifact simplification status. Later report schema updates should add these fields before accepting ONNX serving decisions.
- EMA's existing PyTorch inference path does not accept every `56`-multiple input when the external padder divisor is changed to `56`; `56x56` and `112x168` failed before ONNX with multiscale feature-size mismatches. The old divisor `32` behavior remains the safe PyTorch default.
- `torch.onnx.export(..., dynamo=True, dynamic_shapes=...)` initially failed on EMA's symbolic shape cache keys. After cache fixes, export succeeded but the graph still baked H/W `112x112`, so export flags alone are not a dynamic-H/W fix.
- The dynamo exporter produced an opset 18 ONNX model despite a requested opset 17 because conversion back to 17 failed for `Resize`. This is acceptable as an investigation artifact only, not a preferred serving artifact.

## Decision Log

- 2026-06-01: Create a separate Stage 2.5 ExecPlan rather than continuing directly in the Stage 2 ExecPlan. Rationale: the Stage 2.5 plan is authoritative for this intermediate stabilization stage, and Stage 2 must remain paused before BentoML.
- 2026-06-01: Treat EMA ONNX as an investigation with allowed blocked/deferred outcome. Rationale: Stage 2.5 explicitly forbids static bucket fallback and permits EMA PyTorch-only serving if dynamic/constrained-dynamic ONNX is not usable.
- 2026-06-01: Plan real-image equivalence as additive to synthetic equivalence. Rationale: synthetic checks stress operators/shapes, while real pairs answer practical visual-service risk.
- 2026-06-01: Keep `batch_inference.py` as directory-wide orchestration and add separately named true model batch APIs. Rationale: current naming is overloaded; a separate request/result contract avoids confusing workflow batching with tensor/model batching.
- 2026-06-01: Prefer pair-by-timestep flattening for Nx batch inference. Rationale: the Stage 2.5 plan requires this over a Python loop over timesteps where feasible.
- 2026-06-01: Keep sequential fallback as a first-class path. Rationale: low-VRAM environments, debugging, and regression checks need behavior that is close to the current implementation.
- 2026-06-01: Integrate benchmarks with the existing MLflow helper layer. Rationale: Stage 1 already centralized MLflow setup, bounded connectivity behavior, params, metrics, and artifact logging.
- 2026-06-01: Classify EMA ONNX dynamic H/W as blocked/deferred after Milestone 2. Rationale: legacy dynamic-axes export fails at changed H/W, 56-multiple external padding fails, larger legacy trace-shape export is not equivalent, and modern dynamo export remains static-H/W. A true fix would require a broad upstream feature-extractor/window/grid rewrite outside the intended narrow milestone.
- 2026-06-01: Keep EMA serving on PyTorch backend for now. Rationale: PyTorch EMA inference remains valid with the existing divisor `32`, while ONNX is only static-like and unsuitable for arbitrary user video resolutions.

## Outcomes & Handoff

Not yet complete. This section must be updated at Stage 2.5 closeout with:

- what was implemented;
- what was validated;
- which ONNX paths are usable, blocked, or deferred;
- benchmark results and artifact locations;
- documentation updates;
- remaining risks;
- exact next steps for resuming Stage 2 Milestone 8.

Until this section is completed and accepted, Stage 2 should remain paused before the BentoML compatibility proof.
