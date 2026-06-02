# Stage 2.5 Inference Runtime Stabilization

Stage 2.5 stabilized the inference runtime before Stage 2 resumed with the BentoML compatibility proof. It resolved or classified the three issues that blocked a useful serving proof:

- EMA-VFI ONNX is accepted only as constrained-dynamic with project-owned external divisor `112` padding/unpadding.
- Practical-RIFE v4.26 ONNX has a dynamic-batch dynamo artifact for CPU ORT model-batch calls; CUDA-provider evidence is still deferred.
- Local video inference supports chunked model-batch execution for EMA-VFI and Practical-RIFE, with explicit sequential fallback still available.

Stage 2.5 is complete and accepted. Stage 2 has resumed from Milestone 8 using this handoff.

## Handoff Recommendation

Recommended first BentoML proof:

- Start with Practical-RIFE v4.26 PyTorch through the existing adapter/runtime API.
- Use a small tensor or image payload first, fixed 2x mode, and the same structured request/result path used by local inference.
- Keep the proof minimal: import, instantiate/load, validate request parameters, call inference, and return or inspect the generated tensor.

Secondary ONNX proof, if needed:

- Use Practical-RIFE v4.26 dynamic-batch ONNX on CPU first.
- Use `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx`.
- Keep `practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx.data` adjacent to the graph file.

EMA serving guidance:

- Use EMA PyTorch for general serving.
- Use EMA ONNX only when the service route can enforce external divisor `112` padding/unpadding and can package the adjacent `.onnx.data` file.

Known handoff limitations:

- CUDA-provider ONNX validation remains deferred.
- MLflow server benchmark logging was not smoke-tested; benchmark report generation works with `--disable-mlflow`.
- Practical-RIFE ONNX larger synthetic shapes still have strict-allclose drift, despite current real-pair fixtures passing.
- No BentoML, production service, queue, storage, database, frontend, monitoring, or deployment code was implemented inside Stage 2.5 itself.

Stage 2 Milestone 8 now implements the Practical-RIFE serving-readiness facade and minimal BentoML examples described in `docs/stage2_inference_runtime_refactor.md`.

## Current Baseline

Current ONNX exports live under `model_exports/onnx/`:

- `model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx`
- `model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx.data`
- `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx`
- `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx.data`

Practical-RIFE legacy opset 17 original/simplified artifacts remain loadable for comparison through explicit legacy artifact-resolution flags or `--onnx-path`, but the default export and default artifact resolution path is now dynamo.

ONNX validation outputs live under `outputs/onnx_validation/`. Synthetic checks use `ema validate-onnx` and `rife validate-onnx`; real image-pair checks use `ema validate-onnx-real` and `rife validate-onnx-real`. New Milestone 3 real-pair reports include the input group, command, torch device, artifact kind, graph I/O, pair ids, source frame paths, padded shapes, output shapes, MAE, max absolute error, MSE, PSNR, SSIM, and visual artifact paths.

Runtime benchmark outputs live under `outputs/benchmarks/`. Milestone 8 uses `benchmark runtime` for full local video-pipeline benchmarks. The command runs `run_video_inference(...)`, writes real output videos plus `benchmark_report.json` and `benchmark_metrics.csv`, and records decode, preprocessing, model inference, postprocessing, encode/flush, optional audio remux, sampled quality evaluation when enabled, and total timings. MLflow logging is enabled by default and can be disabled with `--disable-mlflow` for local smoke runs.

Current EMA-VFI evidence:

- The legacy simplified ONNX artifact runs at the export-like `32x32` shape.
- A changed `64x64` shape fails in ONNX Runtime with a `LayerNormalization` shape mismatch.
- Milestone 2 tested external 56-multiple padding, larger legacy re-export, and a modern dynamo export route.
- The follow-up Task 2.5 refactor produced a bounded constrained-dynamic EMA ONNX artifact when project-owned external padding uses divisor `112`.
- EMA ONNX is now classified as constrained-dynamic, not fully dynamic. The existing default divisor `32` still fails and should not be used with the constrained ONNX artifact.

Task 2.5 accepted EMA artifact:

```text
model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx
```

The graph exposes:

```text
left[batch, 3, 112*height_units, 112*width_units]
right[batch, 3, 112*height_units, 112*width_units]
timestep[batch, 1, 1, 1]
intermediate_frame[batch, 3, 112*height_units, 112*width_units]
```

Accepted validation command:

```bash
uv run python -m video_interpolation.cli ema validate-onnx --torch-device cpu --onnx-path model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx --provider cpu --shape 64x64 --shape 112x168 --shape 320x512 --divisor 112 --output-dir outputs/onnx_validation/stage2_5_m3_smoke
```

Result: all three shapes passed through one artifact; mean MAE `1.0100862e-06`, max abs error `4.1246414e-05`, and unpadded output shapes matched the original H/W.

The old EMA opset 17 artifact and Stage 2.5 investigation export/output directories were removed before Milestone 3. Future EMA ONNX checks should use the promoted artifact above with external divisor `112` and write fresh reports under a new output directory for the current validation purpose.

Current Practical-RIFE evidence:

- ONNX Runtime can execute more than one H/W shape.
- CPU original-artifact reports are much closer than the current CUDA simplified reports.
- A pre-Milestone-3 dynamo export produced `practical_rife_v4_26_dynamo_dynamic_hw_opset18_h384w512.onnx` plus `.onnx.data` with symbolic `128*height_units` and `128*width_units`, but fixed batch `1`.
- Milestone 7.5 superseded it with `practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx`, whose `left`, `right`, `timestep`, and output batch axes are symbolic.
- The new dynamo artifact runs on `128x128`, `128x256`, and `320x512`; strict synthetic `1e-3` allclose still fails on larger shapes, with mean MAE `4.6398597e-05` and max abs `0.0087888837`.
- Milestone 3 real-image CPU checks pass on all current pair-test fixtures; CUDA-provider checks remain deferred.

## Local Inputs

Real image pairs used by Stage 2.5 equivalence checks live under `raw_data/pair_test/`. The current structure is:

```text
raw_data/pair_test/001/frame1.png
raw_data/pair_test/001/frame2.png
raw_data/pair_test/002/frame1.png
raw_data/pair_test/002/frame2.png
raw_data/pair_test/003/frame1.png
raw_data/pair_test/003/frame2.png
```

All inspected pair-test images are `512 x 320` RGB PNGs.

Short video smoke inputs live under `raw_data/tmp_test/`. `DORA_cut.mp4` is the preferred first smoke/benchmark input because it is much smaller than the numbered 1080p clips.

## Milestone 3 Real-Image Equivalence

Real-image checks are additive to synthetic checks. They load `frame1.png` and `frame2.png` from each pair directory, run the same PyTorch and ONNX runtime paths with the same padding and timestep policy, compare generated intermediate frames, and write:

```text
outputs/onnx_validation/stage2_5_m3_real_pairs/
  <model>/
    <provider>/
      equivalence_report.json
      equivalence_metrics.csv
      <pair_id>/
        left.png
        right.png
        pytorch_generated.png
        onnx_generated.png
        absdiff.png
```

Use these commands for the current CPU-safe smoke checks:

```bash
uv run python -m video_interpolation.cli rife validate-onnx-real \
  --torch-device cpu \
  --provider cpu \
  --output-dir outputs/onnx_validation/stage2_5_m3_real_pairs

uv run python -m video_interpolation.cli ema validate-onnx-real \
  --torch-device cpu \
  --provider cpu \
  --divisor 112 \
  --output-dir outputs/onnx_validation/stage2_5_m3_real_pairs
```

`--limit-pairs` can bound smoke runs. `--fail-on-mismatch` makes the command exit non-zero for failed `allclose`; by default the real-image commands write evidence and return success when setup and report generation succeed.

Metric interpretation:

- MAE and MSE summarize average absolute and squared pixel-value differences on normalized `[0, 1]` tensors.
- Max absolute error highlights the worst single-channel difference.
- PSNR and SSIM are image-similarity metrics computed between PyTorch and ONNX generated outputs; higher PSNR and SSIM closer to `1.0` are better.
- `absdiff.png` is normalized for visual inspection and should not be read as a raw metric image.

Current CPU real-pair status on `001`, `002`, and `003`:

- EMA-VFI constrained-dynamic dynamo artifact with external divisor `112`: all three pairs pass strict `1e-3` allclose. Original inputs are `3x320x512`, padded to `1x3x336x560`, then unpadded back to `3x320x512`. Aggregate MAE is `3.6674443e-07`, max abs error is `6.8575144e-05`, PSNR mean is `125.4239`, and SSIM mean is `1.0`.
- Practical-RIFE v4.26 dynamo dynamic-batch artifact with external divisor `128` and scale `1.0`: all three pairs pass strict `1e-3` allclose in `outputs/onnx_validation/stage2_5_m7_5_rife_dynamic_batch_real_pairs/`. Original inputs are `3x320x512`, padded to `1x3x384x512`, then unpadded back to `3x320x512`. Aggregate MAE is `6.4718541e-07`, max abs error is `9.4920397e-05`, PSNR mean is `116.6313`, and SSIM mean is `1.0`.

CUDA-provider real-pair validation was not run in this environment. Provider-specific CUDA behavior remains deferred to a CUDA-capable machine.

## Batch Terminology

`src/video_interpolation/batch_inference.py` currently means directory-wide inference orchestration: discovering input videos, selecting targets, constructing output paths, and writing measurement CSVs.

It is not true model batch inference. True model batching now has a separate runtime API contract in `src/video_interpolation/inference_runtime/api.py`:

- `ModelBatchRequest` accepts BCHW `left` and `right` tensors and validates matching batch size/shape.
- `flatten_pair_timesteps()` returns pair-major flattened tensors for one later model call. For Nx, row order is pair first and timestep second: `(pair 0, t0)`, `(pair 0, t1)`, ..., `(pair 1, t0)`, ...
- `ModelBatchResult` stores reconstructed outputs as `outputs[pair_index][timestep_index]` and can rebuild that structure from flattened BCHW model outputs.

Milestone 4 added this contract. Milestone 5 wires it into EMA-VFI and Practical-RIFE PyTorch runtimes. Milestone 6 routes local EMA/RIFE video inference through this contract in chunked batched mode. Milestones 7 and 7.5 add ONNX Runtime batch-request support for the viable current ONNX paths.

## Milestone 5 PyTorch Batch Runtime

EMA-VFI and Practical-RIFE PyTorch runtimes now expose `predict_batch(ModelBatchRequest)`.

Batch behavior:

- Fixed 2x uses one flattened row per pair at timestep `0.5`.
- Nx uses pair-major pair×timestep flattening. For `B` frame pairs and interpolation factor `N`, the runtime prepares `B * (N - 1)` rows ordered as all timesteps for pair `0`, then all timesteps for pair `1`, and so on.
- Results are reconstructed as `outputs[pair_index][timestep_index]`.
- `predict_pair`, `predict_intermediate_frames`, `predict_frame_pair`, `predict`, and `__call__` remain available for sequential compatibility.
- Adapter-level `predict_frame_pairs_batch(...)` accepts a `ModelBatchRequest`. Existing `predict_batch([(left, right), ...])` remains fixed 2x and now uses the true model-batch runtime for EMA/RIFE adapters.

Memory control:

- `inference_batch_size` in EMA/RIFE model configs optionally caps flattened rows per PyTorch model call.
- Leaving `inference_batch_size` empty/null runs the full flattened pair×timestep batch at once.
- Request-level `backend_options={"inference_batch_size": N}` can override the config for direct `ModelBatchRequest` use.
- EMA `fast_tta` forces effective batch size `1` for correctness with the upstream fast-TTA implementation.

CPU smoke status in this environment:

- CUDA is unavailable.
- EMA-VFI CPU batch smokes passed on two `32x32` synthetic pairs at factors `2`, `4`, and `8`; each factor used one flattened model call and returned `3x32x32` outputs.
- Practical-RIFE CPU batch smokes passed on two `32x32` synthetic pairs at factors `2`, `4`, and `8`; each factor used one flattened model call and returned `3x32x32` outputs.

## Milestone 7 ONNX Batch Runtime

EMA-VFI and Practical-RIFE ONNX runtimes now expose `predict_batch(ModelBatchRequest)` alongside the existing sequential `predict(FramePairRequest)`.

Implemented paths:

- EMA-VFI constrained-dynamic ONNX: implemented as true multi-row ONNX Runtime batching. The accepted EMA graph has symbolic batch plus symbolic `112*height_units` and `112*width_units`, so fixed 2x and Nx flattened rows can run in one ORT call unless `inference_batch_size` caps them.
- Practical-RIFE v4.26 dynamo ONNX: implemented as true multi-row ONNX Runtime batching with the Milestone 7.5 dynamic-batch artifact. The runtime rejects fixed-batch artifacts instead of falling back to static-batch-1 chunks.

Skipped/deferred:

- The old Practical-RIFE fixed-batch dynamo artifact is obsolete for runtime use and fails fast at load time.
- AMT-S ONNX batch support is out of scope for Stage 2.5.
- CUDA-provider ONNX batch validation remains deferred in this environment because CUDA is unavailable.

Both ONNX runtimes preserve pair-major pair×timestep flattening:

- fixed 2x: one flattened row per pair at timestep `0.5`;
- Nx: `pair_count * (interpolation_factor - 1)` rows ordered by pair first, then timestep;
- result reconstruction uses `ModelBatchResult.outputs[pair_index][timestep_index]`.

Tensor-level CPU smoke status on accepted artifacts:

- EMA artifact `model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx`, original unsimplified with external data, CPU provider: fixed 2x with two `112x112` pairs used one ORT call; Nx factor 4 with two pairs used one ORT call for six flattened rows.
- Practical-RIFE artifact `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx`, original unsimplified with external data, CPU provider: fixed 2x flattened batches `1`, `2`, and `4` each used one ORT call; Nx factor 4 with one pair used one ORT call for three rows; Nx factor 4 with two pairs used one ORT call for six rows. Synthetic drift is recorded in `outputs/onnx_validation/stage2_5_m7_5_rife_dynamic_batch_batch_smoke/batch_dynamic_report.json`.

## Milestone 6 Video-Level Chunked Batch Inference

Local video inference for EMA-VFI and Practical-RIFE now has two execution modes:

- `batched`: default. The video loop groups neighboring source-frame pairs into chunks and calls adapter `predict_frame_pairs_batch(ModelBatchRequest)` when available.
- `sequential`: explicit fallback. The video loop calls one `FramePairRequest` per neighboring pair, preserving the older behavior.

Adapters that do not expose `predict_frame_pairs_batch(...)`, such as current baselines and AMT-S, automatically run through `sequential_fallback` when a config or CLI requests `batched`.

Chunking behavior:

- The first source frame is written once before pair processing starts.
- Each batch chunk contains consecutive neighboring pairs.
- The final source frame of one chunk becomes the carried first source frame of the next chunk, so a sequence `[0, 1, 2, 3, 4]` with two pairs per chunk is processed as `[0, 1, 2]` then `[2, 3, 4]`.
- Outputs for each pair are written immediately in pair order: generated frames in timestep order, then the next original source frame.
- Fixed 2x writes one generated frame per pair. Nx writes `N - 1` generated frames per pair.
- Output FPS is always `input_fps * interpolation_factor`.

`inference_batch_size` controls flattened model rows, matching the Milestone 5 runtime contract. For fixed 2x, one source pair uses one row. For Nx, one source pair uses `interpolation_factor - 1` rows. A 4x run with `--inference-batch-size 6` therefore processes up to two source pairs per video chunk and passes the same flattened-row cap to the model runtime through `ModelBatchRequest.backend_options`.

Safe smoke examples:

```bash
uv run python -m video_interpolation.cli rife infer-video \
  --config configs/inference/practical_rife_v4_26_2x.yaml \
  --input raw_data/tmp_test/DORA_cut.mp4 \
  --output /tmp/rife_batched_2x.mp4 \
  --mode fixed_2x \
  --interpolation-factor 2 \
  --execution-mode batched \
  --inference-batch-size 2 \
  --codec libx264 \
  --limit-pairs 2 \
  --disable-mlflow

uv run python -m video_interpolation.cli ema infer-video \
  --config configs/inference/ema_vfi_small_2x.yaml \
  --input raw_data/tmp_test/DORA_cut.mp4 \
  --output /tmp/ema_batched_4x.mp4 \
  --mode arbitrary_nx \
  --interpolation-factor 4 \
  --execution-mode batched \
  --inference-batch-size 3 \
  --codec libx264 \
  --limit-pairs 1 \
  --disable-mlflow
```

Use `--execution-mode sequential` to compare frame ordering or reduce memory pressure. If CUDA runs out of memory in batched mode, retry with a smaller `--inference-batch-size`, lower resolution input, or sequential mode. No automatic downscaling is performed.

Measurement metadata now records `requested_execution_mode`, actual `execution_mode`, `inference_batch_size`, `batch_chunks_processed`, `model_batch_requests`, interpolation mode/factor, runtime backend, existing timing fields, FPS, pair/frame counts, and audio preservation counts. Directory-wide `batch_inference.py` remains orchestration over many files/targets; it is separate from true model batching and now writes the same execution metadata to each `inference_measurements.csv`.

CPU smoke status in this environment:

- CUDA is unavailable.
- Practical-RIFE v4.26 passed one-pair video smokes on `raw_data/tmp_test/DORA_cut.mp4` for fixed 2x and Nx factor 4 with `execution_mode=batched`.
- EMA-VFI-small passed one-pair video smokes on `raw_data/tmp_test/DORA_cut.mp4` for fixed 2x and Nx factor 4 with `execution_mode=batched` and checkpoint `EMA-VFI/ours_small_t.pkl`.
- Smoke outputs were written under `outputs/inference/stage2_5_m6_smoke/`.

## Milestone 8 Video Benchmarks

Milestone 8 benchmarks the complete local video pipeline through:

```bash
uv run python -m video_interpolation.cli benchmark runtime
```

The default input is `raw_data/tmp_test/DORA_cut.mp4`. A single video can be selected with `--input`; directory mode uses `--input-dir` and optional `--limit-videos`. Long videos can be bounded with `--limit-pairs`.

The benchmark command uses the same Stage 2.5 video inference path as normal local inference:

- OpenCV decoding and tensor preprocessing in `run_video_inference(...)`;
- `FramePairRequest` for `--execution-mode sequential`;
- `ModelBatchRequest` for `--execution-mode batched`;
- EMA-VFI and Practical-RIFE PyTorch adapters for `--backend torch`;
- accepted EMA-VFI and Practical-RIFE ONNX runtime artifacts for `--backend onnx`;
- PyAV/FFmpeg video encoding, flush, and audio remuxing.

**For pytorch backend** `--device cuda/cpu` is supported. So:

- For **torch** backend runs use combination of `--backend torch`, `--device cuda/cpu`
- For **ONNX** backend runs use combination of `--backend onnx`, `--provider cuda/cpu`

It does not implement a separate interpolation loop. It wraps the existing video inference workflow and disables nested inference MLflow logging so one aggregate benchmark run owns the benchmark reports.

For `practical_rife_v4_26`, sampled quality evaluation is enabled by default in benchmark runs. It samples original source triplets, applies a lightweight adjacent-frame SSIM scene-cut rejection filter, writes accepted triplets as `<triplet_output_dir>/<source_video_id>/<triplet_id>/im1.png`, `im2.png`, and `im3.png`, and records aggregate PSNR/SSIM plus timing overhead. Use `--disable-quality-evaluation` to benchmark without this step. EMA benchmark runs keep quality evaluation disabled by default because the current online quality path is Practical-RIFE-only.

Supported runtime combinations:

- EMA-VFI-small PyTorch sequential and batched;
- EMA-VFI-small constrained-dynamic ONNX sequential and batched, using the current divisor `112` artifact;
- Practical-RIFE v4.26 PyTorch sequential and batched;
- Practical-RIFE v4.26 dynamic-batch ONNX sequential and batched;
- fixed 2x and arbitrary Nx factors `2..8`.

Default one-video smoke:

```bash
uv run python -m video_interpolation.cli benchmark runtime \
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

Same profile without quality evaluation overhead:

```bash
uv run python -m video_interpolation.cli benchmark runtime \
  --model practical_rife_v4_26 \
  --backend onnx \
  --execution-mode batched \
  --input raw_data/tmp_test/DORA_cut.mp4 \
  --limit-pairs 2 \
  --provider cpu \
  --codec libx264 \
  --output-dir outputs/benchmarks/stage2_5_m8_video_rife_onnx_batch_no_quality \
  --disable-quality-evaluation \
  --disable-mlflow
```

Directory-mode example:

```bash
uv run python -m video_interpolation.cli benchmark runtime \
  --model practical_rife_v4_26 \
  --backend onnx \
  --execution-mode batched \
  --input-dir raw_data/tmp_test \
  --limit-videos 2 \
  --limit-pairs 2 \
  --provider cpu \
  --codec libx264 \
  --output-dir outputs/benchmarks/stage2_5_m8_video_dir_rife_onnx_batch \
  --disable-mlflow
```

Nx/batch comparison example:

```bash
uv run python -m video_interpolation.cli benchmark runtime \
  --model ema_vfi_small \
  --backend onnx \
  --execution-mode batched \
  --input raw_data/tmp_test/DORA_cut.mp4 \
  --mode arbitrary_nx \
  --interpolation-factor 4 \
  --limit-pairs 2 \
  --inference-batch-size 6 \
  --provider cpu \
  --codec libx264 \
  --output-dir outputs/benchmarks/stage2_5_m8_video_ema_onnx_nx4_batch \
  --disable-mlflow
```

Report fields:

- identity and execution fields: model, backend, execution mode, interpolation mode/factor, requested batch size, input path, relative input path, output video path, and status/error;
- ONNX provenance: provider and artifact path;
- counts: source frames, pairs processed, generated frames, frames written, batch chunks, and model batch requests;
- timing: decode, preprocessing/tensor conversion, model inference, postprocessing/frame conversion, video encode/flush, audio remux, quality evaluation, and total time;
- quality: quality enabled flag, quality PSNR/SSIM means, sampled triplet count, triplet output directory, and quality error when warn-policy evaluation fails;
- throughput: pairs/sec, generated frames/sec, model-only pairs/sec, and model-only generated frames/sec;
- memory: peak PyTorch CUDA VRAM when a CUDA PyTorch benchmark is run in a CUDA-capable environment.

MLflow behavior:

- MLflow logging uses the project helper layer in `src/video_interpolation/mlflow.py`.
- By default the benchmark logs params, aggregate timing metrics, and JSON/CSV reports to experiment `stage2-5-inference-benchmarks`.
- Generated videos are written under `outputs/benchmarks/video/videos/<profile>/...` by default and are logged to MLflow only when `--log-output-videos` is set.
- Use `--disable-mlflow` when the local MLflow server is not running.
- Basic benchmark report generation does not require MLflow.

Known limitations:

- The benchmark command runs one execution profile per invocation. Compare sequential versus batched or torch versus ONNX by running separate commands and comparing CSV/JSON reports.
- CUDA-provider ONNX benchmark acceptance remains deferred until a CUDA-capable environment can run the same commands with `--provider cuda`.
- Practical-RIFE ONNX larger-shape drift remains a validation finding; benchmarks classify pipeline performance, not equivalence quality.

## Closeout

Detailed implementation and validation history remain in the completed Stage 2.5 ExecPlan:

```text
.agent/docs/exec-plans/completed/02_5_inference_runtime_stabilization.execplan.md
```

Milestone 3 real-image ONNX-vs-PyTorch equivalence, Milestone 4 model-batch API contracts, Milestone 5 PyTorch batch runtime execution, Milestone 6 video-level chunked batch inference, Milestone 7 ONNX batch-request support where viable, Milestone 7.5 Practical-RIFE dynamic-batch ONNX export, Milestone 8 video-pipeline benchmarks with optional MLflow logging, and Milestone 9 documentation/handoff are implemented.

Stage 2 resumed at `.agent/docs/exec-plans/active/02_inference_runtime_refactor.execplan.md` Milestone 8 with Practical-RIFE v4.26 PyTorch as the default serving proof and Practical-RIFE ONNX as the alternate proof path.
