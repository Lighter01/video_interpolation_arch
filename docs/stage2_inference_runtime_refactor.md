# Stage 2 Inference Runtime Refactor

Stage 2 separated model inference from local video I/O, training code, and future serving wrappers. It now provides the reusable inference runtime and serving-readiness handoff for backend/service developers.

Current runtime layout:

```text
src/video_interpolation/inference_runtime/
  api.py                    request/result API, model-batch API, and mode/backend validation
  backends/                 shared runtime backend abstractions
  ema.py                    EMA-VFI PyTorch runtime
  rife.py                   Practical-RIFE PyTorch runtime
  onnx_export.py            ONNX neural-core wrappers and export helpers
  onnx_validation.py        ONNX Runtime equivalence metrics and reports
  rife_upstream/            project-owned Practical-RIFE v4.26 runtime source
src/video_interpolation/inference_benchmark.py
  video-pipeline runtime benchmarks over current local inference paths
```

The public adapter methods remain the compatibility boundary for Stage 1 workflows. Existing calls such as `predict_pair`, `predict_batch`, `predict`, and `__call__` continue to work as fixed 2x middle-frame prediction. Internally, EMA-VFI and Practical-RIFE single-pair prediction construct a `FramePairRequest`, call the model-specific runtime, and receive a `FramePairResult`.

## Final Serving Recommendation

Use Practical-RIFE v4.26 through the project serving facade as the Stage 2 production target:

- production target model: `practical_rife_v4_26`;
- default serving backend: PyTorch runtime backend;
- default device: `cuda`;
- alternative backend: ONNX Runtime with `CUDAExecutionProvider`;
- serving execution mode: `sequential`;
- interpolation mode: `arbitrary_nx`;
- service-level `interpolation_factor` range: `2..4`;
- Practical-RIFE `scale` is a runtime/request parameter;
- current batched inference is validated for local runtime and benchmark paths but is not recommended for serving yet;
- EMA-VFI is not the current serving target;
- AMT-S is legacy and out of scope for Stage 2 serving.

## Stage 2.5 Handoff

Stage 2 was paused before the minimal BentoML proof so Stage 2.5 could stabilize ONNX behavior, real-image equivalence, true model batching, video chunking, and video-pipeline benchmarks. Stage 2.5 has been accepted; the completed handoff is in `.agent/docs/exec-plans/completed/02_5_inference_runtime_stabilization.execplan.md`.

The Stage 2.5 decisions that affect serving are:

- Practical-RIFE v4.26 PyTorch remains the recommended serving path because it avoids ONNX provider and external-data packaging variables.
- Practical-RIFE ONNX is available as a secondary path with the dynamic-batch artifact:

```text
model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx
model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx.data
```

- EMA-VFI is available for local/runtime experimentation but is not the current serving target. EMA ONNX is usable only when a caller enforces external divisor `112` padding/unpadding and packages the adjacent `.onnx.data` file.
- Fixed-batch Practical-RIFE dynamo artifacts are obsolete and unsupported. Legacy ONNX artifacts remain loadable only through explicit legacy artifact selection.

## Practical-RIFE Serving Facade

Stage 2 Milestone 8 adds a minimal serving-facing wrapper for Practical-RIFE v4.26 in `src/video_interpolation/serving.py`.

Use `PracticalRIFEVideoInferenceRunner` when a service process should initialize the model/runtime once and reuse it across requests. Use `run_practical_rife_video_inference(...)` for one-off developer smoke calls.

Serving defaults:

- model: `practical_rife_v4_26`;
- backend: `torch`;
- PyTorch device: `cuda`;
- alternative backend: `onnx`;
- ONNX provider: `CUDAExecutionProvider`;
- execution mode: `sequential`;
- interpolation mode: `arbitrary_nx`;
- service-level interpolation factor: integer `2..4`;
- Practical-RIFE scale: `1.0` by default, allowed values `0.25`, `0.5`, `1.0`, `2.0`, and `4.0`;
- first-triplet quality evaluation: enabled by default with warn fail policy;
- default codec for examples: `libx264`.

The service-level factor range is intentionally narrower than the runtime's general `2..8` support. This keeps the compatibility examples conservative while preserving broader local/runtime APIs for non-serving experiments.

Example one-off call using the default PyTorch CUDA serving recommendation:

```python
from video_interpolation.serving import run_practical_rife_video_inference

result = run_practical_rife_video_inference(
    input_path="raw_data/tmp_test/DORA_cut.mp4",
    output_path="/tmp/dora_rife_4x.mp4",
    interpolation_factor=4,
    scale=1.0,
)
```

For a persistent service process, initialize the runner once and reuse it:

```python
from video_interpolation.serving import (
    PracticalRIFEServingConfig,
    PracticalRIFEVideoInferenceRunner,
)

runner = PracticalRIFEVideoInferenceRunner(
    PracticalRIFEServingConfig(
        backend="torch",
        device="cuda",
    )
)

result = runner.run(
    input_path="raw_data/tmp_test/DORA_cut.mp4",
    output_path="/tmp/dora_rife_4x.mp4",
    interpolation_factor=4,
    scale=1.0,
)
```

Serving quality evaluation is implemented below the BentoML layer in the shared inference stack. It evaluates up to 16 first overlapping source triplets by default, resizes quality-only frames so the larger side is at most 360 pixels, and returns aggregate `quality_psnr_mean`, `quality_ssim_mean`, `quality_triplets_written`, `quality_triplet_output_dir`, and `quality_error` through `VideoInferenceResult`. Triplet PNG persistence is disabled by default; pass `write_quality_triplets=True` and `quality_triplet_output_dir=...` when a backend worker wants Vimeo-style triplets written inside a request-specific temporary directory. Pass `enable_quality_evaluation=False` to `runner.run(...)` or `run_practical_rife_video_inference(...)` to disable quality evaluation for a request.

Use ONNX Runtime only when the deployment will package the accepted artifact and the adjacent external-data file:

```python
runner = PracticalRIFEVideoInferenceRunner(
    PracticalRIFEServingConfig(
        backend="onnx",
        provider="CUDAExecutionProvider",
    )
)
```

For ONNX serving, `scale` must match the scale baked into the loaded artifact. The current Practical-RIFE ONNX artifact is exported with scale `1.0`, so a request with `scale=0.5` should use a matching artifact or fail clearly. Keep the adjacent `.onnx.data` file next to the `.onnx` graph when packaging or copying ONNX artifacts.

## BentoML Examples

Milestone 8 adds two developer-facing BentoML compatibility examples outside the package:

```text
examples/bentoml/practical_rife_torch_service/service.py
examples/bentoml/practical_rife_onnx_service/service.py
```

The PyTorch example is the recommended/default serving proof. It constructs `PracticalRIFEVideoInferenceRunner` with `backend="torch"` and `device="cuda"`.

The ONNX example constructs the same serving runner with `backend="onnx"` and `provider="CUDAExecutionProvider"`. It demonstrates the alternate backend only; CUDA-provider validation still depends on a CUDA-capable environment with ONNX Runtime CUDA available.

Both examples accept simple path strings, runtime `interpolation_factor`, runtime `scale`, and `enable_quality_evaluation`. They return a small metadata dictionary including output path, backend, execution mode, pairs processed, frames written, `psnr_mean`, `ssim_mean`, `quality_triplets_written`, `quality_triplet_output_dir`, and `quality_error`. They are compatibility examples only: no upload API, queue, database, object storage, auth, frontend, Docker deployment, or production orchestration is included.

Stage 2 closeout validation status:

- Full tests passed with `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: `156 passed`.
- Lint passed with `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests`.
- Milestone 9 documentation-only rerun used the same fallback cache because the default uv cache path is read-only in the sandbox; `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` reported `156 passed, 59 warnings`, and `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` reported `All checks passed`.
- Focused serving tests validate config defaults, service-level factor/scale checks, fake-video facade execution, persistent runner reuse, and BentoML example imports/defaults.
- CUDA smokes are deferred in this environment because `torch.cuda.is_available()` returned `False`. ONNX Runtime lists `CUDAExecutionProvider`, but CUDA-provider serving behavior still needs a CUDA-capable smoke run before production claims.

Backend/service developers should build the production layer around `src/video_interpolation/serving.py`, not around model internals. The remaining service work is outside Stage 2: request/upload API, object storage, output lifecycle, job records, queue/worker orchestration, BentoML model-store or artifact packaging policy, Docker/deployment wiring, auth, observability, and CUDA deployment validation.

## Request API

Use `FramePairRequest` for tensor-pair inference:

- `mode=fixed_2x` requires `interpolation_factor=2` and timestep `0.5`;
- `mode=arbitrary_nx` validates integer factors `2..8` and generates `N - 1` timesteps: `1/N, 2/N, ..., (N-1)/N`;
- `backend_kind=torch` is the default runtime backend for local PyTorch inference.
- `backend_kind=onnx` is supported by the Stage 2 ONNX runtime classes for exported EMA-VFI and Practical-RIFE neural-core artifacts.

The result object stores generated intermediate frame tensors, timesteps, model name, backend kind, original shape, padded shape, and elapsed runtime.

In `arbitrary_nx` mode, `interpolation_factor` is a runtime/request argument. Configs may define supported modes, defaults, bounds, and checkpoint mappings, but callers choose the concrete factor per request:

```python
FramePairRequest(
    left=left,
    right=right,
    mode="arbitrary_nx",
    interpolation_factor=4,
    backend_options={"scale": 0.5},
)
```

The model adapters also expose `predict_intermediate_frames(left, right, interpolation_factor=N)` for tensor-pair Nx inference. For `N=4`, the result contains three intermediate frames at timesteps `0.25`, `0.5`, and `0.75`.

Stage 2.5 adds true PyTorch model-batch execution for EMA-VFI and Practical-RIFE through `ModelBatchRequest` and `ModelBatchResult`, plus ONNX batch-request support where the accepted artifacts are viable. The batch contract accepts BCHW left/right tensors, flattens Nx work in pair-major pair×timestep order, and reconstructs outputs as `outputs[pair_index][timestep_index]`. Adapter `predict_frame_pairs_batch(...)` exposes this contract directly, while adapter `predict_batch([(left, right), ...])` remains fixed 2x and now uses the true batch runtime for EMA/RIFE.

For Practical-RIFE, `backend_options["scale"]` is a request/runtime option. Allowed values match upstream Practical-RIFE: `0.25`, `0.5`, `1.0`, `2.0`, and `4.0`. The upstream README recommends `scale=0.5` for high-resolution inputs such as 4K.

Local pair-smoke CLI commands accept the factor now:

```bash
uv run python -m video_interpolation.cli ema infer-pair --interpolation-factor 4
uv run python -m video_interpolation.cli rife infer-pair --interpolation-factor 4 --scale 0.5
```

These commands use synthetic tensor pairs and do not write videos.

## Local Video Inference

Local EMA-VFI and Practical-RIFE video inference now use the same request/result runtime API as tensor-pair inference. Video decoding, tensor conversion, frame interleaving, PyAV/FFmpeg encoding, and audio remuxing remain project-owned in `src/video_interpolation/inference.py`.

For each neighboring input-frame pair `(frame_i, frame_i+1)`:

- `fixed_2x` uses `interpolation_factor=2`, timestep `0.5`, and writes one generated frame;
- `arbitrary_nx` validates a runtime/CLI factor in `2..8`, generates `N - 1` frames at `1/N, 2/N, ..., (N-1)/N`, and writes them in timestep order;
- output order is `original_i`, generated frames, then `original_i+1`;
- output FPS is `input_fps * interpolation_factor`.
- `execution_mode=batched` is the default for EMA-VFI and Practical-RIFE local video inference and routes neighboring pairs through `ModelBatchRequest` chunks;
- `execution_mode=sequential` keeps the older one-pair-at-a-time path for debugging and low-VRAM runs.

The factor is a runtime argument, not a factor-specific YAML file. Existing 2x configs remain compatible and now declare:

```yaml
interpolation_mode: fixed_2x
interpolation_factor: 2
execution_mode: batched
inference_batch_size: null
```

`inference_batch_size` caps flattened model rows. Fixed 2x consumes one row per source pair; Nx consumes `interpolation_factor - 1` rows per source pair. Chunk boundaries overlap by one source frame, so a chunk ending at frame `k` hands frame `k` to the next chunk as its first source frame.

Examples:

```bash
uv run python -m video_interpolation.cli ema infer-video \
  --config configs/inference/ema_vfi_small_2x.yaml \
  --input raw_data/tmp_test/DORA_cut.mp4 \
  --output /tmp/dora_ema_2x.mp4 \
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
  --output /tmp/dora_ema_4x.mp4 \
  --mode arbitrary_nx \
  --interpolation-factor 4 \
  --execution-mode batched \
  --inference-batch-size 3 \
  --codec libx264 \
  --limit-pairs 2 \
  --disable-mlflow

uv run python -m video_interpolation.cli rife infer-video \
  --config configs/inference/practical_rife_v4_26_2x.yaml \
  --input raw_data/tmp_test/DORA_cut.mp4 \
  --output /tmp/dora_rife_4x.mp4 \
  --mode arbitrary_nx \
  --interpolation-factor 4 \
  --execution-mode batched \
  --inference-batch-size 6 \
  --scale 0.5 \
  --codec libx264 \
  --limit-pairs 2 \
  --disable-mlflow
```

Practical-RIFE local video inference can optionally run lightweight quality evaluation after the output video is generated:

```bash
uv run python -m video_interpolation.cli rife infer-video \
  --config configs/inference/practical_rife_v4_26_2x.yaml \
  --input raw_data/tmp_test/DORA_cut.mp4 \
  --output /tmp/dora_rife_4x_quality.mp4 \
  --mode arbitrary_nx \
  --interpolation-factor 4 \
  --execution-mode sequential \
  --scale 1.0 \
  --codec libx264 \
  --limit-pairs 2 \
  --enable-quality-evaluation \
  --quality-sample-count 16 \
  --quality-max-image-side 360 \
  --write-quality-triplets \
  --quality-triplet-output-dir /tmp/dora_quality_triplets \
  --quality-fail-policy warn \
  --disable-mlflow
```

Quality evaluation uses the first `--quality-sample-count` overlapping input triplets `(frame_i, frame_i+1, frame_i+2)`, so it decodes at most `sample_count + 2` source frames and does not seek across the video. It resizes each source frame for quality evaluation only, reducing the larger side to `--quality-max-image-side` pixels while preserving aspect ratio; use `--quality-max-image-side 0` to score full-resolution quality frames. It currently assumes one scene and does not run adjacent-frame scene-cut SSIM filtering. Triplet PNG persistence is disabled by default; add `--write-quality-triplets` to write `<triplet_output_dir>/<source_video_id>/<triplet_id>/im1.png`, `im2.png`, and `im3.png`. The quality request compares the model prediction for `left/right -> middle` against resized `im2`, keeps the same inference mode as the main run, and always uses interpolation factor `2` because each source triplet has exactly one real middle frame.

Directory-wide inference accepts the same runtime factor:

```bash
uv run python -m video_interpolation.cli infer-all-videos \
  --input-dir raw_data/tmp_test \
  --target models \
  --mode arbitrary_nx \
  --interpolation-factor 4 \
  --execution-mode batched \
  --inference-batch-size 6 \
  --rife-scale 0.5 \
  --codec libx264 \
  --limit-pairs 2 \
  --disable-mlflow
```

Batch outputs use an output suffix such as `_4x.mp4`, and measurement CSVs include `interpolation_mode`, `interpolation_factor`, `requested_execution_mode`, actual `execution_mode`, `inference_batch_size`, `batch_chunks_processed`, `model_batch_requests`, `runtime_backend`, timing fields, and request `runtime_options` such as Practical-RIFE `scale`. With no explicit target, `arbitrary_nx` batch inference runs active model targets only; baselines and AMT-S remain fixed-2x/legacy paths unless selected explicitly.

## Runtime Benchmarks

Stage 2.5 Milestone 8 benchmarks complete local video inference profiles without introducing a separate interpolation implementation. The command calls `run_video_inference(...)`, so decode, preprocessing, model calls, postprocessing, video encode/flush, optional audio remux, and total pipeline time are measured in the same path used by local inference.

Use:

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

The command supports `--backend torch|onnx`, `--execution-mode sequential|batched`, `--mode fixed_2x|arbitrary_nx`, `--interpolation-factor`, `--inference-batch-size`, one input video through `--input`, directory mode through `--input-dir`, `--limit-videos`, and `--limit-pairs`. Reports are written to `benchmark_report.json` and `benchmark_metrics.csv` under the selected `--output-dir`; generated benchmark videos are written under `videos/<profile>/`.

For `practical_rife_v4_26`, benchmarks enable first-triplet quality evaluation by default so timing reports include its overhead. Triplet PNG writing is disabled by default, so `quality_triplets_written` is normally `0` even when PSNR/SSIM were computed. Use `--disable-quality-evaluation` to benchmark the same profile without the quality step. EMA benchmark profiles keep quality evaluation disabled by default because the current online quality feature is Practical-RIFE-only. Benchmark CSV/JSON and MLflow metrics include `quality_evaluation_enabled`, `quality_evaluation_sec`, `quality_psnr_mean`, `quality_ssim_mean`, `quality_triplets_written`, `quality_triplet_output_dir`, and `quality_error`.

MLflow logging is enabled by default through the shared project MLflow helper. The benchmark disables nested inference-run logging and writes one aggregate benchmark run. Use `--disable-mlflow` for local smoke runs when the tracking server is not running; generated videos are logged only with `--log-output-videos`.

## EMA-VFI Nx Policy

EMA-VFI has two small checkpoints:

- `ours_small.pkl` for the original fixed 2x training/fine-tuning assumptions;
- `ours_small_t.pkl` for timestep-capable inference.

Stage 2 inference configs now select `ours_small_t.pkl` so the same inference checkpoint can serve `fixed_2x` as `t=0.5` and arbitrary Nx factors without per-request checkpoint switching. Stage 1 EMA training/fine-tuning configs remain fixed 2x and continue to use `ours_small.pkl`.

CUDA was unavailable in the current implementation environment, so real EMA CUDA smoke confirmation for `ours_small_t.pkl` fixed 2x and Nx is still required in a CUDA environment.

## Practical-RIFE Nx Policy

Practical-RIFE v4.26 remains the Stage 2 default. The same selected Practical-RIFE runtime/checkpoint handles fixed 2x and arbitrary Nx through direct timestep inference. Practical-RIFE v4.25 remains available through its alternative configs.

`scale` is external to the checkpoint and can be supplied per request. Config `model.scale` remains the default fallback, but local CLI and future services should pass request-specific values with `rife infer-video --scale`, `rife infer-pair --scale`, `infer-all-videos --rife-scale`, or `FramePairRequest.backend_options`.

## Practical-RIFE Source Policy

Practical-RIFE no longer imports Python model source from `model_weights/Practical-RIFE/.../train_log`.

Runtime source now lives in:

```text
src/video_interpolation/inference_runtime/rife_upstream/
```

The copied runtime source is based on the local Practical-RIFE v4.26 `train_log` code, with inference-only wrapping and device/dtype/shape-aware warp-grid creation. Weight directories remain checkpoint artifact locations containing `flownet.pkl`.

Stage 2 defaults to Practical-RIFE v4.26:

```text
configs/models/practical_rife_v4_26.yaml
configs/inference/practical_rife_v4_26_2x.yaml
configs/validation/practical_rife_v4_26_candidate.yaml
```

Practical-RIFE v4.25 remains available through the matching `*_v4_25*` configs.

## ONNX Export

Milestone 6 adds developer-facing ONNX export boundaries for EMA-VFI and Practical-RIFE. ONNX export targets only the neural computation that maps already prepared/padded tensors plus one timestep to one generated intermediate-frame tensor.

These remain outside ONNX and in project-owned Python:

- video decoding and image loading;
- tensor/image conversion;
- request parsing and interpolation-factor validation;
- timestep-loop generation for Nx;
- padding/unpadding unless a later blocker proves a wrapper needs it;
- frame interleaving;
- output video encoding and audio remuxing;
- CLI/service orchestration and BentoML code.

The export wrappers are:

- `EMAVFIOnnxWrapper`: wraps EMA `model.net`, accepts `left`, `right`, and `timestep`, concatenates the two input tensors, calls the EMA neural core, and returns only `pred`;
- `PracticalRIFEOnnxWrapper`: wraps Practical-RIFE `model.flownet`, accepts `left`, `right`, and `timestep`, builds the Practical-RIFE `scale_list` from an explicit scale value, calls IFNet, and returns only `merged[-1]`.

Default artifact layout:

```text
model_exports/onnx/
  ema_vfi_small/
    ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx
    ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx.data
  practical_rife_v4_26/
    practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx
    practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx.data
```

Dynamic height/width export is attempted first by default with `--shape-mode dynamic_hw` and `--exporter dynamo`. Static sample-shape export is available through `--shape-mode static`, but it is a fallback/tooling option rather than the preferred serving shape policy. Dynamo artifacts may be split into a small `.onnx` graph plus a companion `.onnx.data` weights file; keep both files adjacent for ONNX Runtime loading.

Developer commands:

```bash
uv run python -m video_interpolation.cli ema export-onnx \
  --config configs/models/ema_vfi_small.yaml \
  --device cpu \
  --opset-version 18 \
  --exporter dynamo \
  --dynamic-hw-multiple 112 \
  --height 336 \
  --width 560 \
  --artifact-stem ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560 \
  --no-simplify

uv run python -m video_interpolation.cli rife export-onnx \
  --config configs/models/practical_rife_v4_26.yaml \
  --device cpu \
  --opset-version 18 \
  --exporter dynamo \
  --dynamic-hw-multiple 128 \
  --height 384 \
  --width 512 \
  --scale 1.0 \
  --artifact-stem practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512 \
  --no-simplify
```

Common options:

- `--opset-version`: default `18`;
- `--exporter`: `dynamo` or `legacy`; default `dynamo`;
- `--shape-mode`: `dynamic_hw` or `static`;
- `--dynamic-hw-multiple`: optional H/W multiple constraint for dynamo dynamic shapes;
- `--artifact-stem`: optional output filename stem override;
- `--batch-size`, `--height`, `--width`: sample already prepared/padded input shape;
- `--timestep`: sample timestep, default `0.5`;
- `--simplify/--no-simplify`: simplification is off by default and is supported only for the legacy exporter.

After export, ONNX checker validation is attempted. With the legacy exporter, `--simplify` can run `onnx-simplifier`; a simplifier failure is reported, but the original valid ONNX export remains the fallback artifact. With the dynamo exporter, keep the original unsimplified artifact and validate it directly.

The current project dependency file already lists `onnx`, `onnxruntime`, `onnxruntime-gpu`, `onnx-simplifier`, and `onnxscript`.

Current model-specific status:

- EMA-VFI: the accepted ONNX artifact is a dynamo opset 18 constrained-dynamic graph with symbolic `112*height_units` and `112*width_units`. It requires external divisor `112` padding and unpadding.
- Practical-RIFE: the accepted current export format is a dynamo opset 18 graph with symbolic batch plus symbolic `128*height_units` and `128*width_units`. Runtime use requires this dynamic-batch artifact; the older fixed-batch dynamo artifact is rejected at load time.

## ONNX Runtime Validation

Milestone 7 adds an internal ONNX Runtime backend at `src/video_interpolation/inference_runtime/backends/onnx.py`. It loads exported neural-core artifacts, validates requested providers, converts prepared tensors to ONNX Runtime arrays, runs `InferenceSession.run(...)`, and returns tensors through the same `FramePairResult` API for sequential calls and `ModelBatchResult` for batch requests.

Provider selection is explicit:

- validation commands default to `CPUExecutionProvider` for safe local smoke checks;
- pass `--provider CUDAExecutionProvider --provider CPUExecutionProvider` to request CUDA with CPU fallback in environments where CUDA ONNX Runtime is actually usable;
- aliases `cpu` and `cuda` are accepted;
- a requested provider that is not listed by ONNX Runtime fails before session creation.

The commands prefer dynamo artifacts by default, and the resolver prefers `dynamic_batch_hw` dynamo artifacts when present. Legacy artifacts remain loadable with `--artifact-exporter legacy --artifact-opset-version 17`; `--prefer-simplified/--prefer-original` only affects legacy artifact resolution. Passing `--onnx-path` always loads that explicit artifact path.

Developer validation commands:

```bash
uv run python -m video_interpolation.cli ema validate-onnx \
  --config configs/models/ema_vfi_small.yaml \
  --provider CPUExecutionProvider \
  --shape 32x32

uv run python -m video_interpolation.cli rife validate-onnx \
  --config configs/models/practical_rife_v4_26.yaml \
  --provider CPUExecutionProvider \
  --shape 128x128
```

Repeat `--shape` to validate dynamic H/W behavior:

```bash
uv run python -m video_interpolation.cli ema validate-onnx \
  --config configs/models/ema_vfi_small.yaml \
  --provider CPUExecutionProvider \
  --shape 32x32 \
  --shape 64x64 \
  --output-dir outputs/onnx_validation/ema_dynamic_check

uv run python -m video_interpolation.cli rife validate-onnx \
  --config configs/models/practical_rife_v4_26.yaml \
  --provider CPUExecutionProvider \
  --shape 128x128 \
  --shape 128x256 \
  --output-dir outputs/onnx_validation/rife_dynamic_check
```

Validation compares PyTorch-generated intermediate frames against ONNX Runtime-generated intermediate frames for the same synthetic inputs, timesteps, padding policy, model config, and export artifact. It reports:

- MAE;
- maximum absolute error;
- MSE;
- `torch.allclose(..., atol=1e-3, rtol=1e-3)`.

Reports are written under `outputs/onnx_validation/<model_name>/` or the selected `--output-dir`:

```text
equivalence_report.json
equivalence_metrics.csv
sample_outputs/<model_name>/*_torch.png
sample_outputs/<model_name>/*_onnx.png
sample_outputs/<model_name>/*_absdiff_normalized.png
```

Sample images are written only when tensor `allclose` fails but both tensors are available.

Current CPU validation status:

- EMA-VFI `ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx`: `64x64`, `112x168`, and `320x512` original inputs pass through one artifact with external divisor `112`, mean MAE `1.0100862e-06`, and max absolute error `4.1246414e-05`.
- Practical-RIFE `practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx`: `128x128`, `128x256`, and `320x512` all run through one artifact. Strict synthetic `1e-3` allclose passes for `128x128` and fails for larger shapes; mean MAE is `4.6398597e-05`, max absolute error is `0.0087888837`. Milestone 7.5 batch smokes show fixed 2x flattened batches `1`, `2`, and `4`, plus Nx factor 4 flattened batches `3` and `6`, each executing in one ORT call.
- Stage 2.5 Milestone 3 adds `ema validate-onnx-real` and `rife validate-onnx-real` for `raw_data/pair_test` image pairs. EMA CPU real-pair reports live under `outputs/onnx_validation/stage2_5_m3_real_pairs/`; the current Practical-RIFE dynamic-batch artifact was revalidated under `outputs/onnx_validation/stage2_5_m7_5_rife_dynamic_batch_real_pairs/`. Both models pass all three `512 x 320` fixtures at strict `1e-3` allclose with PSNR/SSIM recorded.
- Stage 2.5 Milestones 7 and 7.5 add `predict_batch(ModelBatchRequest)` to the EMA and Practical-RIFE ONNX runtimes. EMA and Practical-RIFE now use true multi-row ONNX Runtime calls with the accepted dynamic-batch artifacts. Practical-RIFE rejects fixed-batch artifacts instead of providing static-batch-1 fallback support.

CUDA ONNX Runtime was not validated in this environment. ONNX Runtime lists CUDA/TensorRT providers, but PyTorch reports CUDA unavailable, so Milestone 7 validation used CPU provider only.

## Boundaries

Keep these outside model runtime backends:

- video decoding and encoding;
- frame interleaving;
- audio remuxing;
- BentoML service wrappers.

The current PyTorch and ONNX runtime boundaries target only the neural network core plus model-specific tensor padding/unpadding around that core. Stage 2 is ready for handoff to backend/service development with Practical-RIFE v4.26 PyTorch CUDA as the recommended serving path and ONNX Runtime CUDA as the explicit alternate path.
