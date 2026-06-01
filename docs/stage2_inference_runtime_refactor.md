# Stage 2 Inference Runtime Refactor

Stage 2 is separating model inference from local video I/O, training code, and future serving wrappers.

Current runtime layout:

```text
src/video_interpolation/inference_runtime/
  api.py                    request/result API and mode/backend validation
  backends/                 shared runtime backend abstractions
  ema.py                    EMA-VFI PyTorch runtime
  rife.py                   Practical-RIFE PyTorch runtime
  onnx_export.py            ONNX neural-core wrappers and export helpers
  onnx_validation.py        ONNX Runtime equivalence metrics and reports
  rife_upstream/            project-owned Practical-RIFE v4.26 runtime source
```

The public adapter methods remain the compatibility boundary for Stage 1 workflows. Existing calls such as `predict_pair`, `predict_batch`, `predict`, and `__call__` continue to work as fixed 2x middle-frame prediction. Internally, EMA-VFI and Practical-RIFE prediction now construct a `FramePairRequest`, call the model-specific PyTorch runtime, and receive a `FramePairResult`.

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

The factor is a runtime argument, not a factor-specific YAML file. Existing 2x configs remain compatible and now declare:

```yaml
interpolation_mode: fixed_2x
interpolation_factor: 2
```

Examples:

```bash
uv run python -m video_interpolation.cli ema infer-video \
  --config configs/inference/ema_vfi_small_2x.yaml \
  --input raw_data/tmp_test/DORA_cut.mp4 \
  --output /tmp/dora_ema_2x.mp4 \
  --mode fixed_2x \
  --interpolation-factor 2 \
  --codec libx264 \
  --limit-pairs 2 \
  --disable-mlflow

uv run python -m video_interpolation.cli ema infer-video \
  --config configs/inference/ema_vfi_small_2x.yaml \
  --input raw_data/tmp_test/DORA_cut.mp4 \
  --output /tmp/dora_ema_4x.mp4 \
  --mode arbitrary_nx \
  --interpolation-factor 4 \
  --codec libx264 \
  --limit-pairs 2 \
  --disable-mlflow

uv run python -m video_interpolation.cli rife infer-video \
  --config configs/inference/practical_rife_v4_26_2x.yaml \
  --input raw_data/tmp_test/DORA_cut.mp4 \
  --output /tmp/dora_rife_4x.mp4 \
  --mode arbitrary_nx \
  --interpolation-factor 4 \
  --scale 0.5 \
  --codec libx264 \
  --limit-pairs 2 \
  --disable-mlflow
```

Directory-wide inference accepts the same runtime factor:

```bash
uv run python -m video_interpolation.cli infer-all-videos \
  --input-dir raw_data/tmp_test \
  --target models \
  --mode arbitrary_nx \
  --interpolation-factor 4 \
  --rife-scale 0.5 \
  --codec libx264 \
  --limit-pairs 2 \
  --disable-mlflow
```

Batch outputs use an output suffix such as `_4x.mp4`, and measurement CSVs include `interpolation_mode`, `interpolation_factor`, `runtime_backend`, and request `runtime_options` such as Practical-RIFE `scale`. With no explicit target, `arbitrary_nx` batch inference runs active model targets only; baselines and AMT-S remain fixed-2x/legacy paths unless selected explicitly.

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
    ema_vfi_small_dynamic_hw_opset17.onnx
    ema_vfi_small_dynamic_hw_opset17.simplified.onnx
  practical_rife_v4_26/
    practical_rife_v4_26_dynamic_hw_opset17.onnx
    practical_rife_v4_26_dynamic_hw_opset17.simplified.onnx
```

Dynamic height/width export is attempted first by default with `--shape-mode dynamic_hw`. Static sample-shape export is available through `--shape-mode static`, but it is a fallback/tooling option rather than the preferred serving shape policy.

Developer commands:

```bash
uv run python -m video_interpolation.cli ema export-onnx \
  --config configs/models/ema_vfi_small.yaml \
  --device cpu \
  --height 32 \
  --width 32 \
  --output-dir model_exports/onnx

uv run python -m video_interpolation.cli rife export-onnx \
  --config configs/models/practical_rife_v4_26.yaml \
  --device cpu \
  --height 128 \
  --width 128 \
  --scale 1.0 \
  --output-dir model_exports/onnx
```

Common options:

- `--opset-version`: default `17`;
- `--shape-mode`: `dynamic_hw` or `static`;
- `--batch-size`, `--height`, `--width`: sample already prepared/padded input shape;
- `--timestep`: sample timestep, default `0.5`;
- `--simplify/--no-simplify`: simplification is enabled by default.

After export, ONNX checker validation is attempted. If `onnx-simplifier` is available, simplification runs by default. A simplifier failure is reported, but the original valid ONNX export remains the fallback artifact for later Milestone 7 work.

The current project dependency file already lists `onnx`, `onnxruntime`, `onnxruntime-gpu`, and `onnx-simplifier`. Milestone 6 uses `onnx` and `onnx-simplifier` for export validation/simplification only; ONNX Runtime loading is still deferred.

Current model-specific status:

- EMA-VFI: wrapper/export API is implemented around `model.net`; export uses the inference checkpoint selected by the adapter config or `--checkpoint`. A dynamic-H/W export with the 32x32 sample shape succeeds, but PyTorch emits tracer warnings from EMA feature-extractor shape math and cached attention-mask logic. Treat the graph as dynamic-axes-exported but not yet proven dynamic-H/W-safe until Milestone 7 ONNX Runtime checks run multiple shapes.
- Practical-RIFE: wrapper/export API is implemented around project-owned `rife_upstream` IFNet source and does not import Python source from `model_weights/.../train_log`; request/export scale is explicit. A dynamic-H/W export with the 128x128 sample shape succeeds and is ready for Milestone 7 ONNX Runtime checks.
- ONNX Runtime inference and PyTorch-vs-ONNX tensor equivalence are not implemented in Milestone 6. They remain Milestone 7 work.

## ONNX Runtime Validation

Milestone 7 adds an internal ONNX Runtime backend at `src/video_interpolation/inference_runtime/backends/onnx.py`. It loads exported neural-core artifacts, validates requested providers, converts prepared tensors to ONNX Runtime arrays, runs `InferenceSession.run(...)`, and returns tensors through the same `FramePairResult` API.

Provider selection is explicit:

- validation commands default to `CPUExecutionProvider` for safe local smoke checks;
- pass `--provider CUDAExecutionProvider --provider CPUExecutionProvider` to request CUDA with CPU fallback in environments where CUDA ONNX Runtime is actually usable;
- aliases `cpu` and `cuda` are accepted;
- a requested provider that is not listed by ONNX Runtime fails before session creation.

The commands prefer simplified artifacts when they exist and fall back to the original artifact only when `--prefer-original` is supplied or the simplified file is absent.

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

- EMA-VFI `ema_vfi_small_dynamic_hw_opset17.simplified.onnx`: `32x32` passes tensor equivalence with MAE `3.1393333e-07` and max absolute error `1.7881393e-06`. `64x64` fails in ONNX Runtime with a LayerNormalization shape error from the exported EMA feature extractor, so EMA dynamic H/W is not proven and currently needs an export/runtime fix or a documented static/padded fallback.
- Practical-RIFE `practical_rife_v4_26_dynamic_hw_opset17.simplified.onnx`: `128x128` passes tensor equivalence with MAE `8.4759959e-06` and max absolute error `0.00091010332`. `128x256` runs through ONNX Runtime but fails strict `1e-3` allclose due max absolute error `0.0037825704` while MAE remains `5.5331097e-05`; sample PyTorch/ONNX/difference images are written for inspection. The original ONNX artifact shows the same result, so this is not caused by simplification alone.

CUDA ONNX Runtime was not validated in this environment. ONNX Runtime lists CUDA/TensorRT providers, but PyTorch reports CUDA unavailable, so Milestone 7 validation used CPU provider only.

## Boundaries

Keep these outside model runtime backends:

- video decoding and encoding;
- frame interleaving;
- audio remuxing;
- BentoML service wrappers.

The current PyTorch and ONNX runtime boundaries target only the neural network core plus model-specific tensor padding/unpadding around that core. BentoML proof work remains planned later.
