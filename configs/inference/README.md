# Stage 1 Inference Configs

Inference YAML files define local developer inference runs. Runtime roots and MLflow endpoint still come from `.env`.

## Batch Directory Inference

Purpose: run every active inference target over every supported video discovered under an explicit input directory. Stage 2 active defaults are Practical-RIFE v4.26, EMA-VFI-small, and the `duplicate_left`, `blend`, and `farneback` baselines. AMT-S and Practical-RIFE v4.25 remain available by explicit target/config selection.

Command:

```bash
uv run python -m video_interpolation.cli infer-all-videos \
  --input-dir raw_data/tmp_test \
  --codec libx264 \
  --disable-mlflow
```

Inputs and flags:

| Flag | Type | Default | Effect |
| --- | --- | --- | --- |
| `--input-dir` | path | required, no default | Recursively discovers supported videos under this directory |
| `--output-root` | path | `outputs/inference` | Root directory for generated videos |
| `--limit-videos` | integer or omitted | omitted | Processes only the first N discovered videos for smoke runs |
| `--limit-pairs` | integer or omitted | omitted | Passes the same pair cap to every individual inference run |
| `--mode` | string | `fixed_2x` | Selects `fixed_2x` or `arbitrary_nx` video inference |
| `--interpolation-factor` | integer or omitted | model config default | Runtime interpolation factor, validated in `2..8` |
| `--rife-scale` | float or omitted | Practical-RIFE config default | Request-time Practical-RIFE scale. Allowed: `0.25`, `0.5`, `1.0`, `2.0`, `4.0`; upstream recommends `0.5` for high-resolution inputs such as 4K |
| `--codec` | string or omitted | config-driven | Overrides the FFmpeg/PyAV encoder for every output |
| `--target`, `--method` | repeatable string | omitted means active defaults | Selects one or more targets/subgroups to run |
| `--disable-mlflow` | flag | false | Disables MLflow logging for every individual run |
| `--continue-on-error/--fail-fast` | boolean flag | `--continue-on-error` | Either continue remaining runs after failures or stop at the first failure |

Target selections:

- Active neural targets: `practical_rife_v4_26`, `ema_vfi_small`.
- Legacy/alternative neural targets: `practical_rife_v4_25`, `amt_s`.
- Short model aliases: `rife`, `amt`, `ema`.
- Baseline targets: `baseline_duplicate_left`, `baseline_blend`, `baseline_farneback`.
- Short baseline aliases: `duplicate_left`, `blend`, `farneback`.
- Groups: `models`, `baselines`, `all`.
- With `--mode arbitrary_nx` and no explicit target, the command runs active model targets only because baselines and AMT-S are fixed-2x/legacy paths.

Examples:

```bash
uv run python -m video_interpolation.cli infer-all-videos \
  --input-dir raw_data/tmp_test \
  --target ema \
  --target blend \
  --codec libx264 \
  --disable-mlflow

uv run python -m video_interpolation.cli infer-all-videos \
  --input-dir raw_data/tmp_test \
  --target baselines \
  --codec libx264 \
  --disable-mlflow

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

Output layout:

```text
outputs/inference/
  practical_rife_v4_26/<relative_input_dir>/<video_stem>_2x.mp4
  practical_rife_v4_26/inference_measurements.csv
  ema_vfi_small/<relative_input_dir>/<video_stem>_2x.mp4
  ema_vfi_small/inference_measurements.csv
  baselines/duplicate_left/<relative_input_dir>/<video_stem>_2x.mp4
  baselines/duplicate_left/inference_measurements.csv
  baselines/blend/<relative_input_dir>/<video_stem>_2x.mp4
  baselines/blend/inference_measurements.csv
  baselines/farneback/<relative_input_dir>/<video_stem>_2x.mp4
  baselines/farneback/inference_measurements.csv
```

The suffix follows the runtime factor, for example `_2x.mp4`, `_4x.mp4`, or `_8x.mp4`. The command preserves relative subdirectories from `--input-dir`, so repeated filenames in different folders do not collide. It reuses the same PyAV/FFmpeg writer, audio remuxing, frame ordering, timing metrics, and MLflow behavior as the single-video commands below.

Each `inference_measurements.csv` contains one row per input video for that target with `interpolation_mode`, `interpolation_factor`, `execution_mode`, `requested_execution_mode`, `inference_batch_size`, `batch_chunks_processed`, `model_batch_requests`, `runtime_backend`, `model_inference_elapsed_sec`, `total_elapsed_sec`, pair counts, FPS values, throughput, audio preservation counts, output path, status, MLflow run id, and error text when a run fails.
Practical-RIFE request options such as `scale` are recorded in the `runtime_options` column.

ONNX export and ONNX Runtime equivalence validation are model-runtime developer workflows, not local video inference workflows. Use `ema export-onnx` / `rife export-onnx` and `ema validate-onnx` / `rife validate-onnx` with files under `configs/models/`; exported artifacts are written under `model_exports/onnx/`, and validation reports are written under `outputs/onnx_validation/`.

Runtime benchmarks are also model-runtime developer workflows. Use `benchmark runtime` to compare torch/ONNX, sequential/batched execution, fixed 2x/Nx factors, and flattened batch-size caps on real video inputs through the normal local video inference pipeline:

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

The default input is `raw_data/tmp_test/DORA_cut.mp4`. Use `--input-dir` with `--limit-videos` for directory mode. Benchmark reports are written as `benchmark_report.json` and `benchmark_metrics.csv` under `outputs/benchmarks/` or the selected `--output-dir`, and generated videos are written under `videos/<profile>/`. MLflow logging is enabled by default and can be skipped with `--disable-mlflow`; output videos are logged only with `--log-output-videos`.

Common failures:

- A model target fails to initialize because CUDA is unavailable or a checkpoint is missing.
- The selected encoder is unavailable; retry with `--codec libx264`.
- MLflow logging is enabled but the tracking server is not running; use `--disable-mlflow` for local visual comparison smoke runs.
- If any run fails, the command exits non-zero after printing a final summary table. With the default `--continue-on-error`, successful outputs from other targets/videos remain on disk.

## `baseline_2x.yaml`

Purpose: run local 2x video interpolation with one non-neural baseline method using the same PyAV/FFmpeg writer, frame ordering, audio remuxing, and MLflow behavior as model inference.

Baseline-specific `model` fields:

| Field | Type | Allowed/default | Effect | Rerun needed |
| --- | --- | --- | --- | --- |
| `model.model_name` | string | default `baseline_blend` | Names the baseline in logs, metrics, reports, and MLflow params | Yes |
| `model.baseline_name` | string | `duplicate_left`, `blend`, or `farneback`; default `blend` | Selects the baseline predictor used for every neighboring frame pair | Yes |

All top-level video-output fields match model inference configs: `input_path`, `output_path`, `limit_pairs`, `interpolation_mode`, `interpolation_factor`, `execution_mode`, `inference_batch_size`, `output_fps_multiplier`, `codec`, `container`, `pix_fmt`, `frame_format`, `encoder_options_by_codec`, `encoder_options`, and `mlflow`. Baselines remain fixed 2x; `output_fps_multiplier` is kept for config compatibility, while runtime model paths compute output FPS from `interpolation_factor`.

Command:

```bash
uv run python -m video_interpolation.cli baseline infer-video \
  --config configs/inference/baseline_2x.yaml \
  --baseline farneback \
  --input raw_data/tmp_test/Dora.mp4 \
  --output /tmp/dora_farneback_2x.mp4 \
  --codec libx264 \
  --limit-pairs 2 \
  --disable-mlflow
```

Side effects and outputs:

- Writes a new video to `output_path`.
- Interleaves original and generated frames as `left, generated_middle, right, ...`.
- Uses PyAV/FFmpeg encoding and preserves/remuxes compatible input audio streams.
- Logs codec/container/pixel-format/frame-format settings, resolved encoder options, audio preservation counts, timing metrics, frame/pair counts, and FPS metrics to MLflow when enabled.
- Does not modify source videos, datasets, manifests, model repositories, or model weights.

Common failures:

- `model.baseline_name` or `--baseline` is not one of `duplicate_left`, `blend`, or `farneback`.
- The selected FFmpeg encoder is unavailable; try `--codec libx264`.
- The input audio codec is incompatible with the selected output container. Try a compatible container such as `matroska`.
- MLflow logging is enabled but the tracking server is not running.

## `ema_vfi_small_2x.yaml`

Purpose: run local fixed 2x or arbitrary Nx video interpolation with EMA-VFI-small. Fixed 2x inserts one generated middle frame; arbitrary Nx inserts `N - 1` generated frames between each neighboring input-frame pair.

Local video inference calls the Stage 2 `FramePairRequest`/`FramePairResult` API. Video decoding, tensor conversion, frame ordering, PyAV/FFmpeg encoding, and audio remuxing remain model-independent project code.

Fields:

| Field | Type | Allowed/default | Effect | Rerun needed |
| --- | --- | --- | --- | --- |
| `input_path` | path string | example `raw_data/tmp_test/Dora.mp4`; relative to repo root unless absolute | Input video to decode | Yes |
| `output_path` | path string | example `outputs/inference/ema_vfi_small/dora_2x.mp4` | Output video written by PyAV/FFmpeg | Rerun to write elsewhere |
| `model` | mapping | same fields as `configs/models/ema_vfi_small.yaml` | Adapter and checkpoint settings | Yes |
| `limit_pairs` | integer or `null` | default `null`; positive integer for smoke runs | Caps the number of neighboring frame pairs interpolated | Yes |
| `interpolation_mode` | string | `fixed_2x` or `arbitrary_nx`; default `fixed_2x` | Selects one-frame 2x or request-time Nx interpolation | Yes |
| `interpolation_factor` | integer or `null` | `2..8`; config default `2` | Runtime factor. `fixed_2x` requires `2`; `arbitrary_nx` writes `N - 1` generated frames per pair | Yes |
| `execution_mode` | string | `batched` or `sequential`; default `batched` | Uses chunked model-batch video inference for EMA/RIFE when available, or explicit pair-by-pair fallback when `sequential` | Yes |
| `inference_batch_size` | integer or `null` | default `null`; effective fallback `1` unless the model config sets one | Maximum flattened model rows per video batch. For Nx, one source pair consumes `interpolation_factor - 1` rows | Yes |
| `output_fps_multiplier` | float | default `2.0` | Compatibility field retained from Stage 1; effective output FPS is input FPS multiplied by `interpolation_factor` | Yes |
| `codec` | string | default `h264_nvenc`; examples `libx264`, `h264_nvenc`, `libx265`, `hevc_nvenc` | FFmpeg/PyAV encoder name | Yes |
| `container` | string or `null` | default `null`; examples `mp4`, `matroska`, `mov`, `webm` | Output container format. `null` lets PyAV infer from `output_path` | Yes |
| `pix_fmt` | string | default `yuv420p` | Encoded video pixel format; `yuv420p` is the safest MP4 playback choice | Yes |
| `frame_format` | string | default `rgb24`; allowed `rgb24`, `bgr24` | NumPy frame color format passed into `av.VideoFrame.from_ndarray()` | Yes |
| `encoder_options_by_codec` | mapping | optional per-codec option maps | Lets GPU/CPU codec options coexist without rewriting config | Yes |
| `encoder_options` | mapping | default `{}` | Direct PyAV/FFmpeg encoder option overrides applied after built-in and per-codec defaults | Yes |
| `mlflow.enabled` | boolean | default `true` | Logs params, counts, output video, and config to MLflow | Rerun |
| `mlflow.experiment_name` | string | default `stage1-ema-inference` | MLflow experiment | Rerun |
| `mlflow.run_name` | string or `null` | optional run name | MLflow run name | Rerun |
| `mlflow.tags` | mapping | string tags | Extra MLflow tags | Rerun |

Encoder option defaults:

- `h264_nvenc`: `preset=p3`, `rc=vbr`, `cq=23`, `bf=0`.
- `hevc_nvenc`: `preset=p3`, `rc=vbr`, `cq=26`, `bf=0`.
- `libx264`: `preset=veryfast`, `crf=22`, `bf=0`.
- `libx265`: `preset=veryfast`, `crf=28`, `bf=0`.
- If `g` is not set anywhere, the command sets it to `int(output_fps * 2)` for an approximate two-second GOP.
- `mp4v` is not supported anymore because it is an OpenCV fourcc, not the intended FFmpeg encoder config. Use `libx264` for a CPU-compatible H.264 fallback.

Command:

```bash
uv run python -m video_interpolation.cli ema infer-video \
  --config configs/inference/ema_vfi_small_2x.yaml \
  --input raw_data/tmp_test/Dora.mp4 \
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
  --input raw_data/tmp_test/Dora.mp4 \
  --output /tmp/dora_ema_4x.mp4 \
  --mode arbitrary_nx \
  --interpolation-factor 4 \
  --execution-mode batched \
  --inference-batch-size 6 \
  --codec libx264 \
  --limit-pairs 2 \
  --disable-mlflow
```

Side effects and outputs:

- Writes a new video to `output_path`.
- Original and generated frames are interleaved as `left, generated frames in timestep order, right, ...`.
- Output FPS is `input_fps * interpolation_factor`.
- In `batched` mode, EMA/RIFE video inference decodes source frames in overlapping chunks. The last source frame of one chunk becomes the first source frame of the next chunk, so boundary pairs are neither dropped nor duplicated.
- `inference_batch_size` caps flattened model rows. For example, 4x interpolation uses three rows per source pair, so `--inference-batch-size 6` processes two source pairs per video chunk when the adapter supports batch execution.
- `sequential` mode keeps the older pair-by-pair request path for low-VRAM debugging. Adapters that do not expose `predict_frame_pairs_batch(...)` automatically record `execution_mode=sequential_fallback`.
- Uses PyAV/FFmpeg for output encoding and preserves/remuxes input audio streams when compatible with the selected output container.
- When `limit_pairs` is set for a smoke run, the output video is intentionally partial and remuxed audio is capped to the partial output duration.
- Measures model inference elapsed time around Stage 2 request/result runtime calls, total processing time, model pairs/sec, and total pairs/sec.
- Logs codec/container/pixel-format/frame-format settings, resolved encoder options, audio preservation counts, timing metrics, frame/pair counts, and FPS metrics to MLflow when MLflow is enabled.
- Does not modify source videos, datasets, manifests, model repositories, or model weights.
- Requires CUDA for the current EMA adapter.
- Stage 2 inference selects `EMA-VFI/ours_small_t.pkl` so fixed 2x can run as `t=0.5` and arbitrary Nx can use the same timestep-capable checkpoint. EMA training/fine-tuning configs remain on `ours_small.pkl`.

Tensor-pair Nx smoke:

```bash
uv run python -m video_interpolation.cli ema infer-pair \
  --config configs/models/ema_vfi_small.yaml \
  --interpolation-factor 4
```

Inspect:

```bash
ls -lh /tmp/dora_ema_2x.mp4
ffprobe /tmp/dora_ema_2x.mp4
```

Common failures:

- CUDA is unavailable.
- The input video cannot be decoded.
- The selected FFmpeg encoder is unavailable in the local PyAV/FFmpeg build. Try `--codec libx264`.
- The input audio codec is incompatible with the selected output container. Try a compatible container such as `matroska`.
- MLflow logging is enabled but the tracking server is not running.

## `amt_s_2x.yaml`

Purpose: run local 2x video interpolation with AMT-S using the same PyAV/FFmpeg writer, frame ordering, audio remuxing, and MLflow behavior as EMA inference.

AMT-specific `model` fields:

| Field | Type | Allowed/default | Effect | Rerun needed |
| --- | --- | --- | --- | --- |
| `model.model_name` | string | default `amt_s` | Names the model in logs, metrics, reports, and MLflow params | Yes |
| `model.repo_name` | string | default `AMT` | Upstream AMT repo directory under `MODEL_REPOS_ROOT` | Yes |
| `model.config_path` | path string | default `cfgs/AMT-S.yaml`; relative to AMT repo unless absolute | Builds the AMT-S network architecture | Yes |
| `model.checkpoint_path` | path string | default `AMT/amt-s.pth`; relative to `MODEL_WEIGHTS_ROOT` unless absolute | AMT-S checkpoint loaded for inference | Yes |
| `model.device` | string | default `cuda` | Torch device for AMT-S inference | Yes |
| `model.embt` | float | default `0.5` | Middle-frame interpolation timestamp | Yes |
| `model.scale_factor` | float | default `1.0` | AMT internal inference scale factor | Yes |
| `model.divisor` | integer | default `16` | Input padding divisor before AMT inference | Yes |
| `model.strict_checkpoint` | boolean | default `true` | Strict state-dict loading | Yes |

All top-level video-output fields match `ema_vfi_small_2x.yaml`: `input_path`, `output_path`, `limit_pairs`, `interpolation_mode`, `interpolation_factor`, `execution_mode`, `inference_batch_size`, `output_fps_multiplier`, `codec`, `container`, `pix_fmt`, `frame_format`, `encoder_options_by_codec`, `encoder_options`, and `mlflow`. AMT-S remains a fixed-2x legacy path.

Command:

```bash
uv run python -m video_interpolation.cli amt infer-video \
  --config configs/inference/amt_s_2x.yaml \
  --input raw_data/tmp_test/Dora.mp4 \
  --output /tmp/dora_amt_2x.mp4 \
  --codec libx264 \
  --limit-pairs 2 \
  --disable-mlflow
```

Side effects and outputs:

- Writes a new video to `output_path`.
- Interleaves original and generated frames as `left, generated_middle, right, ...`.
- Uses PyAV/FFmpeg encoding and preserves/remuxes compatible input audio streams.
- Logs codec/container/pixel-format/frame-format settings, resolved encoder options, audio preservation counts, timing metrics, frame/pair counts, and FPS metrics to MLflow when enabled.
- Does not modify source videos, datasets, manifests, model repositories, or model weights.

Common failures:

- CUDA is unavailable while `model.device: cuda`.
- The selected FFmpeg encoder is unavailable; try `--codec libx264`.
- AMT-S checkpoint loading fails because the checkpoint path is wrong or the file is not a compatible AMT-S state dict.

## `practical_rife_v4_26_2x.yaml`

Purpose: run local fixed 2x or arbitrary Nx video interpolation with Practical-RIFE v4.26 using the same PyAV/FFmpeg writer, frame ordering, audio remuxing, and MLflow behavior as EMA inference.

Practical-RIFE v4.26 is the Stage 2 default. The same selected runtime/checkpoint handles fixed 2x (`t=0.5`) and arbitrary Nx direct timesteps from the runtime factor.

Practical-RIFE-specific `model` fields:

| Field | Type | Allowed/default | Effect | Rerun needed |
| --- | --- | --- | --- | --- |
| `model.model_name` | string | default `practical_rife_v4_26` | Names the model in logs, metrics, reports, and MLflow params | Yes |
| `model.repo_name` | string | default `Practical-RIFE` | Upstream Practical-RIFE repo directory under `MODEL_REPOS_ROOT` | Yes |
| `model.checkpoint_path` | path string | default `Practical-RIFE/RIFEv4.26/train_log`; relative to `MODEL_WEIGHTS_ROOT` unless absolute | Directory containing `flownet.pkl`; runtime source is project-owned | Yes |
| `model.supported_modes` | list of strings | `fixed_2x`, `arbitrary_nx` | Declares supported tensor-pair runtime modes | Yes |
| `model.default_interpolation_factor` | integer | default `2` | Pair-smoke/request default only when a caller omits `--interpolation-factor` | Yes |
| `model.min_interpolation_factor` / `model.max_interpolation_factor` | integer | `2` / `8` | Runtime/request validation bounds | Yes |
| `model.device` | string | default `cuda` | Torch device for Practical-RIFE inference | Yes |
| `model.timestep` | float | default `0.5` | Middle-frame interpolation timestamp | Yes |
| `model.scale` | float | default `1.0` | Practical-RIFE scale fallback when no request/CLI scale is supplied | Yes |
| `model.divisor` | integer | default `128` | Input padding divisor before Practical-RIFE inference | Yes |
| `model.inference_batch_size` | integer or `null` | default `null` | Optional model-runtime flattened-row cap used when the top-level video config does not override it | Yes |
| `model.strict_checkpoint` | boolean | default `false` | Non-strict state-dict loading accepts extra training-only keys | Yes |

All top-level video-output fields match `ema_vfi_small_2x.yaml`: `input_path`, `output_path`, `limit_pairs`, `interpolation_mode`, `interpolation_factor`, `execution_mode`, `inference_batch_size`, `runtime_options`, `output_fps_multiplier`, `codec`, `container`, `pix_fmt`, `frame_format`, `encoder_options_by_codec`, `encoder_options`, and `mlflow`.

Command:

```bash
uv run python -m video_interpolation.cli rife infer-video \
  --config configs/inference/practical_rife_v4_26_2x.yaml \
  --input raw_data/tmp_test/Dora.mp4 \
  --output /tmp/dora_rife_2x.mp4 \
  --mode fixed_2x \
  --interpolation-factor 2 \
  --execution-mode batched \
  --inference-batch-size 2 \
  --codec libx264 \
  --limit-pairs 2 \
  --disable-mlflow

uv run python -m video_interpolation.cli rife infer-video \
  --config configs/inference/practical_rife_v4_26_2x.yaml \
  --input raw_data/tmp_test/Dora.mp4 \
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

Side effects and outputs:

- Writes a new video to `output_path`.
- Interleaves original and generated frames as `left, generated_middle, right, ...`.
- In batched mode, output order remains `left, generated frames in timestep order, right`; the implementation batches neighboring source pairs only at the model-call level.
- Uses PyAV/FFmpeg encoding and preserves/remuxes compatible input audio streams.
- Logs codec/container/pixel-format/frame-format settings, resolved encoder options, audio preservation counts, timing metrics, frame/pair counts, and FPS metrics to MLflow when enabled.
- Does not modify source videos, datasets, manifests, model repositories, or model weights.
- Fixed 2x is the `interpolation_factor=2`, `t=0.5` case. Arbitrary Nx uses the request/CLI-provided factor and direct timesteps.
- `--scale` is a request-time Practical-RIFE option, not a factor-specific config. Valid values are `0.25`, `0.5`, `1.0`, `2.0`, and `4.0`; the upstream README recommends `0.5` for high-resolution inputs such as 4K.
- Output FPS is `input_fps * interpolation_factor`; generated frames are written in timestep order before the next original frame.

Tensor-pair Nx smoke:

```bash
uv run python -m video_interpolation.cli rife infer-pair \
  --config configs/models/practical_rife_v4_26.yaml \
  --interpolation-factor 4 \
  --scale 0.5
```

Common failures:

- CUDA is unavailable while `model.device: cuda`.
- The selected FFmpeg encoder is unavailable; try `--codec libx264`.
- Practical-RIFE checkpoint loading fails because the checkpoint directory is wrong or does not contain `flownet.pkl`.
