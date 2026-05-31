# Stage 1 Inference Configs

Inference YAML files define local developer inference runs. Runtime roots and MLflow endpoint still come from `.env`.

## Batch Directory Inference

Purpose: run every implemented Stage 1 inference target over every supported video discovered under an explicit input directory. Current targets are Practical-RIFE v4.25, AMT-S, EMA-VFI-small, and the `duplicate_left`, `blend`, and `farneback` baselines.

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
| `--codec` | string or omitted | config-driven | Overrides the FFmpeg/PyAV encoder for every output |
| `--target`, `--method` | repeatable string | omitted means all | Selects one or more targets/subgroups to run |
| `--disable-mlflow` | flag | false | Disables MLflow logging for every individual run |
| `--continue-on-error/--fail-fast` | boolean flag | `--continue-on-error` | Either continue remaining runs after failures or stop at the first failure |

Target selections:

- Neural model targets: `practical_rife_v4_25`, `amt_s`, `ema_vfi_small`.
- Short model aliases: `rife`, `amt`, `ema`.
- Baseline targets: `baseline_duplicate_left`, `baseline_blend`, `baseline_farneback`.
- Short baseline aliases: `duplicate_left`, `blend`, `farneback`.
- Groups: `models`, `baselines`, `all`.

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
```

Output layout:

```text
outputs/inference/
  practical_rife_v4_25/<relative_input_dir>/<video_stem>_2x.mp4
  practical_rife_v4_25/inference_measurements.csv
  amt_s/<relative_input_dir>/<video_stem>_2x.mp4
  amt_s/inference_measurements.csv
  ema_vfi_small/<relative_input_dir>/<video_stem>_2x.mp4
  ema_vfi_small/inference_measurements.csv
  baselines/duplicate_left/<relative_input_dir>/<video_stem>_2x.mp4
  baselines/duplicate_left/inference_measurements.csv
  baselines/blend/<relative_input_dir>/<video_stem>_2x.mp4
  baselines/blend/inference_measurements.csv
  baselines/farneback/<relative_input_dir>/<video_stem>_2x.mp4
  baselines/farneback/inference_measurements.csv
```

The command preserves relative subdirectories from `--input-dir`, so repeated filenames in different folders do not collide. It reuses the same PyAV/FFmpeg writer, audio remuxing, frame ordering, timing metrics, and MLflow behavior as the single-video commands below.

Each `inference_measurements.csv` contains one row per input video for that target with `model_inference_elapsed_sec`, `total_elapsed_sec`, pair counts, FPS values, throughput, audio preservation counts, output path, status, MLflow run id, and error text when a run fails.

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

All top-level video-output fields match model inference configs: `input_path`, `output_path`, `limit_pairs`, `output_fps_multiplier`, `codec`, `container`, `pix_fmt`, `frame_format`, `encoder_options_by_codec`, `encoder_options`, and `mlflow`.

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

Purpose: run local 2x video interpolation with EMA-VFI-small by inserting one generated middle frame between each neighboring input-frame pair.

Fields:

| Field | Type | Allowed/default | Effect | Rerun needed |
| --- | --- | --- | --- | --- |
| `input_path` | path string | example `raw_data/tmp_test/Dora.mp4`; relative to repo root unless absolute | Input video to decode | Yes |
| `output_path` | path string | example `outputs/inference/ema_vfi_small/dora_2x.mp4` | Output video written by PyAV/FFmpeg | Rerun to write elsewhere |
| `model` | mapping | same fields as `configs/models/ema_vfi_small.yaml` | Adapter and checkpoint settings | Yes |
| `limit_pairs` | integer or `null` | default `null`; positive integer for smoke runs | Caps the number of neighboring frame pairs interpolated | Yes |
| `output_fps_multiplier` | float | default `2.0` | Output FPS is input FPS multiplied by this value | Yes |
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
  --codec libx264 \
  --limit-pairs 2 \
  --disable-mlflow
```

Side effects and outputs:

- Writes a new video to `output_path`.
- Original and generated frames are interleaved as `left, generated_middle, right, ...`.
- Uses PyAV/FFmpeg for output encoding and preserves/remuxes input audio streams when compatible with the selected output container.
- When `limit_pairs` is set for a smoke run, the output video is intentionally partial and remuxed audio is capped to the partial output duration.
- Measures model inference elapsed time around EMA `predict_pair()` calls, total processing time, model pairs/sec, and total pairs/sec.
- Logs codec/container/pixel-format/frame-format settings, resolved encoder options, audio preservation counts, timing metrics, frame/pair counts, and FPS metrics to MLflow when MLflow is enabled.
- Does not modify source videos, datasets, manifests, model repositories, or model weights.
- Requires CUDA for the current EMA adapter.

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

All top-level video-output fields match `ema_vfi_small_2x.yaml`: `input_path`, `output_path`, `limit_pairs`, `output_fps_multiplier`, `codec`, `container`, `pix_fmt`, `frame_format`, `encoder_options_by_codec`, `encoder_options`, and `mlflow`.

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

## `practical_rife_v4_25_2x.yaml`

Purpose: run local 2x video interpolation with Practical-RIFE v4.25 using the same PyAV/FFmpeg writer, frame ordering, audio remuxing, and MLflow behavior as EMA and AMT inference.

Practical-RIFE-specific `model` fields:

| Field | Type | Allowed/default | Effect | Rerun needed |
| --- | --- | --- | --- | --- |
| `model.model_name` | string | default `practical_rife_v4_25` | Names the model in logs, metrics, reports, and MLflow params | Yes |
| `model.repo_name` | string | default `Practical-RIFE` | Upstream Practical-RIFE repo directory under `MODEL_REPOS_ROOT` | Yes |
| `model.checkpoint_path` | path string | default `Practical-RIFE/RIFEv4.25/train_log`; relative to `MODEL_WEIGHTS_ROOT` unless absolute | Directory containing `flownet.pkl` and selected model code | Yes |
| `model.device` | string | default `cuda` | Torch device for Practical-RIFE inference | Yes |
| `model.timestep` | float | default `0.5` | Middle-frame interpolation timestamp | Yes |
| `model.scale` | float | default `1.0` | Practical-RIFE internal inference scale | Yes |
| `model.divisor` | integer | default `128` | Input padding divisor before Practical-RIFE inference | Yes |
| `model.strict_checkpoint` | boolean | default `false` | Non-strict state-dict loading accepts extra training-only keys | Yes |

All top-level video-output fields match `ema_vfi_small_2x.yaml`: `input_path`, `output_path`, `limit_pairs`, `output_fps_multiplier`, `codec`, `container`, `pix_fmt`, `frame_format`, `encoder_options_by_codec`, `encoder_options`, and `mlflow`.

Command:

```bash
uv run python -m video_interpolation.cli rife infer-video \
  --config configs/inference/practical_rife_v4_25_2x.yaml \
  --input raw_data/tmp_test/Dora.mp4 \
  --output /tmp/dora_rife_2x.mp4 \
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
- Practical-RIFE checkpoint loading fails because the checkpoint directory is wrong or does not contain `flownet.pkl`.
