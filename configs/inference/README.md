# Stage 1 Inference Configs

Inference YAML files define local developer inference runs. Runtime roots and MLflow endpoint still come from `.env`.

## `ema_vfi_small_2x.yaml`

Purpose: run local 2x video interpolation with EMA-VFI-small by inserting one generated middle frame between each neighboring input-frame pair.

Fields:

| Field | Type | Allowed/default | Effect | Rerun needed |
| --- | --- | --- | --- | --- |
| `input_path` | path string | example `raw_data/tmp_test/Dora.mp4`; relative to repo root unless absolute | Input video to decode | Yes |
| `output_path` | path string | example `outputs/inference/ema_vfi_small/dora_2x.mp4` | Output video written by OpenCV VideoWriter | Rerun to write elsewhere |
| `model` | mapping | same fields as `configs/models/ema_vfi_small.yaml` | Adapter and checkpoint settings | Yes |
| `limit_pairs` | integer or `null` | default `null`; positive integer for smoke runs | Caps the number of neighboring frame pairs interpolated | Yes |
| `output_fps_multiplier` | float | default `2.0` | Output FPS is input FPS multiplied by this value | Yes |
| `codec` | string | default `mp4v` | FourCC used by OpenCV writer | Yes |
| `mlflow.enabled` | boolean | default `true` | Logs params, counts, output video, and config to MLflow | Rerun |
| `mlflow.experiment_name` | string | default `stage1-ema-inference` | MLflow experiment | Rerun |
| `mlflow.run_name` | string or `null` | optional run name | MLflow run name | Rerun |
| `mlflow.tags` | mapping | string tags | Extra MLflow tags | Rerun |

Command:

```bash
uv run python -m video_interpolation.cli ema infer-video \
  --config configs/inference/ema_vfi_small_2x.yaml \
  --input raw_data/tmp_test/Dora.mp4 \
  --output /tmp/dora_ema_2x.mp4 \
  --limit-pairs 2 \
  --disable-mlflow
```

Side effects and outputs:

- Writes a new video to `output_path`.
- Original and generated frames are interleaved as `left, generated_middle, right, ...`.
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
- The output codec is unavailable in the local OpenCV/FFmpeg build.
- MLflow logging is enabled but the tracking server is not running.
