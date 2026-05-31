# Stage 1 Baseline Configs

Baseline YAML files define evaluation parameters. Runtime roots and the MLflow tracking endpoint still come from `.env`.

## `baseline_eval.yaml`

Purpose: evaluate non-neural interpolation baselines on a triplet manifest and write metrics/artifacts.

Fields:

| Field | Type | Allowed/default | Effect | Rerun needed |
| --- | --- | --- | --- | --- |
| `manifest_path` | path string | example `dataset_versions/stage1_default/test_all.csv`; relative to repo root unless absolute | Triplet manifest to evaluate | Yes, rerun evaluation |
| `output_dir` | path string | example `outputs/baselines/stage1_default`; relative to repo root unless absolute | Directory receiving metrics and sample predictions | Rerun command to write elsewhere |
| `baselines` | list of strings | `duplicate_left`, `blend`, `farneback` | Selects baseline predictors | Yes, rerun evaluation |
| `limit_samples` | integer or `null` | default `null`; positive integer for smoke runs | Caps manifest rows evaluated | Yes, rerun evaluation |
| `compute_lpips` | boolean | default `true` | Loads LPIPS and writes LPIPS values; use `--no-lpips` for fast CPU smoke runs | Yes, rerun evaluation |
| `lpips_device` | string | default `cpu`; use `cuda` only when available | Device for LPIPS model | Yes, rerun evaluation |
| `lpips_net` | string | default `alex` | LPIPS backbone name | Yes, rerun evaluation |
| `save_predictions` | boolean | default `true` | Writes visual comparison folders under `sample_predictions/` | Rerun to change artifacts |
| `prediction_limit` | integer | default `8`; `0` writes none | Max visual comparison folders per baseline | Rerun to change artifacts |
| `mlflow.enabled` | boolean | default `true` | Logs params, aggregate metrics, CSVs, manifest/config, and prediction image folders to MLflow | Rerun evaluation |
| `mlflow.experiment_name` | string | default `stage1-baselines` | MLflow experiment | Rerun evaluation |
| `mlflow.run_name` | string or `null` | default `null` | Optional MLflow run name | Rerun evaluation |
| `mlflow.tags` | mapping | string tags | Extra MLflow tags | Rerun evaluation |

Input format:

- `manifest_path` must be a triplet manifest such as `test_all.csv`.
- Frame paths inside the manifest are resolved as `DATASET_ROOT + relative_path`.

Output format:

- `metrics.csv`: one row per evaluated sample and baseline with `prediction_name`, source metadata, PSNR, SSIM, LPIPS, and optional relative prediction path.
- `metrics_summary.csv`: aggregate rows globally, per `source_group`, and per `source_group/source_dataset`.
- `sample_predictions/<baseline>/<sample_id>/`: optional visual comparison examples.
- Each saved sample directory contains `im1.png` left frame, `im2_gt.png` ground-truth middle frame, `im2_generated.png` generated middle frame, and `im3.png` right frame.
- `metrics.csv` stores `prediction_path` as the relative saved sample directory when one was written.

Safe smoke example:

```bash
uv run python -m video_interpolation.cli baseline evaluate \
  --config configs/baselines/baseline_eval.yaml \
  --manifest dataset_versions/stage1_default/test_all.csv \
  --output-dir /tmp/stage1_baseline_smoke \
  --limit-samples 2 \
  --no-lpips \
  --disable-mlflow
```

Inspect:

```bash
head -5 /tmp/stage1_baseline_smoke/metrics.csv
cat /tmp/stage1_baseline_smoke/metrics_summary.csv
find /tmp/stage1_baseline_smoke/sample_predictions -type f
```

Common failures:

- Manifest path is wrong or contains invalid relative frame paths.
- Frame files are missing under `DATASET_ROOT`.
- LPIPS is requested on CUDA when CUDA is unavailable.
- MLflow logging is enabled but the tracking server is not running.

## Baseline Video Inference

Baseline methods are also available through a `ModelAdapter`-compatible wrapper for local video inference.

Command:

```bash
uv run python -m video_interpolation.cli baseline infer-video \
  --config configs/inference/baseline_2x.yaml \
  --baseline blend \
  --input raw_data/tmp_test/Dora.mp4 \
  --output /tmp/dora_blend_2x.mp4 \
  --codec libx264 \
  --limit-pairs 2 \
  --disable-mlflow
```

Supported `--baseline` values:

- `duplicate_left`
- `blend`
- `farneback`

The command uses the same PyAV/FFmpeg writer as model inference, so output videos preserve compatible audio streams and use the same `codec`, `container`, `pix_fmt`, `frame_format`, and encoder option behavior documented in `configs/inference/README.md`.
