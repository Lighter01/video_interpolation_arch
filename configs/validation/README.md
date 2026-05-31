# Stage 1 Validation Configs

Validation YAML files define candidate checkpoint evaluation and approval rules.

## `ema_vfi_small_candidate.yaml`

Purpose: evaluate an EMA-VFI-small checkpoint candidate on `test_all.csv`, write metrics and visual prediction sets, and produce an approval/rejection report.

Fields:

| Field | Type | Allowed/default | Effect | Rerun needed |
| --- | --- | --- | --- | --- |
| `candidate_id` | string | example `ema_vfi_small_candidate` | Used in metrics, report, and prediction folder names | Yes |
| `dataset_version_id` | string | example `stage1_default` | Provenance in the report and MLflow params | Yes |
| `test_manifest_path` | path string | default `dataset_versions/stage1_default/test_all.csv` | Test-only manifest for candidate validation | Yes |
| `output_dir` | path string | default `outputs/candidate_validation/ema_vfi_small/stage1_default` | Receives metrics, report, and sample predictions | Rerun to write elsewhere |
| `model` | mapping | same fields as `configs/models/ema_vfi_small.yaml` | Candidate checkpoint and adapter settings | Yes |
| `thresholds.min_psnr_mean` | float or `null` | example `25.0` | Rejects candidate when global PSNR mean is below this value | Yes |
| `thresholds.min_ssim_mean` | float or `null` | example `0.80` | Rejects candidate when global SSIM mean is below this value | Yes |
| `thresholds.max_lpips_mean` | float or `null` | default `null` | Rejects candidate when global LPIPS mean is above this value | Yes |
| `limit_samples` | integer or `null` | default `null`; positive integer for smoke runs | Caps evaluated test samples | Yes |
| `compute_lpips` | boolean | default `true` | Computes LPIPS when available; use `--no-lpips` for fast smoke runs | Yes |
| `lpips_device` | string | default `cpu`; use `cuda` only when available | LPIPS model device | Yes |
| `lpips_net` | string | default `alex` | LPIPS backbone | Yes |
| `save_predictions` | boolean | default `true` | Writes visual prediction sets | Rerun |
| `prediction_limit` | integer | default `16` | Max saved visual samples | Rerun |
| `mlflow.*` | mapping | see existing MLflow configs | Logs params, metrics, report, CSVs, manifest/config, and sample predictions | Rerun |

Command:

```bash
uv run python -m video_interpolation.cli ema validate-candidate \
  --config configs/validation/ema_vfi_small_candidate.yaml \
  --manifest dataset_versions/stage1_default/test_all.csv \
  --output-dir /tmp/ema_candidate_validation_smoke \
  --limit-samples 1 \
  --no-lpips \
  --disable-mlflow
```

Side effects and outputs:

- Writes `metrics.csv`, `metrics_summary.csv`, and `candidate_validation_report.json`.
- Writes optional visual comparison folders under `sample_predictions/<candidate_id>/<sample_id>/`.
- Each saved sample contains `im1.png`, `im2_gt.png`, `im2_generated.png`, and `im3.png`.
- Does not modify datasets, manifests, model repositories, or model weights.
- Requires CUDA for the current EMA adapter.

Inspect:

```bash
cat /tmp/ema_candidate_validation_smoke/candidate_validation_report.json
head -5 /tmp/ema_candidate_validation_smoke/metrics.csv
find /tmp/ema_candidate_validation_smoke/sample_predictions -type f | head
```

Common failures:

- CUDA is unavailable.
- The candidate checkpoint path is wrong.
- The manifest is not `test_all.csv` or contains missing frame paths.
- MLflow logging is enabled but the tracking server is not running.

## `amt_s_candidate.yaml`

Purpose: evaluate an AMT-S checkpoint candidate on `test_all.csv`, write metrics and visual prediction sets, and produce an approval/rejection report. This also serves as the AMT-S eval-only smoke workflow for Milestone 8.

AMT-specific fields:

| Field | Type | Allowed/default | Effect | Rerun needed |
| --- | --- | --- | --- | --- |
| `candidate_id` | string | example `amt_s_candidate` | Used in metrics, report, and prediction folder names | Yes |
| `dataset_version_id` | string | example `stage1_default` | Provenance in the report and MLflow params | Yes |
| `test_manifest_path` | path string | default `dataset_versions/stage1_default/test_all.csv` | Test-only manifest for AMT-S validation | Yes |
| `output_dir` | path string | default `outputs/candidate_validation/amt_s/stage1_default` | Receives metrics, report, and sample predictions | Rerun to write elsewhere |
| `model` | mapping | same fields as `configs/models/amt_s.yaml` | AMT-S architecture, checkpoint, and device settings | Yes |
| `thresholds.*` | floats or `null` | same meaning as EMA candidate validation | Controls approval/rejection based on global aggregate metrics | Yes |
| `limit_samples` | integer or `null` | default `null`; positive integer for smoke runs | Caps evaluated test samples | Yes |
| `compute_lpips` | boolean | default `true`; use `--no-lpips` for smoke runs | Computes LPIPS when available | Yes |
| `save_predictions` | boolean | default `true` | Writes visual prediction sets | Rerun |
| `prediction_limit` | integer | default `16` | Max saved visual samples | Rerun |
| `mlflow.*` | mapping | see existing MLflow configs | Logs params, metrics, report, CSVs, manifest/config, and sample predictions | Rerun |

Command:

```bash
uv run python -m video_interpolation.cli amt validate-candidate \
  --config configs/validation/amt_s_candidate.yaml \
  --manifest dataset_versions/stage1_default/test_all.csv \
  --output-dir /tmp/amt_candidate_validation_smoke \
  --limit-samples 1 \
  --no-lpips \
  --disable-mlflow
```

Side effects and outputs:

- Writes `metrics.csv`, `metrics_summary.csv`, and `candidate_validation_report.json`.
- Writes optional visual comparison folders under `sample_predictions/<candidate_id>/<sample_id>/`.
- Each saved sample contains `im1.png`, `im2_gt.png`, `im2_generated.png`, and `im3.png`.
- Does not modify datasets, manifests, model repositories, or model weights.
- Uses the AMT adapter and the same metrics/report contract as EMA candidate validation.

Common failures:

- CUDA is unavailable while `model.device: cuda`.
- The AMT-S checkpoint or upstream config path is wrong.
- The manifest is not `test_all.csv` or contains missing frame paths.
- MLflow logging is enabled but the tracking server is not running.

## `practical_rife_v4_25_candidate.yaml`

Purpose: evaluate Practical-RIFE v4.25 on `test_all.csv`, write metrics and visual prediction sets, and produce an approval/rejection report. This also serves as the Practical-RIFE eval-only smoke workflow for Milestone 9.

Practical-RIFE-specific fields:

| Field | Type | Allowed/default | Effect | Rerun needed |
| --- | --- | --- | --- | --- |
| `candidate_id` | string | example `practical_rife_v4_25_candidate` | Used in metrics, report, and prediction folder names | Yes |
| `dataset_version_id` | string | example `stage1_default` | Provenance in the report and MLflow params | Yes |
| `test_manifest_path` | path string | default `dataset_versions/stage1_default/test_all.csv` | Test-only manifest for Practical-RIFE validation | Yes |
| `output_dir` | path string | default `outputs/candidate_validation/practical_rife_v4_25/stage1_default` | Receives metrics, report, and sample predictions | Rerun to write elsewhere |
| `model` | mapping | same fields as `configs/models/practical_rife_v4_25.yaml` | Practical-RIFE checkpoint, timestep, scale, padding, and device settings | Yes |
| `thresholds.*` | floats or `null` | same meaning as EMA/AMT candidate validation | Controls approval/rejection based on global aggregate metrics | Yes |
| `limit_samples` | integer or `null` | default `null`; positive integer for smoke runs | Caps evaluated test samples | Yes |
| `compute_lpips` | boolean | default `true`; use `--no-lpips` for smoke runs | Computes LPIPS when available | Yes |
| `save_predictions` | boolean | default `true` | Writes visual prediction sets | Rerun |
| `prediction_limit` | integer | default `16` | Max saved visual samples | Rerun |
| `mlflow.*` | mapping | see existing MLflow configs | Logs params, metrics, report, CSVs, manifest/config, and sample predictions | Rerun |

Command:

```bash
uv run python -m video_interpolation.cli rife validate-candidate \
  --config configs/validation/practical_rife_v4_25_candidate.yaml \
  --manifest dataset_versions/stage1_default/test_all.csv \
  --output-dir /tmp/rife_candidate_validation_smoke \
  --limit-samples 1 \
  --no-lpips \
  --disable-mlflow
```

Side effects and outputs:

- Writes `metrics.csv`, `metrics_summary.csv`, and `candidate_validation_report.json`.
- Writes optional visual comparison folders under `sample_predictions/<candidate_id>/<sample_id>/`.
- Each saved sample contains `im1.png`, `im2_gt.png`, `im2_generated.png`, and `im3.png`.
- Does not modify datasets, manifests, model repositories, or model weights.
- Uses the Practical-RIFE adapter and the same metrics/report contract as EMA and AMT candidate validation.

Common failures:

- CUDA is unavailable while `model.device: cuda`.
- The Practical-RIFE checkpoint directory is wrong or missing `flownet.pkl`.
- The manifest is not `test_all.csv` or contains missing frame paths.
- MLflow logging is enabled but the tracking server is not running.
