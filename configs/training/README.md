# Stage 1 Training Configs

Training YAML files define model fine-tuning and eval-only runner parameters. Runtime roots and MLflow endpoint still come from `.env`.

## `ema_vfi_small_finetune.yaml`

Purpose: fine-tune EMA-VFI-small from `MODEL_WEIGHTS_ROOT/EMA-VFI/ours_small.pkl` using `train_all.csv` and validate with `val_all.csv`.

Fields:

| Field | Type | Allowed/default | Effect | Rerun needed |
| --- | --- | --- | --- | --- |
| `train_manifest_path` | path string | default `dataset_versions/stage1_default/train_all.csv` | Training samples. Do not point this at `test_all.csv` | Yes |
| `val_manifest_path` | path string | default `dataset_versions/stage1_default/val_all.csv` | Validation samples used during training | Yes |
| `output_dir` | path string | default `outputs/training/ema_vfi_small/stage1_default` | Receives `best_checkpoint.pkl` and `last_checkpoint.pkl` | Rerun to write elsewhere |
| `model` | mapping | same fields as `configs/models/ema_vfi_small.yaml` | Starting checkpoint and adapter settings | Yes |
| `mode` | string | `finetune` or `eval_only`; `scratch_train` is rejected for now | Controls runner behavior | Yes |
| `seed` | integer | default `42` | Seeds Python, NumPy, and Torch RNGs | Yes |
| `batch_size` | integer | default `1` | Batch size for train/validation loaders | Yes |
| `num_workers` | integer | default `0` | DataLoader worker count | Yes |
| `max_epochs` | integer | default `1` | Upper epoch count | Yes |
| `max_steps` | integer or `null` | default/example `1000`; use small values for smoke runs | Caps optimization steps | Yes |
| `learning_rate` | float | default `0.00002` | Passed to upstream EMA `Model.update()` | Yes |
| `validate_every_steps` | integer | default `100` | Validation/checkpoint cadence | Yes |
| `limit_train_samples` | integer or `null` | default `null`; positive integer for smoke runs | Caps train manifest rows loaded | Yes |
| `limit_val_samples` | integer or `null` | default `64` | Caps validation samples for faster feedback | Yes |
| `mlflow.*` | mapping | see existing MLflow configs | Logs params, losses, validation metrics, manifests, config, and checkpoints | Rerun |

Command:

```bash
uv run python -m video_interpolation.cli ema finetune \
  --config configs/training/ema_vfi_small_finetune.yaml \
  --max-steps 1 \
  --limit-train-samples 1 \
  --limit-val-samples 1 \
  --output-dir /tmp/ema_finetune_smoke \
  --disable-mlflow
```

Side effects and outputs:

- Loads frame paths through `DATASET_ROOT + relative_path`.
- Uses only `train_manifest_path` and `val_manifest_path`.
- Saves `best_checkpoint.pkl` and `last_checkpoint.pkl` under `output_dir` during fine-tuning.
- Does not use `test_all.csv`; candidate validation handles test manifests separately.
- Requires CUDA for the current EMA adapter.

Inspect:

```bash
find /tmp/ema_finetune_smoke -maxdepth 1 -type f -print
```

Common failures:

- CUDA is unavailable.
- MLflow logging is enabled but the tracking server is not running.
- A manifest frame path is missing under `DATASET_ROOT`.
- `scratch_train` is selected; Stage 1 currently implements fine-tuning and eval-only for EMA first.
