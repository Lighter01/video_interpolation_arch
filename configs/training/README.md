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

## AMT-S Fine-Tuning Status

AMT-S fine-tuning is not enabled in Milestone 8. Upstream AMT's Vimeo training dataset expects precomputed optical-flow files such as `flow_t0.flo` and `flow_t1.flo`, and `cfgs/AMT-S.yaml` includes a `MultipleFlowLoss` over those flow targets. The current project dataset versions are image-triplet manifests only.

Implemented AMT-S workflows are:

- `uv run python -m video_interpolation.cli amt-preflight`
- `uv run python -m video_interpolation.cli amt adapter-check`
- `uv run python -m video_interpolation.cli amt validate-candidate ...`
- `uv run python -m video_interpolation.cli amt infer-video ...`

Future patch reminder:

- Do not start AMT-S fine-tuning until the project chooses an adaptation style.
- Option A: keep upstream AMT's Vimeo-style training path. Generate `flow_t0.flo` and `flow_t1.flo` under `datasets/sources/vimeo_triplet/flow/<clip>/<sequence>/`, then expose the existing Vimeo layout and split-list files to AMT's upstream training code. This is closest to the authors' training setup, but it bypasses project dataset-version manifests unless extra glue is added.
- Option B: implement a project-native manifest-based AMT training runner. Keep AMT model/loss logic, but load `train_all.csv` and `val_all.csv` through a manifest dataset that also resolves flow paths. This fits the Stage 1 architecture better, but it requires more adapter/training code and an explicit flow-path convention or manifest columns.
- Option C: use a no-flow training policy inspired by AMT's GoPro config, which removes `MultipleFlowLoss`. This avoids flow generation, but changes the training objective and should be approved before implementation.

If training with flows is selected, first make LiteFlowNet flow generation reproducible in the current CUDA/PyTorch environment, ensure its weights are available locally or through an approved download, and generate a small bounded flow subset before attempting full Vimeo flow generation.

## Practical-RIFE Fine-Tuning Status

Practical-RIFE fine-tuning is not enabled in Milestone 9. Upstream Practical-RIFE training currently uses hardcoded `/data` paths, nori/S3-style dataset access, distributed CUDA assumptions, TensorBoard logging, and a training model path that is separate from the shipped `train_log` inference weights. The current Stage 1 Practical-RIFE integration therefore supports eval-only/candidate validation and local video inference first.

Implemented Practical-RIFE workflows are:

- `uv run python -m video_interpolation.cli rife-preflight`
- `uv run python -m video_interpolation.cli rife adapter-check`
- `uv run python -m video_interpolation.cli rife validate-candidate ...`
- `uv run python -m video_interpolation.cli rife infer-video ...`

Future patch reminder:

- Do not start Practical-RIFE fine-tuning until the project chooses an adaptation style.
- Option A: keep more upstream Practical-RIFE training logic and replace only the dataset layer with project manifest loading. This preserves more author logic but still requires removing hardcoded `/data`, nori/S3, distributed, and TensorBoard assumptions.
- Option B: implement a project-native Practical-RIFE training runner around the selected `RIFE_HDv3.Model`/`flownet` update behavior, using `train_all.csv` and `val_all.csv` through `UniversalTripletDataset`. This fits the Stage 1 architecture better, but requires careful loss/optimizer/checkpoint parity review.
- Option C: treat Practical-RIFE as eval/inference-only for Stage 1 and defer fine-tuning until after the EMA route is used for training demonstrations.
