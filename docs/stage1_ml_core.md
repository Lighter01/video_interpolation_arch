# Stage 1 ML Core

This document tracks the implemented Stage 1 developer workflow. Milestone 1 adds the project skeleton, runtime settings, compact artifact contracts, and an EMA-VFI-small compatibility preflight. Milestone 2 adds raw-video preprocessing and source-level indexing commands. Milestone 3 adds global sequence-index construction and manifest-only dataset version building. Milestones 4 and 5 add triplet manifest loading, metrics, baseline evaluation, and MLflow tracking infrastructure.

## Runtime Settings

Runtime values belong in `.env` and are loaded with `pydantic-settings`.

Copy `.env.example` to `.env` and adjust paths if needed:

```bash
DATASET_ROOT=datasets
MODEL_REPOS_ROOT=model_repos
MODEL_WEIGHTS_ROOT=model_weights
MLFLOW_TRACKING_URI=http://localhost:5000
```

These values are machine-local roots and endpoints. Pipeline parameters such as preprocessing options, dataset split ratios, model hyperparameters, and validation thresholds will live in YAML configs added by later milestones.

For MLflow infrastructure, also set the `MLFLOW_POSTGRES_*`, `MINIO_*`, `MLFLOW_ARTIFACT_BUCKET`, and `AWS_*` values shown in `.env.example`. Replace placeholder secret values before starting Docker Compose services.

## CLI

Show resolved settings:

```bash
uv run python -m video_interpolation.cli show-settings
```

Run the early EMA-VFI-small preflight:

```bash
uv run python -m video_interpolation.cli ema-preflight
```

The preflight checks the EMA-VFI repository path, the `ours_small.pkl` checkpoint path, Torch import, EMA-VFI imports, and whether the current runtime can initialize/load the upstream model. If CUDA is unavailable, the command reports a clear blocked status because the upstream EMA-VFI `Trainer.Model` currently hardcodes CUDA setup.

### `ema adapter-check`

Purpose: check EMA-VFI-small adapter prerequisites without loading a full inference or training workflow.

Syntax:

```bash
uv run python -m video_interpolation.cli ema adapter-check [--config PATH] [--fail-on-blocked]
```

Inputs:

- Adapter YAML, default `configs/models/ema_vfi_small.yaml`.
- EMA repository under `MODEL_REPOS_ROOT`, default `model_repos/EMA-VFI`.
- EMA checkpoint from the adapter config, default `MODEL_WEIGHTS_ROOT/EMA-VFI/ours_small.pkl`.
- CUDA runtime for a fully usable adapter status; without CUDA the check reports `blocked`.

Flags:

- `--config PATH`: adapter config path. Default: `configs/models/ema_vfi_small.yaml`.
- `--fail-on-blocked`: exits with a non-zero status when the adapter is blocked, for example when CUDA is unavailable. Default: disabled.

Side effects and outputs:

- No files are written.
- Prints repository path, checkpoint path, Torch version, CUDA status, and EMA import status.
- In the Codex sandbox this usually reports `blocked` because CUDA is unavailable; the user's real WSL CUDA environment has already passed `ema-preflight`.

### `ema infer-video`

Purpose: run local 2x video inference with EMA-VFI-small by inserting one generated frame between every neighboring input-frame pair.

Syntax:

```bash
uv run python -m video_interpolation.cli ema infer-video [--config PATH] [--input PATH] [--output PATH] [--limit-pairs N] [--checkpoint PATH] [--disable-mlflow]
```

Inputs:

- Inference YAML, default `configs/inference/ema_vfi_small_2x.yaml`.
- Input video from config or `--input`. Config default: `raw_data/tmp_test/Dora.mp4`.
- EMA repository under `MODEL_REPOS_ROOT`, default `model_repos/EMA-VFI`.
- EMA checkpoint from config or `--checkpoint`. Config default: `MODEL_WEIGHTS_ROOT/EMA-VFI/ours_small.pkl`.
- CUDA runtime for model loading and inference.
- MLflow tracking server when MLflow logging is enabled.

Flags:

- `--config PATH`: inference config path. Default: `configs/inference/ema_vfi_small_2x.yaml`.
- `--input PATH`: overrides the input video path from the config. If omitted, uses config `input_path`.
- `--output PATH`: overrides the output video path from the config. If omitted, uses config `output_path`, default `outputs/inference/ema_vfi_small/dora_2x.mp4`.
- `--limit-pairs N`: processes only the first `N` neighboring frame pairs. Default: config `limit_pairs`, currently `null`, meaning process all readable pairs.
- `--checkpoint PATH`: overrides the EMA checkpoint path. Relative paths resolve under `MODEL_WEIGHTS_ROOT`; default `EMA-VFI/ours_small.pkl`.
- `--disable-mlflow`: skips MLflow logging for local smoke runs. Default: disabled, so config `mlflow.enabled` controls logging.

Safe smoke example:

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
- Interleaves original and generated frames as `left, generated_middle, right, ...`.
- Output FPS defaults to `input_fps * 2.0`.
- Logs params, counts, output video, and config to MLflow unless disabled.
- Requires CUDA for the current EMA adapter.

Inspect success:

```bash
ls -lh /tmp/dora_ema_2x.mp4
ffprobe /tmp/dora_ema_2x.mp4
```

### `ema finetune`

Purpose: fine-tune EMA-VFI-small from local pretrained weights using `train_all.csv` and `val_all.csv`.

Syntax:

```bash
uv run python -m video_interpolation.cli ema finetune [--config PATH] [--output-dir PATH] [--max-steps N] [--limit-train-samples N] [--limit-val-samples N] [--checkpoint PATH] [--disable-mlflow]
```

Inputs:

- Training YAML, default `configs/training/ema_vfi_small_finetune.yaml`.
- Training manifest from config `train_manifest_path`. Config default: `dataset_versions/stage1_default/train_all.csv`.
- Validation manifest from config `val_manifest_path`. Config default: `dataset_versions/stage1_default/val_all.csv`.
- Frame files resolved as `DATASET_ROOT + relative_path`.
- EMA repository under `MODEL_REPOS_ROOT`, default `model_repos/EMA-VFI`.
- Starting checkpoint from config or `--checkpoint`. Config default: `MODEL_WEIGHTS_ROOT/EMA-VFI/ours_small.pkl`.
- CUDA runtime for model loading, forward/backward pass, and checkpoint reload.
- MLflow tracking server when MLflow logging is enabled.

Flags:

- `--config PATH`: training config path. Default: `configs/training/ema_vfi_small_finetune.yaml`.
- `--output-dir PATH`: overrides where checkpoints are written. If omitted, uses config `output_dir`, default `outputs/training/ema_vfi_small/stage1_default`.
- `--max-steps N`: overrides the optimization-step cap. If omitted, uses config `max_steps`, default `1000`.
- `--limit-train-samples N`: overrides the number of train manifest rows loaded. If omitted, uses config `limit_train_samples`, default `null`, meaning all train rows.
- `--limit-val-samples N`: overrides the number of validation manifest rows loaded. If omitted, uses config `limit_val_samples`, default `64`.
- `--checkpoint PATH`: overrides the starting checkpoint. Relative paths resolve under `MODEL_WEIGHTS_ROOT`; default `EMA-VFI/ours_small.pkl`.
- `--disable-mlflow`: skips MLflow logging for local smoke runs. Default: disabled, so config `mlflow.enabled` controls logging.

Safe smoke example:

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

- Loads frames through `DATASET_ROOT + relative_path`.
- Uses `train_manifest_path` and `val_manifest_path` only.
- Does not use `test_all.csv`; test data is reserved for candidate validation.
- Writes `best_checkpoint.pkl` and `last_checkpoint.pkl` under `output_dir` during fine-tuning.
- Logs params, train losses, validation PSNR/SSIM, manifests, config, and checkpoints to MLflow unless disabled.
- Requires CUDA for the current EMA adapter.

### `ema validate-candidate`

Purpose: evaluate an EMA-VFI-small checkpoint candidate on `test_all.csv`, write metrics/artifacts, and approve or reject based on configured thresholds.

Syntax:

```bash
uv run python -m video_interpolation.cli ema validate-candidate [--config PATH] [--manifest PATH] [--output-dir PATH] [--limit-samples N] [--checkpoint PATH] [--no-lpips] [--disable-mlflow]
```

Inputs:

- Candidate validation YAML, default `configs/validation/ema_vfi_small_candidate.yaml`.
- Test manifest from config or `--manifest`. Config default: `dataset_versions/stage1_default/test_all.csv`.
- Frame files resolved as `DATASET_ROOT + relative_path`.
- Candidate checkpoint from config or `--checkpoint`. Config default: `MODEL_WEIGHTS_ROOT/EMA-VFI/ours_small.pkl`.
- Thresholds from config, default `min_psnr_mean: 25.0`, `min_ssim_mean: 0.80`, `max_lpips_mean: null`.
- CUDA runtime for model loading and inference.
- MLflow tracking server when MLflow logging is enabled.

Flags:

- `--config PATH`: validation config path. Default: `configs/validation/ema_vfi_small_candidate.yaml`.
- `--manifest PATH`: overrides the test manifest path. If omitted, uses config `test_manifest_path`, default `dataset_versions/stage1_default/test_all.csv`.
- `--output-dir PATH`: overrides where metrics, report, and sample predictions are written. If omitted, uses config `output_dir`, default `outputs/candidate_validation/ema_vfi_small/stage1_default`.
- `--limit-samples N`: caps evaluated manifest rows. If omitted, uses config `limit_samples`, default `null`, meaning all test rows.
- `--checkpoint PATH`: overrides the candidate checkpoint. Relative paths resolve under `MODEL_WEIGHTS_ROOT`; default `EMA-VFI/ours_small.pkl`.
- `--no-lpips`: disables LPIPS for faster CPU smoke checks. Default: disabled, so config `compute_lpips: true` computes LPIPS.
- `--disable-mlflow`: skips MLflow logging for local smoke runs. Default: disabled, so config `mlflow.enabled` controls logging.

Safe smoke example:

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
- Logs params, metrics, report, CSVs, manifest/config, and sample predictions to MLflow unless disabled.
- Requires CUDA for the current EMA adapter.

Inspect success:

```bash
cat /tmp/ema_candidate_validation_smoke/candidate_validation_report.json
head -5 /tmp/ema_candidate_validation_smoke/metrics.csv
find /tmp/ema_candidate_validation_smoke/sample_predictions -type f | head
```

### `data index-vimeo-triplets`

Purpose: create a source-level `sequence_index.csv` from the existing Vimeo triplet source.

Syntax:

```bash
uv run python -m video_interpolation.cli data index-vimeo-triplets [--config PATH] [--output PATH] [--limit N]
```

Inputs:

- `--config`: YAML config, default `configs/data/index_vimeo_triplet.yaml`.
- Existing source directory under `DATASET_ROOT`, default `datasets/sources/vimeo_triplet/`.
- Vimeo list rows such as `00001/0001`, each pointing to a directory with `im1.png`, `im2.png`, and `im3.png`.

Flags:

- `--output PATH`: optional CSV path. If omitted, writes `DATASET_ROOT/sources/vimeo_triplet/sequence_index.csv`.
- `--limit N`: positive integer cap for smoke runs. If omitted, all configured list rows are indexed.

Build a source-level index for existing Vimeo triplets:

```bash
uv run python -m video_interpolation.cli data index-vimeo-triplets
```

For a safe smoke run that does not modify the dataset tree, write a small index to `/tmp`:

```bash
uv run python -m video_interpolation.cli data index-vimeo-triplets --limit 3 --output /tmp/vimeo_sequence_index.csv
```

Side effects and outputs:

- Reads the configured split lists and image metadata.
- Does not copy or modify image frames.
- Writes a CSV following the `sequence_index.csv` contract, plus an `original_split` column.
- Prints a Rich progress bar and final summary table with source dataset, records read/written, invalid/skipped records, and output path.

Inspect success:

```bash
head -5 /tmp/vimeo_sequence_index.csv
wc -l /tmp/vimeo_sequence_index.csv
```

Common failures:

- Missing `sequences/`, train list, or test list under the configured source directory.
- Missing triplet frame files when `validate_images: true`.
- Absolute, URI, or parent-directory paths in split-list rows.

### `data build-global-index`

Purpose: combine one or more source-level `sequence_index.csv` files into `global_sequence_index.csv`.

Syntax:

```bash
uv run python -m video_interpolation.cli data build-global-index [--config PATH] [--output PATH] [--limit N] [--source-index PATH ...]
```

Inputs:

- `--config`: YAML config, default `configs/data/global_index.yaml`.
- Source indexes discovered from `source_index_globs` and/or explicit `--source-index` values.
- Source indexes must use relative `relative_sequence_dir` values.

Flags:

- `--output PATH`: optional output CSV path. If omitted, writes `DATASET_ROOT/global_sequence_index.csv`.
- `--limit N`: positive row cap after deterministic sorting, useful for smoke runs.
- `--source-index PATH`: explicit source index path; repeat the flag to combine several files.

Build the default global index:

```bash
uv run python -m video_interpolation.cli data build-global-index
```

Safe smoke example:

```bash
uv run python -m video_interpolation.cli data build-global-index \
  --source-index sources/tmp_test/sequence_index.csv \
  --output /tmp/stage1_dataset_versions/global_sequence_index.csv
```

Side effects and outputs:

- Reads source index CSVs only; it does not copy or edit frames.
- Writes a CSV following the `global_sequence_index.csv` contract.
- Preserves extra columns such as Vimeo `original_split`.
- Prints a Rich progress bar and final summary with source indexes read, records read/written, filtered records, and output path.

Inspect success:

```bash
head -5 /tmp/stage1_dataset_versions/global_sequence_index.csv
wc -l /tmp/stage1_dataset_versions/global_sequence_index.csv
```

### `data build-dataset-version`

Purpose: create logical train/val/test triplet manifests without moving image files.

Syntax:

```bash
uv run python -m video_interpolation.cli data build-dataset-version [--config PATH] [--dataset-version-id ID] [--source-index PATH] [--output-root PATH] [--limit-sequences N]
```

Inputs:

- `--config`: YAML config, default `configs/data/dataset_version.yaml`.
- Global sequence index CSV, default `DATASET_ROOT/global_sequence_index.csv`.
- Frame files under `DATASET_ROOT`; Vimeo `im1.png` naming and preprocessing `frame_000.png` naming are both supported.

Flags:

- `--dataset-version-id ID`: override output version directory name.
- `--source-index PATH`: override the global sequence index path.
- `--output-root PATH`: override the dataset version root. Relative paths resolve from the repo root.
- `--limit-sequences N`: cap source sequence rows for smoke runs.

Build a default dataset version:

```bash
uv run python -m video_interpolation.cli data build-dataset-version
```

Safe smoke example using the global index above:

```bash
uv run python -m video_interpolation.cli data build-dataset-version \
  --source-index /tmp/stage1_dataset_versions/global_sequence_index.csv \
  --output-root /tmp/stage1_dataset_versions \
  --dataset-version-id smoke_version \
  --limit-sequences 10
```

Side effects and outputs:

- Creates `dataset_versions/<dataset_version_id>/` by default, or the overridden output root.
- Writes `train_all.csv`, `val_all.csv`, `test_all.csv`, and `dataset_config.yaml`.
- Does not move or copy image files.
- Splits by `source_video_id`, so records from the same source video do not leak across train/val/test.
- Writes frame paths relative to `DATASET_ROOT`.
- For tiny smoke versions, if the train split contains only old or only new pool rows, mixing keeps those available rows instead of emitting an empty train manifest.
- Prints a Rich progress bar and final summary with source records, generated samples, split counts, source-video counts, and output paths.

Inspect success:

```bash
find /tmp/stage1_dataset_versions/smoke_version -maxdepth 1 -type f -print
head -5 /tmp/stage1_dataset_versions/smoke_version/train_all.csv
cat /tmp/stage1_dataset_versions/smoke_version/dataset_config.yaml
```

Common failures:

- No rows remain after source-group filtering.
- `wide_triplet` is requested for an even-length sequence.
- `validate_frame_paths: true` and frames are missing under `DATASET_ROOT`.
- A global index or manifest path contains an absolute path, URI, or parent-directory reference.

### `data inspect-triplet-manifest`

Purpose: load a triplet manifest through `UniversalTripletDataset` and print tensor shapes plus source metadata for a few samples.

Syntax:

```bash
uv run python -m video_interpolation.cli data inspect-triplet-manifest --manifest PATH [--limit-samples N]
```

Inputs:

- `--manifest`: required `train_all.csv`, `val_all.csv`, or `test_all.csv`.
- Frame paths inside the manifest are resolved as `DATASET_ROOT + relative_path`.

Flags:

- `--limit-samples N`: positive integer, default `3`.

Example:

```bash
uv run python -m video_interpolation.cli data inspect-triplet-manifest \
  --manifest dataset_versions/stage1_default/test_all.csv \
  --limit-samples 2
```

Side effects and outputs:

- No files are written.
- The command opens the referenced PNG frames and prints tensor shapes as `C x H x W`.

Common failures:

- The manifest is missing required columns.
- A frame path is invalid or points outside `DATASET_ROOT`.
- A referenced PNG file is missing.

### `baseline evaluate`

Purpose: evaluate frame duplication, frame blending, and Farneback optical-flow baselines on a triplet manifest.

Syntax:

```bash
uv run python -m video_interpolation.cli baseline evaluate [--config PATH] [--manifest PATH] [--output-dir PATH] [--limit-samples N] [--baseline NAME ...] [--no-lpips] [--no-predictions] [--disable-mlflow]
```

Inputs:

- `--config`: YAML config, default `configs/baselines/baseline_eval.yaml`.
- `--manifest`: optional override for a triplet manifest, usually `test_all.csv`.
- Frame paths are loaded through `UniversalTripletDataset` using `DATASET_ROOT`.

Flags:

- `--output-dir PATH`: override where metrics and sample predictions are written.
- `--limit-samples N`: positive sample cap for smoke runs.
- `--baseline NAME`: repeatable; allowed values are `duplicate_left`, `blend`, and `farneback`.
- `--no-lpips`: disables LPIPS and leaves the LPIPS CSV field empty for fast CPU smoke runs.
- `--no-predictions`: skips sample visual comparison export.
- `--disable-mlflow`: skips MLflow logging for tiny local tests.

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

Side effects and outputs:

- Writes `metrics.csv` and `metrics_summary.csv` under `output_dir`.
- Writes optional visual comparison samples under `sample_predictions/<baseline>/<sample_id>/`.
- Each saved sample directory contains `im1.png` left frame, `im2_gt.png` ground-truth middle frame, `im2_generated.png` generated middle frame, and `im3.png` right frame.
- `metrics.csv` stores `prediction_path` as the relative saved sample directory when one was written.
- Logs params, aggregate metrics, metrics CSVs, config, manifest, and sample predictions to MLflow when enabled.
- Does not modify datasets, manifests, or source image files.
- Prints a Rich progress bar and final summary table.

Inspect success:

```bash
head -5 /tmp/stage1_baseline_smoke/metrics.csv
cat /tmp/stage1_baseline_smoke/metrics_summary.csv
find /tmp/stage1_baseline_smoke/sample_predictions -maxdepth 3 -type f | head
```

Common failures:

- LPIPS is enabled on `cuda` when CUDA is unavailable.
- MLflow logging is enabled but the tracking server is not running.
- Manifest paths or frame paths are invalid.

### `mlflow smoke-log`

Purpose: verify that local Python code can log a minimal run to `MLFLOW_TRACKING_URI`.

Syntax:

```bash
uv run python -m video_interpolation.cli mlflow smoke-log [--experiment-name NAME] [--run-name NAME] [--artifact-path PATH]
```

Example:

```bash
uv run python -m video_interpolation.cli mlflow smoke-log
```

Side effects and outputs:

- Creates a small local artifact file, default `/tmp/stage1_mlflow_smoke.txt`.
- Logs one param, one metric, and the artifact to MLflow.
- Prints the tracking URI, experiment, and run id on success.

Common failures:

- `MLFLOW_TRACKING_URI` points to a server that is not running.
- MLflow server cannot reach PostgreSQL or MinIO.
- Required MLflow/MinIO/PostgreSQL variables are missing from `.env`.

### `data preprocess-videos`

Purpose: sample raw videos into PNG frame sequences and create a source-level `sequence_index.csv`.

Syntax:

```bash
uv run python -m video_interpolation.cli data preprocess-videos [--config PATH] [--limit-videos N] [--only-video NAME] [--video-glob GLOB] [--max-duration-sec SEC] [--max-frames N] [--debug-progress]
```

Inputs:

- `--config`: YAML config, default `configs/data/preprocess_anime.yaml`.
- Raw video files under the configured `raw_input_dir`.
- Supported extensions: `.mkv`, `.mov`, `.mp4`, `.webm`.

Flags:

- `--limit-videos N`: positive integer cap for smoke runs. If omitted, all discovered videos are considered.
- `--only-video NAME`: process one video by filename or path relative to `raw_input_dir`.
- `--video-glob GLOB`: process videos whose filename or relative path matches a glob.
- `--max-duration-sec SEC`: limit each video to the first N seconds for debugging.
- `--max-frames N`: limit each video to the first N frames for debugging.
- `--debug-progress`: print timed per-video diagnostics for metadata, scene detection, candidate sampling, selected-frame decoding, SSIM filtering, PNG writing, and per-video summary.

Preprocess raw anime videos into PNG sequences and a source `sequence_index.csv`:

```bash
uv run python -m video_interpolation.cli data preprocess-videos --config configs/data/preprocess_anime.yaml
```

Use `--limit-videos` for small local checks before running over the full raw video set. Preprocessing writes under `DATASET_ROOT/sources/anime/` and keeps index paths relative to `DATASET_ROOT`.

`--limit-videos` limits the number of files only. For long episodes, combine it with `--max-duration-sec`, `--max-frames`, `--only-video`, or `--video-glob`.

Diagnostic examples:

```bash
uv run python -m video_interpolation.cli data preprocess-videos --config configs/data/preprocess_test.yaml --limit-videos 1 --debug-progress
uv run python -m video_interpolation.cli data preprocess-videos --config configs/data/preprocess_test.yaml --only-video Dora.mp4 --debug-progress
uv run python -m video_interpolation.cli data preprocess-videos --config configs/data/preprocess_test.yaml --only-video one_piece_test_1m.mkv --max-duration-sec 10 --debug-progress
```

Side effects and outputs:

- Creates `DATASET_ROOT/sources/anime/sequences/...` as needed.
- Writes PNG frames named `frame_000.png`, `frame_001.png`, and so on.
- Rewrites `DATASET_ROOT/sources/anime/sequence_index.csv` after processing.
- Existing generated sequence directories with the same names may be overwritten. For experiments, use a separate `output_source_dir` in a copied config.
- Prints a Rich video progress bar, a per-video sequence progress bar, per-video status messages, and a final summary table with discovered/processed/skipped/failed videos, scenes detected, sequences written, static triplets rejected, output directory, and index path.
- With `--debug-progress`, prints container format, video/audio codecs, dimensions, FPS, frame-count source, whether a frame limit was applied, elapsed time for major steps, selected-frame decoding strategy, recovered/missing frame counts, and per-video totals.
- Static triplet filtering rejects a candidate when either adjacent pair is too similar: `SSIM(im1, im2) >= threshold` or `SSIM(im2, im3) >= threshold`.
- Selected-frame decoding uses `decode_strategy` from the config. `auto` uses segment-based seek/decode for later selected frames and falls back to sequential decoding if seeking misses frames.

Inspect success:

```bash
find datasets/sources/anime/sequences -maxdepth 3 -type f | head
head -5 datasets/sources/anime/sequence_index.csv
```

Common failures:

- Missing raw input directory.
- Invalid resize value; use `null` or `WIDTHxHEIGHT`.
- Invalid frame-step settings.
- Video metadata or codec read errors.
- MKV files may lack `stream.frames`; the command uses duration × FPS estimates when available instead of fully decoding the video just to count frames.
- FFmpeg/PyAV may print audio-stream warnings during container probing. The preprocessing path decodes only the selected video stream, so audio warnings are non-blocking unless the command records a per-video failure.

## Artifact Contracts

Milestone 1 defines compact contracts for Stage 1 artifacts:

- `sequence_index.csv`
- `global_sequence_index.csv`
- `train_all.csv`, `val_all.csv`, `test_all.csv`
- `dataset_config.yaml`
- candidate validation report JSON
- metrics CSV

These contracts define required columns/fields and relative-path rules. They are support code for later preprocessing, indexing, dataset versioning, metrics, and validation work.

## Data Configs

Implemented data configs:

- `configs/data/index_vimeo_triplet.yaml` for indexing existing Vimeo triplets without copying frames;
- `configs/data/preprocess_anime.yaml` for sampling raw anime videos into PNG sequences;
- `configs/data/preprocess_test.yaml` for MP4/MKV diagnostics and smoke preprocessing;
- `configs/data/global_index.yaml` for combining source-level sequence indexes;
- `configs/data/dataset_version.yaml` for building manifest-only dataset versions.

These YAML files contain pipeline parameters only. Runtime roots such as `DATASET_ROOT` still come from `.env`.

Detailed field documentation lives in `configs/data/README.md`.

## Baseline Configs

Implemented baseline configs:

- `configs/baselines/baseline_eval.yaml` for duplication, blending, and Farneback baseline evaluation on a triplet manifest.

Detailed field documentation lives in `configs/baselines/README.md`.

## EMA-VFI-small Configs

Implemented EMA configs:

- `configs/models/ema_vfi_small.yaml` for adapter, checkpoint, device, padding, and TTA settings;
- `configs/inference/ema_vfi_small_2x.yaml` for local 2x video inference;
- `configs/training/ema_vfi_small_finetune.yaml` for fine-tuning and eval-only runner parameters;
- `configs/validation/ema_vfi_small_candidate.yaml` for candidate validation thresholds, outputs, prediction samples, and MLflow behavior.

Detailed field documentation lives in `configs/models/README.md`, `configs/inference/README.md`, `configs/training/README.md`, and `configs/validation/README.md`.

## MLflow Infrastructure

Stage 1 MLflow infrastructure lives under `infra/mlflow/`.

Start services after setting `.env` values:

```bash
docker compose --env-file .env -f infra/mlflow/docker-compose.yml up -d --build
```

Stop services:

```bash
docker compose --env-file .env -f infra/mlflow/docker-compose.yml down
```

The stack provides:

- MLflow Tracking Server on `http://localhost:5000`;
- PostgreSQL metadata storage;
- MinIO artifact storage and bucket initialization.

If the stack is not running, use `--disable-mlflow` only for tiny local smoke checks. Real baseline, training, evaluation, and validation runs should log to MLflow.
