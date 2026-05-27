# PROJECT_MAP.md

Compact repository index for coding agents. This file describes the current project structure only; it is an index, not full documentation.

## Repository Root

- `.agent/` — agent instructions, project plans, execution plans, and project map.
- `.env.example` — example runtime environment values for local paths, MLflow URI, and local MLflow infrastructure variables.
- `configs/` — Stage 1 YAML configs for data workflows plus planned models, training, validation, and baselines.
- `datasets/` — local datasets and prepared training data.
- `dataset_versions/` — manifest-only dataset versions with train/val/test split CSVs and dataset config YAML.
- `docs/` — human-facing workflow documentation.
- `infra/` — local infrastructure definitions for Stage 1 services.
- `model_repos/` — external VFI model repositories used as implementation references and integration targets.
- `model_weights/` — pretrained model weights and model-specific runtime files.
- `src/` — local Python package for Stage 1 ML core code.
- `tests/` — focused tests for critical Stage 1 behavior.

## `.agent/`

Agent infrastructure for staged development.

- `AGENTS.md` — global agent instructions and project workflow rules.
- `docs/` — planning protocol, project map, and execution plans.
- `stage_plans/` — human-authored implementation plans for project stages.

## `.agent/docs/`

Agent-facing documentation and development logs.

- `PLANS.md` — ExecPlan protocol and planning workflow.
- `PROJECT_MAP.md` — current repository index.
- `exec-plans/` — agent-authored implementation plans and development logs.

### `.agent/docs/exec-plans/active/`

- `01_ml_core_selected.execplan.md` — selected active ExecPlan for Stage 1 ML Core.

## `.agent/stage_plans/`

Human-authored stage-level implementation plans.

## `raw_data/`

Raw videos, splitted by domains, to preprocess into training triplets.

### `raw_data/anime`

Anime domain video files.

## `datasets/`

Local datasets and prepared training data.

- `global_sequence_index.csv` — combined source sequence index when built by the Stage 1 data workflow.

### `datasets/sources/`

Directory with training data, grouped by source.

### `datasets/sources/vimeo_triplet/`

Vimeo-90K Triplet dataset in the original triplet structure.

- `sequences/` — triplet frame directories.
- `tri_trainlist.txt` — original Vimeo train split list.
- `tri_testlist.txt` — original Vimeo test split list.
- `readme.txt` — dataset notes.

### `datasets/sources/SNU_FILM/`

SNU-FILM triplet evaluation-only dataset.

- `test/GOPRO_test/` — GOPRO dataset's subset of image sequences for VFI evaluation.
- `test/YouTube_test/` — YouTube dataset's subset of image sequences for VFI evaluation.
- `test-*.txt` — SNU-FILM original test split lists, characterised by the complexity of scenes.

### `datasets/sources/ucf101_interp_ours`

UCF101 triplet evaluation-only dataset. Data inside is grouped by the video source.

## `dataset_versions/`

Manifest-only train/val/test dataset versions. Image files remain under `datasets/sources/...` and are referenced by relative paths.

## `model_repos/`

External VFI model repositories copied into the project.

### `model_repos/AMT/`

AMT model repository.

- `train.py` — upstream training entrypoint.
- `cfgs/` — model and training configs.
- `datasets/` — upstream dataset code.
- `networks/` — model definitions.
- `trainers/` — training logic.
- `losses/` — loss functions.
- `metrics/` — metric implementations.
- `scripts/` — helper scripts.
- `flow_generation/` — flow preprocessing/generation utilities.
- `demos/` — upstream demo scripts.
- `benchmarks/` — upstream benchmark code.
- `environment.yaml` — upstream dependency reference.

### `model_repos/EMA-VFI/`

EMA-VFI model repository.

- `train.py` — upstream training entrypoint.
- `Trainer.py` — upstream training logic.
- `dataset.py` — upstream dataset code.
- `config.py` — upstream configuration code.
- `demo_2x.py` — 2x inference demo.
- `demo_Nx.py` — Nx inference demo.
- `model/` — model definitions.
- `benchmark/` — evaluation scripts.
- `example/` — example assets.

### `model_repos/Practical-RIFE/`

Practical-RIFE model repository.

- `inference_video.py` — upstream video inference script.
- `inference_img.py` — upstream image inference script.
- `inference_img_SR.py` — image inference with SR support.
- `inference_video_enhance.py` — enhanced video inference script.
- `model/` — model implementation.
- `training/` — training code copied from upstream archive.
- `demo/` — demo assets or scripts.
- `requirements.txt` — upstream dependency reference.

## `model_weights/`

Pretrained model weights and model-specific runtime files.

### `model_weights/AMT/`

Pretrained AMT weights.

- `amt-s.pth` — AMT-S checkpoint.
- `amt-l.pth` — AMT-L checkpoint.
- `amt-g.pth` — AMT-G checkpoint.
- `gopro_amt-s.pth` — AMT-S checkpoint trained/adapted for GoPro-style data.

### `model_weights/EMA-VFI/`

Pretrained EMA-VFI weights.

- `ours_small.pkl` — small EMA-VFI checkpoint.
- `ours_small_t.pkl` — small EMA-VFI checkpoint variant.
- `ours.pkl` — full EMA-VFI checkpoint.
- `ours_t.pkl` — full EMA-VFI checkpoint variant.

### `model_weights/Practical-RIFE/`

Pretrained Practical-RIFE weights and accompanying model files.

- `RIFEv4.25/train_log/` — RIFE v4.25 weights and model files.
- `RIFEv4.26/train_log/` — RIFE v4.26 weights and model files.

## `src/video_interpolation/`

Local Stage 1 Python package.

- `adapters/` — shared model adapter interface and EMA-VFI-small adapter.
- `baselines.py` — duplication, blending, and Farneback baseline evaluation over triplet manifests.
- `cli.py` — Typer developer CLI with settings display, EMA-VFI-small preflight/adapter/inference/training/validation commands, data workflows, triplet manifest inspection, baseline evaluation, and MLflow smoke logging.
- `contracts.py` — compact artifact contracts and relative-path validation helpers.
- `data/` — source preprocessing and source-level indexing code.
- `ema_preflight.py` — lightweight EMA-VFI-small import/checkpoint compatibility check.
- `image_io.py` — shared tensor/image conversion and triplet-style prediction sample writing helpers.
- `inference.py` — local EMA-VFI-small 2x video inference workflow.
- `metrics.py` — PSNR, SSIM, optional LPIPS scoring, metric aggregation, and CSV export.
- `mlflow.py` — MLflow tracking setup and logging helpers for Stage 1 runs.
- `settings.py` — `pydantic-settings` runtime settings loaded from `.env`.
- `training.py` — EMA-VFI-small fine-tuning and eval-only runner over triplet manifests.
- `validation.py` — EMA-VFI-small candidate validation metrics, reports, prediction samples, and threshold decisions.

### `src/video_interpolation/adapters/`

Stage 1 model adapter implementations.

- `base.py` — common adapter interface and environment-report structures.
- `ema_vfi.py` — EMA-VFI-small adapter using `model_repos/EMA-VFI` and explicit local checkpoints.

### `src/video_interpolation/data/`

Stage 1 source data processing modules.

- `preprocessing.py` — raw-video discovery, scene-safe sequence sampling, static-triplet filtering, PyAV frame extraction, PNG writing, and source `sequence_index.csv` output.
- `indexing.py` — source-level index builder for existing Vimeo triplets plus global sequence-index builder.
- `versioning.py` — manifest-only dataset version builder with source-video-level splitting and triplet manifest generation.
- `datasets.py` — `UniversalTripletDataset` for loading triplet manifests through `DATASET_ROOT` with synchronized transforms.

## `configs/`

Stage 1 YAML config layout and implemented workflow configs.

- `README.md` — describes config groups and `.env` boundary.
- `baselines/README.md` — operational field reference for baseline evaluation configs.
- `baselines/baseline_eval.yaml` — parameters for duplication, blending, and Farneback baseline evaluation.
- `data/README.md` — operational field reference for implemented Stage 1 data configs.
- `data/dataset_version.yaml` — parameters for manifest-only train/val/test dataset version construction.
- `data/global_index.yaml` — parameters for combining source indexes into a global sequence index.
- `data/index_vimeo_triplet.yaml` — parameters for existing Vimeo triplet source indexing.
- `data/preprocess_anime.yaml` — parameters for anime raw-video preprocessing.
- `data/preprocess_test.yaml` — small-video preprocessing config for diagnostics and smoke tests.
- `inference/README.md` — operational field reference for local EMA inference config.
- `inference/ema_vfi_small_2x.yaml` — parameters for local EMA-VFI-small 2x video inference.
- `models/README.md` — operational field reference for model adapter configs.
- `models/ema_vfi_small.yaml` — EMA-VFI-small adapter/checkpoint/device config.
- `training/README.md` — operational field reference for EMA fine-tuning config.
- `training/ema_vfi_small_finetune.yaml` — EMA-VFI-small fine-tuning and eval-only runner config.
- `validation/README.md` — operational field reference for EMA candidate validation config.
- `validation/ema_vfi_small_candidate.yaml` — EMA-VFI-small candidate validation thresholds and outputs config.

## `infra/`

Local infrastructure definitions.

### `infra/mlflow/`

Stage 1 MLflow infrastructure with PostgreSQL metadata storage and MinIO artifact storage.

- `docker-compose.yml` — local MLflow/PostgreSQL/MinIO service stack.
- `Dockerfile` — MLflow server image with PostgreSQL and S3 artifact dependencies.
- `README.md` — setup, start/stop, and smoke-check instructions.

## `docs/`

Human-facing project documentation.

- `stage1_ml_core.md` — Stage 1 workflow notes for current implemented milestones.

## `tests/`

Focused behavior tests.

- `test_contracts.py` — compact artifact contract and relative path validation tests.
- `test_datasets_metrics_baselines.py` — triplet dataset loading, metric sanity, and baseline evaluation smoke tests.
- `test_indexing.py` — Vimeo source-index generation tests.
- `test_preprocessing.py` — scene-safe sampling, quota termination, frame-step validation, and static-triplet filtering tests.
- `test_validation_and_adapters.py` — candidate validation decision logic and prediction artifact smoke tests with a fake adapter.
- `test_versioning.py` — global index and dataset-version split/manifest generation tests.
