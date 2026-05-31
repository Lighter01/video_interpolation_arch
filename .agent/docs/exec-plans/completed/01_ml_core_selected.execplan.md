# Title and Metadata

- Stage: Stage 1 - ML Core
- Status: Completed / closed for Stage 2 handoff
- Created: 2026-05-22
- Updated: 2026-05-31
- Stage plan: `.agent/stage_plans/01_ml_core_stage_plan.md`
- Global plan: `.agent/general_plan.md`
- Prior ExecPlan: None found in `.agent/docs/exec-plans/completed/`
- Base plan: option A draft, retired after selection
- Secondary reference: option C draft, retired after selection
- Scope authority: Direct user instruction for selecting/revising this ExecPlan and `.agent/stage_plans/01_ml_core_stage_plan.md`

## Stage Goal

Build the local ML core as a reproducible pipeline from raw videos and existing triplets to dataset manifests, PyTorch loading, baseline evaluation, model fine-tuning, candidate validation, MLflow tracking, and local 2x video inference.

This selected plan keeps option A's layered implementation strategy: create the project-owned package and settings first, then implement preprocessing/indexing, dataset versioning, dataset/metrics/baselines, MLflow, EMA-VFI-small, AMT-S, Practical-RIFE, and final docs/handoff.

Stage 1 is not a web service stage. It must not implement FastAPI, Celery, Redis, application PostgreSQL, frontend UI, automatic retraining triggers, monitoring, BentoML serving, WebDataset, Triton, TensorRT, Kubernetes, or distributed workers. MLflow infrastructure with PostgreSQL and MinIO remains expected for Stage 1 experiment tracking and artifacts.

Stage 1 fits the full project by producing the local model lifecycle primitives that later service stages will call: dataset versions, model adapters, candidate checkpoints, validation reports, MLflow runs, and local inference behavior.

## Source Documents and Authority

Read before creating this selected plan:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/general_plan.md`
- `.agent/stage_plans/01_ml_core_stage_plan.md`
- retired option A draft, removed from `active/` after selection
- retired option C draft, removed from `active/` after selection
- `.python-version`
- `pyproject.toml`
- `main.py`
- `README.md`
- `model_repos/AMT/environment.yaml`
- `model_repos/AMT/cfgs/AMT-S.yaml`
- `model_repos/AMT/train.py`
- `model_repos/AMT/datasets/vimeo_datasets.py`
- `model_repos/AMT/demos/demo_2x.py`
- `model_repos/EMA-VFI/README.md`
- `model_repos/EMA-VFI/config.py`
- `model_repos/EMA-VFI/dataset.py`
- `model_repos/EMA-VFI/train.py`
- `model_repos/EMA-VFI/Trainer.py`
- `model_repos/EMA-VFI/demo_2x.py`
- `model_repos/Practical-RIFE/README.md`
- `model_repos/Practical-RIFE/requirements.txt`
- `model_repos/Practical-RIFE/inference_img.py`
- `model_repos/Practical-RIFE/training/dataset.py`
- `model_repos/Practical-RIFE/training/train.py`

The current user instruction controls this selected-plan revision. The Stage 1 plan controls implementation scope and acceptance. Option A controls the main structure. Option C contributes only lightweight artifact contract ideas and selected validation discipline; it does not replace option A or turn this into a contract-first workflow.

The obsolete option A/B/C draft files were removed from `.agent/docs/exec-plans/active/` on 2026-05-22 after this selected ExecPlan became the single active plan.

## Context and Current Repository State

Repository root: `/home/lighter_01/projects/itmo/ai_architecture/video_interpolation`.

Current project code after Milestone 9:

- `main.py` only prints a hello message.
- `README.md` is empty.
- `.python-version` is `3.13`.
- `pyproject.toml` exists with Python `>=3.13`, modern Torch/torchvision, OpenCV, Pillow, numpy, pandas, pydantic, `pydantic-settings`, Typer, Rich, SQLAlchemy, MinIO, MLflow, PyAV, PySceneDetect, scikit-image, LPIPS, timm, OmegaConf, imageio, pytest, ruff, and setuptools package discovery for `src/`.
- `uv.lock` exists and was updated for the Milestone 1 dependency baseline.
- `.env.example` exists with `DATASET_ROOT`, `MODEL_REPOS_ROOT`, `MODEL_WEIGHTS_ROOT`, and `MLFLOW_TRACKING_URI`.
- `src/video_interpolation/` exists with settings, compact artifact contracts, CLI, EMA-VFI-small, AMT-S, and Practical-RIFE preflight utilities, source data modules, global indexing, dataset versioning, triplet dataset loading, metrics, baseline evaluation/video inference, MLflow logging helpers, EMA/AMT/Practical-RIFE/baseline model adapters, shared local inference workflow, EMA fine-tuning runner, and shared candidate validation workflow.
- `src/video_interpolation/data/preprocessing.py` supports raw-video discovery, scene-safe sequence sampling, static-triplet filtering, PyAV frame extraction, PNG writing, and source-level `sequence_index.csv` output.
- `src/video_interpolation/data/indexing.py` supports source-level indexing for existing Vimeo triplets without copying frames and global sequence-index construction from source indexes.
- `src/video_interpolation/data/versioning.py` supports manifest-only dataset version construction with source-video-level splitting, triplet extraction policies, relative manifest paths, and lightweight train-pool mixing.
- `src/video_interpolation/data/datasets.py` implements `UniversalTripletDataset` over triplet manifests, resolving frame paths from `DATASET_ROOT` and applying synchronized transforms to left/middle/right frames.
- `src/video_interpolation/metrics.py` implements PSNR, SSIM, optional LPIPS scoring, metric aggregation, and CSV export.
- `src/video_interpolation/baselines.py` implements duplication, blending, and Farneback baseline prediction/evaluation over triplet manifests.
- `src/video_interpolation/batch_inference.py` implements shared helpers for directory-wide inference video discovery, target selection, output path layout, per-run MLflow names, and per-target timing measurement CSVs.
- `src/video_interpolation/adapters/base.py` defines the shared Stage 1 model adapter interface and environment-report structures.
- `src/video_interpolation/adapters/ema_vfi.py` implements the EMA-VFI-small adapter, explicit checkpoint loading from `MODEL_WEIGHTS_ROOT/EMA-VFI/ours_small.pkl`, sequential pair/batch prediction, training/eval helpers, checkpoint save/reload support, and GPU cleanup.
- `src/video_interpolation/adapters/amt.py` implements the AMT-S adapter, explicit upstream config loading from `model_repos/AMT/cfgs/AMT-S.yaml`, checkpoint loading from `MODEL_WEIGHTS_ROOT/AMT/amt-s.pth`, safe PyTorch weights loading, sequential pair/batch prediction, and GPU cleanup.
- `src/video_interpolation/adapters/rife.py` implements the Practical-RIFE v4.25 adapter, explicit selected `train_log` loading from `MODEL_WEIGHTS_ROOT/Practical-RIFE/RIFEv4.25/train_log`, safe PyTorch weights loading, sequential pair/batch prediction, and GPU cleanup.
- `src/video_interpolation/inference.py` implements shared local 2x video inference by interleaving original and generated frames, encoding with PyAV/FFmpeg, and remuxing compatible audio streams.
- `src/video_interpolation/training.py` implements EMA-VFI-small fine-tuning and eval-only runner support over `train_all.csv` and `val_all.csv`.
- `src/video_interpolation/validation.py` implements shared candidate validation over `test_all.csv`, metrics/report writing, triplet-style sample predictions, and threshold decisions for EMA/AMT adapters.
- `src/video_interpolation/image_io.py` provides shared tensor/image conversion and triplet-style prediction sample writing.
- `src/video_interpolation/mlflow.py` implements MLflow configuration, bounded connectivity behavior, baseline/generic logging helpers, and a smoke-log helper.
- `tests/test_contracts.py`, `tests/test_preprocessing.py`, `tests/test_indexing.py`, and `tests/test_versioning.py` exist for focused compact-contract, sampler/filtering, source-index, global-index, and dataset-version validation.
- `tests/test_datasets_metrics_baselines.py` exists for triplet dataset loading, synchronized transform, metric sanity, and baseline evaluation smoke behavior.
- `tests/test_validation_and_adapters.py` exists for candidate validation threshold decisions and triplet-style prediction artifact behavior using a fake adapter.
- `configs/README.md` describes the Stage 1 YAML config layout.
- `configs/data/index_vimeo_triplet.yaml`, `configs/data/preprocess_anime.yaml`, `configs/data/preprocess_test.yaml`, `configs/data/global_index.yaml`, and `configs/data/dataset_version.yaml` exist for implemented data workflows.
- `configs/baselines/baseline_eval.yaml` and `configs/baselines/README.md` exist for baseline evaluation configuration and operations; `configs/inference/baseline_2x.yaml` exists for local baseline video inference.
- `configs/models/ema_vfi_small.yaml`, `configs/models/amt_s.yaml`, `configs/models/practical_rife_v4_25.yaml`, and `configs/models/README.md` exist for EMA-VFI-small, AMT-S, and Practical-RIFE adapter configuration.
- `configs/inference/ema_vfi_small_2x.yaml`, `configs/inference/amt_s_2x.yaml`, `configs/inference/practical_rife_v4_25_2x.yaml`, and `configs/inference/README.md` exist for local EMA/AMT/Practical-RIFE 2x video inference.
- `configs/training/ema_vfi_small_finetune.yaml` and `configs/training/README.md` exist for EMA fine-tuning and eval-only runs.
- `configs/validation/ema_vfi_small_candidate.yaml`, `configs/validation/amt_s_candidate.yaml`, `configs/validation/practical_rife_v4_25_candidate.yaml`, and `configs/validation/README.md` exist for EMA, AMT, and Practical-RIFE candidate validation.
- `infra/mlflow/docker-compose.yml`, `infra/mlflow/Dockerfile`, and `infra/mlflow/README.md` exist for local MLflow/PostgreSQL/MinIO infrastructure.
- `docs/stage1_ml_core.md` documents current Milestone 1 through Milestone 9 settings, CLI, preflight, indexing, preprocessing, global indexing, dataset versioning, dataset inspection, baseline evaluation/video inference, MLflow infrastructure, EMA adapter/inference/training/validation workflows, AMT and Practical-RIFE adapter/inference/candidate-validation workflows, and contract behavior.
- AMT-S eval/inference/candidate-validation integration is implemented through Milestone 8. AMT-S fine-tuning is deferred because upstream training expects precomputed flow files.
- Practical-RIFE v4.25 eval/inference/candidate-validation integration is implemented through Milestone 9. Practical-RIFE fine-tuning is deferred because upstream training is not manifest-ready and assumes hardcoded `/data`, nori/S3 data access, distributed CUDA, TensorBoard logging, and separate training model paths.

Important dependency observations:

- Stage-required Milestone 1 dependencies were added to `pyproject.toml` and locked/synced with `uv`.
- `pyproject.toml` currently includes `webdataset>=1.0.2`, but WebDataset is out of scope for Stage 1 canonical data and must not be used for the MVP dataset format.
- Upstream dependency files target older stacks: AMT uses Python 3.8.5, PyTorch 1.11, numpy 1.21.5, OmegaConf, imageio, and wandb; EMA-VFI documents Python 3.8, torch 1.8, numpy 1.23.1, skimage, timm; Practical-RIFE says Python `<=3.11` and numpy `<=1.23.5`. This conflicts with the current Python 3.13 / modern numpy and torch project state and must be checked early.
- `scenedetect` 0.7 pulls in `opencv-python`; the project also retains `opencv-python-headless`, so later image/video work should watch for OpenCV package conflicts.

Current data:

- `raw_data/anime/` contains 29 `.mkv` or `.mp4` files grouped by title under `Berserk`, `DBZ`, `Manie_Manie`, `One_Piece`, `Shadows_house`, and `Wonder_egg_priority`. Size is about 21G.
- `datasets/sources/anime/` exists but is empty.
- `datasets/sources/vimeo_triplet/` contains `sequences/`, `tri_trainlist.txt`, `tri_testlist.txt`, and `readme.txt`. There are 73,191 triplet directories under `sequences/`, 51,313 train list rows, and 3,783 test list rows. Size is about 33G.
- `datasets/sources/SNU_FILM/` contains `test-easy.txt`, `test-medium.txt`, `test-hard.txt`, `test-extreme.txt`, and `test/`. Each list has 310 rows. The list entries use a `data/SNU-FILM/...` prefix while the repository directory is `datasets/sources/SNU_FILM/...`, so benchmark path resolution needs care.
- `datasets/sources/ucf101_interp_ours/` contains 379 sample directories with `frame_00.png`, `frame_01_gt.png`, `frame_01_ours.png`, and `frame_02.png`.
- `datasets/global_sequence_index.csv` and `dataset_versions/stage1_default/` exist in the workspace from previous Milestone 3 smoke/full runs.

Current model repositories:

- `model_repos/EMA-VFI/` has upstream `train.py`, `Trainer.py`, `dataset.py`, `config.py`, demos, benchmark scripts, and model code. `config.py` defaults to `ours`; `ours_small` is commented and must be selected programmatically for the first integration target. `Trainer.load_model()` loads `ckpt/<name>.pkl`, so project adapters must support explicit checkpoint paths from `model_weights/EMA-VFI/`.
- `model_repos/AMT/` has configs under `cfgs/`, upstream `train.py`, dataset code, demo scripts, trainer code, losses, metrics, networks, and flow generation utilities. `cfgs/AMT-S.yaml` expects a Vimeo layout under `data/vimeo_triplet`; its training dataset expects flow files under a `flow/` directory. The current Vimeo source has triplet images but no observed flow directory.
- `model_repos/Practical-RIFE/` has inference scripts and a `training/` directory, but its training dataset code hardcodes external `/data/...`, S3/nori, and domain paths. Practical-RIFE will need the largest adaptation to use project manifests.

Current model weights:

- `model_weights/EMA-VFI/ours_small.pkl` and `ours_small_t.pkl` are present; `ours_small.pkl` is the first target.
- `model_weights/AMT/amt-s.pth` and `gopro_amt-s.pth` are present; `amt-s.pth` is the first AMT target.
- `model_weights/Practical-RIFE/RIFEv4.25/train_log/` and `model_weights/Practical-RIFE/RIFEv4.26/train_log/` each contain `flownet.pkl` and model Python files.

Git status before this selected plan already showed user/workspace changes in agent docs, data relocation, `pyproject.toml`, and `uv.lock`. Implementation must preserve those changes and avoid reverting unrelated files.

## Stage Requirements Restated

Required functionality:

- Add video preprocessing from `raw_data/` to sampled PNG sequences under `datasets/sources/<source_group>/sequences/` plus source-level `sequence_index.csv`.
- Use PyAV for video reading, PySceneDetect for scene detection, and Pillow/OpenCV for image writing and image operations.
- Support configurable `sequence_length >= 3`, default 3.
- Support optional resize as `null` or `WIDTHxHEIGHT`.
- For `sequence_length = 3`, support `max_frame_step` in `{0,1,2}`, random step selection, sequence-span-aware scene filtering, and static SSIM rejection using threshold 0.95.
- Reject frame-step configuration when `sequence_length > 3`.
- Use bounded quota sampling with `min_sequences_per_video`, `max_sequences_per_video`, `quota_scale`, and `random_seed`.
- Build `datasets/global_sequence_index.csv` from source-level indexes with deterministic ordering and relative paths.
- Build manifest-only dataset versions under `dataset_versions/<dataset_version_id>/` with `train_all.csv`, `val_all.csv`, `test_all.csv`, and `dataset_config.yaml`.
- Split at `source_video_id` level, not at triplet level.
- Convert sequence records to triplet manifest records using `first_triplet`, `center_triplet`, `wide_triplet`, and `all_local_triplets`; default to `wide_triplet`.
- Implement compact artifact contracts for `sequence_index.csv`, `global_sequence_index.csv`, split manifests, `dataset_config.yaml`, metrics CSVs, and candidate validation report JSON.
- Implement `UniversalTripletDataset` over triplet manifests using runtime `DATASET_ROOT`, synchronized transforms, and metadata returns.
- Implement baseline evaluation for duplication, blending, and OpenCV Farneback optical flow on manifests.
- Compute PSNR, SSIM, and LPIPS per sample, aggregate globally and per domain, export CSVs, and log to MLflow.
- Prepare Docker Compose infrastructure for MLflow Tracking Server, PostgreSQL metadata, MinIO artifact storage, and an MLflow artifact bucket.
- Implement a common `ModelAdapter` interface with `validate_environment`, `build_model`, `load_checkpoint`, `save_checkpoint`, `train`, `predict_pair`, `predict_batch`, `predict`, and `__call__`.
- Add an early EMA-VFI-small compatibility smoke check that imports EMA-VFI code, selects `ours_small`, initializes the model, and loads `model_weights/EMA-VFI/ours_small.pkl` before heavy data-pipeline work.
- Integrate models in strict order: `EMAVFIAdapter` for EMA-VFI-small first, then `AMTAdapter` for AMT-S, then `PracticalRIFEAdapter`.
- Do not begin AMT-S or Practical-RIFE integration before EMA-VFI-small works through dataset loading, fine-tuning, candidate validation, MLflow logging, and local inference.
- Implement training runner modes `finetune` and `eval_only`; represent `scratch_train` in configs and CLI but it may be implemented after fine-tuning works.
- Keep training on `train_all.csv` and `val_all.csv`; keep candidate validation on `test_all.csv`.
- Log training, baseline evaluation, model evaluation, and candidate validation to MLflow unless explicitly disabled for tiny local smoke checks.
- Implement candidate validation on test manifests with report JSON, metrics CSV, MLflow artifacts, sample predictions, and approval/rejection policy.
- Implement local 2x video inference CLI for EMA-VFI-small first, then AMT-S and Practical-RIFE.
- Add concise human-facing docs under `docs/` for CLI commands, configs, workflows, artifacts, and troubleshooting.
- Update `.agent/docs/PROJECT_MAP.md` during implementation when important files/directories are added or changed. This planning-only revision does not update the project map.

Mandatory constraints:

- Keep canonical datasets as PNG frames plus CSV/YAML manifests.
- Use relative paths in all manifests; no absolute local paths or S3 URIs in manifests.
- Use `.env` plus `pydantic-settings` for runtime/environment values.
- Use YAML configs for pipeline and experiment parameters, not runtime roots or secrets.
- Do not use WebDataset for Stage 1 canonical data.
- Do not silently downgrade dependencies to old upstream versions.
- Ask before resolving dependency conflicts by downgrading packages or rewriting model code for newer APIs.
- Keep model-specific logic inside adapters.
- Keep dataset preprocessing and dataset version construction outside adapters.
- Treat upstream model repos carefully and document any adaptation.

## Non-Goals and Deferred Work

Out of scope for Stage 1:

- FastAPI backend.
- Celery workers.
- Redis task queue.
- PostgreSQL application database.
- Production MinIO data synchronization for datasets.
- Frontend UI.
- Automatic retraining triggers.
- Monitoring and alerting.
- BentoML production serving.
- Triton or TensorRT deployment.
- WebDataset canonical storage.
- Kubernetes.
- Distributed workers.

Safely deferred within Stage 1 until the first EMA-VFI-small route works, then completed or revisited after the EMA route:

- AMT-S adaptation was completed for eval/inference/candidate validation in Milestone 8; fine-tuning remains deferred.
- Practical-RIFE adaptation was completed for eval/inference/candidate validation in Milestone 9; fine-tuning remains deferred.
- Optimized batched inference if sequential `predict_pair()` loops are sufficient.
- MLflow Model Registry checkpoint loading if local checkpoint path loading is already working; local path loading must come first.
- Full scratch training implementation. Config and CLI should acknowledge `scratch_train`, but the required implementation path is fine-tuning and eval-only.
- Baseline video inference may come after baseline evaluation.

## Architecture and Implementation Strategy

This selected plan builds a project-owned layered package, for example `src/video_interpolation/`, and keeps upstream repositories as model implementation sources. It borrows option C's contract idea only as a compact implementation support layer, not as a separate contract-first milestone.

Proposed local modules:

- `video_interpolation.settings`: `pydantic-settings` application settings loaded from `.env`.
- `video_interpolation.contracts`: compact constants/validators for required file columns, YAML fields, relative path checks, and report/metrics shape. This module supports data and validation modules but does not become a large standalone workflow.
- `video_interpolation.cli`: Typer command group for preprocessing, indexing, dataset-version building, baseline evaluation, training, candidate validation, and local inference.
- `video_interpolation.data.preprocessing`: PyAV/PySceneDetect video scanning, scene interval logic, sequence sampler, optional resize, SSIM static filter, PNG writing, and source-level `sequence_index.csv`.
- `video_interpolation.data.indexing`: source index validation and global index builder.
- `video_interpolation.data.versioning`: split policy, triplet extraction policies, mixing policies, dataset config writing, and manifest validation.
- `video_interpolation.data.datasets`: `UniversalTripletDataset` and synchronized augmentations.
- `video_interpolation.metrics`: PSNR, SSIM, LPIPS, aggregation, CSV export.
- `video_interpolation.baselines`: duplication, blending, Farneback baseline predictors and baseline evaluation runner.
- `video_interpolation.mlflow`: MLflow setup, parameter/metric/artifact logging helpers.
- `video_interpolation.adapters.base`: `ModelAdapter` protocol/base class and shared image/tensor utilities.
- `video_interpolation.adapters.baseline`: non-neural baseline adapter for duplicate-left, blending, and Farneback inference.
- `video_interpolation.adapters.ema_vfi`: EMA-VFI-small adapter first.
- `video_interpolation.adapters.amt`: AMT-S adapter after EMA is complete.
- `video_interpolation.adapters.rife`: Practical-RIFE adapter last.
- `video_interpolation.training`: config models, training runner orchestration, checkpoint output conventions.
- `video_interpolation.validation`: candidate validation runner, reports, thresholds, sample predictions.
- `video_interpolation.inference`: local video decode/interpolate/encode workflow.

### Settings and Config Policy

Runtime/environment values must come from `.env` and `pydantic-settings`:

- `DATASET_ROOT`
- `MODEL_REPOS_ROOT`
- `MODEL_WEIGHTS_ROOT`
- `MLFLOW_TRACKING_URI`
- future service endpoints and secrets when needed, such as MinIO and PostgreSQL values for MLflow infrastructure.

Planned settings artifacts:

- `.env.example` with safe local defaults or placeholders.
- `video_interpolation.settings.Settings` using `pydantic-settings`.
- Documentation explaining which values belong in `.env`.

YAML configs remain the source of pipeline and experiment parameters:

- preprocessing parameters;
- dataset version parameters;
- training hyperparameters;
- model adapter/model architecture configs;
- validation thresholds;
- baseline evaluation options.

Do not replace environment variables with YAML configs. YAML files may contain relative output names, dataset version ids, split ratios, model names, hyperparameters, or thresholds, but not machine-local roots, service secrets, or environment-specific endpoints.

### File Artifact Contracts

Contracts are compact definitions used by builders, loaders, and validators. They should live near shared schema/constants code and be enforced when reading or writing artifacts.

Required `sequence_index.csv` columns:

```text
sequence_id
source_group
source_dataset
source_video_id
relative_sequence_dir
sequence_length
frame_step
width
height
fps
frame_start
frame_end
scene_id
resize
created_at
```

Rules:

- `relative_sequence_dir` must be relative to `DATASET_ROOT`.
- `sequence_id` should be stable and unique within the source index.
- `frame_start`, `frame_end`, `sequence_length`, dimensions, and FPS must be numeric where applicable.
- `frame_step` is meaningful for triplets and should be empty/null or a valid configured value when not applicable.

`global_sequence_index.csv` contract:

- Same required columns as source-level `sequence_index.csv`.
- May include additional provenance columns only if documented.
- Must preserve `source_group`, `source_dataset`, and relative paths.
- Must have deterministic ordering where practical.

Required columns for `train_all.csv`, `val_all.csv`, and `test_all.csv`:

```text
sample_id
source_group
source_dataset
source_video_id
sequence_id
left_frame_path
mid_frame_path
right_frame_path
t_value
width
height
fps
frame_step
```

Rules:

- Frame paths must be relative to `DATASET_ROOT`.
- `t_value` is fixed to `0.5` for Stage 1.
- `sample_id` must be stable and unique within a manifest.
- All split manifests must be validated for source-video-level leakage across train/val/test.

Required `dataset_config.yaml` fields:

- dataset version id;
- source index path;
- split seed;
- split ratios;
- selected source groups;
- triplet policy;
- mixing mode;
- old/new ratios if used;
- preprocessing assumptions relevant to the version;
- created timestamp.

Candidate validation report JSON should include:

- candidate id or run id;
- model name and adapter;
- checkpoint reference;
- dataset version id;
- test manifest path;
- validation thresholds;
- aggregate metrics;
- per-domain metrics where available;
- approval decision;
- decision reasons;
- artifact paths for metrics CSV and sample predictions;
- created timestamp.

Metrics CSV should include:

- one row per evaluated sample where per-sample output is produced;
- `sample_id`, source metadata, metric columns such as `psnr`, `ssim`, `lpips`, and prediction artifact path when available;
- aggregate/per-domain summaries may be separate CSVs or clearly marked rows, but the format must be documented.

### Data Flow

1. Raw videos in `raw_data/anime/` are sampled to `datasets/sources/anime/sequences/...` and `datasets/sources/anime/sequence_index.csv`.
2. Existing Vimeo triplets are indexed without copying image files by creating a source-level index from `datasets/sources/vimeo_triplet/tri_trainlist.txt`, `tri_testlist.txt`, and `sequences/`.
3. Global index builder combines source-level indexes into `datasets/global_sequence_index.csv`.
4. Dataset version builder creates logical manifests under `dataset_versions/<id>/`.
5. `UniversalTripletDataset` loads those manifests for baseline evaluation and training.
6. Baseline and model evaluations produce metrics CSVs, reports, sample predictions, and MLflow runs.
7. Adapters load explicit checkpoint paths from `MODEL_WEIGHTS_ROOT`.
8. Local inference CLI reads an input video, uses an adapter to synthesize one middle frame per adjacent pair, and writes a 2x output video.

### Model Integration Strategy

Early compatibility smoke check:

- During the first implementation milestone, add a lightweight EMA-VFI-small preflight command or script that imports EMA-VFI modules, selects `ours_small`, initializes the model, and loads `model_weights/EMA-VFI/ours_small.pkl`.
- This check should not train, run a full evaluation, or require a dataset version.
- Purpose: detect Python, dependency, model import, CUDA/CPU, and checkpoint-format incompatibilities before large preprocessing and dataset-versioning work.

Full integration remains ordered:

- EMA-VFI-small: create the adapter first. Select `ours_small` by overriding upstream `MODEL_CONFIG` in process, instantiate `Trainer.Model(-1)`, load `MODEL_WEIGHTS_ROOT/EMA-VFI/ours_small.pkl` from explicit path, and implement pair inference with upstream padding utilities. Fine-tuning may reuse upstream `Model.update()` but must use project manifests and MLflow logging rather than upstream TensorBoard as the primary tracker.
- AMT-S: after EMA works end to end, use `cfgs/AMT-S.yaml`, `MODEL_WEIGHTS_ROOT/AMT/amt-s.pth`, and AMT build utilities. Investigate flow-file requirements before training. If AMT fine-tuning requires optical flow targets, either generate required flow artifacts with `model_repos/AMT/flow_generation/` or ask the user before changing AMT losses.
- Practical-RIFE: after AMT works, use `MODEL_WEIGHTS_ROOT/Practical-RIFE/RIFEv4.25/train_log/` or `RIFEv4.26`. Avoid upstream hardcoded `/data` and S3/nori dataset paths by routing through project manifests. Training adaptation likely requires local dataset replacement.

### MLflow Strategy

- Add MLflow Compose infrastructure early enough that baselines and training can log real runs.
- All long-running baseline, training, validation, and inference benchmark runs accept MLflow config and log to `MLFLOW_TRACKING_URI`.
- Tiny local smoke checks may explicitly disable MLflow, but stage completion requires MLflow-backed runs.
- Log configs, manifests, metrics CSVs, reports, checkpoints, and sample predictions.

## Milestones and Work Breakdown

### Milestone 1 - Project Skeleton, Settings, Dependencies, and EMA Preflight

Objective: Add the Stage 1 package skeleton, CLI shell, settings policy, compact contracts, dependency plan, and early EMA-VFI-small compatibility smoke check.

Likely files/modules:

- `src/video_interpolation/`
- `video_interpolation.settings`
- `video_interpolation.contracts`
- `video_interpolation.cli`
- `.env.example`
- `configs/`
- `docs/stage1_ml_core.md`
- `pyproject.toml`
- focused `tests/`

Expected output:

- Importable package and Typer CLI.
- `pydantic-settings` based settings loaded from `.env`.
- `.env.example` with `DATASET_ROOT`, `MODEL_REPOS_ROOT`, `MODEL_WEIGHTS_ROOT`, and `MLFLOW_TRACKING_URI`.
- Compact artifact contract definitions for Stage 1 CSV/YAML/JSON outputs.
- Documented dependency additions and any conflicts.
- EMA-VFI-small preflight command or function that imports, initializes, and loads `ours_small.pkl`.

Validation checkpoint:

- `uv run python -m video_interpolation.cli --help`.
- Settings load from `.env` or explicit environment variables.
- EMA preflight reports success or a clear compatibility blocker.
- Minimal focused tests for relative-path validation and contract helper behavior, avoiding shallow field-presence-only tests.
- `uv run ruff check` once code exists.

### Milestone 2 - Video Preprocessing and Source Indexing

Objective: Convert raw videos into sampled PNG sequences and index existing Vimeo triplets without moving data.

Likely files/modules:

- `video_interpolation.data.preprocessing`
- `video_interpolation.data.indexing`
- `configs/data/preprocess_anime.yaml`
- `configs/data/build_source_indexes.yaml`

Expected output:

- `datasets/sources/anime/sequences/...`
- `datasets/sources/anime/sequence_index.csv`
- `datasets/sources/vimeo_triplet/sequence_index.csv`

Validation checkpoint:

- Focused tests for scene-safe sampling, sampler termination when quota cannot be reached, and SSIM static triplet filtering.
- Small smoke run against a tiny raw video subset or temporary synthetic video.
- Artifact validation confirms required columns and relative paths.

### Milestone 3 - Global Index and Dataset Version Builder

Objective: Create a global sequence index and manifest-only dataset versions.

Likely files/modules:

- `video_interpolation.data.indexing`
- `video_interpolation.data.versioning`
- `configs/data/dataset_version.yaml`

Expected output:

- `datasets/global_sequence_index.csv`
- `dataset_versions/<dataset_version_id>/train_all.csv`
- `dataset_versions/<dataset_version_id>/val_all.csv`
- `dataset_versions/<dataset_version_id>/test_all.csv`
- `dataset_versions/<dataset_version_id>/dataset_config.yaml`

Validation checkpoint:

- Focused test that split logic prevents `source_video_id` leakage across train/val/test.
- Focused test that triplet manifest generation produces valid relative paths.
- Smoke run on a small index subset and selected Vimeo/anime records.

### Milestone 4 - UniversalTripletDataset, Metrics, and Baselines

Objective: Make manifests loadable and establish reusable metrics and baseline evaluation.

Likely files/modules:

- `video_interpolation.data.datasets`
- `video_interpolation.metrics`
- `video_interpolation.baselines`
- `configs/baselines/baseline_eval.yaml`

Expected output:

- `UniversalTripletDataset`.
- PSNR, SSIM, LPIPS metric functions and aggregators.
- Duplication, blending, and Farneback baseline evaluation over manifests.

Validation checkpoint:

- Focused test loading a small synthetic manifest with real PNG triplets.
- Metrics sanity checks on controlled image pairs.
- Baseline evaluation smoke run on a small manifest subset.

### Milestone 5 - MLflow Infrastructure and Logging

Objective: Start tracking real Stage 1 runs and establish artifact logging conventions.

Likely files/modules:

- `infra/mlflow/docker-compose.yml` or `docker-compose.mlflow.yml`
- `video_interpolation.mlflow`
- `configs/mlflow/local.yaml` if needed for non-secret parameters
- `docs/stage1_ml_core.md`

Expected output:

- MLflow/PostgreSQL/MinIO local infrastructure.
- MLflow helper code that logs params, metrics, CSV artifacts, configs, reports, sample predictions, and checkpoints.
- Baseline metrics and sample predictions logged to MLflow.

Validation checkpoint:

- `docker compose` smoke start for MLflow services.
- Baseline evaluation smoke run logs to `MLFLOW_TRACKING_URI`.
- MLflow run contains expected params, metrics, CSV artifacts, and sample predictions.

### Milestone 6 - EMA-VFI-small Adapter, Evaluation, and Local Inference

Objective: Make the first selected model work through the project adapter and local inference route before any other model integration starts.

Likely files/modules:

- `video_interpolation.adapters.base`
- `video_interpolation.adapters.ema_vfi`
- `video_interpolation.training`
- `video_interpolation.validation`
- `video_interpolation.inference`
- `configs/models/ema_vfi_small.yaml`
- `configs/training/ema_vfi_small_finetune.yaml`

Expected output:

- EMA-VFI-small adapter implementing the shared adapter interface.
- Explicit checkpoint loading from `MODEL_WEIGHTS_ROOT/EMA-VFI/ours_small.pkl`.
- `predict_pair()` and sequential `predict_batch()`.
- Eval-only run on a small manifest.
- Local 2x video inference CLI.

Validation checkpoint:

- Adapter environment check.
- Pair inference smoke test on two PNGs.
- Candidate validation smoke run on a small `test_all.csv`.
- Local inference smoke run on a short video or synthetic frame sequence.

### Milestone 7 - EMA-VFI-small Fine-Tuning and Candidate Validation

Objective: Fine-tune EMA-VFI-small with project manifests and record the candidate lifecycle.

Likely files/modules:

- `video_interpolation.training`
- `video_interpolation.adapters.ema_vfi`
- `video_interpolation.validation`
- `video_interpolation.mlflow`

Expected output:

- Fine-tuning run from pretrained checkpoint.
- Best and last checkpoint artifacts.
- Candidate validation report JSON and metrics CSV.
- MLflow run with configs, manifests, metrics, sample predictions, and checkpoints.

Validation checkpoint:

- Tiny overfit run on a deliberately small manifest subset.
- Candidate validation uses only `test_all.csv`.
- Candidate validation decision logic approves/rejects based on configured thresholds.
- Checkpoint reload produces valid prediction.

### Milestone 8 - AMT-S Adapter and Fine-Tuning Route

Objective: Extend the established pipeline to AMT-S without weakening shared interfaces.

Likely files/modules:

- `video_interpolation.adapters.amt`
- `configs/models/amt_s.yaml`
- `configs/inference/amt_s_2x.yaml`
- `configs/validation/amt_s_candidate.yaml`
- `configs/training/amt_s_finetune.yaml` only if the flow-file training dependency is resolved or an approved loss-policy change is made
- possible thin wrapper files under `model_repos/AMT/`

Expected output:

- AMT-S preflight, eval-only/candidate validation, and local inference path.
- AMT-S fine-tuning only if upstream flow-file requirements can be satisfied with a bounded approved change.

Validation checkpoint:

- Environment validation documents flow-file requirements.
- Pair inference smoke test with `MODEL_WEIGHTS_ROOT/AMT/amt-s.pth`.
- Fine-tuning smoke run after flow/dependency issue is resolved.

### Milestone 9 - Practical-RIFE Adapter and Fine-Tuning Route

Objective: Complete the third required model integration.

Likely files/modules:

- `video_interpolation.adapters.rife`
- `configs/models/practical_rife_v4_25.yaml`
- `configs/training/practical_rife_finetune.yaml`
- possible thin wrapper files under `model_repos/Practical-RIFE/`

Expected output:

- Practical-RIFE preflight, eval-only/candidate validation, and local inference path.
- Practical-RIFE fine-tuning only if upstream hardcoded data/distributed/TensorBoard assumptions can be replaced with a bounded approved change.

Validation checkpoint:

- Pair inference smoke test with local train_log weights.
- Training blocker is documented if hardcoded `/data`, nori/S3, distributed CUDA, and TensorBoard assumptions are not safely removable in Milestone 9.
- Candidate validation report logged to MLflow.

### Milestone 10 - Documentation, Project Map, and Stage Handoff

Objective: Make the stage reproducible for the project owner and future agents.

Likely files/modules:

- `docs/stage1_ml_core.md`
- `.agent/docs/PROJECT_MAP.md`
- Active ExecPlan progress, discoveries, decisions, and handoff

Expected output:

- CLI/config/workflow docs.
- Updated project map.
- Final Stage 1 handoff.

Validation checkpoint:

- Docs commands match actual CLI.
- Final smoke workflow produces expected artifacts.
- ExecPlan progress, discoveries, decisions, and handoff are current.

### First Actionable Tasks

1. Add `src/video_interpolation/` package skeleton, Typer CLI entrypoint, `video_interpolation.settings`, `video_interpolation.contracts`, `.env.example`, and config directories.
2. Reconcile Stage 1 dependencies in `pyproject.toml`, adding only stage-required trusted packages and documenting any conflicts before installation.
3. Implement the early EMA-VFI-small preflight to import EMA-VFI, select `ours_small`, initialize the model, and load `MODEL_WEIGHTS_ROOT/EMA-VFI/ours_small.pkl`.
4. Implement preprocessing config models and the small set of critical sampler tests before touching large raw data.
5. Add source index generation for existing Vimeo triplets so a tiny dataset version can be built before raw anime preprocessing is run at scale.

## Validation Strategy

Validation should emphasize meaningful behavior and smoke workflows over broad unit-test coverage. Do not add tests that only check hardcoded config values, trivial field presence, dataclass existence, or private implementation details.

Focused automated tests:

- Scene-safe sampling does not cross scene boundaries.
- Sampler terminates cleanly when the configured quota cannot be reached.
- SSIM static triplet filtering rejects overly static triplets.
- Dataset version splitting keeps each `source_video_id` in only one split.
- Triplet manifest generation produces valid relative paths.
- `UniversalTripletDataset` loads a small synthetic manifest with real PNG triplets.
- Metrics give sane results on controlled image pairs.
- Candidate validation decision logic applies configured thresholds correctly.

Smoke runs:

- EMA-VFI-small preflight imports, initializes, and loads `ours_small.pkl`.
- Build source index from a small subset of existing Vimeo rows.
- Build a small dataset version with train/val/test manifests.
- Run baseline evaluation on a small manifest and inspect MLflow artifacts.
- Run EMA-VFI-small pair inference from `MODEL_WEIGHTS_ROOT/EMA-VFI/ours_small.pkl`.
- Run EMA-VFI-small tiny fine-tune/overfit and checkpoint reload.
- Run local 2x inference on a very short video.
- Repeat eval/fine-tune/inference smoke route for AMT-S, then Practical-RIFE only after EMA is complete.

Manual checks:

- Inspect sampled anime sequences for scene-cut leakage and resize correctness.
- Inspect sample predictions from baselines and models.
- Inspect MLflow UI for parameters, metrics, artifacts, and checkpoint files.
- Inspect generated videos for frame ordering and output FPS.

Stage completion should pass:

- `uv run ruff check`
- `uv run pytest`
- Documented small end-to-end workflow from source index through EMA-VFI-small fine-tuning, candidate validation, MLflow logging, and local inference.
- Equivalent model lifecycle smoke workflows for AMT-S and Practical-RIFE.

## Expected Artifacts

Expected completion artifacts include:

- Local Python package under `src/video_interpolation/`.
- Typer CLI entrypoints for Stage 1 workflows.
- `video_interpolation.settings` using `pydantic-settings`.
- `.env.example`.
- Compact contract/schema definitions for Stage 1 artifacts.
- Config files under `configs/`.
- Focused tests under `tests/`.
- MLflow infrastructure Compose file.
- Human-facing docs under `docs/`.
- `datasets/sources/anime/sequence_index.csv` and sampled anime PNG sequences when preprocessing is run.
- `datasets/sources/vimeo_triplet/sequence_index.csv`.
- `datasets/global_sequence_index.csv`.
- `dataset_versions/<dataset_version_id>/train_all.csv`, `val_all.csv`, `test_all.csv`, and `dataset_config.yaml`.
- Baseline `metrics.csv`, per-domain metrics, sample predictions, and MLflow runs.
- EMA-VFI-small, AMT-S, and Practical-RIFE adapter configs.
- Fine-tuned checkpoints, candidate validation reports, validation metrics CSVs, and sample predictions.
- Local 2x output videos from each integrated model.
- Updated `.agent/docs/PROJECT_MAP.md` during implementation closeout or when important structure changes.

## Risks, Assumptions, and Recovery

Assumptions:

- The local machine has enough disk for sampled anime PNGs and dataset versions.
- CUDA availability may vary; environment validation must report device status instead of failing unclearly.
- The project owner wants to keep Python 3.13 unless explicitly approving a compatibility environment change.
- Existing data under `datasets/sources/` is the intended current layout despite older references to `datasets/vimeo_triplet/`.
- Runtime roots and endpoints are machine-local and belong in `.env`, not YAML pipeline configs.

Risks:

- Upstream repos may not import under Python 3.13, numpy 2.4.6, or torch 2.11.
- EMA-VFI uses global `config.py` and checkpoint path assumptions.
- AMT-S training expects flow files that are not present in the current Vimeo directory.
- Practical-RIFE training code contains hardcoded external paths and nori/S3 dependencies.
- SNU-FILM manifests contain path prefixes that do not match local directory names.
- Large dataset operations can be slow and expensive if not subset-capable.
- MLflow/PostgreSQL/MinIO may be unavailable locally.

Recovery:

- Run the EMA-VFI-small preflight in Milestone 1 before heavy data work. If it fails because of dependency or Python compatibility, document the exact blocker and ask before downgrading packages, changing Python versions, or making broad upstream patches.
- Keep every large command subset-capable and restart-safe.
- Write indexes/manifests to temp files and replace atomically where practical.
- If MLflow is unavailable, allow explicit local dry-run mode for tiny smoke checks but require MLflow logging for stage completion.
- If dependency conflicts appear, stop and ask before downgrading or making broad upstream rewrites.
- If AMT flow generation is required, document storage cost and ask before generating large flow artifacts for the full dataset.
- If a model cannot fine-tune on the current Python environment, record the blocker and propose either a separate model-specific environment or an approved compatibility patch.

Dependency decisions:

- Add missing Stage 1 dependencies only when official/trusted and already implied by the stage plan.
- Ask before adding heavy or conflict-prone packages such as LPIPS if dependency resolution conflicts with current numpy/PyTorch.
- Do not replace PyAV, PySceneDetect, MLflow, or LPIPS with custom alternatives unless installation is blocked and the temporary fallback is documented in this ExecPlan.
- Do not install upstream dependency files blindly.

## Progress

- 2026-05-22: Created selected Stage 1 ExecPlan from option A. Kept option A's layered implementation order, borrowed only lightweight artifact contracts and validation discipline from option C, added `.env`/`pydantic-settings` policy, added early EMA-VFI-small compatibility preflight, and narrowed the testing policy to critical behavior plus smoke workflows. Implementation not started.
- 2026-05-22: Removed obsolete alternative ExecPlan files from `.agent/docs/exec-plans/active/`, leaving only `01_ml_core_selected.execplan.md`.
- 2026-05-22: Implemented Milestone 1 package skeleton under `src/video_interpolation/`, `.env.example`, compact artifact contracts, `pydantic-settings` runtime settings, Typer CLI, EMA-VFI-small preflight, focused contract tests, config layout notes, and Stage 1 docs. Added Stage 1 dependency baseline to `pyproject.toml` and updated `uv.lock`.
- 2026-05-22: Validation completed for Milestone 1: CLI help passed; settings display passed; `pytest` passed with 3 tests; `ruff check src tests` passed; EMA-VFI-small preflight reached repository/checkpoint/Torch/import checks and reported `blocked` because CUDA is unavailable while upstream EMA-VFI hardcodes CUDA setup.
- 2026-05-22: Updated `.agent/docs/PROJECT_MAP.md` for the new `src/`, `tests/`, `configs/`, `docs/`, `.env.example`, and selected active ExecPlan structure.
- 2026-05-22: User manually ran `uv run python -m video_interpolation.cli ema-preflight` in the real WSL CUDA environment and confirmed EMA-VFI-small initialized and loaded `model_weights/EMA-VFI/ours_small.pkl` successfully. The earlier `blocked` result is specific to the Codex sandbox lacking CUDA access.
- 2026-05-22: Added Milestone 1 follow-up cleanup to the EMA preflight so model/checkpoint references are released, `gc.collect()` runs, and CUDA cache is emptied when available after the check finishes.
- 2026-05-22: Follow-up validation completed after preflight cleanup: exact `uv run python -m video_interpolation.cli ema-preflight` could not run in the sandbox because uv's default home cache is read-only; rerun with `UV_CACHE_DIR=/tmp/uv-cache` exited 0 and reported the expected sandbox CUDA `blocked` status. `uv run pytest` passed with 3 tests and `uv run ruff check src tests` passed.
- 2026-05-22: Implemented Milestone 2 source data layer: `video_interpolation.data.preprocessing`, `video_interpolation.data.indexing`, Typer `data` commands, data YAML configs, focused preprocessing/indexing tests, and updated Stage 1 docs. No global index or dataset-version builder was started.
- 2026-05-22: Milestone 2 validation completed: with `UV_CACHE_DIR=/tmp/uv-cache`, `uv run pytest` passed with 8 tests, `uv run ruff check src tests` passed, and the Vimeo indexing smoke command wrote a 3-row source index to `/tmp/video_interpolation_vimeo_sequence_index.csv`. Exact `uv run ...` commands without `UV_CACHE_DIR` still fail in the sandbox because uv cannot create files under the read-only home cache.
- 2026-05-22: Updated `.agent/docs/PROJECT_MAP.md` for the new `src/video_interpolation/data/`, `configs/data/`, and focused Milestone 2 tests.
- 2026-05-22: Completed Milestone 2 follow-up for documentation and CLI UX. Strengthened `.agent/AGENTS.md` human-facing documentation requirements, added a CLI logging/progress policy, added a short ExecPlan reminder in `.agent/docs/PLANS.md`, expanded `docs/stage1_ml_core.md`, and added `configs/data/README.md` with field-level config documentation.
- 2026-05-22: Added Rich progress and final summary tables to Milestone 2 CLI commands. `data preprocess-videos` now shows a video-level progress bar plus a per-video sequence progress bar and prints preprocessing counts/output paths; `data index-vimeo-triplets` now shows record progress and prints source-index counts/output path.
- 2026-05-22: Follow-up validation completed: exact `uv run pytest` and `uv run ruff check src tests` still fail in the sandbox because uv's default home cache is read-only; reruns with `UV_CACHE_DIR=/tmp/uv-cache` passed (`pytest`: 8 tests, `ruff`: all checks). CLI help commands passed. Safe Vimeo indexing smoke wrote 3 rows to `/tmp/video_interpolation_vimeo_sequence_index_followup.csv`; safe preprocessing smoke used a temporary `/tmp` video and dataset root, wrote 2 sequences, and showed the new progress/summary output.
- 2026-05-22: Completed Milestone 2 debugging/optimization follow-up for slow long-video preprocessing. Added `--debug-progress`, `--only-video`, `--video-glob`, `--max-duration-sec`, and `--max-frames` CLI options; added matching config fields plus `decode_strategy`; fixed static-triplet rejection to reject when either adjacent pair is above threshold; and documented the new diagnostics.
- 2026-05-22: Diagnostic baseline on `raw_data/tmp_test/Dora.mp4` before optimization: metadata 0.005s from `stream.frames`, scene detection 2.144s over 644 frames, candidate sampling 0.003s, sequential selected-frame decoding 3.567s for 15 frames, SSIM/write 2.611s, total 8.337s. Selected-frame decoding was the largest measured step.
- 2026-05-22: Diagnostic baseline on `raw_data/tmp_test/one_piece_test_1m.mkv` before optimization: metadata 0.046s, `stream.frames=0`, frame count estimated from container duration, scene detection 2.927s over 1472 frames, candidate sampling 0.005s, sequential selected-frame decoding 5.492s for 15 frames, SSIM/write 2.000s, total 10.479s. Selected-frame decoding dominated; SceneDetect was secondary.
- 2026-05-22: Implemented bounded optimizations: container duration conversion was corrected, missing MKV frame counts now use duration x FPS estimates instead of full decoding where available, debug/smoke runs can limit per-video duration or frames, and `decode_strategy=auto` uses grouped segment seek/decode for later selected frames with sequential fallback.
- 2026-05-22: Post-optimization diagnostics: Dora MP4 used exact metadata frame count; scene detection 1.955s, segment seek decoding 2.870s via 4 segments, SSIM/write 2.706s, total 7.544s. One Piece MKV used duration estimate 1540 frames; scene detection 2.724s, segment seek decoding 1.058s via 5 segments, SSIM/write 1.841s, total 5.694s. A `--max-duration-sec 10` MKV smoke limited the run to 239 frames: scene detection 0.654s, sequential decoding 0.988s, SSIM/write 2.648s, total 4.358s.
- 2026-05-22: Final debugging validation completed: exact `uv run pytest`, `uv run ruff check src tests`, and exact diagnostic `uv run python -m video_interpolation.cli data preprocess-videos --config configs/data/preprocess_test.yaml --limit-videos 1 --debug-progress` still fail in the sandbox because uv's default home cache is read-only. Reruns with `UV_CACHE_DIR=/tmp/uv-cache` passed (`pytest`: 8 tests, `ruff`: all checks), CLI help showed the new debug/selection/limit flags, and diagnostic MP4/MKV commands completed with the optimized timings recorded above.
- 2026-05-22: Started Milestone 3 implementation. Added global sequence-index builder, dataset-version builder, `data build-global-index` and `data build-dataset-version` CLI commands, `configs/data/global_index.yaml`, `configs/data/dataset_version.yaml`, and focused tests for source-video split leakage plus relative manifest path generation.
- 2026-05-22: Completed Milestone 3 implementation. `data build-global-index` combines source-level indexes with deterministic ordering and preserved extra source columns. `data build-dataset-version` writes `train_all.csv`, `val_all.csv`, `test_all.csv`, and `dataset_config.yaml`, splits by `source_video_id`, supports `first_triplet`, `center_triplet`, `wide_triplet`, and `all_local_triplets`, infers Vimeo `im*.png` and preprocessing `frame_*.png` naming, validates relative paths, and records portable dataset config references.
- 2026-05-22: Milestone 3 validation completed. Exact sandboxed `uv run pytest`, `uv run ruff check src tests`, and CLI help commands still fail when uv tries to write under the read-only home cache. Escalated exact commands using uv's normal cache passed: `pytest` collected 11 tests and all passed; `ruff check src tests` passed; `data build-global-index --help` and `data build-dataset-version --help` rendered successfully.
- 2026-05-22: Milestone 3 smoke commands completed using existing source indexes and `/tmp` outputs. Global-index smoke read `sources/anime/sequence_index.csv`, `sources/tmp_test/sequence_index.csv`, and `sources/vimeo_triplet/sequence_index.csv`, read 55,118 source rows, wrote 30 rows to `/tmp/stage1_m3_smoke/global_sequence_index.csv`, and preserved relative sequence paths. Dataset-version smoke read that global index, generated 30 samples, and wrote `/tmp/stage1_m3_smoke/smoke_version/{train_all.csv,val_all.csv,test_all.csv,dataset_config.yaml}` with counts train=1, val=7, test=22.
- 2026-05-26: Started and completed Milestone 4. Added `UniversalTripletDataset`, synchronized transform support, PSNR/SSIM/LPIPS metric utilities, metric aggregation/CSV export, duplication/blending/Farneback baseline evaluation, `configs/baselines/baseline_eval.yaml`, `data inspect-triplet-manifest`, and `baseline evaluate`.
- 2026-05-26: Started and completed Milestone 5. Added MLflow logging helpers, bounded MLflow HTTP retry/timeout defaults, `mlflow smoke-log`, local MLflow/PostgreSQL/MinIO Compose infrastructure under `infra/mlflow/`, baseline MLflow logging support, and `.env.example` entries for MLflow infrastructure variables.
- 2026-05-26: Updated human-facing docs for Milestones 4 and 5: `docs/stage1_ml_core.md`, `configs/README.md`, `configs/baselines/README.md`, and `infra/mlflow/README.md`. Updated `.agent/docs/PROJECT_MAP.md` for new modules, configs, tests, CLI commands, infra, and dataset-version artifacts.
- 2026-05-26: Milestone 4/5 validation completed. Exact sandboxed `uv run pytest` and `uv run ruff check src tests` still fail because uv cannot write its home cache; escalated exact commands passed with 14 tests and `ruff` clean. CLI help for `data inspect-triplet-manifest`, `baseline evaluate`, and `mlflow smoke-log` rendered successfully.
- 2026-05-26: Milestone 4/5 smoke commands completed. `data inspect-triplet-manifest --manifest dataset_versions/stage1_default/test_all.csv --limit-samples 2` loaded 5,245 samples and showed `3x256x448` tensors. `baseline evaluate ... --limit-samples 2 --no-lpips --disable-mlflow` wrote `/tmp/stage1_m4_m5_baseline_smoke/{metrics.csv,metrics_summary.csv,sample_predictions/...}` with 6 baseline predictions across duplication, blending, and Farneback. `docker compose --env-file .env.example -f infra/mlflow/docker-compose.yml config` rendered successfully. `mlflow smoke-log` with the current `MLFLOW_TRACKING_URI` failed quickly and clearly because no MLflow service is running in this environment.
- 2026-05-27: Updated baseline sample prediction export format. `baseline evaluate` keeps the same CLI and run logic, but each saved sample is now a folder containing `im1.png`, `im2_gt.png`, `im2_generated.png`, and `im3.png` for natural visual comparison against left/right context and ground truth.
- 2026-05-27: Implemented Milestone 6 EMA-VFI-small adapter and local inference workflow. Added `video_interpolation.adapters.base`, `video_interpolation.adapters.ema_vfi`, shared image I/O helpers, `video_interpolation.inference`, `ema adapter-check`, and `ema infer-video`. The adapter validates repo/checkpoint/CUDA/import prerequisites, loads `MODEL_WEIGHTS_ROOT/EMA-VFI/ours_small.pkl`, exposes `predict_pair`, sequential `predict_batch`, `predict`, `__call__`, and releases CUDA memory in `close()`.
- 2026-05-27: Implemented Milestone 7 EMA-VFI-small fine-tuning and candidate validation workflow. Added `video_interpolation.training`, `video_interpolation.validation`, `ema finetune`, `ema validate-candidate`, configs under `configs/{models,inference,training,validation}/`, and local README files for field-level documentation. Training uses `train_all.csv` and `val_all.csv`; candidate validation uses `test_all.csv`, writes metrics/report/sample predictions, and applies configured thresholds.
- 2026-05-27: Added focused tests for candidate validation decision logic and triplet-style prediction artifacts with a fake adapter. Updated `docs/stage1_ml_core.md` and `.agent/docs/PROJECT_MAP.md` for the new Milestone 6/7 modules, configs, commands, outputs, and limitations.
- 2026-05-27: Fine-tuning smoke initially exposed that upstream EMA training needed the same divisor padding as inference for non-divisible frame sizes. Updated adapter `train_step()` and `eval_step()` to pad left/middle/right tensors before upstream `Model.update()` and unpad predictions afterward.
- 2026-05-27: Milestone 6/7 validation completed in the real CUDA environment via escalated exact `uv run` commands: `ema-preflight` passed end to end; `ema adapter-check` reported repo/checkpoint/import/CUDA all ok; `ema infer-video --limit-pairs 1 --disable-mlflow` wrote `/tmp/stage1_ema_inference_smoke.mp4` with 1 interpolated pair and 3 frames; `ema validate-candidate --limit-samples 1 --no-lpips --disable-mlflow` wrote `/tmp/stage1_ema_validation_smoke/{metrics.csv,metrics_summary.csv,candidate_validation_report.json,sample_predictions/...}` and approved the pretrained candidate; `ema finetune --max-steps 1 --limit-train-samples 1 --limit-val-samples 1 --disable-mlflow` wrote `/tmp/stage1_ema_finetune_smoke/{best_checkpoint.pkl,last_checkpoint.pkl}`; checkpoint reload validation against the fine-tuned best checkpoint wrote `/tmp/stage1_ema_checkpoint_reload_validation/...` and approved the one-sample candidate.
- 2026-05-27: Final exact validation passed: `uv run pytest` collected 16 tests and all passed; `uv run ruff check src tests` passed; updated `ema finetune --help` rendered with sample-limit overrides. MLflow logging for EMA workflows was deferred because the local MLflow service is not running; smoke commands used `--disable-mlflow`.
- 2026-05-27: Expanded `docs/stage1_ml_core.md` EMA command sections with input and flag descriptions, defaults, and override behavior for `ema adapter-check`, `ema infer-video`, `ema finetune`, and `ema validate-candidate`.
- 2026-05-27: Upgraded `ema infer-video` timing metrics. The workflow now measures model inference elapsed time around EMA `predict_pair()` calls, total command processing time, model pairs/sec, and total pairs/sec; these values appear in the CLI summary and are logged to MLflow with the existing inference metrics when MLflow is enabled.
- 2026-05-27: Upgraded `ema infer-video` output writing from OpenCV `VideoWriter` to PyAV/FFmpeg. Inference still streams neighboring frame pairs through the EMA adapter, writes original/generated/original frames in order, uses FFmpeg encoder names, resolves per-codec encoder options, prints encoding settings with Rich, and remuxes input audio streams when compatible.
- 2026-05-27: Updated `configs/inference/ema_vfi_small_2x.yaml`, `configs/inference/README.md`, and `docs/stage1_ml_core.md` for PyAV output settings: `container`, `pix_fmt`, `frame_format`, `encoder_options_by_codec`, `encoder_options`, `--codec`, audio preservation behavior, and CPU/GPU codec fallback guidance.
- 2026-05-27: Added focused inference tests for config parsing, encoder option resolution, `mp4v` rejection, and a synthetic PyAV writer smoke that verifies a readable output video with preserved audio. Validation passed: exact escalated `uv run ruff check src tests`; exact escalated `uv run pytest` collected 21 tests and all passed; exact `uv run ... ema infer-video --codec libx264 --limit-pairs 1 --disable-mlflow` wrote `outputs/inference/ema_vfi_small/dora_2x.mp4` with 1 preserved AAC audio stream; configured `h264_nvenc` smoke wrote `/tmp/stage1_ema_pyav_nvenc_smoke.mp4` with H.264 video and preserved AAC audio.
- 2026-05-27: Implemented Milestone 8 AMT-S eval/inference integration. Added `video_interpolation.adapters.amt`, `video_interpolation.amt_preflight`, `amt-preflight`, `amt adapter-check`, `amt infer-video`, `amt validate-candidate`, AMT configs under `configs/{models,inference,validation}/`, shared adapter factories in inference/validation, and focused AMT adapter tests. The AMT adapter builds upstream `networks.AMT-S.Model` from `model_repos/AMT/cfgs/AMT-S.yaml`, loads `model_weights/AMT/amt-s.pth` with PyTorch safe weights loading, and exposes `predict_pair`, sequential `predict_batch`, `predict`, and `__call__`.
- 2026-05-27: AMT-S fine-tuning was investigated and deferred. Upstream `cfgs/AMT-S.yaml` and `datasets/vimeo_datasets.py` expect precomputed `flow_t0.flo`/`flow_t1.flo` files and include `MultipleFlowLoss`; the current project dataset versions are image-triplet manifests only. No flow artifacts were generated and no upstream AMT files were patched.
- 2026-05-27: Milestone 8 validation completed in the CUDA environment via escalated exact `uv run` commands. `amt-preflight` passed end to end and reported repo/config/checkpoint/import/CUDA/model initialization/checkpoint load ok; `amt adapter-check` passed and documented the missing flow directory as an AMT fine-tuning blocker; `amt validate-candidate --limit-samples 1 --no-lpips --disable-mlflow` wrote `/tmp/amt_candidate_validation_smoke/{metrics.csv,metrics_summary.csv,candidate_validation_report.json,sample_predictions/...}` and approved the one-sample pretrained candidate; `amt infer-video --codec libx264 --limit-pairs 1 --disable-mlflow` wrote `/tmp/dora_amt_2x.mp4` with 3 frames, H.264/yuv420p video, and 1 preserved AAC audio stream. Final exact validation passed: `uv run pytest` collected 23 tests and all passed; `uv run ruff check src tests` passed; AMT CLI help rendered for `amt`, `amt infer-video`, and `amt validate-candidate`.
- 2026-05-27: Added a future-patch reminder for AMT-S fine-tuning to `configs/training/README.md` and this ExecPlan. The reminder preserves the current decision to move on without adaptation, but records the likely choices for later: upstream Vimeo-style training with generated `.flo` files, project-native manifest-based AMT training with flow paths, or an approved no-flow loss policy.
- 2026-05-27: Implemented Milestone 9 Practical-RIFE v4.25 eval/inference integration. Added `video_interpolation.adapters.rife`, `video_interpolation.rife_preflight`, `rife-preflight`, `rife adapter-check`, `rife infer-video`, `rife validate-candidate`, Practical-RIFE configs under `configs/{models,inference,validation}/`, and focused RIFE adapter tests. The adapter imports upstream `model/*` from `model_repos/Practical-RIFE`, imports selected `train_log` model code from `model_weights/Practical-RIFE/RIFEv4.25/train_log`, loads `flownet.pkl` with PyTorch safe weights loading, strips `module.` prefixes, pads to divisor 128, and exposes `predict_pair`, sequential `predict_batch`, `predict`, and `__call__`.
- 2026-05-27: Practical-RIFE fine-tuning was investigated and deferred. Upstream training uses hardcoded `/data` paths, nori/S3-style dataset access, distributed CUDA setup, TensorBoard logging, and a training model path separate from the shipped `train_log` inference weights. No Practical-RIFE upstream files were patched.
- 2026-05-27: Milestone 9 validation completed in the CUDA environment via escalated exact `uv run` commands. `rife-preflight` passed end to end and reported repo/checkpoint/import/CUDA/model initialization/checkpoint load ok; `rife adapter-check` passed and documented the training refactor as not required for eval/inference; `rife validate-candidate --limit-samples 1 --no-lpips --disable-mlflow` wrote `/tmp/rife_candidate_validation_smoke/{metrics.csv,metrics_summary.csv,candidate_validation_report.json,sample_predictions/...}` and approved the one-sample pretrained candidate; `rife infer-video --codec libx264 --limit-pairs 1 --disable-mlflow` wrote `/tmp/dora_rife_2x.mp4` with 3 frames, H.264/yuv420p video, and 1 preserved AAC audio stream. Final exact validation passed: `uv run pytest` collected 27 tests and all passed; `uv run ruff check src tests` passed; RIFE CLI help rendered for `rife`, `rife infer-video`, and `rife validate-candidate`; `ffprobe` confirmed the smoke output video stream is H.264/yuv420p at 60000/1001 FPS and the audio stream is AAC.
- 2026-05-27: Added a Stage 1 baseline video inference follow-up. Added a ModelAdapter-compatible `BaselineAdapter` for `duplicate_left`, `blend`, and `farneback`, plus `baseline infer-video` and `configs/inference/baseline_2x.yaml`. Baseline video inference now reuses the shared PyAV/FFmpeg inference workflow, including original/generated frame interleaving, doubled FPS, audio remuxing, encoder settings, timing metrics, and optional MLflow logging.
- 2026-05-27: Added an all-target directory inference wrapper. New `infer-all-videos --input-dir ...` discovers supported videos recursively, runs Practical-RIFE v4.25, AMT-S, EMA-VFI-small, and all three baseline methods, writes outputs under `outputs/inference/<target>/...`, preserves relative input subdirectories, and reuses the shared PyAV/FFmpeg inference workflow for each run.
- 2026-05-27: Extended `infer-all-videos` with per-target measurement CSV export. Each model/baseline output directory now gets `inference_measurements.csv` with one row per input video containing model inference elapsed time, total elapsed time, pair/frame counts, FPS, throughput, audio preservation counts, output path, status, MLflow run id, and error text for failed runs.
- 2026-05-27: Extended `infer-all-videos` with repeatable `--target`/`--method` selection. The default remains all targets, but the command can now run selected models, selected baselines, or groups such as `models` and `baselines`; short aliases include `ema`, `amt`, `rife`, `duplicate_left`, `blend`, and `farneback`.
- 2026-05-27: Directory inference wrapper validation completed. Focused `tests/test_inference.py` passed, full `uv run pytest` collected 29 tests and all passed, `uv run ruff check src tests` passed, `infer-all-videos --help` rendered, and a one-video/one-pair smoke over `raw_data/tmp_test` with `--codec libx264 --disable-mlflow` completed 6/6 jobs and wrote outputs for Practical-RIFE, AMT-S, EMA-VFI-small, duplicate-left, blend, and Farneback under `outputs/inference`.
- 2026-05-31: Closed Stage 1 as practically complete for the narrowed Stage 2 inference-runtime planning handoff. No Stage 1 implementation code was changed during closeout. Remaining full MLflow-backed runs still require starting the local MLflow Compose stack, and AMT-S/Practical-RIFE fine-tuning remain deferred for the blockers already recorded below.

## Surprises & Discoveries

- The current data layout is `datasets/sources/...`, while some planning text and git status references older `datasets/vimeo_triplet/...` paths.
- `README.md` is empty and `main.py` is only a hello-world entrypoint.
- `pyproject.toml` targets Python 3.13 with modern Torch/Numpy, while all three upstream model repositories document older Python/dependency expectations.
- `pyproject.toml` includes WebDataset even though Stage 1 explicitly excludes WebDataset as the MVP canonical dataset format.
- Stage-required MLflow, PyAV, PySceneDetect, scikit-image/SSIM, LPIPS, timm, and OmegaConf are not currently listed in `pyproject.toml`.
- AMT-S training dataset expects flow files under a `flow/` directory; no such flow directory was observed in `datasets/sources/vimeo_triplet/`.
- Practical-RIFE training code hardcodes external data paths and nori/S3 dependencies, so it is not manifest-ready.
- `uv run` needs a writable cache; in this sandbox, `UV_CACHE_DIR=/tmp/uv-cache` was required because the default home cache path is read-only.
- The first EMA preflight attempt failed on missing `timm`; after adding and syncing the Stage 1 dependency baseline, EMA imports succeeded.
- EMA-VFI-small initialization is blocked in the current environment because CUDA is unavailable and upstream `Trainer.Model.device()` hardcodes `torch.device("cuda")`.
- In the user's real WSL CUDA environment, EMA-VFI-small preflight passes end to end. The sandbox `blocked` result is an environment limitation, not an EMA checkpoint or project preflight failure.
- The user manually fixed EMA-VFI's deprecated `timm.models.layers` imports by replacing them with `timm.layers`; after that upstream import cleanup, `ema-preflight` still passed successfully.
- `scenedetect` 0.7 depends on `opencv-python`, so `uv sync` installed `opencv-python` alongside the existing `opencv-python-headless` dependency.
- Adding `mlflow` resolved pandas to 2.3.3; the previous lock contained pandas 3.0.3 transitively. This should be reviewed if the project owner expects pandas 3.x, but pandas was not previously a direct project dependency.
- `uv lock` emitted warnings while normalizing invalid version specifiers from transitive package metadata, but lock resolution completed successfully.
- Vimeo source indexing can safely smoke-test against the existing dataset by writing the generated index to `/tmp`; this validates relative-path rows without modifying `datasets/sources/vimeo_triplet/sequence_index.csv`.
- PySceneDetect scene detection is wrapped with a single-scene fallback so preprocessing remains usable for smoke checks or videos where scene detection fails, while still using PySceneDetect when it is available.
- Human-facing docs need field-level config references close to the config files. `configs/data/README.md` is now the local reference for Milestone 2 data config behavior, while `docs/stage1_ml_core.md` links to it and focuses on operational workflow.
- The test MKV is Matroska/H.264/AAC and did not reproduce the reported Opus warning. The preprocessing path selects only the video stream for PyAV decoding; audio streams are listed in debug metadata but not decoded. The observed Opus warning on another MKV is likely an FFmpeg container/probing warning unless it is accompanied by a per-video failure.
- PyAV `container.duration` is in AV time-base units; using `container.duration * av.time_base` was incorrect in this environment because `av.time_base` is `1000000`. The correct conversion here is `container.duration / av.time_base`.
- For 1080p test clips, SSIM filtering and PNG writing are also material runtime contributors. Future optimization may resize for smoke runs, reduce SSIM workload, or add write batching only if needed.
- A tiny dataset-version smoke split can place only new-pool rows in train. The first Milestone 3 smoke exposed that strict `data_accumulation` caps could create an empty train manifest for such tiny versions. The builder now applies old/new ratio caps only when both pools are present in train; if train contains only old or only new rows, it keeps the available rows up to `train_budget`.
- By 2026-05-26, `datasets/global_sequence_index.csv` and `dataset_versions/stage1_default/` exist in the workspace, so Milestone 4 smoke checks can use a real generated manifest without changing dataset files.
- Importing baseline CLI commands imports LPIPS and MLflow packages successfully in the current environment; LPIPS model loading is still deferred until `compute_lpips` is enabled at runtime.
- An MLflow smoke run against `http://localhost:5000` can hang for a long time with default MLflow HTTP retries when no tracking server is running. The MLflow helper now sets bounded request timeout/retry defaults before contacting the tracking server.
- No MLflow service is currently running in this environment. Baseline MLflow logging support is implemented, but real MLflow logging validation remains deferred until the Compose stack is started.
- EMA-VFI-small adapter/import checks can run in the sandbox, but model initialization, inference, fine-tuning, and candidate validation remain blocked here by CUDA absence. These commands now fail with clear adapter errors instead of entering a partial upstream path.
- The local 2x inference implementation uses the project adapter rather than EMA upstream demo scripts. It reads videos with OpenCV, writes a new video with doubled FPS, and interleaves original/generated frames without modifying source videos.
- The PyAV inference output path preserves audio for the Dora MP4 smoke by remuxing the input AAC stream into the output MP4. The output is encoded as H.264/yuv420p at doubled FPS and is readable by `ffprobe`.
- AMT-S checkpoint loading with current PyTorch safe defaults initially failed because the checkpoint wrapper references `typing.OrderedDict`. Allowlisting `typing.OrderedDict`/`collections.OrderedDict` with `torch.serialization.safe_globals()` keeps safe weights loading enabled and loads the local `state_dict` successfully.
- AMT-S one-pair inference on the 1920x1080 Dora smoke is noticeably slower than EMA in this environment: the model step took about 19.0s for one pair, with total command time about 22.4s using `libx264`. This is acceptable for a smoke check but should be considered before full-video AMT runs.
- Practical-RIFE v4.25 and v4.26 local weight bundles currently have identical `RIFE_HDv3.py` and `IFNet_HDv3.py` files in the workspace. The default still uses v4.25 because upstream README recommends it for most scenes and the selected plan explicitly targets v4.25.
- Practical-RIFE `flownet.pkl` contains extra teacher/caltime training keys that are not present in the inference network files stored in the same `train_log` directory. Non-strict checkpoint loading is therefore the correct default for eval/inference with the local v4.25 bundle.
- Practical-RIFE inference requires input dimensions padded to an adequate divisor; unpadded tiny 32x32 tensors fail inside the upstream network because internal timestep/feature tensor sizes diverge. The adapter pads to divisor 128 and unpads predictions to original size.
- Stage 2 starts from a narrowed serving-readiness scope focused on EMA-VFI and Practical-RIFE. AMT-S remains present in the Stage 1 codebase but is not an active Stage 2 refactor target unless the project owner explicitly changes scope.

## Decision Log

- 2026-05-22: Selected option A as the implementation base. Rationale: it moves directly toward a working Stage 1 pipeline while preserving staged risk reduction.
- 2026-05-22: Borrowed only compact artifact contracts from option C. Rationale: required CSV/YAML/JSON contracts reduce path and schema drift without turning the plan into a contract-first workflow.
- 2026-05-22: Added `.env` plus `pydantic-settings` as the runtime configuration policy. Rationale: machine-local roots, endpoints, and secrets should not be encoded in experiment YAML files.
- 2026-05-22: Added early EMA-VFI-small preflight in Milestone 1. Rationale: dependency/checkpoint incompatibility is a high-impact risk and should be detected before large preprocessing work.
- 2026-05-22: Narrowed automated tests to critical behavior and emphasized smoke workflows. Rationale: Stage 1 needs useful validation, not broad shallow test coverage.
- 2026-05-22: Preserved strict model integration order: EMA-VFI-small, then AMT-S, then Practical-RIFE. Rationale: required by the Stage 1 plan and global model strategy.
- 2026-05-22: Used `src/video_interpolation` with setuptools package discovery and pytest `pythonpath`. Rationale: this preserves the selected plan's package layout while making `uv run python -m video_interpolation.cli` work after sync.
- 2026-05-22: Added official/trusted Stage 1 dependencies directly to `pyproject.toml`: PyAV, PySceneDetect, scikit-image, pandas, MLflow, LPIPS, timm, OmegaConf, and imageio. Rationale: all are either required by the stage plan or needed by the first EMA preflight; upstream dependency files were not installed blindly.
- 2026-05-22: Accepted MLflow's current pandas 2.x resolution for Milestone 1 because pandas was newly promoted to a direct Stage 1 dependency and MLflow is a required Stage 1 tool. Downstream implication: revisit if pandas 3.x is required later.
- 2026-05-22: Kept the EMA preflight as a standalone compatibility utility rather than a model adapter. Rationale: Milestone 1 requires early import/initialization/checkpoint detection but does not implement Milestone 6 adapter behavior.
- 2026-05-22: Treated CUDA absence as a preflight `blocked` result, not a code failure. Rationale: the current environment cannot initialize upstream EMA-VFI without CUDA, but the command reports the precise compatibility boundary for a CUDA-capable run.
- 2026-05-22: Added explicit model preflight resource cleanup and will apply the same pattern to future model-loading smoke checks. Rationale: preflight commands should not retain GPU memory after compatibility checks succeed, fail, or exit through a blocked path.
- 2026-05-22: Kept Milestone 2 source indexes separate from Milestone 3 global indexing. Rationale: source-level `sequence_index.csv` generation is in scope now, but combining sources into `datasets/global_sequence_index.csv` belongs to Milestone 3.
- 2026-05-22: Added the Vimeo indexing CLI with `--limit` and `--output` overrides. Rationale: this provides a safe smoke workflow that exercises real dataset paths without writing generated artifacts into the dataset tree.
- 2026-05-22: Implemented frame-step sampling as stride `frame_step + 1`, so `frame_step=0` means adjacent triplets, `1` skips one frame between neighbors, and `2` skips two frames. Rationale: this matches the stage requirement's `max_frame_step` values while preserving adjacent triplets as the default step.
- 2026-05-22: Vimeo list entries are validated as relative artifact paths before source index rows are built. Rationale: source indexing should not allow absolute, URI, or parent-directory paths to enter generated indexes.
- 2026-05-22: Kept detailed config field documentation in `configs/data/README.md` and linked from central docs. Rationale: local config documentation is easier to maintain as data configs grow, and central docs should stay operational rather than duplicate every field.
- 2026-05-22: Added UI-neutral progress callbacks in data functions and rendered them with Rich in the Typer CLI. Rationale: core preprocessing/indexing code stays reusable while long-running developer commands become inspectable.
- 2026-05-22: Added debug selectors and per-video limits to preprocessing instead of changing the full anime config defaults. Rationale: `--limit-videos` only limits file count, while `--max-duration-sec`, `--max-frames`, `--only-video`, and `--video-glob` provide safe diagnostics without weakening full preprocessing behavior.
- 2026-05-22: Chose `decode_strategy=auto` with grouped segment seek and sequential fallback as the smallest measured optimization. Rationale: diagnostics showed selected-frame decoding dominated runtime, especially for the MKV, and grouped segment seeking avoids seeking for every individual frame.
- 2026-05-22: Kept PySceneDetect in place and bounded it with the effective frame limit when `max_duration_sec` or `max_frames` is used. Rationale: SceneDetect was not the largest measured bottleneck on the test clips, but long-video smoke runs need duration/frame limits.
- 2026-05-22: Added global-index construction to `video_interpolation.data.indexing` rather than a separate module. Rationale: it is a direct extension of source-index handling and shares the same CSV contract and relative-path validation rules.
- 2026-05-22: Added dataset-version construction to `video_interpolation.data.versioning`. Rationale: split policy, triplet extraction, manifest writing, and dataset config provenance are separate from source indexing and should remain outside future PyTorch Dataset/model code.
- 2026-05-22: Dataset-version smoke artifacts were written under `/tmp/stage1_m3_smoke` instead of the project dataset tree. Rationale: this validates the Milestone 3 CLI on existing source indexes without modifying large dataset artifacts during a development smoke run.
- 2026-05-22: For train mixing, ratio caps are applied only when both old and new pools are present in the train split. Rationale: tiny smoke versions must remain usable and should not emit an empty `train_all.csv` solely because one conceptual pool is absent after source-video-level splitting.
- 2026-05-26: `UniversalTripletDataset` returns a dictionary with `left`, `middle`, `right`, and `metadata` keys, with tensors in `C,H,W` float `[0,1]` format. Rationale: this is simple for PyTorch/DataLoader usage and keeps metrics/domain aggregation metadata attached to each sample.
- 2026-05-26: Synchronized transforms are expressed as one callable over `TripletFrames`. Rationale: this avoids applying random transforms independently to left/middle/right frames and keeps augmentation behavior explicit for future training code.
- 2026-05-26: Metrics CSVs now require `prediction_name` in addition to sample/source metadata and metric columns. Rationale: baseline and future model evaluation CSVs need to distinguish multiple prediction sources in one file while preserving a compact metrics contract.
- 2026-05-26: Baseline evaluation defaults to MLflow enabled in config, but the CLI supports `--disable-mlflow` for tiny smoke runs. Rationale: Stage 1 requires real MLflow-backed runs, while local tests should remain possible when infrastructure is not running.
- 2026-05-26: LPIPS is implemented as an optional runtime scorer and is disabled in tiny CPU smoke validation with `--no-lpips`. Rationale: LPIPS is a required metric for real runs, but smoke tests should not need to load the model.
- 2026-05-26: MLflow infrastructure lives under `infra/mlflow/` with PostgreSQL and MinIO instead of the project root. Rationale: keeping service files grouped makes setup docs and future service expansion easier to navigate.
- 2026-05-26: MLflow helper sets bounded HTTP timeout and retry defaults before logging. Rationale: when the tracking server is unavailable, CLI commands should fail promptly with setup guidance instead of appearing stuck.
- 2026-05-27: Baseline `prediction_path` now points to a saved sample directory rather than a single generated PNG. Rationale: visual inspection should compare the generated middle frame with its neighboring input frames and ground truth in one folder.
- 2026-05-27: Kept EMA integration inside `EMAVFIAdapter` and did not modify upstream EMA-VFI code. Rationale: Stage 1 model-specific logic should be isolated in adapters, and the upstream repo already passed the user's real CUDA preflight after the manual `timm` import cleanup.
- 2026-05-27: Implemented EMA batch prediction as a sequential loop over `predict_pair`. Rationale: the selected plan explicitly allows optimized batch inference to be deferred until pair inference works end to end.
- 2026-05-27: Implemented local video inference with the project adapter rather than upstream `demo_2x.py`. Rationale: Stage 1 should exercise the same adapter path that evaluation, validation, and later serving will use.
- 2026-05-27: Candidate validation saves each visual sample as `im1.png`, `im2_gt.png`, `im2_generated.png`, and `im3.png`, matching the revised baseline inspection format. Rationale: model candidates should be visually inspectable with the same left/middle/right context as baselines.
- 2026-05-27: Padded EMA train/eval batches inside the adapter instead of rewriting upstream EMA model code. Rationale: inference already requires padded dimensions, and keeping padding at the adapter boundary fixes manifest image sizes without modifying the external repository.
- 2026-05-27: Replaced OpenCV video output with PyAV/FFmpeg encoding for `ema infer-video` and rejected legacy `mp4v` instead of mapping it to `libx264`. Rationale: output encoding now needs FFmpeg codec options, playback-compatible pixel formats, and audio remuxing; silently translating OpenCV fourcc values would hide a config semantic change.
- 2026-05-27: Implemented AMT-S as a thin project adapter over upstream AMT build utilities instead of patching `model_repos/AMT`. Rationale: eval/inference can use the upstream config/network/checkpoint directly, and Stage 1 model-specific glue belongs in adapters.
- 2026-05-27: Loaded AMT-S checkpoints with PyTorch `weights_only=True` plus safe globals for the checkpoint's `OrderedDict` wrapper. Rationale: this avoids disabling PyTorch safe loading for local model weights while remaining compatible with the upstream checkpoint format.
- 2026-05-27: Deferred AMT-S fine-tuning instead of generating flow files or removing AMT's flow loss. Rationale: upstream AMT training expects precomputed optical-flow targets; creating those artifacts or changing loss policy is non-trivial and should be approved separately.
- 2026-05-27: Recorded AMT-S fine-tuning as a future patch rather than implementing it now. Rationale: the project needs an explicit adaptation-style decision before changing training: either preserve upstream Vimeo layout and generate `flow/<clip>/<sequence>/{flow_t0.flo,flow_t1.flo}`, implement a manifest-based AMT training runner with flow-path resolution, or approve a no-flow AMT loss policy inspired by the upstream GoPro config.
- 2026-05-27: Generalized inference and candidate validation through adapter factories while preserving EMA command behavior. Rationale: AMT-S should reuse Stage 1's existing metrics, PyAV writer, MLflow, and report paths without weakening the shared `ModelAdapter` interface.
- 2026-05-27: Selected Practical-RIFE `RIFEv4.25/train_log` as the default Stage 1 RIFE target. Rationale: upstream README recommends 4.25 for most scenes, matching local weights are present, and this follows the approved Milestone 9 plan.
- 2026-05-27: Loaded Practical-RIFE checkpoints directly into `model.flownet` with `weights_only=True`, `module.` prefix stripping, and non-strict loading by default. Rationale: the shipped checkpoint contains training-only teacher/caltime keys not used by the inference network; non-strict loading keeps inference compatibility without patching upstream files.
- 2026-05-27: Deferred Practical-RIFE fine-tuning instead of rewriting upstream training. Rationale: upstream training has hardcoded `/data`, nori/S3, distributed CUDA, TensorBoard, and separate training-model assumptions that require a deliberate future adaptation decision.
- 2026-05-31: Closed this ExecPlan and moved it from `active/` to `completed/` for Stage 2 handoff. Rationale: the current user instruction considers Stage 1 practically complete and asks to start narrowed Stage 2 planning. Downstream implication: future work should create a new Stage 2 ExecPlan only after analysis and clarification questions are answered.

## Outcomes & Handoff

- Milestone 1 is implemented.
- Milestone 2 is implemented.
- Milestone 3 is implemented.
- Milestone 4 is implemented.
- Milestone 5 is implemented.
- Milestone 6 is implemented.
- Milestone 7 is implemented.
- Milestone 8 is implemented for AMT-S eval/inference/candidate validation; AMT-S fine-tuning is deferred/blocked by missing upstream flow targets.
- Milestone 9 is implemented for Practical-RIFE v4.25 eval/inference/candidate validation; Practical-RIFE fine-tuning is deferred/blocked by upstream training assumptions that are not manifest-ready.
- Delivered through Milestone 9: package skeleton, `.env.example`, settings, compact contracts, CLI, EMA/AMT/Practical-RIFE preflight utilities, dependency baseline, data config layout, source preprocessing/indexing modules, global sequence-index builder, manifest-only dataset-version builder, `UniversalTripletDataset`, metrics module, baseline evaluation and baseline video inference, all-target directory video inference with per-target measurement CSVs, MLflow helper module, MLflow/PostgreSQL/MinIO Compose infrastructure, EMA-VFI-small adapter, AMT-S adapter, Practical-RIFE adapter, baseline adapter, local 2x video inference with PyAV/FFmpeg output and audio preservation for EMA/AMT/Practical-RIFE/baselines, EMA fine-tuning/eval-only runner, EMA/AMT/Practical-RIFE candidate validation, focused tests, Stage 1 docs, config field docs, Rich CLI progress/summaries, debug preprocessing diagnostics, selected-video filters, per-video frame/duration limits, segment-seek selected-frame extraction, and updated project map.
- Validated: CLI help, settings loading/display, contract tests, preprocessing/indexing/versioning/dataset/metrics/baseline/validation/inference/AMT-adapter/RIFE-adapter tests, ruff, lock/sync, sandbox EMA preflight reporting a clear CUDA blocker after successful dependency/import checks, user-confirmed real WSL CUDA EMA preflight success, exact real CUDA `ema-preflight` success, safe Vimeo source-index CLI smoke run to `/tmp`, safe preprocessing CLI smoke run using a temporary `/tmp` video and dataset root, MP4/MKV diagnostic preprocessing runs under `raw_data/tmp_test`, safe global-index smoke to `/tmp/stage1_m3_smoke/global_sequence_index.csv`, safe dataset-version smoke to `/tmp/stage1_m3_smoke/smoke_version/`, triplet dataset loading smoke on `dataset_versions/stage1_default/test_all.csv`, baseline evaluation smoke to `/tmp/stage1_m4_m5_baseline_smoke`, EMA one-pair video inference smoke to `/tmp/stage1_ema_inference_smoke.mp4`, EMA PyAV `libx264` one-pair smoke to `outputs/inference/ema_vfi_small/dora_2x.mp4`, EMA PyAV `h264_nvenc` one-pair smoke to `/tmp/stage1_ema_pyav_nvenc_smoke.mp4`, EMA one-sample candidate validation smoke to `/tmp/stage1_ema_validation_smoke`, EMA one-step fine-tuning smoke to `/tmp/stage1_ema_finetune_smoke`, EMA checkpoint reload validation smoke to `/tmp/stage1_ema_checkpoint_reload_validation`, AMT preflight and adapter-check, AMT one-sample candidate validation smoke to `/tmp/amt_candidate_validation_smoke`, AMT one-pair video inference smoke to `/tmp/dora_amt_2x.mp4`, Practical-RIFE preflight and adapter-check, Practical-RIFE one-sample candidate validation smoke to `/tmp/rife_candidate_validation_smoke`, Practical-RIFE one-pair video inference smoke to `/tmp/dora_rife_2x.mp4`, all-target directory inference smoke to `outputs/inference/{practical_rife_v4_25,amt_s,ema_vfi_small,baselines/...}/001_2x.mp4`, `ffprobe` verification of AMT and Practical-RIFE output H.264/yuv420p video plus AAC audio, and MLflow Compose config rendering.
- MLflow logging smoke against `MLFLOW_TRACKING_URI=http://localhost:5000` was attempted previously and failed clearly because no MLflow service is running in this environment. Milestone 6/7 EMA and Milestone 8 AMT smoke commands used `--disable-mlflow`; start `infra/mlflow/docker-compose.yml` before validating real MLflow-backed training/inference/validation runs.
- Stage 1 is closed as practically complete for the next planning stage. The project map and human-facing docs already describe the Stage 1 modules, configs, commands, outputs, and known limitations closely enough for Stage 2 planning.
- The next session should create `.agent/docs/exec-plans/active/02_inference_runtime_refactor.execplan.md` only after the Stage 2 analysis is accepted and the project owner answers the open design questions.
- This ExecPlan is closed. Reopen or amend it only if the project owner explicitly requests a Stage 1 correction.
