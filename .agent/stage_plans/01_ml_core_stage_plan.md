# Stage 1 Plan — ML Core

## 1. Stage Goal

Stage 1 builds the local ML core of the Video Frame Interpolation project.

The result of this stage must be a reproducible local workflow:

```text
video files
→ sampled frame sequences
→ global dataset index
→ dataset version
→ PyTorch Dataset
→ baseline evaluation
→ model training / fine-tuning
→ candidate validation
→ MLflow logging
→ local 2x video inference
```

Stage 1 is not a web service stage. It focuses on data preparation, model training, model evaluation, experiment tracking, and local inference.

## 2. Out of Scope for Stage 1

Do not implement these components during Stage 1 unless the user explicitly changes the scope:

- FastAPI backend;
- Celery workers;
- Redis task queue;
- PostgreSQL application database;
- production MinIO data synchronization;
- frontend UI;
- automatic retraining triggers;
- monitoring and alerting;
- BentoML production serving;
- Triton / TensorRT deployment;
- WebDataset;
- distributed workers.

MLflow infrastructure is the only service-like infrastructure expected during Stage 1. It should be prepared with Docker Compose using MLflow, PostgreSQL, and MinIO for artifact storage.

## 3. Existing Inputs

The repository currently contains:

```text
raw_data/
  anime/
    Berserk/
    DBZ/
    Manie_Manie/
    One_Piece/
    Shadows_house/
    Wonder_egg_priority/

datasets/
  vimeo_triplet/
    sequences/
    tri_trainlist.txt
    tri_testlist.txt
  anime/
  SNU_FILM/
    test/
    test-easy.txt
    text-extreme.txt
    test-hard.txt
    text-medium.txt
  ucf101_interp_ours/

model_repos/
  AMT/
  EMA-VFI/
  Practical-RIFE/

model_weights/
  AMT/
  EMA-VFI/
  Practical-RIFE/
```

The copied model repositories contain upstream training and inference code. They are treated as external model modules that may be adapted carefully.

The first complete end-to-end training/inference route must be implemented with **EMA-VFI-small**. After that route works, the same pipeline must be extended to **AMT-S**, and then to **Practical-RIFE**.

IFRNet is not part of this project.

### 3.1. Raw data

Raw unprocessed video files are stored inside `raw_data/` directory. Videos inside are grouped by the data source/domain. Currently only a single new domain is prepared - anime domain.
Inside anime domain videos are grouped by the title, or in other words - source.
Data from thid directory is expected to be used to prepare new training data domain using stage's preprocessing pipeline.
Videos inside have two extensions: `.mkv` and `.mp4`, but the pipeline should be able to support more extensions, at least `.avi`, since it's one of the formats supported by the upstream models.

### 3.2. Datasets

The project currently contains predefined offline datasets, stored inside `datasets/` directory. Each dataset is grouped by the source of the data:

- vimeo-90k
- anime
- snu-film
- ucf101

Anime dataset source is currently empty, since this domain's data preprocessing and dataset creation is one of the stage's tasks.

Vimeo dataset is stored inside `datasets/vimeo_triplet/`. It holds two `.txt` files - `tri_testlist.txt` and `tri_trainlist.txt` - with triplet split lists. Also there is a folder `sequences/` containing dataset triplets, grouped by the source video clip and subsequence of the clip.

SNU-FILM dataset is expected to be used only for evaluation purpose. It stores 4 txt-files of triplet lists, grouped by the complexity of test example. Inside `test/` folder data is grouped by the source of videos: GOPRO (coming from GOPRO dataset) and YouTube. Each subfolder inside contains a sequence of images, not a triplet. It's expected that upstream model's repositories supporting those datasets, know how to work with this dataset's format.

UCF101 dataset is another dataset, expected to be used only for evaluation purpose. It stores only subfolders with test triplets and no manifest files. Inside each subfolder with images stored four images: left frame, right frame, middle ground truth frame and middle generated frame, provided by the authors of the dataset using their trained VFI model. These images may be used for visual comparison purpose to compare results of project's selected models with others results.

## 4. Required Stage Deliverables

Stage 1 is complete only when all of the following are available:

1. dataset preprocessing from video directories into sampled PNG frame sequences;
2. global sequence index builder;
3. dataset version builder that creates train/val/test triplet manifests;
4. PyTorch dataset for triplet-level manifests;
5. baseline evaluation on manifests;
6. MLflow tracking infrastructure and logging helpers;
7. fine-tuning pipeline for EMA-VFI-small;
8. fine-tuning pipeline for AMT-S;
9. fine-tuning pipeline for Practical-RIFE;
10. candidate validation on test manifests;
11. local 2x video inference CLI for at least EMA-VFI-small, then AMT-S and Practical-RIFE;
12. concise human-facing documentation for all CLI commands and configs;
13. updated `.agent/docs/PROJECT_MAP.md`.

## 5. Environment and Dependency Preparation

Use `uv` for dependency management.

Before implementing ML code, the agent must inspect dependency requirements from:

```text
model_repos/AMT/environment.yaml
model_repos/EMA-VFI/README.md
model_repos/Practical-RIFE/requirements.txt
```

The agent must not blindly install these files. Instead, it must create a unified dependency list using current compatible package versions where possible.

Required dependency families for Stage 1:

- PyTorch;
- torchvision;
- Pillow;
- OpenCV;
- PyAV;
- PySceneDetect;
- scikit-image or equivalent SSIM implementation;
- numpy;
- pandas;
- pydantic;
- pydantic-settings;
- python-dotenv;
- rich;
- sqlalchemy;
- psycopg[binary];
- tqdm;
- typer;
- mlflow;
- lpips or another explicitly chosen LPIPS implementation.

Do not install `cudatoolkit` as a Python dependency. CUDA is provided by the system / PyTorch wheel source. The project owner will configure PyTorch CUDA sources in `pyproject.toml`.

If dependency conflicts appear, the agent must ask the user before downgrading packages. The agent may adapt project code to newer package APIs, but must not silently downgrade dependencies to old upstream repo versions.

## 6. MLflow Infrastructure

Stage 1 must include Docker Compose infrastructure for MLflow:

- MLflow Tracking Server;
- PostgreSQL for MLflow metadata;
- MinIO for MLflow artifacts;
- a MinIO bucket for MLflow artifacts.

The local training code must log to MLflow through `MLFLOW_TRACKING_URI`.

Stage 1 does not require using MinIO as the source of truth for training datasets. Dataset files may remain on the local filesystem during Stage 1.

## 7. Dataset Preprocessing Core

### 7.1 Purpose

The preprocessing module converts raw videos into sampled frame sequences stored as PNG files.

It is used for:

- preparing additional training data from local videos;
- preparing additional domain datasets such as anime;
- later reusing parts of the logic for inference input preparation.

### 7.2 Technology

Use:

- PyAV for video reading;
- PySceneDetect for scene detection;
- Pillow/OpenCV for image writing and lightweight image operations;
- PNG as the frame storage format.

Do not implement preprocessing by dumping all video frames to disk first. The module must save only sampled sequences.

### 7.3 Sequence Length

The module must support:

```text
sequence_length >= 3
```

Default:

```text
sequence_length = 3
```

If `sequence_length = N`, the saved sequence must contain:

```text
im1.png
im2.png
...
imN.png
```

The training pipeline remains 2x interpolation only. Longer sequences are stored for future flexibility, but triplet manifests are generated later by Dataset Version Builder.

### 7.4 Optional Resize

Input videos may have arbitrary resolution. The preprocessing module must support optional resizing.

Configuration:

```text
resize = null
resize = WIDTHxHEIGHT
```

Rules:

- default is `resize = null`;
- if `resize = null`, frames are saved at original resolution;
- if resize is set, every saved frame in the sequence is resized to the configured resolution.

### 7.5 Scene Detection

For each video:

1. run scene detection;
2. create safe intervals inside scene boundaries;
3. discard scenes shorter than the required sequence span;
4. sample only inside scenes;
5. never save a sequence that crosses a scene cut.

### 7.6 Frame Step

Frame step is supported only when:

```text
sequence_length = 3
```

Configuration:

```text
max_frame_step ∈ {0, 1, 2}
```

Meaning:

```text
0 → i, i+1, i+2
1 → i, i+2, i+4
2 → i, i+3, i+6
```

If `max_frame_step > 0`, each sampled triplet chooses its actual step randomly from:

```text
0..max_frame_step
```

Default:

```text
max_frame_step = 0
```

For `sequence_length > 3`, frame step must not be used. The implementation must reject this configuration clearly.

### 7.7 Sequence Span

For triplets with frame step:

```text
sequence_span = 1 + (sequence_length - 1) * (frame_step + 1)
```

For ordinary neighboring sequences:

```text
sequence_span = sequence_length
```

Scene filtering and safe interval logic must use `sequence_span`, not only `sequence_length`.

### 7.8 Sampling Quota

Do not extract all possible sequences from a video. Use a bounded sampling quota.

Fixed quota formula for Stage 1:

```text
target_sequences = min(
    max_sequences_per_video,
    max(
        min_sequences_per_video,
        ceil(quota_scale * log2(valid_frames / sequence_span + 1))
    )
)
```

Default values:

```text
min_sequences_per_video = 5
max_sequences_per_video = 200
quota_scale = 20
random_seed = 42
```

All values must be configurable.

### 7.9 Sampling Algorithm

For each video:

1. run scene detection;
2. build safe scene intervals;
3. filter intervals shorter than `sequence_span`;
4. compute target sequence quota;
5. repeatedly choose an interval with probability proportional to remaining interval length;
6. sample a valid start frame inside that interval;
7. save the selected sequence;
8. remove the used frame span from the available intervals;
9. stop when the quota is reached or when no valid intervals remain.

If the quota cannot be reached because valid intervals are exhausted, finish without error and report how many sequences were saved.

### 7.10 Static Triplet SSIM Filter

For triplets only, reject overly static triplets.

Rule:

```text
reject triplet if:
SSIM(im1, im2) >= 0.95
or
SSIM(im2, im3) >= 0.95
```

This filter applies only when `sequence_length = 3`.

For `sequence_length > 3`, do not apply this SSIM filter in Stage 1.

### 7.11 Preprocessing Output Format

For each source group:

```text
datasets/sources/<source_group>/
  sequences/
    <source_video_id>/<sequence_id>/im1.png
    <source_video_id>/<sequence_id>/im2.png
    ...
    <source_video_id>/<sequence_id>/imN.png
  sequence_index.csv
```

Preprocessing must not create train/val/test splits.

### 7.12 `sequence_index.csv`

Required fields:

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

All paths must be relative. Do not write absolute filesystem paths.

## 8. Global Dataset Index Builder

### 8.1 Purpose

The Global Dataset Index Builder combines all source-level `sequence_index.csv` files into a single index.

Input:

```text
datasets/sources/*/sequence_index.csv
```

Output:

```text
datasets/global_sequence_index.csv
```

### 8.2 Required Behavior

The builder must:

- preserve source metadata;
- validate required columns;
- preserve relative paths;
- preserve `source_group`;
- preserve `source_dataset`;
- produce deterministic output ordering where practical.

`source_group` is the high-level data group, for example:

```text
vimeo
adobe240
gopro
anime
video_games
online_packages
```

## 9. Dataset Version Builder

### 9.1 Purpose

Dataset Version Builder creates a concrete train/val/test dataset version for model training and evaluation.

Input:

```text
datasets/global_sequence_index.csv
dataset_builder_config.yaml
```

Output:

```text
dataset_versions/<dataset_version_id>/
  train_all.csv
  val_all.csv
  test_all.csv
  dataset_config.yaml
```

### 9.2 Split Policy

Splits are logical and manifest-based.

Do not physically move image files into train/val/test folders.

Split must happen at the `source_video_id` level, not at individual sequence or triplet level. This prevents leakage between train, validation, and test.

### 9.3 Triplet-Level Manifest Generation

Training uses triplet-level manifests.

Dataset Version Builder converts sequence-level records into triplet-level records.

If `sequence_length = 3`, the triplet is:

```text
im1.png, im2.png, im3.png
```

If `sequence_length > 3`, support these policies:

```text
first_triplet
center_triplet
wide_triplet
all_local_triplets
```

Definitions:

- `first_triplet`: `im1, im2, im3`;
- `center_triplet`: three neighboring frames around the sequence center; requires enough frames;
- `wide_triplet`: `im1, imM, imN`, where `M` is the exact middle frame; this policy requires odd `sequence_length`;
- `all_local_triplets`: all consecutive local triplets `(im1, im2, im3)`, `(im2, im3, im4)`, etc.

Default:

```text
triplet_policy = wide_triplet
```

If `wide_triplet` is requested for even `sequence_length`, the builder must fail with a clear configuration error.

### 9.4 Triplet Manifest Fields

Required fields for `train_all.csv`, `val_all.csv`, and `test_all.csv`:

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

Fixed value:

```text
t_value = 0.5
```

### 9.5 Dataset Version Config

`dataset_config.yaml` must record:

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

## 10. Data Mixing Policy

### 10.1 Concepts

Dataset Builder must support these conceptual pools:

```text
old_pool = base_static + online_accepted
new_pool = online_pending
```

In Stage 1, `new_pool` may be simulated by a local domain dataset such as anime.

### 10.2 Training Budget

Training budget is measured in triplet samples.

For each dataset version:

```text
train_budget = size of selected base train subset
```

The dataset version should remain bounded by this budget unless the config explicitly says otherwise.

### 10.3 `data_accumulation` Mode

Default mixing mode:

```text
data_accumulation
```

Rules:

```text
old_min_ratio = 0.70
new_max_ratio = 0.30
```

- if new data is less than 30% of budget, include all new data and fill the rest with old data;
- if new data is more than 30% of budget, sample 30% new and 70% old;
- old data must not fall below 70%;
- do not oversample in this mode.

### 10.4 `quality_drift` Mode

This mode is designed for future retraining trigger logic.

Rules:

```text
new_min_ratio = 0.20
old_max_ratio = 0.80
```

If new data is below 20% of budget, oversampling new data is allowed until it reaches 20%.

Oversampling is allowed only in `quality_drift` mode.

### 10.5 Accepted Data

After successful fine-tuning, new data used in training will later be moved conceptually from:

```text
online_pending
```

to:

```text
online_accepted
```

This is not required to be automated in Stage 1, but the data model and config should not prevent it.

## 11. PyTorch Dataset

### 11.1 Required Class

Implement:

```text
UniversalTripletDataset
```

It reads triplet-level manifests only.

The dataset must not decide how to extract triplets from longer sequences. That logic belongs to Dataset Version Builder.

### 11.2 Required Behavior

The dataset must:

1. read `train_all.csv`, `val_all.csv`, or `test_all.csv`;
2. use runtime `DATASET_ROOT`;
3. construct file paths as `DATASET_ROOT + relative_path`;
4. read PNG images;
5. return left, middle, right frames;
6. return metadata needed for metrics and domain aggregation;
7. apply synchronized augmentations to all three frames.

## 12. Baseline Module

### 12.1 Components

Baseline module has two separate components:

```text
Baseline Evaluation
Baseline Inference
```

### 12.2 Baseline Evaluation

Evaluation works on triplet manifests.

Required methods:

1. frame duplication;
2. frame blending;
3. Farneback optical flow.

Implementation for evaluation:

- Python/OpenCV for duplication;
- Python/OpenCV for blending;
- Python/OpenCV Farneback optical flow for optical-flow baseline.

Evaluation input:

```text
test_all.csv
DATASET_ROOT
```

For each sample:

```text
left_frame + right_frame → predicted_middle_frame
```

Then compare prediction with `mid_frame`.

Metrics:

```text
PSNR
SSIM
LPIPS
```

All baseline evaluation results must be logged to MLflow.

### 12.3 Baseline Inference

Baseline video inference is used for local video processing.

Use FFmpeg filters where suitable:

- `tblend` for frame blending;
- `minterpolate` for motion interpolation;
- Python/OpenCV/Pillow fallback for frame duplication.

Baseline inference is not the first critical path. Baseline evaluation is required earlier.

## 13. Metrics

Required metrics:

```text
PSNR
SSIM
LPIPS
```

Metric code must support:

- per-sample metrics;
- aggregated metrics;
- per-domain aggregation based on manifest metadata;
- CSV export;
- MLflow logging.

## 14. Training Runner

### 14.1 Purpose

Training Runner is an orchestration wrapper over selected model training code.

It adapts upstream model repositories to this project by providing:

- consistent configs;
- consistent dataset inputs;
- consistent checkpoint handling;
- MLflow logging;
- unified adapter interface.

### 14.2 Training Modes

Training Runner must support:

```text
finetune
scratch_train
eval_only
```

Required in Stage 1:

```text
finetune
eval_only
```

`scratch_train` must be represented in configs and CLI, but may be implemented after fine-tuning works.

### 14.3 Train / Validation / Test Separation

Training uses:

```text
train_all.csv
val_all.csv
```

During training:

- train split is used for optimization;
- validation split is used for intermediate model selection;
- test split must not be used.

Candidate Validation uses:

```text
test_all.csv
```

### 14.4 Configuration

All training parameters must come from config files.

Required config groups:

- model;
- mode;
- checkpoint source;
- dataset paths;
- optimizer;
- scheduler if used;
- training steps or epochs;
- augmentation;
- MLflow logging;
- output directory.

CLI commands are developer conveniences. Core logic must be callable through Python functions/classes.

### 14.5 Checkpoints

Checkpoint sources:

1. local file path;
2. MLflow Model Registry.

Stage 1 may start with local checkpoint paths. MLflow checkpoint loading must be supported later in Stage 1 if practical.

## 15. Model Adapters

### 15.1 Purpose

Model adapters provide a common interface between the project pipeline and external VFI model repositories.

The adapter must hide model-specific details such as checkpoint format, model construction, training script invocation, tensor conventions, preprocessing, postprocessing, and inference calls.

The adapter must not own dataset preprocessing or dataset version construction. Dataset preparation is handled by the preprocessing modules, Global Dataset Index Builder, Dataset Version Builder, and `UniversalTripletDataset`.

While some upstream models may have scripts for training or evaluation, and the adapter may wrap them for conveniece istead of writing training/validation logic from scratch, some models may not have such scripts. In case of `Practical-RIFE` model, there is no code for evaluation on any of the datasets other models support. This means, that for this specific model such logic should be implemented from scratch, based on existing code and using existing interfaces, provided by the upstream repo. Then this logic should be wrapped into adapter.

Note, that it's preferable to adapt or adjust existing upstream model's code instead of writing logic from scratch. Writing new logic as a part of the upstream models is acceptable, when there is no existing implementation, supported by these repos, but required by the stage goals.

### 15.2 Interface

Implement a common adapter interface:

```text
ModelAdapter
```

Required methods:

```text
validate_environment()
build_model()
load_checkpoint()
save_checkpoint()
train()
predict_pair()
predict_batch()
predict()
__call__()
```

Method meanings:

```text
validate_environment()
```

Checks that the adapter can run in the current environment. It should verify required model repository paths, checkpoint paths, config files, imports, and device availability. It must not install packages, modify the environment, or silently patch dependency issues.

```text
build_model()
```

Creates the model object for the selected architecture and configuration.

```text
load_checkpoint()
```

Loads pretrained, fine-tuned, or candidate weights into the model.

```text
save_checkpoint()
```

Saves adapter-compatible checkpoints after training or fine-tuning.

```text
train()
```

Runs the model-specific training or fine-tuning procedure using the provided training config, `train_all.csv`, and `val_all.csv`.

The training method uses only train and validation splits. It must not use `test_all.csv`.

```text
predict_pair()
```

Runs inference for one pair of input frames and returns the synthesized middle frame.

This method is the first required inference path and must work before batch inference is optimized.

```text
predict_batch()
```

Runs inference for a batch of frame pairs.

The batch represents neighboring frame pairs from an input video, for example:

```text
[
  [im0, im1],
  [im1, im2],
  [im2, im3],
  ...
  [imN-1, imN]
]
```

In the first implementation, `predict_batch()` may internally loop over `predict_pair()`. Later it can be replaced with an optimized batched implementation without changing the public interface.

If an upstream model does not expose batch inference directly, do not block the first working sequential path. Optimized batch inference should be added only after sequential inference works end to end.

```text
predict()
```

Canonical inference method. It should call `predict_batch()`.

```text
__call__()
```

Alias for `predict()`, so the adapter can be used as a callable inference object.

### 15.3 Dataset Responsibility Boundary

The adapter must not build dataset versions.

The following responsibilities belong outside the adapter:

- video preprocessing;
- scene detection;
- sequence sampling;
- train/val/test split construction;
- old/new data mixing;
- creation of `train_all.csv`, `val_all.csv`, `test_all.csv`;
- global dataset indexing.

The adapter may include small internal conversion helpers only when a specific upstream model repository requires a temporary model-specific dataset config or path format. Such helpers must not replace the project-level Dataset Version Builder.

### 15.4 Training and Validation Boundary

Training and candidate validation are separate stages.

During training:

- use `train_all.csv` for training;
- use `val_all.csv` for intermediate validation and checkpoint selection;
- log training and validation metrics to MLflow.

During candidate validation:

- use `test_all.csv`;
- evaluate the final candidate checkpoint;
- compute test metrics on fixed benchmark/domain splits;
- produce a candidate validation report;
- decide whether the candidate is approved or rejected.

Candidate validation may use the adapter inference methods, especially `predict_batch()`.

### 15.5 Future Runtime Backend Compatibility

Stage 1 uses PyTorch inference through adapters.

The adapter interface must be designed so that inference can later be executed through either PyTorch or ONNX Runtime without changing the public prediction API:

```text
predict_pair()
predict_batch()
predict()
__call__()
```

ONNX export and ONNX Runtime inference are not required in Stage 1. They are later serving optimizations.

The intended future structure is:

```text
ModelAdapter
  -> preprocessing / tensor formatting
  -> runtime backend
       -> PyTorch first
       -> ONNX Runtime later
  -> postprocessing
```

Do not duplicate preprocessing or postprocessing in a future BentoML service. Those responsibilities must remain inside adapters.

### 15.6 Future BentoML Serving Compatibility

BentoML serving must call model adapters through the public prediction API. BentoML must not duplicate model-specific preprocessing, tensor formatting, checkpoint loading rules, runtime calls, or postprocessing logic.

The intended future serving call chain is:

```text
BentoML Service / Runner
  -> ModelAdapter.predict_batch()
  -> PyTorch Runtime first
  -> ONNX Runtime later if exported model is available
```

A future BentoML custom Runnable/Runner should create the adapter inside the runner process from configuration and BentoML Model Store references. Do not design Stage 1 around passing an already constructed adapter instance into a serving runner.

Primary serving strategy for the first service stage:

```text
one BentoML service
one runner
one active production model
```

Multi-GPU extension strategy:

```text
one BentoML service per active model
one runner inside each service
one GPU assignment per service/container
```

Dynamic loading and unloading of several large VFI models inside one runner is not the MVP strategy.

### 15.7 Implementation Order

Implement model integration in this exact order:

1. `EMAVFIAdapter` for EMA-VFI-small;
2. `AMTAdapter` for AMT-S;
3. `PracticalRIFEAdapter` for Practical-RIFE.

Do not start adapting AMT-S or Practical-RIFE before the EMA-VFI-small pipeline can run end to end through dataset loading, fine-tuning, validation, MLflow logging, and local inference.

### 15.8 Repository Adaptation Policy

The model repositories under `model_repos/` may be adapted carefully.

Allowed:

- add wrapper scripts;
- add adapter-friendly entrypoints;
- adjust dataset paths;
- adapt checkpoint loading;
- replace or duplicate TensorBoard logging with MLflow logging;
- patch small compatibility issues;
- add thin compatibility layers required for project integration.

Not allowed:

- large unexplained rewrites;
- deleting upstream code;
- changing model behavior without documenting why;
- breaking original inference/training scripts unnecessarily.

If a compatibility issue requires either downgrading a package or adapting model code, ask the user before choosing.

## 16. MLflow Logging

Every training, evaluation, baseline, and candidate validation run must log to MLflow unless explicitly disabled for a tiny local unit test.

### 16.1 Params

Log at least:

```text
model_name
model_adapter
mode
dataset_version_id
base_checkpoint
learning_rate
batch_size
num_epochs
num_steps
crop_size
sequence_length
triplet_policy
frame_step
resize
old_new_ratio
augmentation_config
```

### 16.2 Metrics

Log at least:

```text
train_loss
val_loss
val_psnr
val_ssim
val_lpips
test_psnr_base
test_ssim_base
test_lpips_base
test_psnr_new
test_ssim_new
test_lpips_new
training_time_sec
```

Metrics that are not available for a specific run should be omitted, not logged as fake values.

### 16.3 Artifacts

Log at least:

```text
training_config.yaml
dataset_config.yaml
train_all.csv
val_all.csv
test_all.csv
metrics.csv
candidate_validation_report.json
candidate_validation_metrics.csv
best_checkpoint
last_checkpoint
sample_predictions/
```

### 16.4 TensorBoard Replacement

If upstream model code writes to TensorBoard, adapt or duplicate logging so that MLflow is the primary tracker.

TensorBoard is not the project’s main tracking tool.

## 17. Candidate Validation

### 17.1 Purpose

Candidate Validation is a separate module that runs after training.

It evaluates a trained checkpoint on test data only.

### 17.2 Inputs

```text
candidate checkpoint
optional baseline/previous checkpoint
test manifests
validation config
```

### 17.3 Outputs

```text
candidate_validation_report.json
candidate_validation_metrics.csv
MLflow artifacts
approved/rejected decision
```

### 17.4 Evaluation Domains

Evaluation must be reported separately by domain when domain metadata exists.

Minimum expected domains:

```text
vimeo_test
anime_test
ucf101_test
snu_film_test
online_holdout_test
```

The fixed base benchmark must remain stable across runs to allow fair comparison between model generations.

### 17.5 Approval Rule

Initial rule:

```text
approved if:
  new-domain metric improves
  and base-domain degradation is within configured threshold
```

Default thresholds:

```text
max_base_psnr_drop = 0.20
max_base_ssim_drop = 0.005
max_base_lpips_increase = 0.02
```

Thresholds must be configurable.

## 18. Local Inference Core

### 18.1 Purpose

Stage 1 must support local video interpolation without frontend, API, or queue.

### 18.2 Command

Expected developer-facing command:

```text
interpolate_video
```

### 18.3 Required Behavior

Pipeline:

1. accept input video path;
2. extract frames to numbered PNG files:
   ```text
   1.png, 2.png, 3.png, ...
   ```
3. process consecutive frame pairs;
4. generate one middle frame for each pair;
5. interleave original and generated frames;
6. assemble a 2x FPS output video.

### 18.4 Inference Mode

First implementation:

```text
sequential pair inference
```

After it works:

```text
batch pair inference
```

Batch inference is optional in Stage 1 and should not block the sequential path.

### 18.5 Video Assembly

Video assembly belongs to the inference module.

Use FFmpeg CLI or PyAV. Prefer the simpler reliable implementation first.

## 19. BentoML Future Integration

BentoML is the selected production model serving and inference API tool for later service stages.

Stage 1 does not implement BentoML serving. Stage 1 must only ensure that model adapters expose a stable inference API that can later be called from BentoML.

### 19.1 Release Flow

Future release flow:

```text
training / experiments
→ MLflow Model Registry
→ approved model
→ release script loads approved model from MLflow
→ save model to BentoML Model Store
→ BentoML service
→ Docker image
```

MLflow remains the source of truth for training history, experiments, candidate validation, and approved model versions. BentoML Model Store is used for release models prepared for inference serving.

### 19.2 Serving Responsibility Boundary

FastAPI remains responsible for non-inference application logic in later stages:

- upload handling;
- input validation;
- request metadata;
- status polling;
- result links;
- orchestration.

BentoML is responsible for model inference APIs and serving.

BentoML serving code must call adapters. It must not reimplement model-specific preprocessing or postprocessing.

### 19.3 Serving Deployment Strategy

Initial serving strategy:

```text
one BentoML service
one runner
one active production model
```

This is the primary MVP path.

Multi-GPU extension:

```text
ema-service  -> EMA runner  -> assigned GPU
amt-service  -> AMT runner  -> assigned GPU
rife-service -> RIFE runner -> assigned GPU
```

This extension uses one service per active model to simplify GPU memory isolation, independent restarts, and model-specific resource configuration.

A single BentoML service with multiple runners is not forbidden, but it is not the preferred MVP path. Dynamic switching between several large models inside one runner is not planned.

### 19.4 ONNX Runtime

ONNX export and ONNX Runtime inference are not required in Stage 1.

The adapter API must remain stable so that future ONNX support can be added internally as a runtime backend without changing external inference calls.

## 20. Tests and Validation Policy

Write a small number of useful tests only for critical behavior.

Do not write tests that only check hardcoded config values, trivial field presence, or implementation details with no behavioral value.

Good Stage 1 tests include:

- scene interval filtering does not cross scene boundaries;
- sampler terminates safely when quota cannot be reached;
- SSIM static triplet filter rejects the correct samples;
- Dataset Version Builder splits by `source_video_id`, not by sample;
- triplet manifest generation produces valid relative paths;
- `UniversalTripletDataset` loads a small synthetic manifest and returns correctly shaped tensors;
- baseline evaluation runs on a tiny synthetic dataset;
- metric functions return reasonable values on known identical/different images.

Smoke tests are more important than broad unit-test coverage.

Required smoke workflows:

1. preprocess one or two tiny videos;
2. build global index;
3. build tiny dataset version;
4. load dataset through DataLoader;
5. run baseline evaluation;
6. run one eval-only model pass if possible;
7. run local interpolation on a short video.

## 21. Documentation Requirements

Stage 1 documentation must explain how to run each implemented workflow.

For each CLI command or script, document:

- purpose;
- required inputs;
- input format;
- output location;
- all important parameters;
- allowed parameter values;
- default values;
- side effects;
- example command;
- expected successful output.

Documentation must be practical enough for the project owner to run and test the code without reading implementation files.

## 22. Expected Implementation Order

Recommended implementation order:

```text
1. Finalize the Python project structure, `pyproject.toml`, dependency baseline, and local configuration layout.
2. Add MLflow Docker Compose setup with PostgreSQL metadata storage and MinIO artifact storage.
3. Implement MLflow connectivity smoke test and basic MLflow logging helpers.
4. Implement dataset preprocessing core: PyAV video reading, PySceneDetect scene detection, scene-safe sequence sampling, optional resize, frame-step sampling for triplets, SSIM static triplet filtering, PNG sequence writing, and `sequence_index.csv` creation.
5. Implement Vimeo-90K Triplet indexing support for the existing `datasets/sources/vimeo_triplet/` structure.
6. Implement global dataset index builder over `datasets/sources/*/sequence_index.csv`.
7. Implement Dataset Version Builder: logical split by `source_video_id`, triplet-level manifest generation, triplet policies, dataset mixing policy, and `dataset_config.yaml`.
8. Implement `UniversalTripletDataset` for reading triplet manifests with relative paths and synchronized augmentations.
9. Implement metrics module: PSNR, SSIM, LPIPS, per-sample metrics, aggregated metrics, per-domain aggregation, CSV export, and MLflow logging.
10. Implement baseline evaluation on triplet manifests: frame duplication, frame blending, and Farneback optical flow.
11. Implement the common `ModelAdapter` interface and shared adapter utilities.
12. Implement `EMAVFIAdapter` for EMA-VFI-small as the first end-to-end model integration.
13. Implement EMA-VFI-small fine-tuning from local pretrained weights using `train_all.csv` and `val_all.csv`.
14. Implement EMA-VFI-small eval-only and candidate validation on `test_all.csv`.
15. Implement local 2x video inference CLI for EMA-VFI-small with sequential pair inference.
16. Add optional batch inference path for EMA-VFI-small only after sequential pair inference works.
17. Extend the existing adapter pipeline to AMT-S.
18. Extend the existing adapter pipeline to Practical-RIFE.
19. Add consolidated Stage 1 documentation for configs, CLI commands, workflows, outputs, and known limitations.
20. Update `.agent/docs/PROJECT_MAP.md` and complete the Stage 1 ExecPlan handoff.
```
Do not integrate AMT-S or Practical-RIFE before EMA-VFI-small works end to end through dataset loading, fine-tuning, validation, MLflow logging, and local inference.
    
Do not implement BentoML serving, FastAPI, Celery, Redis, PostgreSQL application tables, frontend, automatic retraining triggers, or monitoring during Stage 1 unless the user explicitly changes the stage scope.


## 23. Stage Completion Criteria

Stage 1 is complete when:

1. local data preprocessing works on at least one small video directory;
2. dataset version manifests can be generated reproducibly;
3. DataLoader can read generated manifests;
4. baseline evaluation works and logs to MLflow;
5. EMA-VFI-small can run through fine-tuning/eval/validation locally;
6. AMT-S is integrated into the same workflow;
7. Practical-RIFE is integrated into the same workflow;
8. local 2x video inference works for at least EMA-VFI-small and preferably all three models;
9. MLflow contains runs, metrics, configs, and artifacts;
10. documentation explains how to run the workflows;
11. `PROJECT_MAP.md` is updated;
12. the active ExecPlan contains progress, decisions, validation results, and handoff.
