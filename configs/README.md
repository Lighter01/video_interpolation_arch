# Stage 1 Config Layout

Stage 1 uses YAML configs for pipeline and experiment parameters. Runtime roots and endpoints stay in `.env`.

Planned config groups:

- `data/` - preprocessing, source indexing, global index, and dataset version parameters.
- `models/` - model adapter and model architecture selections.
- `training/` - fine-tuning and eval-only runner parameters.
- `validation/` - candidate validation thresholds and report options.
- `inference/` - local model inference workflow parameters.
- `baselines/` - baseline evaluation options.

Implemented data configs:

- `data/index_vimeo_triplet.yaml` - source-level indexing for existing Vimeo triplets.
- `data/preprocess_anime.yaml` - raw anime video sampling into PNG sequences.
- `data/global_index.yaml` - global sequence-index construction from source indexes.
- `data/dataset_version.yaml` - manifest-only train/val/test dataset version construction.

See `data/README.md` for fields, allowed values, behavioral effects, rerun guidance, inputs, outputs, and inspection commands.

Implemented baseline configs:

- `baselines/baseline_eval.yaml` - duplication, blending, and Farneback baseline evaluation on triplet manifests.
- `inference/baseline_2x.yaml` - local 2x video inference with one baseline method.

See `baselines/README.md` for fields, allowed values, MLflow behavior, outputs, and smoke commands.

Implemented EMA-VFI-small configs:

- `models/ema_vfi_small.yaml` - adapter/checkpoint/device parameters.
- `inference/ema_vfi_small_2x.yaml` - local 2x video inference parameters.
- `training/ema_vfi_small_finetune.yaml` - fine-tuning and eval-only runner parameters.
- `validation/ema_vfi_small_candidate.yaml` - candidate validation thresholds, outputs, and MLflow settings.

See the README files in those directories for field references and safe smoke guidance.

See `inference/README.md` for single-video inference configs and the `infer-all-videos --input-dir ...` directory-wide wrapper that runs every implemented model and baseline over a video folder.

Implemented AMT-S configs:

- `models/amt_s.yaml` - AMT-S adapter/checkpoint/device/config parameters.
- `inference/amt_s_2x.yaml` - local AMT-S 2x video inference parameters.
- `validation/amt_s_candidate.yaml` - AMT-S candidate/eval-only validation thresholds, outputs, and MLflow settings.

AMT-S fine-tuning is deferred until the project either generates the upstream-required flow files or explicitly changes the AMT training loss policy.

Implemented Practical-RIFE configs:

- `models/practical_rife_v4_25.yaml` - Practical-RIFE v4.25 adapter/checkpoint/device parameters.
- `inference/practical_rife_v4_25_2x.yaml` - local Practical-RIFE 2x video inference parameters.
- `validation/practical_rife_v4_25_candidate.yaml` - Practical-RIFE candidate/eval-only validation thresholds, outputs, and MLflow settings.

Practical-RIFE fine-tuning is deferred until the project chooses how to replace upstream hardcoded `/data`, nori/S3, distributed, and TensorBoard training assumptions with project manifests and MLflow.
