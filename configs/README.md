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

See `baselines/README.md` for fields, allowed values, MLflow behavior, outputs, and smoke commands.

Implemented EMA-VFI-small configs:

- `models/ema_vfi_small.yaml` - adapter/checkpoint/device parameters.
- `inference/ema_vfi_small_2x.yaml` - local 2x video inference parameters.
- `training/ema_vfi_small_finetune.yaml` - fine-tuning and eval-only runner parameters.
- `validation/ema_vfi_small_candidate.yaml` - candidate validation thresholds, outputs, and MLflow settings.

See the README files in those directories for field references and safe smoke guidance.
