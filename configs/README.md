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

- `models/ema_vfi_small.yaml` - adapter/checkpoint/device parameters plus fixed 2x/arbitrary Nx capability limits; Stage 2 inference uses `ours_small_t.pkl`.
- `inference/ema_vfi_small_2x.yaml` - local fixed 2x video inference defaults; the same command can run arbitrary Nx with `--mode arbitrary_nx --interpolation-factor N`.
- `training/ema_vfi_small_finetune.yaml` - fine-tuning and eval-only runner parameters.
- `validation/ema_vfi_small_candidate.yaml` - candidate validation thresholds, outputs, and MLflow settings.

EMA training/fine-tuning remains fixed 2x and keeps the training-compatible `ours_small.pkl` checkpoint.

See the README files in those directories for field references and safe smoke guidance.

See `inference/README.md` for single-video inference configs and the `infer-all-videos --input-dir ...` directory-wide wrapper. Local model video inference uses the Stage 2 request/result runtime API for EMA-VFI and Practical-RIFE, writes generated frames in timestep order, and sets output FPS to `input_fps * interpolation_factor`. EMA-VFI and Practical-RIFE can also use chunked PyTorch model-batch video inference through `execution_mode: batched` and `inference_batch_size`; `execution_mode: sequential` remains available for debugging and low-VRAM fallback. Practical-RIFE scale can also be supplied per request/CLI call for high-resolution inputs.

Stage 2 ONNX export is exposed through developer commands that use `models/` configs:

```bash
uv run python -m video_interpolation.cli ema export-onnx --config configs/models/ema_vfi_small.yaml
uv run python -m video_interpolation.cli rife export-onnx --config configs/models/practical_rife_v4_26.yaml
```

Exports write neural-core artifacts under `model_exports/onnx/`; local video decoding, padding policy, timestep loops, frame interleaving, and encoding stay outside ONNX. Dynamo export is the default (`--exporter dynamo`, opset 18, no simplification), while legacy export remains available with `--exporter legacy`; ONNX simplification is supported only for legacy exports via `--simplify`.
ONNX Runtime equivalence checks use the same `models/` configs and write reports under `outputs/onnx_validation/`:

```bash
uv run python -m video_interpolation.cli ema validate-onnx --config configs/models/ema_vfi_small.yaml --shape 32x32
uv run python -m video_interpolation.cli rife validate-onnx --config configs/models/practical_rife_v4_26.yaml --shape 128x128
```

Implemented AMT-S configs:

- `models/amt_s.yaml` - AMT-S adapter/checkpoint/device/config parameters.
- `inference/amt_s_2x.yaml` - local AMT-S 2x video inference parameters.
- `validation/amt_s_candidate.yaml` - AMT-S candidate/eval-only validation thresholds, outputs, and MLflow settings.

AMT-S fine-tuning is deferred until the project either generates the upstream-required flow files or explicitly changes the AMT training loss policy.

Implemented Practical-RIFE configs:

- `models/practical_rife_v4_26.yaml` - Practical-RIFE v4.26 adapter/checkpoint/device parameters plus fixed 2x/arbitrary Nx capability limits; Stage 2 default.
- `inference/practical_rife_v4_26_2x.yaml` - local Practical-RIFE v4.26 fixed 2x video inference defaults; the same command can run arbitrary Nx with `--mode arbitrary_nx --interpolation-factor N`.
- `validation/practical_rife_v4_26_candidate.yaml` - Practical-RIFE v4.26 candidate/eval-only validation thresholds, outputs, and MLflow settings.
- `models/practical_rife_v4_25.yaml` - Practical-RIFE v4.25 adapter/checkpoint/device parameters.
- `inference/practical_rife_v4_25_2x.yaml` - alternative Practical-RIFE v4.25 fixed 2x video inference defaults.
- `validation/practical_rife_v4_25_candidate.yaml` - Practical-RIFE candidate/eval-only validation thresholds, outputs, and MLflow settings.

Practical-RIFE fine-tuning is deferred until the project chooses how to replace upstream hardcoded `/data`, nori/S3, distributed, and TensorBoard training assumptions with project manifests and MLflow.
