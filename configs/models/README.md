# Stage 1 Model Configs

Model YAML files define adapter-level parameters. Runtime roots still come from `.env`.

## `ema_vfi_small.yaml`

Purpose: configure the EMA-VFI-small adapter used for preflight, pair inference, local video inference, fine-tuning, and candidate validation.

Fields:

| Field | Type | Allowed/default | Effect |
| --- | --- | --- | --- |
| `model_name` | string | default `ema_vfi_small` | Names the model in logs, metrics, and reports |
| `repo_name` | string | default `EMA-VFI` | Directory under `MODEL_REPOS_ROOT` containing upstream EMA code |
| `checkpoint_path` | path string | default `EMA-VFI/ours_small_t.pkl` in the Stage 2 inference config; relative to `MODEL_WEIGHTS_ROOT` unless absolute | Checkpoint loaded by the adapter for inference |
| `inference_checkpoint_path` | path string | default `EMA-VFI/ours_small_t.pkl` | Preferred timestep-capable checkpoint for fixed 2x-as-`t=0.5` and arbitrary Nx inference |
| `training_checkpoint_path` | path string | default `EMA-VFI/ours_small.pkl` | Fixed 2x checkpoint kept for EMA training/fine-tuning compatibility |
| `supported_modes` | list of strings | `fixed_2x`, `arbitrary_nx` | Declares runtime modes supported by EMA inference |
| `default_interpolation_factor` | integer | default `2` | Used by pair-smoke/request builders only when a caller omits `--interpolation-factor` |
| `min_interpolation_factor` / `max_interpolation_factor` | integer | `2` / `8` | Documented request bounds; runtime validation rejects values outside this range |
| `device` | string | default `cuda` | EMA upstream currently requires CUDA; CPU is not a working Stage 1 path |
| `tta` | boolean | default `false` for `ours_small_t` | Enables upstream test-time augmentation during pair inference |
| `fast_tta` | boolean | default `false` for `ours_small_t` | Enables upstream fast TTA path |
| `divisor` | integer | default `32` | Input padding divisor before EMA inference |
| `strict_checkpoint` | boolean | default `true` | Uses strict PyTorch state-dict loading |

Operational notes:

- Changing `checkpoint_path`, TTA flags, or strict loading changes evaluation/inference behavior and requires rerunning the relevant command.
- `arbitrary_nx` does not use one config per factor. Pass the concrete factor at request/CLI time, for example `--interpolation-factor 4`. Local video inference uses the same request/result API and writes output at `input_fps * interpolation_factor`.
- EMA training/fine-tuning remains fixed 2x. The training config keeps `checkpoint_path: EMA-VFI/ours_small.pkl` even though Stage 2 inference prefers `ours_small_t.pkl`.
- `tta` means test-time augmentation: the adapter also runs an augmented/flipped inference path and averages predictions. It can improve quality in some cases but increases runtime and GPU memory use.
- `fast_tta` is EMA-VFI's faster test-time augmentation path. It is still extra inference work, but cheaper than full TTA. The config field is `fast_tta`.
- `divisor` controls padding before EMA inference/training updates. EMA expects dimensions divisible by this value; the adapter pads inputs and unpads predictions back to the original size.
- `strict_checkpoint` is passed to PyTorch state-dict loading. Keep it `true` for expected EMA-VFI-small checkpoints so missing or unexpected weights fail clearly.
- The adapter check does not write files:

```bash
uv run python -m video_interpolation.cli ema adapter-check
```

- In environments without CUDA, the check reports `blocked` because upstream EMA-VFI hardcodes CUDA model placement.
- Tensor-pair Nx smoke command:

```bash
uv run python -m video_interpolation.cli ema infer-pair --interpolation-factor 4
```

- ONNX neural-core export command:

```bash
uv run python -m video_interpolation.cli ema export-onnx \
  --config configs/models/ema_vfi_small.yaml \
  --device cpu \
  --height 32 \
  --width 32 \
  --output-dir model_exports/onnx
```

- ONNX Runtime equivalence validation command:

```bash
uv run python -m video_interpolation.cli ema validate-onnx \
  --config configs/models/ema_vfi_small.yaml \
  --provider CPUExecutionProvider \
  --shape 32x32
```

## `amt_s.yaml`

Purpose: configure the AMT-S adapter used for preflight, pair inference, local video inference, and candidate/eval-only validation.

Fields:

| Field | Type | Allowed/default | Effect |
| --- | --- | --- | --- |
| `model_name` | string | default `amt_s` | Names the model in logs, metrics, and reports |
| `repo_name` | string | default `AMT` | Directory under `MODEL_REPOS_ROOT` containing upstream AMT code |
| `config_path` | path string | default `cfgs/AMT-S.yaml`; relative to the AMT repo unless absolute | Upstream AMT network/config file used to build AMT-S |
| `checkpoint_path` | path string | default `AMT/amt-s.pth`; relative to `MODEL_WEIGHTS_ROOT` unless absolute | AMT-S checkpoint loaded by the adapter |
| `device` | string | default `cuda`; `cpu` is useful only for tiny compatibility checks | Torch device for model loading and inference |
| `embt` | float | default `0.5` | Interpolation time between left and right frames; Stage 1 uses middle-frame 2x interpolation |
| `scale_factor` | float | default `1.0` | AMT internal inference scale factor; keep `1.0` unless debugging memory constraints |
| `divisor` | integer | default `16` | Input padding divisor before AMT inference |
| `strict_checkpoint` | boolean | default `true` | Uses strict PyTorch state-dict loading |

Operational notes:

- Changing `checkpoint_path`, `config_path`, `embt`, `scale_factor`, or strict loading changes evaluation/inference behavior and requires rerunning the relevant command.
- The AMT checkpoint is loaded with PyTorch safe weights loading plus an allowlist for the checkpoint's `OrderedDict` wrapper; the adapter does not disable safe loading.
- AMT upstream training expects optical-flow supervision files under a `flow/` directory. Stage 1 Milestone 8 therefore supports AMT-S preflight, eval-only/candidate validation, and local video inference first; fine-tuning is deferred until flow generation or a loss-policy change is approved.

Check the adapter:

```bash
uv run python -m video_interpolation.cli amt adapter-check
```

Run the heavier import/build/checkpoint smoke:

```bash
uv run python -m video_interpolation.cli amt-preflight
```

## `practical_rife_v4_26.yaml`

Purpose: configure the Practical-RIFE v4.26 adapter used for preflight, pair inference, local video inference, and candidate/eval-only validation.

Fields:

| Field | Type | Allowed/default | Effect |
| --- | --- | --- | --- |
| `model_name` | string | default `practical_rife_v4_26` | Names the model in logs, metrics, and reports |
| `repo_name` | string | default `Practical-RIFE` | Directory under `MODEL_REPOS_ROOT` containing upstream Practical-RIFE code |
| `checkpoint_path` | path string | default `Practical-RIFE/RIFEv4.26/train_log`; relative to `MODEL_WEIGHTS_ROOT` unless absolute | Directory containing `flownet.pkl`; Python runtime source is project-owned under `src/video_interpolation/inference_runtime/rife_upstream/` |
| `supported_modes` | list of strings | `fixed_2x`, `arbitrary_nx` | Declares runtime modes supported by Practical-RIFE inference |
| `default_interpolation_factor` | integer | default `2` | Used by pair-smoke/request builders only when a caller omits `--interpolation-factor` |
| `min_interpolation_factor` / `max_interpolation_factor` | integer | `2` / `8` | Documented request bounds; runtime validation rejects values outside this range |
| `device` | string | default `cuda`; `cpu` is useful only for tiny compatibility checks | Torch device for model loading and inference |
| `timestep` | float | default `0.5` | Interpolation time between left and right frames; Stage 1 uses middle-frame 2x interpolation |
| `scale` | float | default `1.0` | Default Practical-RIFE scale fallback when no request/CLI scale is supplied |
| `divisor` | integer | default `128` | Input padding divisor before Practical-RIFE inference |
| `strict_checkpoint` | boolean | default `false` | Non-strict loading accepts extra teacher/training keys present in the shipped v4.25 `flownet.pkl` |

Operational notes:

- Changing `checkpoint_path`, `timestep`, default `scale`, `divisor`, or strict loading changes evaluation/inference behavior and requires rerunning the relevant command.
- Stage 2 defaults to `RIFEv4.26`; v4.25 remains available through `configs/models/practical_rife_v4_25.yaml`.
- Fixed 2x is the `interpolation_factor=2`, `t=0.5` case. Arbitrary Nx uses direct request timesteps from the caller-provided factor. Local video inference writes generated frames in timestep order and sets output FPS to `input_fps * interpolation_factor`.
- Practical-RIFE does not use one config per factor. Pass the concrete factor at request/CLI time, for example `--interpolation-factor 4`.
- Practical-RIFE scale is also request-time controllable through `FramePairRequest.backend_options`, `rife infer-pair --scale`, `rife infer-video --scale`, or `infer-all-videos --rife-scale`. Valid values match upstream: `0.25`, `0.5`, `1.0`, `2.0`, and `4.0`; upstream recommends `0.5` for high-resolution inputs such as 4K.
- `strict_checkpoint` defaults to `false` because the local checkpoints contain teacher/caltime training keys that are not used by the inference network.
- Practical-RIFE Python runtime source is no longer imported from `model_weights/.../train_log`.
- Practical-RIFE upstream training is not manifest-ready in Milestone 9; eval-only/candidate validation and local video inference are implemented first.

Check the adapter:

```bash
uv run python -m video_interpolation.cli rife adapter-check
```

Run the heavier import/build/checkpoint smoke:

```bash
uv run python -m video_interpolation.cli rife-preflight
```

Tensor-pair Nx smoke command:

```bash
uv run python -m video_interpolation.cli rife infer-pair --interpolation-factor 4 --scale 0.5
```

ONNX neural-core export command:

```bash
uv run python -m video_interpolation.cli rife export-onnx \
  --config configs/models/practical_rife_v4_26.yaml \
  --device cpu \
  --height 128 \
  --width 128 \
  --scale 1.0 \
  --output-dir model_exports/onnx
```

ONNX Runtime equivalence validation command:

```bash
uv run python -m video_interpolation.cli rife validate-onnx \
  --config configs/models/practical_rife_v4_26.yaml \
  --provider CPUExecutionProvider \
  --shape 128x128
```

ONNX export writes artifacts under `model_exports/onnx/<model_name>/`, attempts dynamic height/width export by default, and runs ONNX simplification unless `--no-simplify` is passed. Export targets only the neural core; video I/O, padding/unpadding policy, timestep loops, and frame interleaving remain outside ONNX.
ONNX Runtime validation compares PyTorch and ONNX-generated intermediate tensors with `atol=1e-3` and `rtol=1e-3`, writes `equivalence_report.json` plus `equivalence_metrics.csv` under `outputs/onnx_validation/`, and writes sample PyTorch/ONNX/difference PNGs when tensor `allclose` fails.

## `practical_rife_v4_25.yaml`

Purpose: keep Practical-RIFE v4.25 available as an alternative checkpoint selection.

The fields match `practical_rife_v4_26.yaml`; only `model_name` and `checkpoint_path` differ.
