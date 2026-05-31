# Stage 1 Model Configs

Model YAML files define adapter-level parameters. Runtime roots still come from `.env`.

## `ema_vfi_small.yaml`

Purpose: configure the EMA-VFI-small adapter used for preflight, pair inference, local video inference, fine-tuning, and candidate validation.

Fields:

| Field | Type | Allowed/default | Effect |
| --- | --- | --- | --- |
| `model_name` | string | default `ema_vfi_small` | Names the model in logs, metrics, and reports |
| `repo_name` | string | default `EMA-VFI` | Directory under `MODEL_REPOS_ROOT` containing upstream EMA code |
| `checkpoint_path` | path string | default `EMA-VFI/ours_small.pkl`; relative to `MODEL_WEIGHTS_ROOT` unless absolute | Checkpoint loaded by the adapter |
| `device` | string | default `cuda` | EMA upstream currently requires CUDA; CPU is not a working Stage 1 path |
| `tta` | boolean | default `false` for `ours_small` | Enables upstream test-time augmentation during pair inference |
| `fast_tta` | boolean | default `false` for `ours_small` | Enables upstream fast TTA path |
| `divisor` | integer | default `32` | Input padding divisor before EMA inference |
| `strict_checkpoint` | boolean | default `true` | Uses strict PyTorch state-dict loading |

Operational notes:

- Changing `checkpoint_path`, TTA flags, or strict loading changes evaluation/inference behavior and requires rerunning the relevant command.
- `tta` means test-time augmentation: the adapter also runs an augmented/flipped inference path and averages predictions. It can improve quality in some cases but increases runtime and GPU memory use.
- `fast_tta` is EMA-VFI's faster test-time augmentation path. It is still extra inference work, but cheaper than full TTA. The config field is `fast_tta`.
- `divisor` controls padding before EMA inference/training updates. EMA expects dimensions divisible by this value; the adapter pads inputs and unpads predictions back to the original size.
- `strict_checkpoint` is passed to PyTorch state-dict loading. Keep it `true` for expected EMA-VFI-small checkpoints so missing or unexpected weights fail clearly.
- The adapter check does not write files:

```bash
uv run python -m video_interpolation.cli ema adapter-check
```

- In environments without CUDA, the check reports `blocked` because upstream EMA-VFI hardcodes CUDA model placement.

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

## `practical_rife_v4_25.yaml`

Purpose: configure the Practical-RIFE v4.25 adapter used for preflight, pair inference, local video inference, and candidate/eval-only validation.

Fields:

| Field | Type | Allowed/default | Effect |
| --- | --- | --- | --- |
| `model_name` | string | default `practical_rife_v4_25` | Names the model in logs, metrics, and reports |
| `repo_name` | string | default `Practical-RIFE` | Directory under `MODEL_REPOS_ROOT` containing upstream Practical-RIFE code |
| `checkpoint_path` | path string | default `Practical-RIFE/RIFEv4.25/train_log`; relative to `MODEL_WEIGHTS_ROOT` unless absolute | Directory containing `flownet.pkl`, `RIFE_HDv3.py`, and `IFNet_HDv3.py` |
| `device` | string | default `cuda`; `cpu` is useful only for tiny compatibility checks | Torch device for model loading and inference |
| `timestep` | float | default `0.5` | Interpolation time between left and right frames; Stage 1 uses middle-frame 2x interpolation |
| `scale` | float | default `1.0` | Practical-RIFE internal inference scale; lower values can reduce memory at high resolution |
| `divisor` | integer | default `128` | Input padding divisor before Practical-RIFE inference |
| `strict_checkpoint` | boolean | default `false` | Non-strict loading accepts extra teacher/training keys present in the shipped v4.25 `flownet.pkl` |

Operational notes:

- Changing `checkpoint_path`, `timestep`, `scale`, `divisor`, or strict loading changes evaluation/inference behavior and requires rerunning the relevant command.
- The default `checkpoint_path` selects `RIFEv4.25` because upstream Practical-RIFE recommends it for most scenes and matching local weights are available.
- `strict_checkpoint` defaults to `false` because the local v4.25 checkpoint contains teacher/caltime training keys that are not used by the inference network.
- Practical-RIFE upstream training is not manifest-ready in Milestone 9; eval-only/candidate validation and local video inference are implemented first.

Check the adapter:

```bash
uv run python -m video_interpolation.cli rife adapter-check
```

Run the heavier import/build/checkpoint smoke:

```bash
uv run python -m video_interpolation.cli rife-preflight
```
