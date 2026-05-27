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
