# Title and Metadata

- Task: Slow-motion output playback mode
- Status: Active
- Created: 2026-06-03
- Scope authority: current user task and `.agent/tasks/misc/TASK_1.md`

## Goal

Add an output playback mode to shared video inference so generated frame order stays unchanged while output timing can be encoded either as real-time FPS multiplication or slow-motion original-FPS playback.

## Requirements

- Add `output_playback_mode` with allowed values `real_time` and `slow_motion`.
- Keep `real_time` as the default and preserve existing behavior.
- In `slow_motion`, encode at input FPS and do not create/remux audio streams.
- Do not change model adapters, model runtimes, ONNX export, batching, scene detection, quality evaluation, weights, or interpolation frame ordering.
- Expose the mode through core inference config, local video CLI commands, video benchmarks, serving facade, and BentoML examples.

## Implementation Plan

1. Add playback-mode enum/config validation, output-FPS resolver, result metadata, progress payload metadata, and slow-motion audio skip behavior.
2. Wire CLI, directory inference measurements, benchmark config/records/reports, serving facade, and BentoML examples.
3. Add focused tests for FPS resolution, config validation, slow-motion audio behavior, CLI pass-through, benchmark reporting, and serving/BentoML pass-through.
4. Update Stage 2/2.5 docs, config README, project map, and this ExecPlan with validation results.

## Progress Log

- 2026-06-03: Created active ExecPlan after reading the task and inspecting current inference, benchmark, serving, BentoML, CLI, config, and test paths.
- 2026-06-03: Added `VideoOutputPlaybackMode`, output-FPS resolution, slow-motion audio skip behavior, result/report metadata, CLI flags, benchmark pass-through, serving/BentoML pass-through, config defaults, docs, project map updates, and focused tests.

## Validation

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_inference.py tests/test_inference_benchmark.py tests/test_serving.py` — passed, 58 tests, 29 warnings.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` — passed, 172 tests, 59 warnings.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` — passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import torch; print(torch.cuda.is_available())"` — reported `False`; Practical-RIFE CUDA CLI smoke was skipped because the current config defaults to CUDA and `rife infer-video` has no device override.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli baseline infer-video --input raw_data/tmp_test/DORA_cut.mp4 --output /tmp/baseline_slow_motion_smoke.mp4 --limit-pairs 1 --codec libx264 --output-playback-mode slow_motion --disable-mlflow` — passed; output FPS matched input FPS, audio streams preserved were `0/1`, and audio remux time was `0.000s`.

## Outcomes & Handoff

- Added `VideoOutputPlaybackMode` with `real_time` and `slow_motion`, plus `resolve_output_fps(...)`.
- `real_time` preserves current FPS/audio behavior; `slow_motion` writes the same frame sequence at input FPS and skips audio stream creation/remuxing.
- Exposed `--output-playback-mode` on local video inference, directory inference, and video benchmarks.
- Updated serving and BentoML examples to accept and return output playback metadata.
- Updated configs, human-facing docs, benchmark/report metadata, project map, and tests.
- Active pending user acceptance; move this ExecPlan to `completed/` after acceptance if following the project convention.
