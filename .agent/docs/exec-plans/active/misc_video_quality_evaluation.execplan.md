# Title and Metadata

- Task: Optional Practical-RIFE video quality evaluation
- Status: Active
- Created: 2026-06-02
- Scope authority: current user task and `.agent/tasks/misc/TASK_0.md`

## Goal

Add optional quality evaluation below the BentoML layer by extending the shared video inference pipeline. The feature evaluates the first contiguous source triplets from an input video, reuses the already-loaded Practical-RIFE runtime against the real middle frame, writes accepted source triplets in Vimeo-style layout, records aggregate PSNR/SSIM, and exposes the result through local inference, benchmarks, serving, and BentoML examples.

## Requirements

- Practical-RIFE only.
- Quality evaluation is disabled by default for generic local inference.
- Serving enables quality evaluation by default with warn fail policy.
- Benchmarks enable quality evaluation by default only for Practical-RIFE; EMA/unsupported benchmark paths keep it disabled unless explicitly configured later.
- Sampled triplet evaluation uses `FramePairRequest` with the same mode as the main inference run and quality factor `2`.
- No MinIO upload/download, good/bad classification, retraining triggers, online history, drift logic, LPIPS, or BentoML-local metric implementation.

## Implementation Plan

1. Add `video_quality` subpackage with config/result dataclasses, first-contiguous-triplet decoding, opt-in triplet writing, and quality evaluation orchestration.
2. Extend `VideoInferenceConfig`, `VideoInferenceTiming`, and `VideoInferenceResult`; run quality evaluation in `run_video_inference(...)` after main processing and before adapter cleanup.
3. Add CLI flags for Practical-RIFE local video inference and benchmark quality disable/config fields.
4. Add serving config/run options and BentoML example response fields.
5. Add focused unit/integration tests with tiny synthetic videos and fake adapters.
6. Update Stage 2/2.5 docs and project map.

## Progress Log

- 2026-06-02: Created active ExecPlan after reading task and current runtime state.
- 2026-06-02: Added `video_quality` subpackage, public selected-frame decode wrapper, inference result/timing fields, local Practical-RIFE CLI flags, benchmark quality reporting/defaults, serving defaults, BentoML response fields, tests, docs, and project map updates.
- 2026-06-02: User benchmark found sparse random sampling too decode-heavy: the quality-enabled smoke spent `53.070498s` in quality evaluation versus `0.000000s` when disabled. Updated the plan to replace random whole-video sampling with first-contiguous-triplet evaluation that decodes at most `sample_count + 2` frames.
- 2026-06-02: Follow-up investigation found first-contiguous quality evaluation remained too slow. Updated the experiment to disable triplet PNG writing by default and remove adjacent-frame scene-cut SSIM filtering, assuming benchmark videos are single-scene for now.
- 2026-06-02: Added quality-only proportional frame resizing before quality prediction and metrics, defaulting to max image side `360`, without changing main video inference frames.

## Validation

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_video_quality.py tests/test_inference.py tests/test_inference_benchmark.py tests/test_serving.py` — passed, 58 tests, 29 warnings.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` — passed, 163 tests, 59 warnings.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` — passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli rife infer-video --help` — passed; quality flags are exposed on Practical-RIFE local video inference.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli ema infer-video --help` — passed; quality flags are not exposed on EMA local video inference.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli benchmark runtime --model practical_rife_v4_26 --backend torch --execution-mode sequential --input raw_data/tmp_test/DORA_cut.mp4 --limit-pairs 1 --device cpu --codec libx264 --output-dir outputs/benchmarks/misc_quality_smoke --disable-mlflow
UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli benchmark runtime --model practical_rife_v4_26 --backend torch --execution-mode sequential --input raw_data/tmp_test/DORA_cut.mp4 --limit-pairs 1 --device cpu --codec libx264 --output-dir outputs/benchmarks/misc_quality_disabled_smoke --disable-quality-evaluation --disable-mlflow
```

Smoke results:

- Quality enabled: passed; 1 pair, 1 generated frame, 16 quality triplets, model inference `2.705499s`, quality evaluation `53.070498s`, total `56.518345s`.
- Quality disabled: passed; 1 pair, 1 generated frame, 0 quality triplets, model inference `2.262280s`, quality evaluation `0.000000s`, total `2.856395s`.

First-contiguous-triplet refactor smoke:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_video_quality.py tests/test_inference.py tests/test_inference_benchmark.py tests/test_serving.py` — passed, 59 tests, 29 warnings.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` — passed, 164 tests, 59 warnings.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` — passed.
- Quality enabled after refactor: passed; 1 pair, 1 generated frame, 16 quality triplets, model inference `2.262292s`, quality evaluation `59.075172s`, total `61.913702s`.
- Quality disabled after refactor: passed; 1 pair, 1 generated frame, 0 quality triplets, model inference `2.167035s`, quality evaluation `0.000000s`, total `2.723949s`.
- The refactor removed sparse whole-video selected-frame decoding from the quality path, but this CPU smoke still spends most quality time in 16 Practical-RIFE quality model evaluations.

No-triplet-write/no-scene-filter follow-up:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_video_quality.py tests/test_inference.py tests/test_inference_benchmark.py tests/test_serving.py` — passed, 60 tests, 29 warnings.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` — passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` — passed, 165 tests, 59 warnings.
- Quality enabled after disabling triplet PNG writes and scene-cut SSIM filtering: passed; 1 pair, 1 generated frame, 0 quality triplets written, model inference `2.363421s`, quality evaluation `39.136039s`, total `42.121549s`.
- Quality disabled after follow-up: passed; 1 pair, 1 generated frame, 0 quality triplets written, model inference `2.114648s`, quality evaluation `0.000000s`, total `2.652468s`.
- Result: removing writes/filtering reduced CPU quality time from `59.075172s` to `39.136039s`, but quality evaluation is still too slow and likely dominated by 16 full-resolution quality predictions plus final full-resolution metrics.

Downscaled quality-frame follow-up:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_video_quality.py tests/test_inference.py tests/test_inference_benchmark.py tests/test_serving.py` — passed, 62 tests, 29 warnings.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` — passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` — passed, 167 tests, 59 warnings.
- Quality enabled with max quality image side `360`: passed; 1 pair, 1 generated frame, 0 quality triplets written, model inference `2.251434s`, quality evaluation `2.854908s`, total `5.717573s`.
- Quality disabled after downscale follow-up: passed; 1 pair, 1 generated frame, 0 quality triplets written, model inference `2.133927s`, quality evaluation `0.000000s`, total `2.712819s`.
- Result: resizing quality-only frames before prediction/metrics reduced CPU quality time from `39.136039s` to `2.854908s` for the same smoke profile.

## Outcomes & Handoff

- Implemented optional Practical-RIFE first-triplet quality evaluation below BentoML and exposed metrics through `VideoInferenceResult`.
- Added Practical-RIFE local inference quality flags, benchmark default-on Practical-RIFE quality measurement with `--disable-quality-evaluation`, serving default quality evaluation with warn fail policy, and BentoML example response fields.
- Quality evaluation keeps the main inference mode and uses factor `2` for the source-triplet metric, avoiding fixed-2x checkpoint switching for arbitrary-Nx inference runs.
- Quality triplet selection now uses the first `sample_count` overlapping source triplets and no longer performs random sparse frame selection or selected-frame seek decoding.
- Quality evaluation now skips scene-cut filtering and writes Vimeo-style triplet PNGs only when explicitly requested.
- Quality prediction and PSNR/SSIM now run on resized quality-only frames by default, preserving aspect ratio and capping the larger side at 360 pixels.
- Human-facing docs and project map now describe the new CLI flags, benchmark behavior, serving response fields, triplet layout, and limitations.
- Active pending user acceptance; move this ExecPlan to `completed/` after acceptance if following the project convention.
