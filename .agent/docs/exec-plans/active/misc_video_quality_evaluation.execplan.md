# Title and Metadata

- Task: Optional Practical-RIFE video quality evaluation
- Status: Active
- Created: 2026-06-02
- Scope authority: current user task and `.agent/tasks/misc/TASK_0.md`

## Goal

Add optional sampled quality evaluation below the BentoML layer by extending the shared video inference pipeline. The feature samples source triplets from an input video, evaluates the already-loaded Practical-RIFE runtime against the real middle frame, writes accepted source triplets in Vimeo-style layout, records aggregate PSNR/SSIM, and exposes the result through local inference, benchmarks, serving, and BentoML examples.

## Requirements

- Practical-RIFE only.
- Quality evaluation is disabled by default for generic local inference.
- Serving enables quality evaluation by default with warn fail policy.
- Benchmarks enable quality evaluation by default only for Practical-RIFE; EMA/unsupported benchmark paths keep it disabled unless explicitly configured later.
- Sampled triplet evaluation uses `FramePairRequest` with the same mode as the main inference run and quality factor `2`.
- No MinIO upload/download, good/bad classification, retraining triggers, online history, drift logic, LPIPS, or BentoML-local metric implementation.

## Implementation Plan

1. Add `video_quality` subpackage with config/result dataclasses, deterministic triplet sampling, public selected-frame decoding, triplet writing, scene-cut filtering, and quality evaluation orchestration.
2. Extend `VideoInferenceConfig`, `VideoInferenceTiming`, and `VideoInferenceResult`; run quality evaluation in `run_video_inference(...)` after main processing and before adapter cleanup.
3. Add CLI flags for Practical-RIFE local video inference and benchmark quality disable/config fields.
4. Add serving config/run options and BentoML example response fields.
5. Add focused unit/integration tests with tiny synthetic videos and fake adapters.
6. Update Stage 2/2.5 docs and project map.

## Progress Log

- 2026-06-02: Created active ExecPlan after reading task and current runtime state.
- 2026-06-02: Added `video_quality` subpackage, public selected-frame decode wrapper, inference result/timing fields, local Practical-RIFE CLI flags, benchmark quality reporting/defaults, serving defaults, BentoML response fields, tests, docs, and project map updates.

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

## Outcomes & Handoff

- Implemented optional sampled Practical-RIFE quality evaluation below BentoML and exposed metrics through `VideoInferenceResult`.
- Added Practical-RIFE local inference quality flags, benchmark default-on Practical-RIFE quality measurement with `--disable-quality-evaluation`, serving default quality evaluation with warn fail policy, and BentoML example response fields.
- Quality evaluation keeps the main inference mode and uses factor `2` for the sampled triplet metric, avoiding fixed-2x checkpoint switching for arbitrary-Nx inference runs.
- Human-facing docs and project map now describe the new CLI flags, benchmark behavior, serving response fields, triplet layout, and limitations.
- Active pending user acceptance; move this ExecPlan to `completed/` after acceptance if following the project convention.
