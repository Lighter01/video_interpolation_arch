# Start Stage 2.5 Milestone 6 Implementation

Continue Stage 2.5 — ONNX Stabilization, Real-Image Equivalence, and Batched Inference.

Use `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md` as the main working document.

Briefly re-read:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/stage_plans/02_5_inference_runtime_stabilization_stage_plan.md`
- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`

Implement only **Milestone 6 — Video-Level Chunked Batch Inference**.

Do not start Milestone 7.

## Task

Update local video inference so it can use the true model batch inference path implemented in Milestone 5.

This milestone is about video-level chunking, frame ordering, output FPS, and local inference integration.

## Scope

Implement chunked batched video inference for active PyTorch model runtimes:

- EMA-VFI;
- Practical-RIFE.

The implementation must support:

- fixed 2x;
- arbitrary Nx;
- configurable `inference_batch_size`;
- chunk overlap by one source frame;
- correct generated-frame ordering for each source frame pair;
- correct output FPS: `input_fps * interpolation_factor`;
- sequential fallback;
- existing PyAV/FFmpeg encoding behavior;
- existing audio remux behavior where already supported;
- measurement metadata that records execution mode, factor, batch size, and relevant timing fields.

Do not implement ONNX batch inference in this milestone.

Do not implement benchmark workflows yet; that belongs to Milestone 8.

## Validation

Add focused tests with fake/lightweight runtime for:

- chunk overlap correctness;
- output frame count for factors `2`, `4`, and `8`;
- generated-frame ordering;
- output FPS multiplication;
- `inference_batch_size` validation;
- sequential fallback compatibility;
- measurement metadata fields;
- equivalence of frame ordering between sequential and batched video paths where practical.

Fake/lightweight runtime tests are required because they verify video-pipeline correctness deterministically without depending on GPU speed, model quality, or real model availability.

Also add or run integration smoke checks with real EMA and Practical-RIFE runtimes when practical:

- prefer CUDA if available;
- if CUDA is unavailable, try CPU only on very short video if it is fast enough (you can use video examples from `raw_data/tmp_test/`);
- use a very small pair limit, for example `--limit-pairs 1` or `--limit-pairs 2`;
- test fixed 2x for both models;
- test Nx with factor `4` for both models where feasible;
- record if real-model CPU video smoke is too slow or blocked by model assumptions.

Run:

```bash
uv run pytest
uv run ruff check src tests
````

If the uv cache is not writable:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
```

Do not run long videos, full directories, or full batch jobs.

## Updates

Update:

- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`
- `.agent/docs/PROJECT_MAP.md` if repository structure changes
- human-facing docs, especially Stage 2.5 inference docs and relevant config README files

Document:

- how batched video inference works;
- how it differs from directory-wide `batch_inference.py`;
- `inference_batch_size`;
- sequential fallback;
- VRAM/resolution considerations;
- safe smoke-run examples.

## End-of-task summary

Summarize:

- how video-level batched inference was implemented;
- how chunk overlap and output ordering are handled;
- how fixed 2x and Nx are handled;
- how `inference_batch_size` is configured;
- what tests were added;
- what smoke checks were run or deferred;
- validation command results;
- docs/project-map/ExecPlan updates;
- what remains for Milestone 7.
