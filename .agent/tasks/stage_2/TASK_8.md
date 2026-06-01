# TASK — Stage 2 Milestone 8: Practical-RIFE PyTorch/ONNX Serving Readiness and BentoML Examples

## Purpose

This task replaces the earlier generic Stage 2 Milestone 8 prompt.

The goal is to prepare the repository so teammates can use the existing codebase to build BentoML inference services around **Practical-RIFE v4.26**.

The current serving recommendation has changed:

> the default serving path should use **Practical-RIFE v4.26 with PyTorch runtime backend on CUDA**, not ONNX.

ONNX remains valuable as an alternative backend and should still be demonstrated separately. Therefore, this milestone should provide two minimal BentoML-oriented examples:

1. Practical-RIFE service example with **PyTorch runtime backend**.
2. Practical-RIFE service example with **ONNX Runtime backend**.

Both examples should be developer-facing compatibility examples, not production services.

## Current decisions

Treat the following as the current serving direction:

- **Default production model:** Practical-RIFE v4.26.
- **Default serving backend:** PyTorch runtime backend.
- **Default PyTorch device:** `cuda`.
- **Alternative backend:** ONNX Runtime.
- **Preferred ONNX provider when ONNX is used:** `CUDAExecutionProvider`.
- **Execution mode for serving:** `sequential`.
- **Interpolation mode:** `arbitrary_nx`.
- **Allowed service-level interpolation factor:** integer `2..4`.
  - The model/runtime may still support up to `8`.
  - The BentoML examples/service boundary should restrict user-facing values to `2..4`.
- **Practical-RIFE `scale`:** external runtime/request parameter.
  - Default: `1.0`.
  - Validation should follow existing project/upstream Practical-RIFE assumptions.
  - If the allowed scale range is not already explicit, inspect the current implementation and document the chosen validation rule.
- **EMA-VFI:** not a serving target for this milestone.
- **AMT-S:** legacy/out of scope.
- **Batch inference:** not used for serving.
  - Benchmarks showed the current batched path is unsuitable for serving.
  - The serving examples must use `sequential` execution only.
  - Do not remove batch inference in this task unless explicitly asked later.

## Required reading

Before implementation, briefly re-read:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/stage_plans/02_inference_runtime_refactor_stage_plan.md`
- `.agent/docs/exec-plans/active/02_inference_runtime_refactor.execplan.md`, if still active
- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`, if still active
- Stage 2 and Stage 2.5 human-facing docs under `docs/`

Use the active Stage 2 ExecPlan as the main working document for Milestone 8. If Stage 2.5 introduced decisions that affect serving, reference them in the Stage 2 ExecPlan instead of duplicating the full Stage 2.5 history.

## Main tasks

### 1. Verify the Practical-RIFE serving interface

Inspect the current code and confirm whether there is already a clean callable Python interface that can run Practical-RIFE video inference with:

- Practical-RIFE v4.26;
- PyTorch runtime backend on `cuda`;
- optional ONNX Runtime backend with `CUDAExecutionProvider`;
- sequential execution mode;
- arbitrary Nx interpolation;
- runtime `interpolation_factor`;
- runtime `scale`;
- input video path;
- output video path.

The interface must be usable from Python code, not only from CLI.

If such an interface does not exist or is awkward, add a small project-owned wrapper/facade that future services can call.

Preferred conceptual shape:

```python
run_practical_rife_video_inference(
    input_path=...,
    output_path=...,
    backend="torch",                  # default for serving
    device="cuda",                    # default for torch backend
    provider="CUDAExecutionProvider", # used only for onnx backend
    execution_mode="sequential",
    interpolation_mode="arbitrary_nx",
    interpolation_factor=...,
    scale=...,
)
```

The exact function/class name may differ if the existing project style suggests a better name.

Requirements:

- `interpolation_factor` must be a runtime argument.
- `scale` must be a runtime argument.
- PyTorch backend must allow explicit device selection and default to `cuda` in serving examples.
- ONNX backend must allow explicit provider selection and default to `CUDAExecutionProvider` in the ONNX example.
- Invalid `interpolation_factor` values outside service range `2..4` should fail clearly at the service/example boundary.
- Invalid `scale`, backend, provider, artifact path, input path, or output path should fail clearly.
- Do not hardcode input/output paths inside the inference logic.
- Do not load model weights/artifacts per frame.
- Avoid loading model weights/artifacts per request if the current runtime supports persistent initialization.

### 2. Confirm or adjust configs for the serving target

Ensure there is a clear config path for the Practical-RIFE v4.26 serving target.

The config or documented defaults should make clear:

- model: `practical_rife_v4_26`;
- default backend for serving: `torch`;
- default PyTorch device: `cuda`;
- alternative backend: `onnx`;
- default ONNX provider: `CUDAExecutionProvider`;
- execution mode: `sequential`;
- interpolation mode: `arbitrary_nx`;
- service-level factor range: `2..4`;
- default `scale`: `1.0`;
- ONNX artifact path for the ONNX example.

Do not create one config per interpolation factor.

### 3. Repository structure cleanup

Clean up the project structure enough that teammates can understand where inference-related code lives.

Do not perform a risky large-scale rename unless it is clearly safe and covered by tests.

Suggested approach:

- keep the existing public CLI working;
- keep backward-compatible import wrappers if moving modules;
- separate serving/inference-facing helpers from training/data-only code where practical;
- remove accidental `__pycache__` directories from the repository if they are tracked or present in the working tree and should not be committed;
- ensure `.gitignore` covers Python cache/build artifacts if it does not already;
- update `PROJECT_MAP.md` after any structural change.

Acceptable result:

- modest cleanup;
- clearer inference/serving-facing wrapper;
- updated documentation;
- no full package rewrite.

Do not move model repositories, model weights, or exported ONNX artifacts unless explicitly required.

### 4. Add two minimal BentoML examples outside `src/video_interpolation`

Create minimal BentoML examples demonstrating how future backend developers should call Practical-RIFE inference.

Important:

- Examples must **not** be placed inside `src/video_interpolation`.
- Use a clear location such as:

```text
examples/bentoml/practical_rife_torch_service/
examples/bentoml/practical_rife_onnx_service/
```

or another clearly named non-package directory.

#### Example A — PyTorch backend service

This is the recommended/default serving example.

It should demonstrate:

- initializing Practical-RIFE v4.26 with PyTorch runtime backend;
- setting PyTorch device to `cuda`;
- accepting runtime `interpolation_factor`;
- accepting runtime `scale`;
- using sequential execution mode;
- calling the project inference wrapper/facade;
- writing/returning an output video path or minimal response.

#### Example B — ONNX backend service

This is the alternative backend example.

It should demonstrate:

- initializing Practical-RIFE v4.26 with ONNX Runtime backend;
- using `CUDAExecutionProvider`;
- accepting runtime `interpolation_factor`;
- accepting runtime `scale`;
- using sequential execution mode;
- calling the same project inference wrapper/facade or the same serving-facing abstraction;
- writing/returning an output video path or minimal response.

Both examples are **developer examples**, not production services.

Do not implement:

- MinIO integration;
- request queue;
- database records;
- auth/session handling;
- frontend;
- production Docker build;
- production deployment pipeline.

If BentoML APIs differ by installed version, implement examples in the simplest style compatible with the current installed version and document version assumptions.

### 5. Smoke validation

Add or run bounded validation proving that the serving target is usable.

At minimum:

- import the serving-facing wrapper/facade;
- validate config/default resolution;
- validate request parameter checks for `interpolation_factor` and `scale`;
- import/construct both BentoML examples;
- verify the PyTorch example uses `backend="torch"` and `device="cuda"` by default;
- verify the ONNX example uses `backend="onnx"` and `provider="CUDAExecutionProvider"` by default;
- run a safe Practical-RIFE PyTorch CUDA sequential smoke on a short video if CUDA is available;
- run a safe Practical-RIFE ONNX CUDA sequential smoke on a short video if CUDA and ONNX Runtime CUDA are available;
- if CUDA is unavailable, run import/config tests and clearly record CUDA smoke as deferred.

Use a short input such as `raw_data/tmp_test/DORA_cut.mp4` if a real video smoke is run.

Do not run long videos or full directories.

## Constraints

Do not implement in this task:

- full production BentoML service;
- FastAPI backend;
- Celery/Redis;
- PostgreSQL application schema;
- MinIO upload/output orchestration;
- frontend;
- monitoring;
- retraining triggers;
- EMA-VFI serving proof;
- AMT-S serving proof;
- training/fine-tuning changes;
- batch inference redesign;
- new ONNX export work unless needed to fix a broken artifact reference.

Do not change model weights.

Do not remove existing Stage 1/Stage 2 functionality unless it is clearly obsolete and the removal is documented and tested.

## Validation commands

Run:

```bash
uv run pytest
uv run ruff check src tests
```

If the uv cache is not writable:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
```

If the repository contains known lint debt outside `src`/`tests`, run the scoped lint command that is standard for this project and record the reason.

Run BentoML example import/construct smoke checks if practical.

Run CUDA video inference smoke only if CUDA is available.

## Documentation updates

Update human-facing documentation so teammates can understand:

- current serving recommendation;
- why Practical-RIFE is the production target;
- why PyTorch + CUDA + sequential is now the recommended serving path;
- when ONNX + CUDA may be used as an alternative;
- why current batch inference is not used for serving;
- how `interpolation_factor` is passed;
- service-level `interpolation_factor` range `2..4`;
- how `scale` is passed;
- how PyTorch device is selected and why serving should use `cuda`;
- where ONNX artifacts live for the ONNX example;
- where the two BentoML examples live;
- how to run local smoke tests;
- what remains for the real backend team to implement.

Mandatory Stage 2 documentation requirement:

- Add a dedicated section to `docs/stage2_inference_runtime_refactor.md` explaining how to create a BentoML service using the project inference module.
- The section must describe the expected service architecture at a developer level:
  - import the serving-facing Practical-RIFE wrapper/facade from the project package;
  - initialize Practical-RIFE once at service startup;
  - keep the model/runtime resident instead of loading weights per request;
  - pass `interpolation_factor` and `scale` as request/runtime parameters;
  - use PyTorch backend with `device="cuda"` as the default service path;
  - optionally use ONNX backend with `provider="CUDAExecutionProvider"`;
  - call the inference module for full video inference;
  - return or store the generated output video path;
  - leave MinIO, queues, DB records, auth, and production orchestration to the backend team.
- The section must point to the BentoML examples added in this milestone and explain which example is recommended for the first service implementation.

Update:

- `.agent/docs/PROJECT_MAP.md`
- active Stage 2 ExecPlan
- `docs/stage2_inference_runtime_refactor.md`
- relevant docs under `docs/`
- relevant config README files if config behavior changes

## End-of-task summary

At the end, summarize:

- whether the Practical-RIFE PyTorch CUDA sequential inference interface is ready;
- whether the Practical-RIFE ONNX CUDA sequential inference interface is available as an alternative;
- what wrapper/facade was added or confirmed;
- how `interpolation_factor` and `scale` are passed;
- how PyTorch `device` and ONNX `provider` are configured;
- what project structure cleanup was performed;
- where the PyTorch BentoML example lives;
- where the ONNX BentoML example lives;
- how the examples should be used by teammates;
- what tests and smoke checks were run;
- whether CUDA smoke was run or deferred;
- whether `docs/stage2_inference_runtime_refactor.md` now contains the dedicated BentoML-service creation section;
- what documentation and project-map updates were made;
- what remains before handing the repository to backend/service developers.
