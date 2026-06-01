# Title and Metadata

- Stage: Stage 2 - Inference Runtime Refactor and Serving Readiness
- Status: Completed ExecPlan - Stage 2 Milestone 9 documentation, project map, and handoff complete; CUDA smokes deferred
- Created: 2026-05-31
- Updated: 2026-06-01
- Stage plan: `.agent/stage_plans/02_inference_runtime_refactor_stage_plan.md`
- Global plan: `.agent/general_plan.md`
- Prior ExecPlan: `.agent/docs/exec-plans/completed/01_ml_core_selected.execplan.md`
- Scope authority: current user answers from 2026-05-31 plus `.agent/stage_plans/02_inference_runtime_refactor_stage_plan.md`

## Stage Goal

Refactor the Stage 1 inference code into a reusable inference runtime subsystem that can be called by local CLI workflows now and by future BentoML services later.

The stage must produce a clean request/result-based inference API, keep existing adapter calls usable where practical, make runtime backend boundaries explicit, support PyTorch inference first, add PyTorch Nx interpolation before ONNX work, analyze and implement ONNX export/runtime where feasible, and provide a minimal BentoML compatibility proof after the new API is stable.

This is not the full backend stage. It must not implement production BentoML services, FastAPI, Celery, Redis, application PostgreSQL tables, object-storage upload/output orchestration, frontend, retraining triggers, or monitoring.

## Source Documents and Authority

Read before this ExecPlan was created:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/general_plan.md`
- `.agent/stage_plans/01_ml_core_stage_plan.md`
- `.agent/stage_plans/02_inference_runtime_refactor_stage_plan.md`
- `.agent/02_inference_runtime_refactor_TASK.md`
- `.agent/docs/exec-plans/completed/01_ml_core_selected.execplan.md`
- `src/video_interpolation/adapters/base.py`
- `src/video_interpolation/adapters/ema_vfi.py`
- `src/video_interpolation/adapters/rife.py`
- `src/video_interpolation/inference.py`
- `src/video_interpolation/batch_inference.py`
- `src/video_interpolation/image_io.py`
- `src/video_interpolation/training.py`
- `src/video_interpolation/validation.py`
- `src/video_interpolation/cli.py`
- `configs/models/ema_vfi_small.yaml`
- `configs/models/practical_rife_v4_25.yaml`
- `configs/inference/ema_vfi_small_2x.yaml`
- `configs/inference/practical_rife_v4_25_2x.yaml`
- `model_repos/EMA-VFI/Trainer.py`
- `model_repos/EMA-VFI/model/flow_estimation.py`
- `model_repos/EMA-VFI/model/warplayer.py`
- `model_repos/EMA-VFI/demo_2x.py`
- `model_repos/EMA-VFI/demo_Nx.py`
- `model_repos/Practical-RIFE/inference_img.py`
- `model_repos/Practical-RIFE/inference_video.py`
- `model_repos/Practical-RIFE/model/warplayer.py`
- `model_weights/Practical-RIFE/RIFEv4.25/train_log/RIFE_HDv3.py`
- `model_weights/Practical-RIFE/RIFEv4.25/train_log/IFNet_HDv3.py`
- `model_weights/Practical-RIFE/RIFEv4.26/train_log/RIFE_HDv3.py`
- `model_weights/Practical-RIFE/RIFEv4.26/train_log/IFNet_HDv3.py`

The current user answers control ambiguous design decisions. The Stage 2 plan controls the stage scope. The completed Stage 1 ExecPlan controls the handoff facts from the previous stage.

Note: the task text referenced `.agent/docs/general_plan.md`, but the repository contains `.agent/general_plan.md`.

## Context and Current Repository State

Repository root: `/home/lighter_01/projects/itmo/ai_architecture/video_interpolation`.

Stage 1 is closed as practically complete. The completed handoff is in `.agent/docs/exec-plans/completed/01_ml_core_selected.execplan.md`.

Current inference-related code:

- `src/video_interpolation/adapters/base.py` defines the Stage 1 `ModelAdapter` interface with `load_checkpoint`, `save_checkpoint`, `train`, `eval`, `predict_pair`, sequential `predict_batch`, `predict`, `__call__`, and `close`.
- `src/video_interpolation/adapters/ema_vfi.py` builds upstream `Trainer.Model`, loads `model_weights/EMA-VFI/ours_small.pkl` into `_model.net`, uses upstream `InputPadder`, calls `_model.inference(...)` for 2x prediction, and also contains EMA training/eval helper methods.
- `src/video_interpolation/adapters/rife.py` imports Practical-RIFE repository modules plus selected `train_log` model code from the weight bundle, loads `flownet.pkl` into `model.flownet`, pads to divisor 128, and calls `_model.inference(...)`.
- `src/video_interpolation/inference.py` implements model-independent local video inference with OpenCV decoding, PyAV/FFmpeg encoding, audio remuxing, original/generated frame interleaving, timing metrics, and MLflow logging.
- `src/video_interpolation/batch_inference.py` implements directory-wide video discovery, output path layout, target selection, and measurement CSV writing.
- `src/video_interpolation/training.py` is EMA-specific and uses `EMAVFIAdapter.train_step()` / `eval_step()`.
- `src/video_interpolation/validation.py` is model-generic at the adapter level and calls `ModelAdapter.predict_pair()`.
- `src/video_interpolation/cli.py` wires Stage 1 commands for EMA, AMT, RIFE, baselines, data, MLflow, and all-target inference.

Current configs:

- `configs/models/ema_vfi_small.yaml` points to `EMA-VFI/ours_small.pkl`.
- `configs/models/practical_rife_v4_25.yaml` points to `Practical-RIFE/RIFEv4.25/train_log`.
- `configs/inference/ema_vfi_small_2x.yaml` and `configs/inference/practical_rife_v4_25_2x.yaml` describe fixed 2x local video inference.

Current model weights:

- EMA-VFI weights include `ours_small.pkl` and `ours_small_t.pkl`; the `_t` checkpoint is the expected EMA Nx checkpoint.
- Practical-RIFE weights include `RIFEv4.25/train_log` and `RIFEv4.26/train_log`; local model code files are identical, but `flownet.pkl` differs. Stage 2 should make v4.26 the default and keep v4.25 available.

Current upstream export observations:

- EMA-VFI `Trainer.Model` creates `self.net` from `MODEL_CONFIG['MODEL_TYPE']`; for the small config this is `MultiScaleFlow(feature_extractor(...), ...)`.
- EMA-VFI `Trainer.Model.inference()` concatenates inputs and calls `self.net(imgs, timestep=timestep)`.
- EMA-VFI `MultiScaleFlow.forward()` returns `(flow_list, mask_list, merged, pred)`.
- EMA-VFI contains hardcoded `.cuda()` calls in flow/timestep construction and CUDA assumptions in `Trainer.Model.device()`.
- Practical-RIFE `RIFE_HDv3.Model` creates `self.flownet = IFNet()` and `inference()` concatenates inputs, builds `scale_list`, calls `self.flownet(imgs, timestep, scale_list)`, and returns `merged[-1]`.
- Practical-RIFE `IFNet.forward()` accepts concatenated image tensors, a timestep, and a scale list; it returns `(flow_list, mask, merged)`.
- Practical-RIFE upstream video scripts include frame queues, static-frame handling, recursive/multi-frame interpolation, padding, and output writing. Those orchestration parts should remain project-owned, not upstream-owned.

## Stage Requirements Restated

Required functionality:

- Introduce a request/result-based inference API suitable for BentoML and future ONNX Runtime.
- Preserve existing Stage 1 calls such as `predict_pair`, `predict_batch`, `predict`, and `__call__` as wrappers where practical.
- Create `src/video_interpolation/inference_runtime/` as the new inference subsystem.
- Use one shared runtime backend abstraction with model-specific runtime classes underneath.
- Make PyTorch runtime the working default.
- Keep the public model adapter API as the main project boundary, but move internal runtime execution responsibilities out of adapter methods where possible.
- Keep video decoding, tensor formatting, padding/unpadding, timestep loops, frame interleaving, output encoding, and serving orchestration outside ONNX graphs.
- Add PyTorch Nx support after the runtime refactor and before ONNX export.
- Support fixed 2x and arbitrary Nx inference with `interpolation_factor` in `[2, 8]`.
- Treat arbitrary/Nx interpolation as an inference/runtime feature only. EMA training and fine-tuning remain fixed 2x, existing training configs/runners must remain compatible, and arbitrary/Nx requests must not be routed through training code.
- Treat the concrete `interpolation_factor` as a runtime/request argument, not as a hardcoded model config choice. Configs may define supported modes, default factor, min/max factor, and checkpoint mapping, but the actual factor for a call must come from the request API, CLI argument, or future service request.
- Do not plan one separate config file per factor such as 4x or 8x configs. A generic Nx-capable config may provide defaults and limits, while `FramePairRequest` or the caller supplies the requested factor.
- Validate `interpolation_factor` before model inference starts: it must be an integer, `2 <= interpolation_factor <= 8`, `fixed_2x` must reject factors other than `2`, and `arbitrary_nx` must produce `N - 1` intermediate frames.
- Prefer EMA `ours_small_t.pkl` as the default EMA inference checkpoint for arbitrary/Nx and, if smoke testing confirms correctness, for fixed 2x inference as the special case `t = 0.5`.
- Keep EMA `ours_small.pkl` available for backward compatibility and Stage 1 training/fine-tuning paths. EMA inference and EMA training may have different checkpoint needs.
- Make Practical-RIFE v4.26 the default if only weights differ from v4.25 code; keep v4.25 as an alternative config.
- Use the same selected Practical-RIFE runtime/checkpoint for fixed 2x and arbitrary/Nx inference. For RIFE, fixed 2x is the `interpolation_factor = 2`, `t = 0.5` case; arbitrary/Nx uses request-provided factors and direct timesteps.
- Mark AMT-S as legacy and exclude it from active Stage 2 defaults. Do not delete AMT model files or repository directories without explicit confirmation.
- Analyze dynamic height/width ONNX export first. Prefer dynamic H/W when the model architecture and export path support it. Use configured static/padded fallback only with documented evidence.
- Store ONNX artifacts under `model_exports/onnx/`.
- Implement ONNX export/runtime only after PyTorch refactor and Nx support.
- Compare PyTorch and ONNX intermediate-frame tensors with `atol=1e-3`, `rtol=1e-3` where possible, and always report MAE and max absolute error.
- Add image-level comparison/sample inspection if tensor closeness is unstable.
- Implement a minimal BentoML compatibility proof after the new inference API is stable. It should instantiate and call the refactored adapter/runtime, not implement production upload, queue, storage, FastAPI, or deployment flow.
- Document all upstream model repository patches in this ExecPlan.
- Proactively remove hardcoded CUDA assumptions from EMA inference/export paths during the EMA runtime refactor, using input-device-aware tensor creation or `.to(input_tensor.device)` patterns rather than another global fixed device.
- Audit EMA-VFI and Practical-RIFE warping/grid helper code used in inference/export paths. Make grid creation and caches device-, dtype-, and shape-aware where current helper code depends on hardcoded global CUDA state or unsafe global grid reuse.
- Do not blindly rewrite Practical-RIFE `grid_sample` or warping code before export evidence exists. Attempt export through the planned wrapper first, record exact blockers, then patch only the minimal helper/wrapper/upstream code needed.
- Stabilize Practical-RIFE runtime source imports so Python model code is no longer imported from `model_weights/Practical-RIFE/.../train_log`. Prefer project-owned runtime source under `src/video_interpolation/inference_runtime/rife_upstream/` if it gives cleaner production-facing control.
- Treat `model_weights/Practical-RIFE/...` as weights/checkpoint/runtime artifact directories only after the RIFE runtime refactor.
- Inspect `pyproject.toml` before adding any Stage 2 dependency. Dependencies may be added through `uv` only when directly required for ONNX export, ONNX Runtime inference, or the minimal BentoML compatibility proof.

Mandatory constraints:

- Do not implement full BentoML production service before the runtime API stabilizes.
- Do not implement FastAPI, Celery, Redis, PostgreSQL application schema, frontend, retraining triggers, or monitoring.
- Do not implement ONNX export before runtime boundaries and PyTorch Nx are in place.
- Do not make fixed ONNX shapes the default plan without evidence that dynamic H/W is not reliable.
- Do not delete AMT-S repos, weights, or model files without explicit confirmation.
- Do not silently change model defaults without documenting the decision and configs.
- Do not import Practical-RIFE Python source from `model_weights/.../train_log` after the Stage 2 RIFE runtime stabilization milestone.
- Do not silently downgrade or change PyTorch, CUDA-related packages, NumPy, model-critical packages, or existing Stage 1 dependencies to add ONNX/BentoML dependencies. Stop and ask before such dependency changes.

## Non-Goals and Deferred Work

Out of scope for this stage:

- Full production BentoML service.
- FastAPI backend.
- Celery/Redis request queue.
- PostgreSQL application tables.
- MinIO upload/output service flow.
- Frontend.
- User authentication/session logic.
- Online quality history.
- Retraining triggers or retraining pool.
- Monitoring and alerting.
- Complete Docker Compose production stack.
- AMT-S runtime refactor, ONNX export, or Nx inference.
- Training/fine-tuning changes except what is needed to preserve existing Stage 1 EMA training behavior.

Deferred until after this stage:

- Full backend stage from the global plan.
- Production model registry/BentoML model-store integration.
- Multi-service/multi-GPU deployment layout.
- Extensive full-video or full-dataset validation.

## Architecture and Implementation Strategy

### Target Package Layout

Create a project-owned inference runtime package:

```text
src/video_interpolation/inference_runtime/
  __init__.py
  api.py
  backends/
    __init__.py
    base.py
    torch.py
    onnx.py
  common.py
  padding.py
  ema.py
  rife.py
  factory.py
  onnx_export.py
```

The exact file list may change during implementation, but responsibilities should remain clear.

### Request/Result API

Design the new API before changing adapter internals. Expected concepts:

- `InferenceMode`: `fixed_2x` and `arbitrary_nx`.
- `RuntimeBackendKind`: `torch` and `onnx`.
- `FramePairRequest`: left tensor, right tensor, mode, interpolation factor, optional timesteps, device/backend options.
- `FramePairResult`: generated intermediate frame tensors, timesteps, model name, backend name, original shape, padded shape, elapsed time where useful.
- Optional lower-level `RuntimeInputs` / `RuntimeOutputs` for model-specific runtime classes.

For `arbitrary_nx`, the request object is the authority for the concrete factor:

```python
FramePairRequest(
    ...,
    mode=InferenceMode.ARBITRARY_NX,
    interpolation_factor=4,
)
```

Local CLI support should eventually expose this as an argument such as `--interpolation-factor 4`. Future BentoML proof/service code should read the value from request parameters or a request body. Model configs may provide defaults and allowed limits only; they must not hide a fixed factor per model. If a caller omits the factor, config defaults may be used to build the request before runtime validation.

Public adapter methods should delegate:

- `predict_pair(left, right)` calls the new API with fixed 2x semantics and returns the single generated tensor.
- `predict_batch(pairs)` can remain sequential initially.
- `predict(left, right)` and `__call__` remain wrappers.
- New `predict_intermediate_frames(...)` or equivalent should return `N - 1` frames for arbitrary Nx.

Training-specific EMA methods `train_step()` and `eval_step()` should not be forced through the serving runtime. Arbitrary/Nx interpolation is inference-only, and Stage 1 training/fine-tuning must keep fixed 2x semantics and existing training-compatible checkpoint assumptions.

### Backend Abstraction

Use one runtime backend abstraction shared by EMA and RIFE:

```text
RuntimeBackend
  load(...)
  run(...)
  close()
```

PyTorch backend responsibilities:

- Own the explicit neural network callable.
- Run `torch.no_grad()` inference.
- Return raw model output tensors.

ONNX backend responsibilities:

- Load an ONNX artifact with ONNX Runtime.
- Accept already padded/formatted tensors.
- Return raw output tensors.
- Keep provider selection configurable.

Model-specific runtime responsibilities:

- Build/load the model.
- Normalize checkpoint state dicts.
- Format model inputs and timesteps.
- Apply model-specific padding/unpadding.
- Call the selected backend.
- Convert backend outputs into `FramePairResult`.

### Device, Warping, and Grid Helper Policy

Hardcoded device assumptions must be treated as runtime/export risks during the model-specific PyTorch refactors, not as late ONNX cleanup.

For EMA-VFI:

- Replace or bypass inference/export-path `.cuda()` and fixed CUDA tensor creation with input-device-aware logic.
- Preferred patterns are `device = input_tensor.device`, `tensor.to(device)`, and direct tensor creation with `device=input_tensor.device` and compatible dtype.
- Do not replace `.cuda()` with another global fixed device unless a documented model-specific reason requires it.
- Prefer project-owned wrappers when sufficient, but minimal upstream patches in `model_repos/EMA-VFI/` are allowed if documented and inference correctness is preserved.

For EMA-VFI and Practical-RIFE warp/grid helpers:

- Audit helper code used by inference/export for module-level `device`, global CUDA tensors, and global cached grids.
- If caches are retained, key them by device, dtype, height, width, and any other shape-affecting property needed for correctness.
- Grids must be created on the same device as the input tensor and with a compatible dtype.
- Different input resolutions must not reuse incompatible cached grids.
- Do not rewrite the warp math unless the current implementation creates concrete device/export/runtime fragility.

### Dependency Policy

Stage 2 may add dependencies through `uv` only when they are directly required for ONNX export, ONNX Runtime inference, or the minimal BentoML proof.

Expected candidate dependencies:

```text
onnx
onnxruntime or onnxruntime-gpu
bentoml
```

Before adding any dependency:

- inspect `pyproject.toml`;
- check whether the dependency is already present;
- record the dependency decision in this ExecPlan.

If dependency resolution requires downgrading or changing PyTorch, CUDA-related packages, NumPy, model-critical packages, or existing Stage 1 dependencies, stop and ask the project owner before applying the change.

### EMA-VFI Strategy

Initial PyTorch runtime:

- Build EMA small and EMA small-t through project-owned runtime code.
- Keep upstream checkpoint loading behavior but make the neural call boundary explicit.
- Use `Trainer.Model.net` as the torch core initially, or construct the same `MultiScaleFlow` network directly if this removes CUDA/training wrapper coupling cleanly.
- Keep arbitrary/Nx inference separate from EMA training/fine-tuning. Training remains fixed 2x and should continue to use the current training-compatible path and assumptions.
- Prefer `ours_small_t.pkl` as the default EMA inference checkpoint so runtime workers do not dynamically switch checkpoints per request. Use it for arbitrary/Nx and, if smoke testing confirms correctness, for fixed 2x inference as `t = 0.5`.
- Keep `ours_small.pkl` available for backward compatibility and training/fine-tuning. If configs need explicit fields, plan names such as `default_inference_checkpoint`, `fixed_2x_checkpoint`, `arbitrary_checkpoint`, and `training_checkpoint`, or a cleaner equivalent, without changing training semantics.
- Add arbitrary/Nx behavior with request-provided `interpolation_factor` and timesteps `(i + 1) / interpolation_factor`.
- `fixed_2x` remains a compatibility path that produces exactly one frame at timestep `0.5` and rejects request factors other than `2`.
- During EMA runtime work and before ONNX export, resolve EMA inference-path hardcoded CUDA usage proactively enough that CPU-side export analysis and configurable worker devices are not blocked by unconditional `.cuda()` calls.

ONNX export target:

- Prefer a project-owned `nn.Module` wrapper around EMA `net.forward(concat(img0, img1), timestep)` that returns only `pred`.
- Padding/unpadding stays outside ONNX.
- TTA and fast TTA stay outside ONNX or remain unsupported in first ONNX path.
- Dynamic batch/height/width should be attempted if hardcoded CUDA/device issues can be removed or bypassed.
- ONNX wrappers must not be tied to a hardcoded CUDA device. Device and dtype handling should follow the input tensors.

Known EMA risks:

- `Trainer.Model.device()` hardcodes CUDA.
- `MultiScaleFlow.calculate_flow()` and `forward()` create timestep tensors with `.cuda()`.
- `model/warplayer.py` has a module-level `device` and cached grids.
- Dynamic H/W may be affected by attention/window/padding internals and cached warp grids.
- These risks must be handled before or during the EMA runtime refactor and ONNX wrapper work, not deferred to final export validation.

### Practical-RIFE Strategy

Initial PyTorch runtime:

- Make `RIFEv4.26/train_log` the default model config.
- Keep `RIFEv4.25/train_log` as an alternative config.
- Stop importing Practical-RIFE Python source from `model_weights/Practical-RIFE/.../train_log`.
- Move or copy the required runtime model code into a stable location, preferably project-owned `src/video_interpolation/inference_runtime/rife_upstream/` if that gives cleaner production-facing control. An alternative stable upstream-repo location such as `model_repos/Practical-RIFE/model_runtime/` is acceptable if it proves cleaner.
- After this refactor, `model_weights/Practical-RIFE/...` should be used for weights/checkpoints only.
- Preserve non-strict `flownet.pkl` loading because local checkpoints contain extra training-only keys.
- Use the same selected Practical-RIFE runtime/checkpoint for fixed 2x and arbitrary/Nx. v4.26 remains the default Stage 2 runtime target, and v4.25 remains an alternative config.
- Fixed 2x calls timestep `0.5` and is equivalent to request `interpolation_factor = 2`.
- Arbitrary/Nx calls direct timesteps `(i + 1) / interpolation_factor`; local v4.25/v4.26 code supports direct timestep inference.
- The concrete `interpolation_factor` comes from `FramePairRequest`, a local CLI argument, or a future service request. Model configs may provide defaults and min/max limits but must not hide a fixed per-factor behavior.

ONNX export target:

- Prefer a project-owned wrapper around `IFNet` / `flownet` that concatenates image tensors or accepts already concatenated tensors and returns `merged[-1]`.
- `scale_list` should be fixed/configured outside ONNX or represented in a wrapper in a way export can handle.
- Padding/unpadding and timestep loops remain outside ONNX.
- Dynamic H/W should be attempted because the PyTorch path already uses divisor-based padding.
- First attempt export through the planned wrapper boundary and record the exact blocker if export fails. Patch `grid_sample`, dynamic grid construction, Python list inputs/outputs, unsupported operators, hardcoded device logic, or dynamic-shape issues only after the blocker is observed and scoped.

Known Practical-RIFE risks:

- Runtime imports model code from weight directories.
- `model.warplayer.py` uses module-level device and cached grids.
- ONNX export may not like Python lists in outputs/inputs, repeated dynamic warp grids, or `grid_sample`.
- `scale_list` as a Python list may need wrapper constants for export.
- The runtime-source import issue must be resolved during the Practical-RIFE runtime refactor, before ONNX export work depends on the model code location.

### ONNX Export and Runtime Strategy

ONNX work starts only after the PyTorch runtime refactor and PyTorch Nx support are in place.

For each model:

- export only the neural network core through a project-owned wrapper;
- keep padding, timestep loops, tensor formatting, video decoding, frame interleaving, output encoding, and service orchestration in Python;
- attempt dynamic height/width export first when the architecture and export path support it;
- if dynamic H/W fails, record the exact blocker and then propose or implement a documented configured static/padded-shape fallback;
- record all blockers and chosen fixes in this ExecPlan.

Practical-RIFE-specific ONNX policy:

- do not preemptively rewrite `grid_sample` or warp code;
- attempt the export first through the wrapper boundary;
- if export fails due to `grid_sample`, dynamic grid creation, Python lists, unsupported operators, hardcoded device logic, or dynamic shape limitations, patch the minimal wrapper/upstream helper needed and document the evidence.

### Video I/O Strategy

Keep `src/video_interpolation/inference.py` as the local video workflow initially, but change it to call the new adapter API after that API is stable.

Video I/O remains model-independent:

- decode frames;
- convert to tensors;
- call adapter/runtime with a request-provided interpolation factor when Nx video support is explicitly added;
- interleave original/generated frames;
- encode output;
- remux audio where possible;
- write timing/measurement metadata.

Do not move video decoding or encoding into model adapters, runtime backends, ONNX wrappers, or BentoML proof code.

Video-level Nx is deferred until Milestone 5 unless explicitly reprioritized. Milestone 4 should make tensor/frame-pair Nx work first and should not expand video workflow behavior beyond what is needed to keep existing fixed 2x paths compatible.

### BentoML Proof Strategy

After request/result API, PyTorch runtime, and local inference compatibility are stable, add a minimal proof only.

Expected proof:

- Can import the refactored inference component.
- Can instantiate an EMA or Practical-RIFE adapter/runtime from config.
- Can call the request/result API on tensor or image inputs.
- Does not implement production upload handling, async jobs, storage, database records, full REST contract, or deployment pipeline.

If adding BentoML requires a dependency already absent from `pyproject.toml`, check dependency status first and record the decision before changing dependencies.

## Milestones and Work Breakdown

### Milestone 1 - API and Runtime Skeleton

Objective: Add the request/result API and runtime backend skeleton without changing behavior.

Likely files/modules:

- `src/video_interpolation/inference_runtime/api.py`
- `src/video_interpolation/inference_runtime/backends/base.py`
- `src/video_interpolation/inference_runtime/common.py`
- `src/video_interpolation/inference_runtime/padding.py`
- focused tests under `tests/`

Expected output:

- Dataclasses/enums for inference requests, results, modes, and backend kind.
- Backend protocol/base class.
- Shared validation for fixed 2x and arbitrary Nx factors.
- No production behavior change yet.

Validation checkpoint:

- Focused tests for request validation and Nx factor/timestep generation.
- `uv run ruff check src tests`.
- `uv run pytest` or focused tests, using `UV_CACHE_DIR=/tmp/uv-cache` if sandbox cache permissions require it.

### Milestone 2 - PyTorch Runtime Refactor for EMA-VFI

Objective: Move EMA inference internals behind the new runtime API while preserving Stage 1 adapter compatibility.

Likely files/modules:

- `src/video_interpolation/inference_runtime/ema.py`
- `src/video_interpolation/inference_runtime/backends/torch.py`
- `src/video_interpolation/adapters/ema_vfi.py`
- EMA adapter tests

Expected output:

- EMA runtime class with explicit PyTorch neural call boundary.
- Existing `predict_pair`, `predict_batch`, `predict`, and `__call__` preserved.
- Stage 1 training methods still work or are explicitly isolated from the serving runtime path.
- EMA inference/export-path CUDA assumptions are audited and either handled in project wrappers or minimally patched upstream with documented rationale.
- EMA warp/grid helper behavior is audited for device, dtype, and shape safety before ONNX work begins.

Validation checkpoint:

- Existing EMA adapter tests pass.
- EMA pair prediction smoke still works in CUDA environment.
- Existing local EMA video inference path still works after adapter delegation.

### Milestone 3 - PyTorch Runtime Refactor for Practical-RIFE and Default v4.26

Objective: Move RIFE inference internals behind the new runtime API, switch default to v4.26, and keep v4.25 available.

Likely files/modules:

- `src/video_interpolation/inference_runtime/rife.py`
- stable Practical-RIFE runtime-source location, preferably `src/video_interpolation/inference_runtime/rife_upstream/`
- `src/video_interpolation/adapters/rife.py`
- `configs/models/practical_rife_v4_26.yaml`
- `configs/inference/practical_rife_v4_26_2x.yaml`
- preserve or update `configs/models/practical_rife_v4_25.yaml`
- preserve or update `configs/inference/practical_rife_v4_25_2x.yaml`
- CLI target/default updates

Expected output:

- Practical-RIFE runtime class with explicit PyTorch neural call boundary.
- v4.26 becomes active default for Stage 2.
- v4.25 remains runnable as an alternative.
- AMT-S is marked legacy and excluded from active Stage 2 defaults.
- Practical-RIFE Python runtime source is no longer imported from `model_weights/.../train_log`.
- RIFE warp/grid helper behavior is audited for device, dtype, and shape safety before ONNX work begins.

Validation checkpoint:

- Existing RIFE adapter tests pass.
- v4.26 adapter check/preflight works in CUDA environment.
- v4.26 one-pair video inference smoke works.
- v4.25 one-pair smoke remains available if the config is selected.

### Milestone 4 - PyTorch Nx Interpolation

Objective: Add arbitrary/Nx interpolation for EMA and Practical-RIFE through the new request/result API as an inference-only feature.

Likely files/modules:

- `src/video_interpolation/inference_runtime/api.py`
- `src/video_interpolation/inference_runtime/ema.py`
- `src/video_interpolation/inference_runtime/rife.py`
- `src/video_interpolation/adapters/ema_vfi.py`
- `src/video_interpolation/adapters/rife.py`
- tests for timestep generation and output count

Expected output:

- Fixed 2x remains supported.
- `interpolation_mode: fixed_2x | arbitrary_nx`.
- `interpolation_factor` is supplied by `FramePairRequest` or an adapter/CLI caller when building the request; configs may provide only defaults, limits, supported modes, and checkpoint mappings.
- `interpolation_factor` validates as an integer in `2 <= factor <= 8`.
- `fixed_2x` rejects factors other than `2` and returns exactly one frame at timestep `0.5`.
- `arbitrary_nx` generates and returns `N - 1` intermediate frames for request factor `N`.
- EMA inference uses `ours_small_t.pkl` for arbitrary/Nx and should use it for fixed 2x as `t = 0.5` if smoke testing confirms correctness. `ours_small.pkl` stays available for backward compatibility and training/fine-tuning.
- RIFE arbitrary/Nx uses direct timesteps on v4.26 by default; v4.25 remains selectable as an alternative.
- EMA training/fine-tuning remains fixed 2x and is not routed through arbitrary/Nx runtime code.
- Existing video inference may still be 2x-only until video-level Nx flow is explicitly updated in Milestone 5, but tensor/frame-pair Nx must work.
- Do not add one config file per interpolation factor. If a config change is needed, prefer generic supported-mode/default/min/max/checkpoint fields.

Validation checkpoint:

- Unit tests for factors 2, 4, and 8.
- Unit tests that invalid factors fail before model inference starts.
- Unit tests that `fixed_2x` rejects non-2 factors and `arbitrary_nx` returns `factor - 1` frames.
- Tensor smoke for EMA Nx in CUDA environment where available.
- Tensor smoke for RIFE Nx in CUDA environment where available.

### Milestone 5 - Local Video Inference Uses New API

Objective: Rewire local video inference to the request/result API without duplicating model logic.

Likely files/modules:

- `src/video_interpolation/inference.py`
- `src/video_interpolation/batch_inference.py`
- `src/video_interpolation/cli.py`
- inference config files
- tests under `tests/test_inference.py`

Expected output:

- Existing 2x local inference remains compatible.
- Nx video inference is added if it is clean and bounded by config.
- Measurement output records mode/factor/backend.
- AMT-S excluded from active Stage 2 default all-target selection or clearly marked legacy.

Validation checkpoint:

- Existing PyAV writer tests pass.
- One-pair 2x smoke for EMA and RIFE still writes readable video.
- Nx video smoke on a tiny/short input if implemented in this milestone.

### Milestone 6 - ONNX Export Feasibility and Export Wrappers

Objective: Analyze and implement export wrappers for EMA and RIFE where feasible.

Likely files/modules:

- `src/video_interpolation/inference_runtime/onnx_export.py`
- `src/video_interpolation/inference_runtime/ema.py`
- `src/video_interpolation/inference_runtime/rife.py`
- `configs/export/` or `configs/inference_runtime/` if useful
- `model_exports/onnx/`
- docs

Expected output:

- EMA export wrapper around neural core returning only generated frame.
- RIFE export wrapper around neural core returning only generated frame.
- Dynamic H/W export attempted first.
- Static/padded shape fallback documented only if dynamic H/W fails with evidence.
- Export artifacts written under `model_exports/onnx/`.
- EMA export path is not blocked by unconditional `.cuda()` or fixed CUDA tensor creation.
- RIFE export uses the stabilized runtime source location rather than importing source from `model_weights`.
- Practical-RIFE warp/grid code is patched only after export evidence identifies a concrete blocker.
- Any added `onnx` dependency decision is recorded after inspecting `pyproject.toml`.

Validation checkpoint:

- Export command or function either produces ONNX artifact or records exact blocker.
- Export blocker evidence includes exception/operator/device/dynamic-shape details.
- No silent fixed-shape fallback.

### Milestone 7 - ONNX Runtime Backend and Equivalence Checks

Objective: Add ONNX Runtime loading/inference where export succeeds and compare against PyTorch outputs.

Likely files/modules:

- `src/video_interpolation/inference_runtime/backends/onnx.py`
- `src/video_interpolation/inference_runtime/ema.py`
- `src/video_interpolation/inference_runtime/rife.py`
- equivalence test/smoke helper
- docs

Expected output:

- ONNX Runtime backend can load exported artifact.
- Same request/result API can select `backend: onnx`.
- PyTorch-vs-ONNX comparison reports MAE and max absolute error.
- Uses `atol=1e-3`, `rtol=1e-3` where possible.
- Image-level samples are written/inspected if tensor closeness is unstable.
- Any added `onnxruntime` or `onnxruntime-gpu` dependency decision is recorded after inspecting `pyproject.toml`.

Validation checkpoint:

- ONNX equivalence smoke on tiny valid padded inputs for each exported model.
- Dynamic H/W smoke on at least two different input shapes if export supports it.
- Documented fallback if one model cannot run ONNX reliably.

### Milestone 8 - Minimal BentoML Compatibility Proof

Objective: Demonstrate that the refactored runtime can be instantiated and called from BentoML.

Likely files/modules:

- a minimal proof module under a clearly named location, for example `src/video_interpolation/bentoml_poc/` or `services/bentoml_poc/`
- docs
- optional focused smoke test if BentoML can be imported locally

Expected output:

- Minimal service/proof imports the project package.
- Instantiates one configured adapter/runtime.
- Calls the request/result API on an input tensor or simple image payload.
- No production upload handling, queues, storage, database, FastAPI integration, or deployment pipeline.
- Any added `bentoml` dependency decision is recorded after inspecting `pyproject.toml`.

Validation checkpoint:

- Import/smoke command succeeds if BentoML dependency is present.
- If dependency is missing, dependency decision is recorded before adding it.

### Milestone 9 - Documentation, Project Map, and Handoff

Objective: Make the refactor understandable and restartable.

Likely files/modules:

- `docs/`
- `configs/*/README.md`
- `.agent/docs/PROJECT_MAP.md`
- this ExecPlan

Expected output:

- Inference architecture docs.
- Runtime backend docs.
- Fixed 2x and Nx config docs.
- ONNX export/runtime docs with artifact locations and limitations.
- BentoML proof docs.
- Updated project map.
- Completed Stage 2 handoff.

Validation checkpoint:

- `uv run ruff check src tests`
- `uv run pytest`
- representative smoke commands recorded in this ExecPlan.

## Validation Strategy

Validation must be incremental and behavior-focused.

Required automated checks:

- Request/result API validation, including mode/factor constraints.
- Timestep generation for factors 2, 4, and 8.
- Runtime/request `interpolation_factor` validation, including non-integer rejection, range rejection, and fixed 2x rejection of factors other than `2`.
- Arbitrary/Nx output count validation: factor `N` produces exactly `N - 1` intermediate frames.
- Adapter backward-compatible wrapper behavior.
- EMA training/fine-tuning compatibility remains isolated from arbitrary/Nx runtime paths.
- RIFE padding/unpadding behavior preserved.
- Video inference config and writer behavior preserved.
- ONNX comparison helper reports MAE and max absolute error when ONNX path exists.

Required smoke checks:

- EMA fixed 2x pair inference through new API.
- RIFE v4.26 fixed 2x pair inference through new API.
- RIFE v4.25 pair inference remains available through alternative config.
- EMA fixed 2x inference with `ours_small_t.pkl` at `t = 0.5`, if CUDA is available, before making `_t` the default inference checkpoint for fixed 2x.
- EMA Nx pair inference for factors 2, 4, 8 using `ours_small_t.pkl`.
- RIFE Nx pair inference for factors 2, 4, 8.
- Existing local 2x video inference for EMA and RIFE still writes readable outputs.
- ONNX export commands for EMA and RIFE, or documented blockers.
- ONNX Runtime equivalence where export succeeds.
- Minimal BentoML proof imports and calls the runtime after the API stabilizes.

Expected final checks:

- `uv run ruff check src tests`
- `uv run pytest`
- model-specific CUDA smoke commands where the current environment permits.

If sandbox `uv` cache permissions fail, rerun with a writable cache such as `UV_CACHE_DIR=/tmp/uv-cache`. If CUDA is unavailable in the sandbox, record that GPU model smoke checks require the user's CUDA environment.

## Expected Artifacts

Expected completion artifacts:

- `src/video_interpolation/inference_runtime/` package.
- Request/result API dataclasses or equivalent structured interfaces.
- Shared runtime backend abstraction.
- PyTorch runtime backend.
- ONNX Runtime backend where feasible.
- EMA runtime class.
- Practical-RIFE runtime class.
- Updated EMA and RIFE adapters preserving Stage 1 wrapper calls.
- RIFE v4.26 model and inference configs.
- RIFE v4.25 alternative configs preserved.
- Fixed 2x and arbitrary/Nx request/runtime support, with any configs limited to supported modes, defaults, min/max factor bounds, and checkpoint mappings rather than per-factor files.
- EMA inference checkpoint policy documented and implemented so `ours_small_t.pkl` can serve arbitrary/Nx and, after smoke confirmation, fixed 2x at `t = 0.5`; `ours_small.pkl` remains available for training/backward compatibility.
- ONNX export artifacts under `model_exports/onnx/` where export succeeds.
- ONNX feasibility/blocker report in docs or this ExecPlan.
- PyTorch-vs-ONNX comparison outputs or reports.
- Minimal BentoML compatibility proof.
- Updated docs.
- Updated `.agent/docs/PROJECT_MAP.md`.
- Completed Stage 2 ExecPlan handoff.

## Risks, Assumptions, and Recovery

Assumptions:

- Stage 1 behavior should remain available while internals are refactored.
- The project owner accepts a new request/result API as the primary future API.
- Existing adapter methods can become compatibility wrappers.
- EMA and RIFE can share backend abstractions even though their model construction and padding differ.
- Practical-RIFE v4.26 should become default because local model code matches v4.25 and only weights differ.
- Dynamic H/W ONNX export should be attempted before any fixed-shape fallback.
- Device-aware tensor creation is preferable to global device state for inference/runtime/export paths.
- Practical-RIFE runtime source can be stabilized into a project-owned location without changing inference semantics.
- Arbitrary/Nx interpolation is inference-only; EMA training and fine-tuning remain fixed 2x.
- Runtime requests, CLI arguments, or future service payloads supply the concrete interpolation factor. Configs only describe capabilities, defaults, bounds, and checkpoint mappings.
- EMA `ours_small_t.pkl` can likely serve fixed 2x inference at `t = 0.5`, but this should be confirmed with smoke testing before relying on it as the fixed 2x default inference checkpoint.

Risks:

- EMA hardcoded `.cuda()` calls may block CPU export and dynamic ONNX export unless patched or bypassed.
- Replacing `.cuda()` incorrectly with another global device would preserve the same serving/export fragility under a different name.
- EMA and RIFE warp helpers may reuse cached grids across incompatible device, dtype, height, or width combinations.
- EMA attention/window logic may have dynamic-shape export limitations.
- RIFE `grid_sample`/warp code may be unsupported or numerically unstable in ONNX export/runtime.
- RIFE import isolation from weight-directory `train_log` can remain fragile.
- ONNX Runtime dependency may be absent or provider-sensitive.
- BentoML dependency may be absent or may require dependency changes.
- Adding ONNX/BentoML dependencies may pressure PyTorch, CUDA-related packages, NumPy, or existing Stage 1 dependency versions.
- Changing default RIFE version may affect expected validation metrics.
- Removing AMT-S from active defaults may require CLI/doc updates without deleting legacy code.
- Accidentally putting interpolation factors into separate configs or hidden model settings would conflict with future request-driven service behavior.
- Routing arbitrary/Nx through EMA training/fine-tuning code would risk breaking Stage 1 training semantics.
- Switching EMA fixed 2x inference to `ours_small_t.pkl` without smoke evidence could change outputs unexpectedly.

Recovery:

- Keep refactor milestones small and preserve wrappers until new API is validated.
- Do not remove old Stage 1 configs until replacements are validated.
- If EMA direct `net` construction is cleaner than `Trainer.Model`, switch with a documented decision and tests.
- When EMA inference/export code creates tensors, prefer input-device-aware creation or `.to(input_tensor.device)` and preserve dtype where relevant.
- If warp/grid caches are retained, key them by device, dtype, height, width, and any other shape-affecting property needed for correctness.
- For Practical-RIFE ONNX, first attempt export through the wrapper boundary and patch warping/grid logic only after the exact blocker is observed.
- Move/copy Practical-RIFE runtime model source into a stable project-owned or upstream-repo runtime location before ONNX export relies on it.
- If dynamic ONNX export fails, capture the exact error and implement a documented static/padded fallback only after evidence is recorded.
- If ONNX export fails because of upstream hardcoded device logic, patch upstream minimally or create project wrappers that bypass those paths, documenting each change.
- If ONNX, ONNX Runtime, or BentoML dependencies are missing, inspect `pyproject.toml` first and record the dependency decision before adding anything.
- If dependency resolution requires downgrading or changing PyTorch, CUDA-related packages, NumPy, model-critical packages, or existing Stage 1 dependencies, stop and ask before applying the change.
- If CUDA is unavailable in the current environment, separate CPU-only API tests from GPU smoke commands and record which validation was deferred.
- Keep arbitrary/Nx implementation in runtime/adapters/local inference only; if training code needs compatibility adjustments, keep them minimal and fixed-2x preserving.
- If `ours_small_t.pkl` fixed 2x smoke differs materially from `ours_small.pkl`, keep `ours_small.pkl` as the fixed 2x inference fallback and document why dynamic checkpoint switching or separate runtime instances are needed.
- If config changes are required for Nx, use generic fields such as supported modes, default/min/max factor, and explicit checkpoint mappings rather than factor-specific config files.

## Progress

- 2026-05-31: Created Stage 2 ExecPlan from the narrowed Stage 2 plan, completed Stage 1 handoff, current repository inspection, upstream EMA/RIFE inspection, and user clarification answers. Implementation has not started.
- 2026-05-31: Updated the active ExecPlan with additional technical-risk policies before implementation. Captured EMA hardcoded CUDA remediation, warp/grid helper audits, evidence-first RIFE ONNX patching, Practical-RIFE runtime-source stabilization, and Stage 2 dependency decision rules. No source code, configs, dependencies, model repositories, or weights were changed.
- 2026-05-31: Started Milestone 1 only. Added `src/video_interpolation/inference_runtime/` with structured request/result API types, interpolation mode/backend enums, runtime input/output containers, shared backend abstraction, and minimal callable-based PyTorch backend skeleton. No EMA-VFI, Practical-RIFE, AMT-S, local video inference, configs, ONNX, or BentoML code was changed.
- 2026-05-31: Added focused tests in `tests/test_inference_runtime_api.py` for fixed 2x validation, arbitrary Nx factor/timestep generation, invalid request handling, result validation, backend lifecycle behavior, and PyTorch backend no-grad execution.
- 2026-05-31: Focused validation completed for Milestone 1 work so far: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_inference_runtime_api.py` passed with 11 tests; `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/video_interpolation/inference_runtime tests/test_inference_runtime_api.py` passed. Initial `uv run pytest tests/test_inference_runtime_api.py` failed because the default uv cache under `/home/lighter_01/.cache/uv` is read-only in the sandbox; rerun with `/tmp/uv-cache` succeeded.
- 2026-05-31: Completed Milestone 1 validation. Required unprefixed commands `uv run pytest` and `uv run ruff check src tests` both failed at uv startup because the sandbox cannot create temporary files under `/home/lighter_01/.cache/uv`. Required fallback commands succeeded: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` passed with 41 tests, and `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` passed.
- 2026-05-31: Completed Milestone 2. Added `src/video_interpolation/inference_runtime/ema.py` with `EMAVFIPyTorchRuntime`, `EMAVFIPyTorchRuntimeConfig`, and shared EMA tensor preparation. Refactored `EMAVFIAdapter.predict_pair()` to create a fixed-2x `FramePairRequest` and delegate to `predict_frame_pair()`, which returns a `FramePairResult` from the EMA runtime. Existing `predict_batch`, `predict`, and `__call__` remain inherited wrappers over `predict_pair`. `train_step()` and `eval_step()` remain on the upstream training model path and are not routed through the serving runtime.
- 2026-05-31: Patched EMA upstream inference/runtime risk points. `model_repos/EMA-VFI/Trainer.py` now accepts a configurable device while defaulting to `cuda`; `model_repos/EMA-VFI/model/flow_estimation.py` now creates timestep tensors on the feature tensor device/dtype instead of calling `.cuda()`; `model_repos/EMA-VFI/model/warplayer.py` now creates warp grids on `tenFlow.device`/`tenFlow.dtype` and keys the cache by device, dtype, batch, height, and width; `model_repos/EMA-VFI/model/loss.py` now lets `LapLoss` receive the configured model device and creates temporary upsample tensors on the input tensor device/dtype.
- 2026-05-31: Added `tests/test_ema_adapter.py` for EMA request/result runtime behavior, deferred arbitrary Nx handling, adapter wrapper compatibility, training-path isolation, and checkpoint normalization.
- 2026-05-31: Completed Milestone 2 validation. Required unprefixed `uv run pytest` and `uv run ruff check src tests` still fail at uv startup due read-only `/home/lighter_01/.cache/uv`; fallback commands succeeded. `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` passed with 46 tests. `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` passed. EMA CLI smoke checks `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli ema-preflight` and `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli ema adapter-check` both exited 0 but reported `blocked` because CUDA is unavailable and the default EMA configs request CUDA. No CUDA 2x inference smoke was run in this environment.
- 2026-05-31: Completed Milestone 3. Added `src/video_interpolation/inference_runtime/rife.py` with `PracticalRIFEPyTorchRuntime`, `PracticalRIFEPyTorchRuntimeConfig`, and shared RIFE tensor preparation/padding helpers. Refactored `PracticalRIFEAdapter.predict_pair()` to create a fixed-2x `FramePairRequest` and delegate to `predict_frame_pair()`, which returns a `FramePairResult` from the RIFE runtime. Existing `predict_batch`, `predict`, and `__call__` remain inherited wrappers over `predict_pair`.
- 2026-05-31: Stabilized Practical-RIFE runtime imports by copying the v4.26 `IFNet_HDv3.py` source into `src/video_interpolation/inference_runtime/rife_upstream/`, adding an inference-only `RIFE_HDv3.Model` wrapper there, and adding project-owned `warplayer.py`. The adapter no longer imports Python source from `model_weights/Practical-RIFE/.../train_log`; those directories are now checkpoint artifact inputs containing `flownet.pkl`.
- 2026-05-31: Made Practical-RIFE v4.26 the Stage 2 default. Added `configs/models/practical_rife_v4_26.yaml`, `configs/inference/practical_rife_v4_26_2x.yaml`, and `configs/validation/practical_rife_v4_26_candidate.yaml`. Kept v4.25 configs available as alternatives. Updated RIFE CLI defaults and active batch aliases to v4.26; active `models`/`neural`/`all` aliases exclude AMT-S while the explicit `amt`/`amt_s` target remains available.
- 2026-05-31: Audited and patched RIFE warping helper behavior in the project-owned runtime source. `rife_upstream/warplayer.py` now creates grids on `tenFlow.device`/`tenFlow.dtype` and keys the cache by device, dtype, batch, height, and width. No upstream Practical-RIFE repository files were patched in this milestone.
- 2026-05-31: Added and updated documentation for Milestone 3. Created `docs/stage2_inference_runtime_refactor.md`; updated config README files and `.agent/docs/PROJECT_MAP.md` for the v4.26 default, v4.25 alternative, and project-owned RIFE runtime source.
- 2026-05-31: Completed Milestone 3 validation. Required unprefixed `uv run pytest` and `uv run ruff check src tests` still fail at uv startup due read-only `/home/lighter_01/.cache/uv`; fallback commands succeeded. `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` passed with 48 tests. `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` passed. RIFE CLI smoke checks `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli rife-preflight`, `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli rife adapter-check`, and `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli rife adapter-check --config configs/models/practical_rife_v4_25.yaml` all exited 0 but reported `blocked` because CUDA is unavailable and the tested configs request CUDA. No CUDA 2x inference smoke was run in this environment.
- 2026-05-31: Planning-only update before Milestone 4. Clarified that arbitrary/Nx is inference-only, `interpolation_factor` is request/runtime supplied rather than hardcoded in configs, EMA inference should prefer `ours_small_t.pkl` for arbitrary/Nx and fixed 2x at `t = 0.5` after smoke confirmation, EMA training remains fixed 2x/backward-compatible, and Practical-RIFE uses the same selected runtime/checkpoint for fixed 2x and arbitrary/Nx. No source code, configs, model repositories, dependencies, or tests were changed/run for this update.
- 2026-05-31: Completed Milestone 4 PyTorch tensor-pair Nx implementation. EMA and Practical-RIFE PyTorch runtimes now accept validated `FramePairRequest` values in `fixed_2x` and `arbitrary_nx` modes, loop over request timesteps, and return `FramePairResult` objects with `factor - 1` frames for arbitrary/Nx. Existing `predict_pair`, `predict_batch`, `predict`, and `__call__` fixed 2x behavior remains as compatibility wrappers, and both adapters now expose `predict_intermediate_frames(...)` for public tensor-pair Nx inference.
- 2026-05-31: Updated EMA inference checkpoint policy in configs. Stage 2 EMA model and local inference configs now select `EMA-VFI/ours_small_t.pkl` for inference, keep `EMA-VFI/ours_small.pkl` recorded as the training checkpoint, and expose supported modes plus default/min/max interpolation factor fields. The EMA training config remains fixed 2x and continues to use `ours_small.pkl`.
- 2026-05-31: Added pair-smoke CLI commands `ema infer-pair` and `rife infer-pair` with `--mode` and `--interpolation-factor` runtime arguments. These use synthetic tensor pairs and the request/result API, leaving full video-level Nx scheduling and output FPS behavior deferred to Milestone 5.
- 2026-05-31: Updated Stage 2 documentation, config READMEs, and `.agent/docs/PROJECT_MAP.md` for fixed 2x vs arbitrary/Nx semantics, request-time interpolation factors, EMA `_t` inference checkpoint policy, Practical-RIFE v4.26 direct timestep behavior, pair-smoke CLI commands, and Milestone 5 video-level Nx deferral. No upstream model repository files were patched in Milestone 4.
- 2026-05-31: Completed Milestone 4 validation. Focused fallback validation `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_inference_runtime_api.py tests/test_ema_adapter.py tests/test_rife_adapter.py` passed with 27 tests, and `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` passed. Required unprefixed `uv run pytest` and `uv run ruff check src tests` still fail at uv startup because `/home/lighter_01/.cache/uv` is read-only. Required fallback validation succeeded: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` passed with 53 tests and `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` passed. `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import torch; print(torch.cuda.is_available())"` returned `False`, so CUDA EMA/RIFE Nx smoke checks for factors 2, 4, and 8 were deferred. CLI wiring checks for `ema infer-pair --help` and `rife infer-pair --help` succeeded. EMA/RIFE preflight and adapter-check commands exited 0 but reported `blocked` due CUDA unavailability.
- 2026-05-31: Completed Milestone 5 local video inference refactor. `src/video_interpolation/inference.py` now validates `interpolation_mode` and request-time `interpolation_factor`, computes local video timesteps through the Stage 2 runtime API, calls `predict_frame_pair(FramePairRequest(...))` for runtime-aware adapters, interleaves all returned intermediate frames before the next original frame, and sets output FPS to `input_fps * interpolation_factor`. Legacy adapters without `predict_frame_pair` remain fixed-2x-only through `predict_pair`.
- 2026-05-31: Added local video CLI/runtime factor wiring. `ema infer-video`, `rife infer-video`, and `infer-all-videos` now accept `--mode` and `--interpolation-factor`; invalid factors outside `2..8` fail before model inference starts. Batch output paths and MLflow run names now use factor suffixes such as `_4x`, and measurement CSVs include `interpolation_mode`, `interpolation_factor`, and `runtime_backend`. With no explicit target, `infer-all-videos --mode arbitrary_nx` selects active EMA/RIFE model targets only; baselines and AMT-S remain fixed-2x/legacy paths unless explicitly selected.
- 2026-05-31: Updated EMA and Practical-RIFE local inference configs with top-level `interpolation_mode: fixed_2x` and `interpolation_factor: 2` defaults while preserving Stage 1 2x behavior. `output_fps_multiplier` remains in configs as a compatibility field, but Stage 2 local model video output FPS is now driven by the validated interpolation factor.
- 2026-05-31: Added focused local-video tests for request/result runtime calls, factors 2/4/8 frame counts, generated-frame ordering by timestep, output FPS multiplication, config/CLI factor validation, existing fixed-2x config compatibility, and measurement CSV metadata. Updated `docs/stage2_inference_runtime_refactor.md`, config README files, and `.agent/docs/PROJECT_MAP.md` for Milestone 5 behavior. No upstream model repository files were patched in Milestone 5.
- 2026-05-31: Completed Milestone 5 validation. Required unprefixed `uv run pytest` and `uv run ruff check src tests` still fail at uv startup because `/home/lighter_01/.cache/uv` is read-only. Required fallback validation succeeded: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` passed with 60 tests and `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` passed. Focused fallback validation `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_inference.py tests/test_inference_runtime_api.py tests/test_ema_adapter.py tests/test_rife_adapter.py` passed with 42 tests. CLI help checks for `infer-all-videos`, `ema infer-video`, and `rife infer-video` succeeded. `torch.cuda.is_available()` returned `False`; EMA/RIFE preflight and adapter-check commands exited 0 but reported `blocked` due CUDA unavailability, so CUDA video smoke checks were deferred.
- 2026-05-31: Added request-time Practical-RIFE `scale` support as a serving-readiness refinement after Milestone 5, without starting Milestone 6. `FramePairRequest.backend_options["scale"]` now overrides the Practical-RIFE config default per request, validates against upstream-supported values `0.25`, `0.5`, `1.0`, `2.0`, and `4.0`, and is passed into `model.inference(..., scale=...)` for every requested timestep.
- 2026-05-31: Exposed Practical-RIFE scale externally in local CLI paths. `rife infer-pair` and `rife infer-video` now accept `--scale`, while `infer-all-videos` accepts `--rife-scale` for Practical-RIFE targets. Local video inference carries request runtime options into `FramePairRequest.backend_options`, reports them in `VideoInferenceResult`, logs them to MLflow params, and includes them in batch measurement CSVs.
- 2026-05-31: Updated tests and docs for request-time Practical-RIFE scale. Added validation tests for allowed/disallowed scale values, request scale overriding the config default, adapter `predict_intermediate_frames(..., scale=...)`, local video runtime option propagation, and CLI invalid-scale failure. Updated `docs/stage2_inference_runtime_refactor.md`, config README files, and `.agent/docs/PROJECT_MAP.md`.
- 2026-05-31: Started Milestone 6 ONNX export feasibility work. Inspected the current Stage 2 EMA and Practical-RIFE PyTorch runtimes plus the upstream neural-core signatures. Added `src/video_interpolation/inference_runtime/onnx_export.py` with `OnnxExportConfig`, `OnnxExportResult`, dynamic/static shape policy validation, model/version-oriented artifact path generation, ONNX checker integration, optional ONNX simplification, `EMAVFIOnnxWrapper`, and `PracticalRIFEOnnxWrapper`.
- 2026-05-31: Added developer CLI export commands without starting ONNX Runtime. `ema export-onnx` loads the selected EMA adapter/checkpoint and exports `runtime.model.net` through `EMAVFIOnnxWrapper`; `rife export-onnx` loads Practical-RIFE v4.26/v4.25 configs and exports `runtime.model.flownet` through `PracticalRIFEOnnxWrapper`, with explicit `--scale`. Both commands default to `model_exports/onnx/`, `--shape-mode dynamic_hw`, opset 17, and simplification enabled.
- 2026-05-31: Added focused ONNX export tests in `tests/test_onnx_export.py` for export config validation, dynamic/static axes, sample input creation, EMA/RIFE wrapper output boundaries, RIFE scale-list construction, successful fake-module ONNX artifact writing, and structured export failure reporting. No ONNX Runtime inference or PyTorch-vs-ONNX equivalence checks were added.
- 2026-05-31: Real CPU EMA dynamic-H/W export initially failed with `RuntimeError: state_dict changed after running the tracer` because EMA's feature extractor lazily registers shape-specific `attn_mask`/`HW` buffers during the first traced forward pass. Added a project-owned export-helper prewarm forward pass before tracing so lazy inference caches are initialized before export without changing model math or upstream source.
- 2026-05-31: Real CPU ONNX export smokes succeeded for EMA-VFI-small and Practical-RIFE v4.26 with default simplification. Artifacts were written under `model_exports/onnx/ema_vfi_small/` and `model_exports/onnx/practical_rife_v4_26/`. EMA emitted PyTorch tracer warnings about Python shape math and cached attention-mask logic, so dynamic axes are exported but true dynamic-H/W behavior remains to be validated in Milestone 7 with ONNX Runtime multi-shape checks.
- 2026-05-31: Started and completed Milestone 7 ONNX Runtime backend work. Added `src/video_interpolation/inference_runtime/backends/onnx.py` with ONNX Runtime artifact loading, provider validation, tensor-to-array conversion, input/output name checks, session-provider reporting, and clear backend errors for missing artifacts/providers/inputs/shapes. Added EMA and Practical-RIFE ONNX runtime classes that reuse the Stage 2 `FramePairRequest`/`FramePairResult` API, model-specific padding/unpadding, and existing timestep loops.
- 2026-05-31: Added `src/video_interpolation/inference_runtime/onnx_validation.py` with default artifact resolution, PyTorch-vs-ONNX tensor equivalence metrics (`MAE`, max absolute error, MSE, `allclose` with `atol=1e-3`, `rtol=1e-3`), JSON/CSV report writing, and sample PyTorch/ONNX/normalized-absdiff PNG output for tensor mismatches. Added developer CLI commands `ema validate-onnx` and `rife validate-onnx` with explicit providers, artifact override/default resolution, repeated `--shape` dynamic-H/W checks, mode/factor selection, tolerance options, and validation output paths under `outputs/onnx_validation/`.
- 2026-05-31: Real CPU ONNX Runtime validation used the simplified Milestone 6 artifacts by default. EMA `ema_vfi_small_dynamic_hw_opset17.simplified.onnx` matched PyTorch at `32x32` with MAE `3.1393333e-07` and max absolute error `1.7881393e-06`; `64x64` failed in ONNX Runtime with a LayerNormalization shape error (`X.shape={2,64,98}`, `scale.shape={128}`, `bias.shape={128}`), so EMA dynamic H/W remains blocked and needs an export/runtime fix or static/padded fallback evidence before serving use.
- 2026-05-31: Practical-RIFE `practical_rife_v4_26_dynamic_hw_opset17.simplified.onnx` matched PyTorch at `128x128` with MAE `8.4759959e-06` and max absolute error `0.00091010332`. Dynamic `128x256` executed successfully but failed strict `1e-3` allclose due max absolute error `0.0037825704` while MAE stayed low at `5.5331097e-05`; sample torch/onnx/absdiff PNGs were written for inspection. The original ONNX artifact produced the same `128x256` metrics, so the mismatch is not simplification-specific.

## Surprises & Discoveries

- `.agent/docs/general_plan.md` was referenced by the task text, but the actual global plan path is `.agent/general_plan.md`.
- Practical-RIFE v4.25 and v4.26 local `RIFE_HDv3.py` and `IFNet_HDv3.py` files are identical; only `flownet.pkl` differs.
- EMA-VFI and Practical-RIFE both use module-level device/grid helper code in upstream warp paths, which may affect ONNX export and dynamic-shape behavior.
- EMA-VFI has hardcoded CUDA tensor creation inside the neural core, not only in the outer `Trainer.Model` wrapper.
- The current sandbox cannot write uv temporary files under the default `/home/lighter_01/.cache/uv` path, so validation commands need `UV_CACHE_DIR=/tmp/uv-cache` unless the environment permissions change.
- EMA `model/loss.py` also had module-level CUDA/default-device assumptions. `LapLoss` and its temporary upsample tensors were made configurable/input-device-aware because `Trainer.Model` constructs the loss object even when the adapter is used for serving inference. `model/refine.py` still has an unused module-level device variable and was left unchanged.
- Practical-RIFE v4.25 and v4.26 bundled source files were still identical at implementation time; switching the default to v4.26 therefore only changes the default checkpoint/config selection, not the project-owned runtime code.
- Practical-RIFE `train_log/RIFE_HDv3.py` included optimizer, loss, DDP, and training helpers that are not needed for Stage 2 serving. The project-owned `rife_upstream/RIFE_HDv3.py` is intentionally inference-only and keeps `flownet`, `train()`, `eval()`, and `inference()`.
- Local video inference can exercise fixed 2x and arbitrary Nx behavior without CUDA/model weights by using a fake runtime adapter in tests. This is sufficient for frame scheduling, FPS, and metadata validation, but not for model quality or real checkpoint behavior.
- Practical-RIFE upstream `inference_video.py` accepts `--scale`, treats `--UHD` as `scale=0.5`, and asserts scale is one of `0.25`, `0.5`, `1.0`, `2.0`, or `4.0`. Stage 2 mirrors that validation and exposes the value as a request option rather than a fixed model-config-only setting.
- `pyproject.toml` already includes `onnx`, `onnxruntime`, `onnxruntime-gpu`, and `onnx-simplifier`, so Milestone 6 did not add dependencies. ONNX Runtime remains unused by implementation code in this milestone.
- The installed PyTorch exporter emits deprecation warnings for the legacy TorchScript ONNX exporter. Milestone 6 intentionally uses `dynamo=False` for export because it avoids introducing `onnxscript` as an additional dependency during this bounded export-wrapper milestone.
- The ONNX export boundary can be represented as `left/right/timestep -> generated frame` for both active models. EMA uses the last value returned by `model.net(...)` (`pred`), and Practical-RIFE uses the last merged output from IFNet (`merged[-1]`).
- EMA's adapter keeps the upstream repository import context active until close, including the current working directory. Relative ONNX export paths therefore must be resolved to absolute paths before model loading; otherwise EMA exports land under `model_repos/EMA-VFI/`.
- The same EMA CWD issue affects ONNX validation reports if output paths are resolved after entering the adapter context. Milestone 7 resolves validation output directories before model loading and moved early generated EMA reports back under the project `outputs/onnx_validation/` tree.
- `onnxruntime` 1.26.0 is installed and lists `TensorrtExecutionProvider`, `CUDAExecutionProvider`, and `CPUExecutionProvider`, but `torch.cuda.is_available()` is `False` in this environment. Milestone 7 validation therefore used only `CPUExecutionProvider`.
- EMA ONNX dynamic axes do not imply working dynamic H/W. The exported graph runs at the original `32x32` sample shape but fails at `64x64` in a LayerNormalization node, likely because EMA feature-extractor shape math/cached attention-mask logic was traced with shape-dependent constants.
- Practical-RIFE dynamic H/W ONNX Runtime execution works for `128x256`, but strict tensor closeness is slightly unstable at that shape (`max_abs_error` above `1e-3` while MAE remains low). Original and simplified ONNX artifacts behave the same for the tested shape.
- Stage 2.5 pre-Milestone-3 work superseded the original legacy-only ONNX export default: active model exports now default to dynamo/opset 18/no simplification, Practical-RIFE has a dynamo artifact with `.onnx.data`, and legacy `.onnx` / `.simplified.onnx` loading remains available explicitly.

## Decision Log

- 2026-05-31: Use `src/video_interpolation/inference_runtime/` for the new subsystem. Rationale: approved by the project owner and keeps serving/runtime code separate from data/training modules.
- 2026-05-31: Design the request/result API first and preserve Stage 1 adapter methods as wrappers where practical. Rationale: BentoML and ONNX need a clean structured API, while training/validation/local inference still depend on existing adapter calls.
- 2026-05-31: Use a shared runtime backend abstraction with model-specific runtime classes. Rationale: approved by the project owner and avoids separate incompatible EMA/RIFE backend concepts.
- 2026-05-31: Export only neural network cores to ONNX. Rationale: preprocessing, padding, timestep loops, frame interleaving, output encoding, and orchestration are project-owned Python responsibilities.
- 2026-05-31: Attempt dynamic H/W ONNX export before fixed-shape fallback. Rationale: local PyTorch inference supports multiple resolutions through padding, and the owner wants ONNX to preserve that behavior where feasible.
- 2026-05-31: Store ONNX exports under `model_exports/onnx/`. Rationale: approved by the project owner.
- 2026-05-31: Implement PyTorch Nx support after runtime refactor and before ONNX export. Rationale: approved sequencing; ONNX export boundaries are easier to validate after the PyTorch API supports the target modes.
- 2026-05-31: Make Practical-RIFE v4.26 the default and keep v4.25 as an alternative. Rationale: owner approval plus identical local model code files, with different weights.
- 2026-05-31: Allow documented upstream model repository patches when required for clean inference/runtime/export integration. Rationale: owner approval; upstream repos are not production-ready and project-owned orchestration should replace demo logic.
- 2026-05-31: Mark AMT-S legacy and exclude it from active Stage 2 defaults. Rationale: owner approval; AMT-S is not part of this refactor target.
- 2026-05-31: Add only a minimal BentoML compatibility proof after the API stabilizes. Rationale: owner wants proof that BentoML can call the refactored runtime, but not full production service work.
- 2026-05-31: Use `atol=1e-3`, `rtol=1e-3` for PyTorch-vs-ONNX tensor comparison where possible and always report MAE/max absolute error. Rationale: owner-provided comparison policy.
- 2026-05-31: Treat EMA hardcoded CUDA usage as a proactive runtime-refactor risk. Rationale: unconditional `.cuda()` blocks CPU-side export analysis and future configurable BentoML worker devices. Preferred fix is input-device-aware tensor creation or movement, not another global fixed device.
- 2026-05-31: Audit EMA and RIFE warp/grid helpers during model runtime refactors. Rationale: global device state or unsafe cached grids can break multi-resolution inference, CPU export analysis, ONNX Runtime, or multi-device serving. Any retained cache must be device-, dtype-, and shape-aware.
- 2026-05-31: Use evidence-first Practical-RIFE ONNX patching. Rationale: `grid_sample` or dynamic grid construction may or may not be the actual export blocker; first attempt export through the wrapper boundary, then patch the minimal blocker and record evidence.
- 2026-05-31: Stabilize Practical-RIFE runtime source imports during Stage 2. Rationale: importing Python source from `model_weights/.../train_log` is fragile and production-hostile; weights directories should contain weights/checkpoints, while runtime code should live in a stable project-owned or upstream-repo location.
- 2026-05-31: Gate Stage 2 dependency additions through inspection and recorded decisions. Rationale: `onnx`, `onnxruntime`/`onnxruntime-gpu`, and `bentoml` may be needed, but changes that downgrade or alter PyTorch, CUDA packages, NumPy, model-critical packages, or Stage 1 dependencies require user approval.
- 2026-05-31: Milestone 1 keeps fixed 2x semantics strict: `fixed_2x` requires `interpolation_factor=2` and exactly timestep `0.5`. Rationale: this preserves the Stage 1 middle-frame behavior and avoids ambiguous fixed-mode requests before model runtimes are wired in.
- 2026-05-31: Milestone 1 arbitrary/Nx timesteps are generated as direct fractions `(i + 1) / interpolation_factor`, with factor range `[2, 8]`, strict ordering, open interval `(0, 1)`, and required count `factor - 1`. Rationale: this matches the owner-approved Nx semantics and is model-independent.
- 2026-05-31: The initial PyTorch backend skeleton is callable-based and requires explicit `load()` before `run()`. Rationale: model-specific runtime classes in later milestones can own checkpoint/model construction while sharing one backend lifecycle and no-grad execution boundary.
- 2026-05-31: Keep EMA training methods separate from the serving runtime. Rationale: `Trainer.Model.update()` owns optimizer/loss training behavior, while serving only needs the explicit `inference()` neural call boundary; routing training through the serving runtime would add fragility without helping BentoML or ONNX readiness.
- 2026-05-31: Patch EMA upstream code only where it affects inference/runtime/export risk. Rationale: configurable `Trainer.Model` device, device-aware timestep tensors, device/dtype/shape-aware warp-grid caching, and device-aware `LapLoss` construction directly affect PyTorch serving and later ONNX/export analysis. Broader training internals remain out of scope.
- 2026-05-31: Use `src/video_interpolation/inference_runtime/rife_upstream/` as the stable Practical-RIFE runtime source location. Rationale: this removes Python source imports from `model_weights/.../train_log`, keeps weight directories as checkpoint artifact inputs, and gives the project direct control over inference-only wrappers and later ONNX export boundaries.
- 2026-05-31: Use an inference-only Practical-RIFE `RIFE_HDv3.Model` wrapper instead of the full upstream training wrapper. Rationale: Stage 2 serving needs `flownet`, `eval()`, and `inference()` only; optimizer/loss/DDP setup belongs to upstream training code and creates avoidable runtime/import coupling.
- 2026-05-31: Make Practical-RIFE v4.26 the default and preserve v4.25 as an alternative config. Rationale: owner approval plus local source equivalence between v4.25 and v4.26; only checkpoint selection changes by default.
- 2026-05-31: Exclude AMT-S from active Stage 2 batch aliases while preserving explicit AMT commands/configs. Rationale: owner asked to mark AMT-S legacy/excluded from active defaults without deleting support.
- 2026-05-31: Treat arbitrary/Nx as inference-only. Rationale: Stage 1 EMA training/fine-tuning is fixed 2x and should not inherit serving/runtime timestep semantics.
- 2026-05-31: Make `interpolation_factor` a runtime/request value rather than a hidden model config choice. Rationale: future frontend and BentoML requests need per-call factor selection, while configs should only define capabilities, defaults, bounds, and checkpoint mappings.
- 2026-05-31: Do not plan factor-specific configs such as separate 4x and 8x files. Rationale: factor-specific configs would work against request-driven local CLI and future service behavior.
- 2026-05-31: Prefer EMA `ours_small_t.pkl` as the default inference checkpoint for both arbitrary/Nx and fixed 2x at `t = 0.5`, subject to smoke validation. Rationale: future BentoML serving should avoid dynamic checkpoint switching per request. `ours_small.pkl` remains available for training/backward compatibility.
- 2026-05-31: Use the same selected Practical-RIFE runtime/checkpoint for fixed 2x and arbitrary/Nx. Rationale: RIFE v4.25/v4.26 inference accepts direct timesteps; fixed 2x is the request factor `2`, timestep `0.5` case.
- 2026-05-31: Implement PyTorch Nx in runtime loops over validated request timesteps rather than separate factor-specific configs or video-flow changes. Rationale: the request/result API already owns factor/timestep validation, and video frame scheduling belongs to Milestone 5.
- 2026-05-31: Add adapter-level `predict_intermediate_frames(...)` while preserving fixed 2x wrappers unchanged. Rationale: this provides a clear public tensor-pair Nx method without breaking Stage 1 `predict_pair`, `predict_batch`, `predict`, or `__call__` calls.
- 2026-05-31: Update Stage 2 EMA inference configs to load `ours_small_t.pkl`, but leave EMA training config on `ours_small.pkl`. Rationale: inference should avoid checkpoint switching for Nx/fixed 2x service use, while training/fine-tuning remains fixed 2x and compatibility-sensitive.
- 2026-05-31: Add `ema infer-pair` and `rife infer-pair` CLI commands instead of changing video inference now. Rationale: Milestone 4 needs a bounded model/frame-pair smoke path with `--interpolation-factor`; full video-level Nx interleaving and FPS changes are Milestone 5 work.
- 2026-05-31: In Milestone 5, compute local video output FPS from the validated `interpolation_factor` rather than the legacy `output_fps_multiplier` field. Rationale: Stage 2 Nx semantics require output FPS to scale by the actual request/runtime factor; `output_fps_multiplier` remains for backward-compatible config parsing.
- 2026-05-31: Use `predict_frame_pair(FramePairRequest(...))` when an adapter exposes the Stage 2 API and fall back to `predict_pair()` only for fixed-2x legacy adapters. Rationale: EMA/RIFE video inference should exercise the new runtime path, while AMT-S and baselines should keep existing fixed-2x behavior.
- 2026-05-31: Let default `infer-all-videos --mode arbitrary_nx` run active model targets only. Rationale: baselines and AMT-S are fixed-2x/legacy paths and should not cause the default Nx batch workflow to fail; explicit target selection still validates and fails clearly for unsupported combinations.
- 2026-05-31: Use `FramePairRequest.backend_options["scale"]` for request-time Practical-RIFE scale instead of adding a model-specific top-level request field. Rationale: `scale` is specific to Practical-RIFE, while `backend_options` already exists for serving/runtime options; local CLI still exposes the value explicitly with `--scale`/`--rife-scale`, and future BentoML services can pass the same request option with `interpolation_factor`.
- 2026-05-31: Use separate project-owned `EMAVFIOnnxWrapper` and `PracticalRIFEOnnxWrapper` classes for ONNX export rather than exporting full adapters or local video inference. Rationale: wrappers make the export boundary explicit as prepared/padded tensors plus timestep to generated tensor, while padding, Nx loops, video I/O, and serving orchestration remain Python-owned.
- 2026-05-31: Default Milestone 6 export commands to dynamic H/W metadata and provide `--shape-mode static` as an explicit option. Rationale: the owner asked to attempt dynamic H/W first; static export is available only as a configured fallback/tooling path.
- 2026-05-31: Run ONNX simplification by default but treat simplifier failure as non-fatal when the original ONNX file passes checker validation. Rationale: later Milestone 7 should prefer simplified artifacts when available, but simplifier/operator support should not hide or invalidate an otherwise usable original export. Superseded for active Stage 2.5 exports by the 2026-06-01 dynamo/opset18/no-simplification default while preserving legacy simplification as an explicit option.
- 2026-05-31: Keep ONNX Runtime dependency unused in Milestone 6 implementation despite being listed in `pyproject.toml`. Rationale: this milestone is export-boundary work only; runtime loading, provider selection, and PyTorch-vs-ONNX equivalence are Milestone 7.
- 2026-05-31: Prewarm ONNX wrapper modules once with the sample inputs before tracing. Rationale: EMA lazily creates shape-specific inference buffers during the first forward pass, and prewarming keeps the trace state stable without patching upstream math or moving padding/timestep orchestration into ONNX.
- 2026-05-31: Default ONNX Runtime validation CLI commands to `CPUExecutionProvider` while allowing explicit repeated `--provider` options. Rationale: CPU smoke checks are safe in the current environment; CUDA can be requested explicitly when available, and unavailable providers should fail clearly before session creation.
- 2026-05-31: Prefer simplified ONNX artifacts for runtime validation, with `--prefer-original` available. Rationale: Milestone 6 simplification succeeded for both active models, but the original artifact remains a useful fallback and debugging comparison if simplified runtime behavior diverges.
- 2026-05-31: Keep Practical-RIFE `scale` baked into the selected ONNX artifact and reject request-time ONNX scale mismatches. Rationale: the Milestone 6 RIFE ONNX wrapper builds `scale_list` as graph constants; future serving should select or export an artifact that matches the requested scale rather than silently using the wrong scale.
- 2026-05-31: Write sample PyTorch/ONNX/normalized-absdiff PNGs only when tensor outputs are available but `allclose` fails. Rationale: this keeps passing validation output compact while giving concrete inspection artifacts for backend/operator differences such as the Practical-RIFE dynamic-shape mismatch.
- 2026-05-31: Do not relax the `1e-3` tensor tolerance for Practical-RIFE dynamic `128x256` despite low MAE. Rationale: the owner requested starting with `atol=1e-3`, `rtol=1e-3`; the observed mismatch is recorded for follow-up rather than hidden by a looser threshold.
- 2026-06-01: Use Practical-RIFE v4.26 PyTorch CUDA as the default Milestone 8 serving path and keep ONNX as a separate example. Rationale: Stage 2.5 handoff recommends PyTorch first for serving because it avoids ONNX external-data/provider packaging variables, while ONNX remains useful as an explicit alternate backend.
- 2026-06-01: Limit the BentoML example service boundary to sequential arbitrary-Nx factors `2..4`. Rationale: runtime code supports broader factors and batched execution, but current serving examples should be conservative and avoid the Stage 2.5 benchmarked batched path for service requests.
- 2026-06-01: Keep BentoML examples outside `src/video_interpolation` and route both examples through `src/video_interpolation/serving.py`. Rationale: examples should stay developer-facing and thin, while the reusable project-owned serving facade owns validation, adapter/runtime initialization, ONNX artifact/provider handling, and video inference calls.

## Outcomes & Handoff

Stage 2 is complete and ready for user acceptance. It produced a reusable inference runtime subsystem, model-specific PyTorch and ONNX runtime boundaries, request/result APIs, Practical-RIFE v4.26 as the active default, arbitrary-Nx local video inference, ONNX export/runtime tooling, video-pipeline benchmark support carried in from Stage 2.5, and a minimal Practical-RIFE serving-readiness layer with BentoML developer examples.

Final serving recommendation:

- Production target model: Practical-RIFE v4.26 (`practical_rife_v4_26`).
- Default serving backend: PyTorch runtime backend.
- Default device: `cuda`.
- Alternative backend: ONNX Runtime with `CUDAExecutionProvider`.
- Serving execution mode: `sequential`.
- Interpolation mode: `arbitrary_nx`.
- Service-level `interpolation_factor` range: `2..4`.
- `scale` is a runtime/request parameter.
- Current batched inference is validated for local runtime and benchmark paths, but is not recommended for serving yet.
- EMA-VFI is not the current serving target.
- AMT-S is legacy and out of scope for Stage 2 serving.

Stage 2.5 decisions affecting this handoff:

- Stage 2.5 is accepted and archived at `.agent/docs/exec-plans/completed/02_5_inference_runtime_stabilization.execplan.md`.
- Dynamo/opset 18 ONNX export is the default export mode; simplification is legacy-only.
- Practical-RIFE's accepted ONNX artifact is `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx` with adjacent `.onnx.data`; fixed-batch RIFE dynamo artifacts are obsolete and unsupported.
- EMA ONNX is constrained-dynamic and requires project-owned external divisor `112` padding/unpadding; EMA remains useful for experimentation but is not the current serving target.
- CUDA-provider ONNX behavior remains deferred until validation in a CUDA-capable environment.

Serving handoff artifacts:

- `src/video_interpolation/serving.py` contains `PracticalRIFEServingConfig`, `PracticalRIFEVideoInferenceRunner`, `PracticalRIFEOnnxVideoAdapter`, and `run_practical_rife_video_inference(...)`.
- `examples/bentoml/practical_rife_torch_service/service.py` is the recommended PyTorch CUDA BentoML compatibility example.
- `examples/bentoml/practical_rife_onnx_service/service.py` is the alternate ONNX Runtime CUDA compatibility example.
- `docs/stage2_inference_runtime_refactor.md` is the concise human-facing Stage 2 handoff for runtime structure, Python serving calls, ONNX caveats, BentoML example locations, and backend/service next steps.

Validation status at handoff:

- Full automated tests and lint passed after Milestone 8: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` reported `156 passed, 59 warnings`; `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` reported `All checks passed`.
- Stage 2 Milestone 9 documentation-only rerun: unprefixed `uv run pytest` and `uv run ruff check src tests` failed at uv startup because `/home/lighter_01/.cache/uv` is read-only in the sandbox. Fallback validation passed: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` reported `156 passed, 59 warnings`, and `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` reported `All checks passed`.
- CUDA smokes were deferred because `torch.cuda.is_available()` returned `False` in this environment. ONNX Runtime lists CUDA/TensorRT providers, but CUDA execution is not classified from CPU-only evidence.

Known limitations:

- No production BentoML service, FastAPI app, queue, database, object storage flow, frontend, monitoring, auth, or Docker deployment is included in Stage 2.
- The BentoML examples accept path strings and return metadata; they are compatibility examples, not upload APIs.
- Practical-RIFE ONNX bakes `scale` into the artifact. ONNX serving requests must use a scale matching the loaded artifact or fail clearly.
- Current batched inference should stay out of serving until a service-specific batching policy, queue behavior, memory budget, and GPU validation are designed.
- EMA-VFI serving and AMT-S serving are out of the current production recommendation.

Recommended next steps for backend/service developers:

- Wrap `PracticalRIFEVideoInferenceRunner` in the production API/service layer and initialize the runner once per worker process.
- Package model weights and, for ONNX deployments, keep `.onnx` and `.onnx.data` adjacent.
- Implement request/upload handling, output storage, job state, queue/worker orchestration, deployment configuration, auth, and observability outside the inference runtime package.
- Run short CUDA smokes for Practical-RIFE PyTorch and Practical-RIFE ONNX in the target GPU environment before making production performance or provider claims.
- Define a production artifact/versioning policy for PyTorch checkpoints and ONNX artifacts.

Detailed milestone record:

Milestone 1 implemented:

- `src/video_interpolation/inference_runtime/api.py` with `InferenceMode`, `RuntimeBackendKind`, `FramePairRequest`, `FramePairResult`, `RuntimeInputs`, `RuntimeOutputs`, interpolation factor validation, and timestep generation.
- `src/video_interpolation/inference_runtime/backends/base.py` with the shared `RuntimeBackend` abstraction and lifecycle guard.
- `src/video_interpolation/inference_runtime/backends/torch.py` with a minimal callable-based PyTorch backend skeleton using explicit `load()`, `torch.no_grad()` execution, normalized runtime outputs, and `close()`.
- Package exports in `src/video_interpolation/inference_runtime/__init__.py` and backend exports in `src/video_interpolation/inference_runtime/backends/__init__.py`.
- Focused tests in `tests/test_inference_runtime_api.py`.
- `.agent/docs/PROJECT_MAP.md` entry for the new Stage 2 runtime package and tests.

Milestone 2 implemented:

- `src/video_interpolation/inference_runtime/ema.py` with prediction-only `EMAVFIPyTorchRuntime`, `EMAVFIPyTorchRuntimeConfig`, and shared `prepare_ema_image_tensor()`.
- `src/video_interpolation/adapters/ema_vfi.py` now delegates fixed 2x `predict_pair()` through a `FramePairRequest` and `FramePairResult` via `predict_frame_pair()`.
- Existing adapter compatibility methods remain available: `predict_batch()` stays the inherited sequential wrapper, and `predict()` / `__call__` stay inherited wrappers over `predict_pair()`.
- EMA `train_step()` and `eval_step()` remain isolated on the upstream training model path and do not use the serving runtime.
- `src/video_interpolation/ema_preflight.py` has an updated blocked-message for the default CUDA preflight path.
- `model_repos/EMA-VFI/Trainer.py` accepts a configurable device with `cuda` default preserved.
- `model_repos/EMA-VFI/model/flow_estimation.py` uses input-feature device/dtype-aware timestep tensors instead of unconditional `.cuda()`.
- `model_repos/EMA-VFI/model/warplayer.py` uses device/dtype/shape-aware warp-grid creation and cache keys.
- `model_repos/EMA-VFI/model/loss.py` lets `LapLoss` be created on the configured model device and creates temporary tensors on the input tensor device/dtype.
- `tests/test_ema_adapter.py` covers EMA runtime request/result behavior, adapter wrapper compatibility, training-path isolation, arbitrary/Nx frame-pair behavior, checkpoint-policy separation, and checkpoint normalization.

Milestone 3 implemented:

- `src/video_interpolation/inference_runtime/rife.py` with prediction-only `PracticalRIFEPyTorchRuntime`, `PracticalRIFEPyTorchRuntimeConfig`, tensor preparation, padding, and unpadding helpers.
- `src/video_interpolation/inference_runtime/rife_upstream/` as the stable project-owned Practical-RIFE runtime source package.
- `src/video_interpolation/adapters/rife.py` now delegates fixed 2x `predict_pair()` through a `FramePairRequest` and `FramePairResult` via `predict_frame_pair()`.
- Existing Practical-RIFE adapter compatibility methods remain available: `predict_batch()` stays the inherited sequential wrapper, and `predict()` / `__call__` stay inherited wrappers over `predict_pair()`.
- Practical-RIFE v4.26 configs added and made the CLI/default target; v4.25 configs remain available.
- Active batch aliases now use `practical_rife_v4_26` and `ema_vfi_small` for neural/model defaults; AMT-S remains explicit legacy support.
- `docs/stage2_inference_runtime_refactor.md` documents the runtime API, RIFE source policy, current backend boundaries, and deferred ONNX/BentoML work.
- `tests/test_rife_adapter.py` covers RIFE runtime request/result behavior, adapter wrapper compatibility, arbitrary/Nx frame-pair behavior, v4.26 default config, v4.25 alternative config parsing, and checkpoint normalization.

Milestone 4 implemented:

- `src/video_interpolation/inference_runtime/ema.py` now supports `fixed_2x` and `arbitrary_nx` PyTorch request/result inference by looping over request timesteps and returning one intermediate tensor per timestep.
- `src/video_interpolation/inference_runtime/rife.py` now supports `fixed_2x` and `arbitrary_nx` PyTorch request/result inference with the same per-request timestep loop and existing RIFE padding/unpadding behavior.
- `src/video_interpolation/adapters/ema_vfi.py` and `src/video_interpolation/adapters/rife.py` preserve `predict_pair`, `predict_batch`, `predict`, and `__call__` as fixed 2x compatibility wrappers and add `predict_intermediate_frames(...)` for tensor-pair Nx inference.
- `configs/models/ema_vfi_small.yaml` and `configs/inference/ema_vfi_small_2x.yaml` now use `EMA-VFI/ours_small_t.pkl` for Stage 2 inference and record `EMA-VFI/ours_small.pkl` as the training checkpoint. `configs/training/ema_vfi_small_finetune.yaml` remains fixed 2x and still uses `ours_small.pkl`.
- Practical-RIFE v4.26 and v4.25 model/inference configs now declare `fixed_2x`/`arbitrary_nx` support plus default/min/max interpolation-factor fields; no factor-specific configs were added.
- `src/video_interpolation/cli.py` adds `ema infer-pair` and `rife infer-pair` with runtime `--mode` and `--interpolation-factor` arguments for bounded tensor-pair smoke checks. Local video-level Nx remains deferred.
- `docs/stage2_inference_runtime_refactor.md`, config README files, and `.agent/docs/PROJECT_MAP.md` were updated for Nx semantics, request-time factor selection, EMA checkpoint policy, Practical-RIFE v4.26 Nx behavior, and Milestone 5 deferrals.
- `tests/test_ema_adapter.py` and `tests/test_rife_adapter.py` now cover arbitrary/Nx runtime output counts, timestep calls, adapter `predict_intermediate_frames(...)`, backward-compatible fixed 2x wrappers, and EMA inference/training checkpoint-policy separation.

Milestone 5 implemented:

- `src/video_interpolation/inference.py` now accepts and validates `interpolation_mode` and request-time `interpolation_factor`, resolves mode timesteps through the Stage 2 runtime API, and calls adapter `predict_frame_pair(FramePairRequest(...))` when available.
- Local video output now writes `original_i`, all generated intermediate frames in timestep order, then `original_i+1`; fixed 2x writes one generated frame and arbitrary Nx writes `N - 1` frames for factors `2..8`.
- Output FPS is now computed as `input_fps * interpolation_factor`. The legacy `output_fps_multiplier` field remains accepted for config compatibility but is not the Stage 2 authority for model video output FPS.
- `VideoInferenceResult`, MLflow params/metrics, progress payloads, and batch measurement rows now include interpolation mode, factor, timesteps where useful, and runtime backend.
- `VideoInferenceConfig.runtime_options` carries request runtime options such as Practical-RIFE `scale` into `FramePairRequest.backend_options`, and batch measurement rows include the recorded runtime options.
- `src/video_interpolation/cli.py` adds `--mode` and `--interpolation-factor` to `ema infer-video`, `rife infer-video`, and `infer-all-videos`; invalid CLI factors outside `2..8` fail before model inference starts.
- Directory-wide inference output paths and run names now use factor suffixes such as `_2x` or `_4x`. `infer-all-videos --mode arbitrary_nx` defaults to active model targets only when no target is explicitly selected.
- `configs/inference/ema_vfi_small_2x.yaml`, `configs/inference/practical_rife_v4_26_2x.yaml`, and `configs/inference/practical_rife_v4_25_2x.yaml` now declare fixed-2x mode/factor defaults while supporting CLI runtime factor overrides for EMA/RIFE.
- `tests/test_inference.py` covers video-level request/result API usage, factors 2/4/8 frame counts, Nx frame ordering, output FPS multiplication, invalid factor rejection before inference, CLI factor validation, existing fixed-2x config compatibility, and measurement CSV mode/factor/backend metadata.
- `tests/test_inference.py` covers video-level request/result API usage, factors 2/4/8 frame counts, Nx frame ordering, output FPS multiplication, invalid factor rejection before inference, CLI factor validation, request runtime option propagation, existing fixed-2x config compatibility, and measurement CSV mode/factor/backend/runtime-option metadata.
- `tests/test_rife_adapter.py` covers request-time Practical-RIFE scale override, invalid scale rejection before model calls, and adapter `predict_intermediate_frames(..., scale=...)`.
- `docs/stage2_inference_runtime_refactor.md`, config README files, and `.agent/docs/PROJECT_MAP.md` were updated for local fixed 2x/Nx video inference behavior, output FPS policy, request-time Practical-RIFE scale policy, CLI examples, batch metadata, and remaining ONNX/BentoML deferrals.
- No upstream model repository files were patched in Milestone 5.

Milestone 6 implemented:

- `src/video_interpolation/inference_runtime/onnx_export.py` adds ONNX export dataclasses, `OnnxShapeMode`, absolute artifact path generation under `model_exports/onnx/<model_name>/`, sample input construction, dynamic/static axes selection, pre-trace wrapper cache initialization, ONNX checker validation, optional simplification, and structured success/failure results.
- `EMAVFIOnnxWrapper` wraps the EMA neural core `model.net` and exports the boundary `left/right/timestep -> pred`, with padding/unpadding, timestep loops, video I/O, MLflow, validation, and training logic outside ONNX.
- `PracticalRIFEOnnxWrapper` wraps the project-owned Practical-RIFE `flownet` source and exports the boundary `left/right/timestep -> merged[-1]`, with an explicit RIFE scale value used to build the constant `scale_list`.
- `src/video_interpolation/cli.py` adds developer commands `ema export-onnx` and `rife export-onnx`. Both support `--checkpoint`, `--output-dir`, `--device`, `--opset-version`, `--shape-mode dynamic_hw|static`, `--batch-size`, `--height`, `--width`, `--timestep`, and `--simplify/--no-simplify`; RIFE also supports `--scale`.
- `tests/test_onnx_export.py` covers export option validation, artifact paths, dynamic axes, sample inputs, wrapper output boundaries, RIFE scale-list behavior, fake-module artifact export, and structured exporter failure reporting.
- `docs/stage2_inference_runtime_refactor.md`, `configs/README.md`, `configs/inference/README.md`, `configs/models/README.md`, and `.agent/docs/PROJECT_MAP.md` document the ONNX export purpose, neural-core-only boundary, artifact layout, commands, dynamic/static policy, simplification behavior, and Milestone 7 deferrals.
- No upstream model repository files were patched for Milestone 6 so far.
- Real CPU dynamic-H/W export smokes wrote original and simplified ONNX artifacts for `ema_vfi_small` and `practical_rife_v4_26` under `model_exports/onnx/`. EMA export succeeded after prewarming but emitted tracer warnings from feature-extractor shape math/cached masks; dynamic axes are present, while true multi-resolution ONNX Runtime behavior remains unproven until Milestone 7.

Milestone 7 implemented:

- `src/video_interpolation/inference_runtime/backends/onnx.py` adds `OnnxRuntimeBackendConfig`, `OnnxRuntimeBackend`, provider alias/availability validation, model input/output name checks, tensor-to-ONNX-array conversion, `InferenceSession.run(...)`, output tensor conversion, and session-provider metadata.
- `src/video_interpolation/inference_runtime/ema.py` adds `EMAVFIOnnxRuntimeConfig` and `EMAVFIOnnxRuntime`, which reuse the existing EMA padding class, build timestep tensors outside ONNX, call the ONNX backend, unpad predictions, and return `FramePairResult` with `backend_kind=onnx`.
- `src/video_interpolation/inference_runtime/rife.py` adds `PracticalRIFEOnnxRuntimeConfig` and `PracticalRIFEOnnxRuntime`, which reuse RIFE padding/unpadding, enforce that request scale matches the scale baked into the ONNX artifact, call the ONNX backend, and return `FramePairResult` with ONNX metadata.
- `src/video_interpolation/inference_runtime/onnx_validation.py` adds default artifact resolution, PyTorch-vs-ONNX equivalence metrics, JSON/CSV report writing, and sample output image generation for failed tensor closeness.
- `src/video_interpolation/cli.py` adds `ema validate-onnx` and `rife validate-onnx` developer commands with explicit providers, optional ONNX artifact paths, default `model_exports/onnx/` artifact resolution, repeated `--shape` dynamic checks, mode/factor selection, RIFE scale selection, tolerances, and report output roots.
- `tests/test_onnx_runtime.py` covers ONNX artifact resolution, provider validation, tiny ONNX Runtime session execution, missing-input errors, EMA ONNX runtime request/result behavior, RIFE ONNX scale mismatch rejection, equivalence metric computation, and report writing.
- `docs/stage2_inference_runtime_refactor.md`, `configs/README.md`, `configs/inference/README.md`, `configs/models/README.md`, and `.agent/docs/PROJECT_MAP.md` document ONNX Runtime backend purpose, provider selection, artifact resolution, validation commands, tolerance policy, validation outputs, current model status, and dynamic-H/W findings.
- No upstream model repository files were patched for Milestone 7.
- No dependencies were added in Milestone 7; `pyproject.toml` already listed `onnxruntime` and `onnxruntime-gpu`.

Latest validation:

- `uv run pytest` failed at uv startup because `/home/lighter_01/.cache/uv` is read-only in the sandbox.
- `uv run ruff check src tests` failed at uv startup for the same uv cache reason.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` passed: 87 tests.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` passed.
- Focused ONNX Runtime validation tests `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_onnx_runtime.py` passed with 8 tests.
- Full test suite includes the existing ONNX export tests in `tests/test_onnx_export.py` and the new ONNX Runtime tests in `tests/test_onnx_runtime.py`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/video_interpolation/inference_runtime src/video_interpolation/cli.py` passed during implementation.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import torch; print(torch.cuda.is_available())"` returned `False`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY' ... import onnxruntime as ort ... PY` showed `onnxruntime 1.26.0` and available providers `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`; CUDA was not used because PyTorch CUDA is unavailable.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli ema export-onnx --help` and `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli rife export-onnx --help` succeeded and show ONNX export options including `--shape-mode`, `--opset-version`, and `--simplify/--no-simplify`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli ema validate-onnx --help` and `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli rife validate-onnx --help` succeeded and show ONNX Runtime validation options including `--provider`, repeated `--shape`, `--onnx-path`, `--prefer-simplified/--prefer-original`, `--atol`, and `--rtol`.
- `UV_CACHE_DIR=/tmp/uv-cache timeout 180s uv run python -m video_interpolation.cli ema export-onnx --config configs/models/ema_vfi_small.yaml --device cpu --height 32 --width 32 --output-dir model_exports/onnx` succeeded. ONNX checker succeeded and simplification succeeded. The preferred artifact is `model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamic_hw_opset17.simplified.onnx`; the original artifact is also retained.
- `UV_CACHE_DIR=/tmp/uv-cache timeout 180s uv run python -m video_interpolation.cli rife export-onnx --config configs/models/practical_rife_v4_26.yaml --device cpu --height 128 --width 128 --scale 1.0 --output-dir model_exports/onnx` succeeded. ONNX checker succeeded and simplification succeeded. The preferred artifact is `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamic_hw_opset17.simplified.onnx`; the original artifact is also retained.
- Exported artifact sizes at closeout: EMA original `58,858,828` bytes, EMA simplified `58,534,582` bytes, RIFE original `22,875,748` bytes, RIFE simplified `22,848,842` bytes.
- EMA ONNX Runtime equivalence smoke `UV_CACHE_DIR=/tmp/uv-cache timeout 180s uv run python -m video_interpolation.cli ema validate-onnx --config configs/models/ema_vfi_small.yaml --provider CPUExecutionProvider --shape 32x32 --output-dir outputs/onnx_validation` passed using the simplified artifact. Metrics: MAE `3.1393333e-07`, max absolute error `1.7881393e-06`, `allclose=True`.
- EMA dynamic-H/W smoke `UV_CACHE_DIR=/tmp/uv-cache timeout 180s uv run python -m video_interpolation.cli ema validate-onnx --config configs/models/ema_vfi_small.yaml --provider CPUExecutionProvider --shape 32x32 --shape 64x64 --output-dir outputs/onnx_validation/ema_dynamic_check` exited non-zero after writing reports. `32x32` matched, while `64x64` failed in ONNX Runtime with LayerNormalization shape mismatch (`X.shape={2,64,98}`, `scale.shape={128}`, `bias.shape={128}`). Static/fixed padded fallback remains a likely EMA follow-up unless the export path is fixed.
- Practical-RIFE ONNX Runtime equivalence smoke `UV_CACHE_DIR=/tmp/uv-cache timeout 240s uv run python -m video_interpolation.cli rife validate-onnx --config configs/models/practical_rife_v4_26.yaml --provider CPUExecutionProvider --shape 128x128 --output-dir outputs/onnx_validation` passed using the simplified artifact. Metrics: MAE `8.4759959e-06`, max absolute error `0.00091010332`, `allclose=True`.
- Practical-RIFE dynamic-H/W smoke `UV_CACHE_DIR=/tmp/uv-cache timeout 240s uv run python -m video_interpolation.cli rife validate-onnx --config configs/models/practical_rife_v4_26.yaml --provider CPUExecutionProvider --shape 128x128 --shape 128x256 --output-dir outputs/onnx_validation/rife_dynamic_check` exited non-zero after writing reports and sample images. `128x128` matched; `128x256` ran through ONNX Runtime but failed strict allclose with MAE `5.5331097e-05` and max absolute error `0.0037825704`.
- Practical-RIFE original-artifact comparison `UV_CACHE_DIR=/tmp/uv-cache timeout 240s uv run python -m video_interpolation.cli rife validate-onnx --config configs/models/practical_rife_v4_26.yaml --provider CPUExecutionProvider --shape 128x128 --shape 128x256 --prefer-original --output-dir outputs/onnx_validation/rife_dynamic_original_check` produced the same dynamic `128x256` mismatch, so the observed difference is not simplification-specific.
- ONNX Runtime reports were written under `outputs/onnx_validation/`, including `outputs/onnx_validation/ema_vfi_small/equivalence_report.json`, `outputs/onnx_validation/ema_dynamic_check/ema_vfi_small/equivalence_report.json`, `outputs/onnx_validation/practical_rife_v4_26/equivalence_report.json`, and `outputs/onnx_validation/rife_dynamic_check/practical_rife_v4_26/equivalence_report.json`. The RIFE dynamic mismatch also wrote sample PNGs under `outputs/onnx_validation/rife_dynamic_check/sample_outputs/practical_rife_v4_26/`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli infer-all-videos --help`, `... ema infer-video --help`, and `... rife infer-video --help` succeeded and show `--mode` plus `--interpolation-factor`; Practical-RIFE help also shows `--scale`, and batch help shows `--rife-scale`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli ema infer-pair --help` and `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli rife infer-pair --help` succeeded.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli ema-preflight` exited 0 and reported `blocked` because CUDA is unavailable for the default EMA config.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli ema adapter-check` exited 0 and reported `blocked` because CUDA is unavailable for the default EMA config.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli rife-preflight` exited 0 and reported `blocked` because CUDA is unavailable for the default Practical-RIFE v4.26 config.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m video_interpolation.cli rife adapter-check` exited 0 and reported `blocked` because CUDA is unavailable for the default Practical-RIFE v4.26 config.
- CUDA was unavailable, so CUDA EMA and Practical-RIFE export/runtime smoke checks were deferred. CPU export and ONNX Runtime validation smokes were run instead.

Stage 2.5 handoff update:

- Stage 2 was paused before Milestone 8 so Stage 2.5 could stabilize ONNX artifacts, real-image equivalence checks, true model batching, video chunking, and benchmark evidence.
- Stage 2.5 Milestones 1 through 9 are complete and accepted. The full handoff is in `.agent/docs/exec-plans/completed/02_5_inference_runtime_stabilization.execplan.md`; the concise human-facing summary is in `docs/stage2_5_inference_runtime_stabilization.md`.
- Practical-RIFE v4.26 PyTorch is the recommended first model/backend for the Milestone 8 BentoML compatibility proof. It exercises the active default model and request/result API without ONNX provider or external-data packaging variables.
- Practical-RIFE v4.26 ONNX dynamic-batch CPU is the recommended secondary proof path if ORT is included. Use `model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx` and keep the adjacent `.onnx.data` file next to it.
- EMA PyTorch remains the general EMA serving path. EMA ONNX is acceptable only for routes that enforce project-owned external divisor `112` padding/unpadding with `model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx` plus adjacent `.onnx.data`.
- Fixed-batch Practical-RIFE dynamo artifacts are obsolete and unsupported. Legacy `.onnx` / `.simplified.onnx` artifacts remain loadable only through explicit path or legacy resolution flags.
- Batch semantics after Stage 2.5 are pair-major. Fixed 2x uses one flattened row per pair at `t=0.5`; Nx uses `pairs * (factor - 1)` flattened rows and reconstructs `outputs[pair_index][timestep_index]`.
- CUDA-provider ONNX validation and MLflow server logging remain deferred; do not classify CUDA or production tracking behavior from CPU-only reports.

Still not changed before Milestone 8 implementation:

- No BentoML proof yet.
- No AMT-S refactor or active Nx support.
- No model weights or dependencies changed.
- No upstream model repository files patched in Milestone 7.
- No production serving API, upload flow, queue, storage orchestration, database work, frontend, monitoring, or deployment pipeline.

Milestone 8 implementation update:

- Added `src/video_interpolation/serving.py` with `PracticalRIFEServingConfig`, `PracticalRIFEVideoInferenceRunner`, `PracticalRIFEOnnxVideoAdapter`, and `run_practical_rife_video_inference(...)`.
- Serving defaults use Practical-RIFE v4.26, PyTorch backend, `device="cuda"`, sequential `arbitrary_nx`, factor range `2..4`, `scale=1.0`, and `codec="libx264"`.
- ONNX serving remains an alternate path with `provider="CUDAExecutionProvider"` and scale validation against the artifact's baked scale.
- Added developer-facing BentoML examples under `examples/bentoml/practical_rife_torch_service/` and `examples/bentoml/practical_rife_onnx_service/`.
- Updated `.gitignore` so `examples/` can be tracked; local `__pycache__` directories were removed from the working tree and none were tracked.
- Focused serving tests were added in `tests/test_serving.py`.
- Validation passed: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` reported `156 passed, 59 warnings`; `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` reported `All checks passed`; `git diff --check` passed.
- Focused validation also passed for `tests/test_serving.py` and focused ruff on the serving/examples files.
- `torch.cuda.is_available()` returned `False`, so Practical-RIFE PyTorch CUDA and ONNX CUDA short-video smokes were deferred. ONNX Runtime lists `TensorrtExecutionProvider`, `CUDAExecutionProvider`, and `CPUExecutionProvider`, but CUDA model execution was not classified without PyTorch CUDA availability.

Milestone 9 closeout update:

- Final Stage 2 handoff content was added to this ExecPlan and `docs/stage2_inference_runtime_refactor.md`.
- `.agent/docs/PROJECT_MAP.md` was updated to point to the completed Stage 2 ExecPlan and current serving-readiness files.
- Validation was rerun after the documentation-only closeout edits: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` passed with `156 passed, 59 warnings`; `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests` passed.
- Stage 2 is ready for user acceptance and backend/service-developer handoff. No new runtime, backend, serving, export, or batch-inference features were added in Milestone 9.
