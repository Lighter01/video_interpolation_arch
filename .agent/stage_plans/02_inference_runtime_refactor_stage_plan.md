# Stage 2 Plan — Inference Runtime Refactor and Serving Readiness

## 1. Stage Goal

Stage 2 prepares the existing Stage 1 codebase for production-style inference serving.

This stage is a narrowed and practical version of the broader "Inference Service Backend" stage from `general_plan.md`. It does **not** implement the full backend, queue, database, frontend, or deployment stack. Instead, it focuses on restructuring and hardening the inference part of the project so that it can later be wrapped into BentoML services and, where feasible, executed through ONNX Runtime.

The target result is a modular inference subsystem where:

```text
video/frame input
→ model-independent preprocessing/tensor formatting
→ model adapter public API
→ runtime backend: PyTorch first, ONNX next
→ model-independent postprocessing
→ video/frame output
```

The stage must make inference logic explicit, inspectable, and separable from training-data workflows.

---

## 2. Relationship to the Global Plan

The global project plan still defines the long-term direction: model lifecycle through MLflow, serving through BentoML, application orchestration through FastAPI, asynchronous jobs through Celery/Redis, and local Docker Compose infrastructure.

This Stage 2 is a preparatory serving-readiness stage before the full backend implementation. It should make the later full service stage easier by creating a clean inference/runtime component first.

The broader Stage 2 from `general_plan.md` may be implemented later as a separate stage after this refactor is complete.

---

## 3. Current Context

Stage 1 produced a local ML core with data preprocessing, dataset versioning, triplet datasets, metrics, baselines, model adapters, local 2x inference, EMA fine-tuning, and candidate validation.

The current project structure includes:

- `src/video_interpolation/data/` — data preprocessing, indexing, versioning, dataset loading;
- `src/video_interpolation/training.py` — EMA-VFI training/fine-tuning logic;
- `src/video_interpolation/validation.py` — candidate validation;
- `src/video_interpolation/adapters/` — model adapters for EMA-VFI, AMT-S, Practical-RIFE, and baselines;
- `src/video_interpolation/inference.py` — shared local 2x video inference;
- `src/video_interpolation/image_io.py` — image/tensor conversion helpers;
- `configs/inference/` and `configs/models/` — local inference/model configs;
- `model_repos/EMA-VFI/` and `model_repos/Practical-RIFE/` — upstream model repositories;
- `model_weights/EMA-VFI/` and `model_weights/Practical-RIFE/` — local model weights.

Stage 1 also integrated AMT-S, but AMT-S has practical limitations in this project: higher VRAM requirements and more complicated training/runtime assumptions. Stage 2 focuses only on **EMA-VFI** and **Practical-RIFE**.

---

## 4. Models in Scope

### Required

1. **EMA-VFI**
   - existing Stage 1 adapter: `EMAVFIAdapter`;
   - fixed 2x weights: `model_weights/EMA-VFI/ours_small.pkl`;
   - arbitrary/Nx weights: `model_weights/EMA-VFI/ours_small_t.pkl` if Nx mode is implemented in this stage.

2. **Practical-RIFE**
   - existing Stage 1 adapter: `PracticalRIFEAdapter`;
   - current default weights: `model_weights/Practical-RIFE/RIFEv4.25/train_log/`;
   - `RIFEv4.26` may be inspected but must not silently replace the current default without a documented decision.

### Out of Scope

- AMT-S refactor;
- AMT-S ONNX export;
- AMT-S BentoML serving;
- AMT-S Nx interpolation.

Existing AMT-S code must not be deleted or broken, but no new AMT-S work is required in this stage.

---

## 5. Core Problem

The current model inference works through adapters, but the internal inference logic is still tightly coupled to upstream repositories:

- EMA-VFI keeps much of inference inside upstream `Trainer.Model`;
- Practical-RIFE wraps another internal model object and loads weights from `.pkl` files with model code located in the weight directory;
- adapters currently add another abstraction layer, but do not yet expose a clean separation between model-specific preprocessing, runtime forward execution, and postprocessing;
- ONNX export boundaries are not explicit;
- BentoML services would currently have to depend on model-specific implementation details or import heavy upstream code in an ad hoc way.

This stage must refactor inference so that future BentoML/ONNX work does not duplicate or reverse-engineer adapter internals.

---

## 6. Required Outcome

At the end of this stage, the codebase should have a clear inference component with the following properties:

1. **Inference is modular.**
   - Data preparation/training modules remain separate from inference modules.
   - Inference code can be imported and used without importing dataset-building or training workflows unnecessarily.

2. **Adapters expose a stable public prediction API.**
   - Existing adapter API must remain usable.
   - The API should be extended only if needed for backend/runtime support.

3. **Runtime backend is explicit.**
   - PyTorch backend is the working default.
   - ONNX backend is designed and partially or fully implemented where feasible.
   - Runtime selection should be driven by config, not hardcoded branches scattered across model-specific code.

4. **ONNX export boundaries are explicit.**
   - For EMA-VFI and Practical-RIFE, the plan/implementation must state exactly what object/module is exported.
   - If a model cannot be exported without major upstream refactor, this must be documented with evidence and a proposed fallback.

5. **BentoML readiness is improved.**
   - A future BentoML service should be able to instantiate an adapter/runtime from config and call its public prediction API.
   - BentoML wrappers should remain thin and should not duplicate tensor formatting, padding, checkpoint loading, or postprocessing.

6. **Nx inference support is prepared or implemented after the runtime refactor.**
   - Fixed 2x inference must remain supported.
   - Nx inference is only for inference, not training/fine-tuning.
   - `interpolation_factor` must be an integer in `[2, 8]`.
   - Nx work should prioritize EMA-VFI and Practical-RIFE.

---

## 7. Functional Boundaries

The refactor should establish or strengthen the following boundaries.

### 7.1 Data/Training Components

These components remain separate from inference serving:

- raw-video preprocessing for training data;
- source/global index building;
- dataset version building;
- `UniversalTripletDataset` for training/evaluation;
- training/fine-tuning runners;
- candidate validation logic;
- retraining-pool logic in later stages.

They may still use model adapters for evaluation or validation, but they must not be required by the inference-serving component.

### 7.2 Inference Core

The inference core should contain:

- model adapter interfaces;
- runtime backend interfaces;
- tensor/image conversion utilities;
- padding/unpadding logic;
- `predict_pair`, `predict_batch`, and future `predict_intermediate_frames` logic;
- model-specific inference cores for EMA-VFI and Practical-RIFE;
- local video inference using the adapter public API.

### 7.3 Video I/O

Video decoding, frame iteration, interleaving, encoding, and audio remuxing are model-independent inference utilities.

They should stay outside model-specific adapters. Adapters should operate on frames/tensors, not on full video files.

### 7.4 Serving Layer

BentoML service code is not required as a full implementation in this stage unless explicitly approved later.

However, the inference core must be shaped so that a future BentoML Runnable/Service can do:

```text
load config / BentoML model reference
→ create adapter/runtime in service process
→ call adapter.predict_batch(...) or equivalent public API
→ return frames/video output
```

---

## 8. Runtime Backend Design

The project should move toward a backend-aware inference design.

Expected conceptual structure:

```text
ModelAdapter
  -> model-specific preprocessing/tensor formatting
  -> RuntimeBackend
       -> TorchRuntimeBackend
       -> OnnxRuntimeBackend
  -> model-specific postprocessing
```

The exact file/module structure is left to the agent's implementation plan, but the responsibilities must remain clear.

### 8.1 PyTorch Runtime

PyTorch runtime remains the default and must continue to pass existing inference smoke tests.

The PyTorch runtime should wrap explicit model-call boundaries, not hidden full upstream scripts.

### 8.2 ONNX Runtime

ONNX Runtime support should be planned and implemented only after the export boundary is clear.

The stage should include:

- model-specific ONNX export analysis;
- export script or command where feasible;
- ONNX Runtime loading path where feasible;
- equivalence check between PyTorch and ONNX outputs on tiny inputs;
- documentation of unsupported operators, dynamic-shape issues, or export blockers.

### 8.3 Dynamic Shapes

The stage must decide and document whether ONNX exports use:

- fixed input shape;
- dynamic batch dimension only;
- dynamic height/width;
- fixed padded shape for serving.

The first working export may use fixed or bounded shapes if dynamic height/width is too fragile. The decision must be documented because it affects BentoML serving and accepted input sizes.

---

## 9. ONNX Export Requirements

For each in-scope model, the agent must determine:

1. Which Python object is actually the exportable neural network.
2. Which tensors are inputs.
3. Which tensors are outputs.
4. Whether timestep `t` is a tensor input, scalar argument, or fixed constant.
5. Which padding/divisor rules must be applied outside the exported graph.
6. Which preprocessing/postprocessing stays in Python.
7. Whether export supports dynamic shapes.
8. Whether the exported model can be verified against PyTorch outputs.

### 9.1 EMA-VFI

The agent must inspect:

- `model_repos/EMA-VFI/Trainer.py`;
- `model_repos/EMA-VFI/model/`;
- `model_repos/EMA-VFI/demo_2x.py`;
- `model_repos/EMA-VFI/demo_Nx.py`;
- current `EMAVFIAdapter` implementation.

The agent must identify whether the export boundary should be:

- the internal EMA-VFI network object;
- a thin project-owned wrapper around the internal network;
- a separate inference-only wrapper reproducing upstream `Model.inference` / `multi_inference` behavior.

### 9.2 Practical-RIFE

The agent must inspect:

- `model_repos/Practical-RIFE/inference_img.py`;
- `model_repos/Practical-RIFE/inference_video.py`;
- `model_repos/Practical-RIFE/model/`;
- `model_weights/Practical-RIFE/RIFEv4.25/train_log/`;
- current `PracticalRIFEAdapter` implementation.

The agent must identify whether the export boundary should be:

- the loaded `flownet` object;
- a project-owned wrapper around `model.inference(...)`;
- a smaller internal network loaded from `train_log` code.

The plan must explicitly address the fact that Practical-RIFE loads model code from the selected `train_log` directory and stores weights in `flownet.pkl`.

---

## 10. Nx Interpolation Requirements

Nx interpolation is not the first refactor step. It should be implemented only after inference responsibilities and runtime boundaries are clear.

### 10.1 Public Semantics

Use:

```text
interpolation_mode: fixed_2x | arbitrary_nx
interpolation_factor: int
```

Rules:

- `fixed_2x` uses the existing 2x path and 2x checkpoints;
- `arbitrary_nx` generates `N - 1` intermediate frames between each pair of input frames;
- `interpolation_factor` must satisfy `2 <= interpolation_factor <= 8`;
- `interpolation_factor = 2` in `arbitrary_nx` should behave like one intermediate frame through the arbitrary/Nx path;
- training and fine-tuning remain fixed 2x only.

### 10.2 EMA-VFI Nx

EMA-VFI Nx should use the `_t` checkpoint variant when using arbitrary/Nx mode:

```text
model_weights/EMA-VFI/ours_small_t.pkl
```

The agent must inspect upstream `demo_Nx.py` and implement project-compatible behavior through the adapter/runtime layer.

### 10.3 Practical-RIFE Nx

Practical-RIFE Nx should use the model's timestep/multi interpolation path if supported by the current local v4.25 implementation.

The agent must verify this through repository inspection and smoke tests. If v4.25 cannot support direct Nx cleanly but v4.26 can, the agent must document the evidence and ask the user before changing the default version.

---

## 11. BentoML Readiness Requirements

The stage does not have to implement the full production service, but it must prepare for it.

The planned serving architecture is:

```text
one BentoML service
one runner
one active production model
```

Future multi-GPU extension:

```text
one BentoML service per active model/mode
one runner inside each service
one GPU assignment per service/container
```

The inference code must be compatible with this direction:

- BentoML service wrappers should be thin.
- Model loading should happen inside the serving process.
- Adapter/runtime construction should be config-driven.
- No dynamic loading/unloading of multiple large models per request.
- No duplication of model-specific preprocessing/postprocessing in BentoML service files.

A minimal BentoML proof-of-concept may be proposed in the ExecPlan, but should not be implemented until the inference/runtime refactor and ONNX feasibility analysis are complete.

---

## 12. Required Planning Before Implementation

The coding agent must not immediately implement this stage.

The first task is analysis and clarification:

1. Inspect the current codebase.
2. Inspect EMA-VFI and Practical-RIFE upstream code.
3. Inspect current adapters and inference flow.
4. Identify the current coupling points.
5. Identify candidate module boundaries for refactor.
6. Identify ONNX export boundaries for each model.
7. Identify blockers and tradeoffs.
8. Ask the user all necessary clarification questions.
9. Only after user answers, create the Stage 2 ExecPlan.

---

## 13. Non-Goals

Do not implement in this stage unless explicitly approved:

- full FastAPI backend;
- Celery/Redis request queue;
- PostgreSQL application schema;
- MinIO uploads/outputs integration;
- frontend;
- user authentication/session logic;
- online quality history;
- retraining triggers;
- retraining pool;
- monitoring/alerts;
- complete Docker Compose production stack;
- AMT-S runtime refactor;
- AMT-S ONNX export;
- AMT-S Nx inference;
- training/fine-tuning changes not needed for inference runtime separation.

---

## 14. Expected Deliverables

The final implementation of this stage should produce, at minimum:

1. A planned and documented inference-module boundary.
2. Refactored EMA-VFI and Practical-RIFE inference code with explicit inference cores/runtimes.
3. Stable adapter public API suitable for local inference and future BentoML usage.
4. Explicit runtime backend abstraction or equivalent clean design.
5. PyTorch backend that preserves existing 2x behavior.
6. ONNX export feasibility report for EMA-VFI and Practical-RIFE.
7. ONNX export scripts/commands where feasible.
8. ONNX Runtime prediction path where feasible.
9. PyTorch-vs-ONNX equivalence smoke checks where feasible.
10. Nx inference support for EMA-VFI and Practical-RIFE after runtime refactor.
11. Configs for fixed 2x and arbitrary/Nx inference modes.
12. Updated docs explaining inference architecture, runtime backends, ONNX export, and Nx modes.
13. Updated `PROJECT_MAP.md`.
14. A completed Stage 2 ExecPlan with decisions, validation, and handoff.

If ONNX export is not feasible for one model without major rewriting, the deliverable is a documented blocker with evidence and a proposed fallback path, not a silent partial implementation.

---

## 15. Validation Expectations

Validation must be incremental and bounded.

Required validation categories:

- existing 2x inference smoke tests still pass for EMA-VFI and Practical-RIFE;
- adapter public API remains compatible with existing local inference;
- runtime backend selection works for PyTorch;
- ONNX export command either produces an ONNX artifact or fails with documented reason;
- ONNX Runtime output is compared with PyTorch output when export succeeds;
- Nx inference works for factor 2, 4, and 8 where supported;
- tests remain focused on behavior rather than trivial config values;
- `ruff` and `pytest` pass after each substantial implementation step.

Do not run full-dataset evaluation or long inference jobs as part of routine validation.

---

## 16. Documentation Requirements

Documentation must explain:

- how the inference subsystem is organized;
- what belongs to data/training modules vs inference modules;
- how adapters are used;
- what runtime backends exist;
- how PyTorch inference is called;
- how ONNX export is attempted;
- where ONNX artifacts are written;
- how to compare PyTorch and ONNX outputs;
- how fixed 2x and arbitrary/Nx inference modes are configured;
- limitations and known model-specific constraints;
- what is ready for BentoML and what remains future work.

Documentation must be practical enough for a teammate to implement BentoML services on top of the refactored inference component without reading every upstream model file.

---

## 17. Open Questions the Agent Should Resolve With the User

The agent should analyze the repository first and then ask concrete questions. Expected question areas include:

1. Whether existing AMT code should remain untouched, be marked legacy, or be removed from default all-target inference.
2. Where ONNX artifacts should be stored, for example `model_exports/onnx/` or `outputs/onnx/`.
3. Whether ONNX export must support dynamic height/width or whether fixed padded shapes are acceptable initially.
4. Whether Nx mode must be implemented before or after ONNX export.
5. Whether BentoML service skeletons should be part of this stage or only the inference component should be prepared.
6. Whether Practical-RIFE v4.25 must remain the default or whether v4.26 may be used if it is easier for Nx/ONNX.
7. What tolerance should be used for PyTorch-vs-ONNX output comparison.
8. Whether 2x fixed mode and Nx arbitrary mode should be two separate model configs or one config with a mode switch.
9. Whether future services should be split by model, by mode, or by model+mode.
10. Whether project package names and module layout may be reorganized substantially or only incrementally.

---

## 18. Completion Criteria

This stage is complete when:

1. The inference-related code is separated clearly enough to be reused by future BentoML services.
2. EMA-VFI and Practical-RIFE PyTorch inference still work after refactor.
3. ONNX export feasibility is established for both models.
4. ONNX export and ONNX Runtime inference are implemented where feasible, or blockers are documented with evidence.
5. Nx inference is implemented for EMA-VFI and Practical-RIFE where feasible.
6. Documentation explains the new inference/runtime architecture.
7. The Stage 2 ExecPlan records all decisions, validation results, blockers, and handoff notes.
