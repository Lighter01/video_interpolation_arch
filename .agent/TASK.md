Start a new intermediate planning stage: Stage 2.5 — ONNX Stabilization, Real-Image Equivalence, and Batched Inference.

This is a planning-only task.

Do not implement code yet.
Do not modify source code.
Do not modify configs.
Do not modify model repositories.
Do not add dependencies.
Do not run long jobs.

Before planning, read:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/general_plan.md`
- `.agent/stage_plans/02_inference_runtime_refactor_stage_plan.md`
- `.agent/stage_plans/02_5_inference_runtime_stabilization_stage_plan.md`
- `.agent/docs/exec-plans/active/02_inference_runtime_refactor.execplan.md`
- `.agent/docs/exec-plans/completed/01_ml_core_selected.execplan.md`
- `.docs/stage1_ml_core.md`
- `.docs/stage2_inference_runtime_refactor.md`

Then inspect the current repository state, especially:

- `src/video_interpolation/inference_runtime/`
- `src/video_interpolation/adapters/
- `src/video_interpolation/inference.py`
- `src/video_interpolation/batch_inference.py`
- `src/video_interpolation/cli.py`
- `src/video_interpolation/mlflow.py`
- `src/video_interpolation/metrics.py`
- `src/video_interpolation/image_io.py`
- `configs/models/`
- `configs/inference/`
- any ONNX export/runtime configs already present
- `model_exports/onnx/`
- `outputs/onnx_validation/`
- `raw_data/pair_test/`
- `raw_data/tmp_test/`
- `docs/stage2_inference_runtime_refactor.md`
- tests related to inference runtime, ONNX, EMA, RIFE, adapters, video inference, and metrics
- `model_repos/EMA-VFI/model/feature_extractor.py`
- `model_repos/EMA-VFI/model/flow_estimation.py`
- `model_repos/EMA-VFI/model/warplayer.py`
- Practical-RIFE runtime source location established during Stage 2
- `model_repos/Practical-RIFE/`
- `model_weights/EMA-VFI/`
- `model_weights/Practical-RIFE/`

Stage 2 is currently paused before its BentoML compatibility proof milestone.

The purpose of Stage 2.5 is to resolve or classify technical blockers discovered by the end of Stage 2 Milestone 7 before continuing to BentoML.

The Stage 2.5 stage plan is authoritative for this intermediate stage.

Your task:

1. Analyze the current state of Stage 2 and the ONNX/runtime implementation.
2. Analyze the EMA-VFI dynamic H/W ONNX problem using the stage plan notes and existing code/logs.
3. Analyze Practical-RIFE ONNX-vs-PyTorch equivalence status.
4. Analyze current local video inference and current meaning of `batch_inference.py`.
5. Identify where true model batch inference should be implemented.
6. Identify how mini-benchmarks should integrate with existing MLflow helpers.
7. Create a new active ExecPlan:

```text
.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md
```

The ExecPlan must follow `.agent/docs/PLANS.md`.

The ExecPlan must include:

- Stage goal;
- source documents and authority;
- current repository state;
- requirements restated;
- non-goals;
- architecture and implementation strategy;
- ordered milestones;
- validation strategy;
- expected artifacts;
- risks and recovery;
- progress;
- surprises/discoveries;
- decision log;
- outcomes/handoff placeholder.

The ExecPlan should plan, at minimum:

1. EMA-VFI dynamic/constrained-dynamic ONNX investigation and refactor attempt.
2. Real-image ONNX-vs-PyTorch equivalence checks using `raw_data/pair_test`, in addition to synthetic tests.
3. True model batch inference API design.
4. PyTorch batch inference for EMA-VFI and Practical-RIFE.
5. Optional ONNX batch inference only if ONNX support remains viable.
6. Mini-benchmarks for backend/mode/batch-size comparisons.
7. MLflow logging for benchmarks.
8. Human-facing documentation.
9. Stage 2 ExecPlan handoff pointer/update after Stage 2.5 completion.

Important decisions to preserve:

- Do not implement static bucket fallback for EMA-VFI ONNX.
- If EMA-VFI dynamic/constrained-dynamic ONNX cannot be made usable, mark EMA-VFI ONNX as blocked/deferred and keep EMA-VFI PyTorch-only.
- Real-image equivalence checks must be added in addition to synthetic checks, not as a replacement.
- Batch inference should support fixed 2x and Nx.
- Preferred Nx batch strategy is pair×timestep flattening, not a Python loop over timesteps.
- Sequential inference fallback must remain available.
- Batch inference must be VRAM-aware and configurable.
- Benchmarks must record timing breakdowns and log to MLflow where available.
- Stage 2 remains paused and must resume only after Stage 2.5 is complete.

Do not begin implementation after writing the ExecPlan.

After creating the ExecPlan, summarize:

- which files you read;
- what current blockers you identified;
- what milestones you proposed;
- what questions, if any, need user clarification before implementation;
- where the new ExecPlan was created.
