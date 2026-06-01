# Stage 2.5 Inference Runtime Stabilization

Stage 2.5 exists to stabilize the inference runtime before Stage 2 resumes with the BentoML compatibility proof. It focuses on three current issues:

- EMA-VFI ONNX exports have symbolic dynamic axes but do not currently work across changed H/W.
- Practical-RIFE ONNX executes at multiple H/W sizes, but PyTorch-vs-ONNX equivalence is not yet trusted.
- Local video inference is sequential at the model-call level; true model batch inference is still planned.

## Current Baseline

ONNX exports live under `model_exports/onnx/`:

- `model_exports/onnx/ema_vfi_small/`
- `model_exports/onnx/practical_rife_v4_26/`

Both model directories currently contain original and simplified opset 17 dynamic-H/W artifacts.

ONNX validation outputs live under `outputs/onnx_validation/`. Existing reports are synthetic-tensor checks only. They include MAE, max absolute error, MSE, and allclose status, but they do not yet record enough provenance for final serving decisions, such as command, timestamp, Git revision, PyTorch device, real-vs-synthetic input source, RIFE scale, or full artifact classification.

Current EMA-VFI evidence:

- The legacy simplified ONNX artifact runs at the export-like `32x32` shape.
- A changed `64x64` shape fails in ONNX Runtime with a `LayerNormalization` shape mismatch.
- Milestone 2 tested external 56-multiple padding, larger legacy re-export, and a modern dynamo export route.
- EMA ONNX is now classified as export-size/static-like only. It is not suitable for dynamic serving, and EMA should stay on the PyTorch backend unless a broader upstream dynamic-shape rewrite is approved.

Current Practical-RIFE evidence:

- ONNX Runtime can execute more than one H/W shape.
- CPU original-artifact reports are much closer than the current CUDA simplified reports.
- Controlled synthetic and real-image checks are still required before accepting Practical-RIFE ONNX for serving.

## Local Inputs

Real image pairs for future equivalence checks live under `raw_data/pair_test/`. The current structure is:

```text
raw_data/pair_test/001/frame1.png
raw_data/pair_test/001/frame2.png
raw_data/pair_test/002/frame1.png
raw_data/pair_test/002/frame2.png
raw_data/pair_test/003/frame1.png
raw_data/pair_test/003/frame2.png
```

All inspected pair-test images are `512 x 320` RGB PNGs.

Short video smoke inputs live under `raw_data/tmp_test/`. `DORA_cut.mp4` is the preferred first smoke/benchmark input because it is much smaller than the numbered 1080p clips.

## Batch Terminology

`src/video_interpolation/batch_inference.py` currently means directory-wide inference orchestration: discovering input videos, selecting targets, constructing output paths, and writing measurement CSVs.

It is not true model batch inference. Stage 2.5 still needs a separate model-batch API for processing multiple frame pairs in one model call, while keeping sequential inference as a fallback.

## Later Milestones

Detailed implementation remains in the active Stage 2.5 ExecPlan:

```text
.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md
```

Milestone 3 adds real-image ONNX-vs-PyTorch equivalence. It should treat EMA ONNX as deferred/PyTorch-only and focus ONNX acceptance work on Practical-RIFE unless the EMA upstream dynamic-shape rewrite is reopened. True model batch inference, video chunking, ONNX batch support where viable, benchmarks, and MLflow logging are planned for later Stage 2.5 milestones.
