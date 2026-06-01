# Plan Dynamic-Batch ONNX Export for Practical-RIFE and EMA-VFI

Continue Stage 2.5 — ONNX Stabilization, Real-Image Equivalence, and Batched Inference.

Use `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md` as the main working document.

This is a planning task.

Do not implement code yet.
Do not modify source code yet.
Do not modify configs yet.
Do not export new ONNX artifacts yet.
Do not run long jobs.

Briefly re-read:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/stage_plans/02_5_inference_runtime_stabilization_stage_plan.md`
- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`

Then inspect only the files and artifacts directly relevant to ONNX export/runtime batching:

- current ONNX export code;
- current ONNX runtime backend code;
- EMA-VFI runtime/export wrapper;
- Practical-RIFE runtime/export wrapper;
- current ONNX artifacts under `model_exports/onnx/`;
- current ONNX validation reports under `outputs/onnx_validation/`;
- tests related to ONNX export/runtime and model batch inference.

## Task

Plan the required changes to make ONNX-exported models support dynamic batch size for true model batch inference.

The immediate motivation is that Stage 2.5 Milestone 7 showed:

- EMA-VFI ONNX supports true multi-row ORT batching through `ModelBatchRequest`;
- Practical-RIFE ONNX currently supports the batch API and reconstruction, but effective ORT batch is forced to `1` because the accepted Practical-RIFE dynamo artifact has static batch `1`.

Before Milestone 8 benchmarks, plan how to fix Practical-RIFE ONNX export so that ONNX Runtime can process multi-row flattened batches directly.

Also verify whether the accepted EMA-VFI ONNX artifact truly supports dynamic batch size. If EMA-VFI does not support dynamic batch size reliably, include an analogous dynamic-batch export adaptation plan for EMA-VFI as well.

## Required analysis

For each active ONNX model target:

- Practical-RIFE v4.26;
- EMA-VFI constrained-dynamic divisor-112 ONNX path;

determine:

1. Whether the current exported ONNX artifact has a dynamic batch axis.
2. Whether ONNX Runtime accepts runtime `B > 1` for fixed 2x.
3. Whether ONNX Runtime accepts arbitrary flattened batch sizes `B = pairs * (N - 1)` for arbitrary Nx.
4. Whether timestep input is batch-shaped correctly and can vary with `B`.
5. Whether output batch axis is dynamic and matches input `B`.
6. Whether simplified and original artifacts behave the same.
7. Whether CPU provider and CUDA provider behavior should be validated separately.
8. Whether current validation reports already prove this or whether new validation commands are needed.

## Practical-RIFE dynamic-batch export plan

Plan changes for Practical-RIFE first.

The goal is dynamic ONNX batch size, not support for a fixed set of batch sizes.

The exported ONNX model should accept runtime batch dimension `B`, where `B` is determined by the current inference request/chunk and limited only by memory and configured `inference_batch_size`.

Examples:

```text
fixed 2x with 2 source pairs:
  B = 2 flattened rows

fixed 2x with 8 source pairs:
  B = 8 flattened rows

Nx factor 4 with 2 source pairs:
  B = 2 * (4 - 1) = 6 flattened rows

Nx factor 8 with 3 source pairs:
  B = 3 * (8 - 1) = 21 flattened rows
```

These are examples only. The export must not hardcode batch sizes `1`, `2`, `4`, `6`, `8`, or `21`.

Do not implement separate ONNX artifacts per batch size.

Do not implement fixed-batch exports such as:

```text
rife_batch1.onnx
rife_batch2.onnx
rife_batch4.onnx
```

There should be one dynamic-batch artifact per model/export configuration.

Validation should test batch sizes `1`, `2`, and `4` only as smoke checks to prove that the batch axis is dynamic. Passing those checks does not mean the implementation should restrict batch size to those values.

The plan should cover:

- export wrapper input shapes;
- dynamic batch axis declaration;
- timestep input shape and broadcasting rules;
- output dynamic batch axis;
- whether `dynamo=True` / `dynamic_shapes` or legacy `dynamic_axes` should be used;
- how Practical-RIFE `scale` is represented;
- whether original and simplified artifacts should both be generated;
- how to name and store new dynamic-batch artifacts;
- how to keep existing static-batch artifacts available as fallback until the new path is validated;
- how to validate dynamic batch with smoke sizes `1`, `2`, and `4`;
- how to validate fixed 2x and Nx factor `4`;
- how to compare PyTorch-vs-ONNX outputs after dynamic-batch export.

Do not assume that dynamic H/W implies dynamic batch. Treat batch axis as a separate export requirement.

## EMA-VFI dynamic-batch verification plan

EMA-VFI currently has an accepted constrained-dynamic ONNX path with external divisor-112 padding.

Plan a verification step to prove whether EMA ONNX batch is truly dynamic.

The plan should test:

- batch size `1`;
- batch size `2`;
- batch size `4` if cheap;
- at least one constrained-dynamic spatial shape that already passed, such as the accepted divisor-112 path;
- fixed 2x;
- arbitrary Nx factor `4`.

If EMA batch axis is not dynamic, plan the same kind of dynamic-batch export adaptation as for Practical-RIFE, but preserve the current constrained-dynamic H/W policy.

Do not reopen EMA full dynamic H/W work in this task. EMA spatial support remains constrained-dynamic with external divisor-112 padding.

## Validation plan

The planned implementation must include validation for:

- ONNX artifact graph input/output dimensions;
- ORT execution with batch sizes `1`, `2`, and where practical `4`;
- fixed 2x batch;
- arbitrary Nx flattened batch;
- output reconstruction as `outputs[pair_index][timestep_index]`;
- PyTorch-vs-ONNX equivalence metrics:
  - MAE;
  - max absolute error;
  - MSE;
  - PSNR/SSIM where already available;
- provider-specific behavior when CUDA is available;
- fallback behavior when dynamic batch is unavailable.

Keep tests small and safe.

Do not plan long video jobs.

## ExecPlan update

Update `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md` with a short planning entry before implementation begins.

Record:

- why the dynamic-batch export plan is needed;
- current Practical-RIFE static-batch limitation;
- whether EMA batch support is already proven or still needs verification;
- proposed implementation location;
- proposed validation commands;
- risks and fallback behavior;
- whether this should be inserted before Milestone 8 benchmarks or treated as a Milestone 7b follow-up.

Do not start implementation after updating the plan.

## End-of-task summary

Summarize:

- what current artifacts/reports were inspected;
- whether Practical-RIFE dynamic batch is currently unsupported or merely unverified;
- whether EMA dynamic batch is currently supported, unsupported, or unverified;
- what changes are planned for Practical-RIFE export;
- whether EMA needs analogous export changes;
- what validation should be run after implementation;
- what ExecPlan sections were updated;
- whether the next step should be implementation of this dynamic-batch export follow-up before Milestone 8.
