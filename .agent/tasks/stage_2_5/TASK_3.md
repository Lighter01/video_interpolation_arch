# Start Stage 2.5 Milestone 3 Implementation

Continue Stage 2.5 — ONNX Stabilization, Real-Image Equivalence, and Batched Inference.

Use `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md` as the main working document.

Before implementation, briefly re-read:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/stage_plans/02_5_inference_runtime_stabilization_stage_plan.md`
- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`

Then inspect only the files that are directly relevant to **Milestone 3 — Real-Image ONNX-vs-PyTorch Equivalence**.

Implement only Milestone 3.

Do not start Milestone 4.

## Current context

The deeper EMA-VFI ONNX refactor trial after Milestone 2 changed the EMA classification to:

```text
EMA constrained-dynamic ONNX works with external divisor 112 padding.
```

Use the updated ExecPlan as source of truth, but be aware of these known outputs:

- current accepted EMA ONNX artifact:
  `model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx`
- accepted EMA Task 2.5 metrics are summarized in the ExecPlan and Stage 2.5 docs; the old Task 2.5 validation output directory was cleaned before Milestone 3. New Milestone 3 real-pair validation should write fresh reports, for example under:
  `outputs/onnx_validation/stage2_5_m3_real_pairs/`

Milestone 3 should now add real-image equivalence checks for:

- EMA-VFI ONNX, using the accepted constrained-dynamic divisor-112 path;
- Practical-RIFE ONNX, using the current v4.26 new updated path.

## Scope

Add a real-image ONNX-vs-PyTorch equivalence pipeline using image pairs from:

```text
raw_data/pair_test/
```

Real-image checks must be added **in addition to** existing synthetic checks, not as a replacement.

This milestone should answer:

- do PyTorch and ONNX outputs look materially different on realistic image pairs?
- what are the per-pair and aggregate differences?
- are EMA and Practical-RIFE ONNX outputs acceptable enough to keep investigating for serving?
- which provider/artifact/settings produced each result?

Do not implement batch inference in this milestone.

Do not implement benchmarks in this milestone.

Do not implement BentoML proof in this milestone.

## Required behavior

Implement or extend the ONNX validation workflow so it can run on real image pairs.

Expected pair discovery:

- discover subdirectories under `raw_data/pair_test/`;
- each pair directory should contain two input frames, currently expected as `frame1.png` and `frame2.png`;
- verify the actual structure before assuming it;
- fail clearly if a pair is malformed.

For every pair, compare:

```text
PyTorch generated intermediate frame
vs
ONNX generated intermediate frame
```

using the same:

- model target;
- checkpoint/export artifact;
- provider;
- backend config;
- input image pair;
- preprocessing/padding policy;
- timestep;
- interpolation mode/factor;
- Practical-RIFE scale where relevant.

## EMA-specific requirements

For EMA-VFI real-image validation:

- use the accepted constrained-dynamic ONNX path from the deep refactor trial;
- apply the documented external divisor-112 padding policy;
- record original and padded input shapes;
- validate that ONNX output is unpadded back to the expected original H/W;
- do not reopen the EMA dynamic-shape refactor in this milestone;
- do not implement static bucket fallback.

If EMA ONNX validation fails on real pairs, record the exact failure and classify the real-image status without changing the EMA refactor.

## Practical-RIFE-specific requirements

For Practical-RIFE real-image validation:

- use Practical-RIFE v4.26 as the default target;
- preserve v4.25 as an alternative if it already remains available;
- test ONNX-vs-PyTorch on real pairs using the same padding/scale/timestep path used by current runtime validation;
- record provider-specific behavior clearly, especially CPU vs CUDA if both are tested;
- compare original vs simplified ONNX artifact if both are relevant and cheap to test.

Do not attempt to fix Practical-RIFE ONNX mismatch in this milestone. This milestone is about adding real-image evidence.

## Metrics

For each real-image pair, compute and record:

- MAE;
- max absolute error;
- MSE;
- PSNR between PyTorch output and ONNX output;
- SSIM between PyTorch output and ONNX output;
- optional LPIPS only if cheap and already supported, but do not make LPIPS required for smoke runs.

Also write aggregate summaries per:

- model;
- provider;
- artifact;
- shape/padding policy;
- synthetic vs real input group.

## Visual artifacts

For each real-image pair, write visual artifacts such as:

```text
left.png
right.png
pytorch_generated.png
onnx_generated.png
absdiff.png
```

Use a clear output structure, for example:

```text
outputs/onnx_validation/stage2_5_m3_real_pairs/
  <model>/
    <provider_or_backend>/
      <pair_id>/
        ...
      equivalence_metrics.csv
      equivalence_report.json
```

Exact layout can differ if cleaner, but it must be documented.

## Provenance requirements

Every report must include enough metadata to interpret results later:

- model target;
- backend/provider;
- torch device;
- ONNX artifact path;
- original vs simplified artifact;
- input pair id;
- original input shape;
- padded input shape;
- output shape;
- interpolation mode;
- interpolation factor;
- timestep;
- Practical-RIFE scale if relevant;
- EMA divisor/padding policy if relevant;
- command/config used if available.

If existing synthetic reports lack some metadata, do not rewrite all old artifacts. Ensure new Milestone 3 outputs contain sufficient provenance.

## CLI/API expectations

Add a safe developer-facing CLI command or extend the existing ONNX validation command so it can run real-image pair validation.

Expected options should include:

- model target, for example EMA or Practical-RIFE;
- pair root, defaulting to `raw_data/pair_test` if appropriate;
- ONNX artifact path or default artifact resolution;
- provider selection;
- torch device;
- output directory;
- limit pairs for smoke runs;
- optional flags for simplified vs original ONNX artifact if relevant;
- Practical-RIFE scale if relevant;
- EMA divisor if relevant.

Keep commands bounded and safe.

Do not run full videos or datasets.

## Validation

Run the normal project checks:

```bash
uv run pytest
uv run ruff check src tests
```

If the uv cache is not writable, use:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
```

Run small real-image validation smoke checks using `raw_data/pair_test`.

At minimum, if the environment supports it, run:

- Practical-RIFE real-image ONNX-vs-PyTorch validation on the available pairs;
- EMA real-image ONNX-vs-PyTorch validation using the accepted constrained-dynamic divisor-112 ONNX path.

If CUDA is unavailable or a provider is unavailable, run CPU-safe checks where possible and record deferred CUDA checks clearly.

Do not run long jobs.

## Updates required

Update:

- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`
- `.agent/docs/PROJECT_MAP.md` if repository structure changes
- human-facing docs, especially `docs/stage2_5_inference_runtime_stabilization.md`

Document:

- how to run real-image equivalence checks;
- how synthetic and real-image checks differ;
- where outputs are written;
- how to interpret MAE, max error, PSNR, and SSIM;
- EMA constrained-dynamic divisor-112 validation status on real pairs;
- Practical-RIFE real-image equivalence status;
- known limitations and what remains for later milestones.

## End-of-task summary

Summarize:

- what real-image validation pipeline was added;
- which CLI/API commands were added or changed;
- what real image pairs were discovered and tested;
- what metrics and artifacts were written;
- EMA real-image ONNX-vs-PyTorch results;
- Practical-RIFE real-image ONNX-vs-PyTorch results;
- whether CPU/CUDA providers were tested or deferred;
- what validation commands were run;
- what documentation/project-map/ExecPlan updates were made;
- what remains for Milestone 4.
