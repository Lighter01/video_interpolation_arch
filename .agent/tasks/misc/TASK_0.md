# TASK — Optional Video Quality Evaluation in Inference Pipeline

## Purpose

Add an optional quality-evaluation step to the existing video inference pipeline.

This feature must be implemented **below the BentoML layer**. It should be part of the project inference stack and available to:

- local video inference code;
- benchmark workflows;
- serving-facing `PracticalRIFEVideoInferenceRunner`;
- future BentoML services that call the project inference module.

The BentoML examples/services should only consume this feature through the existing inference/runner interfaces. They must not implement their own frame sampling, metric calculation, or triplet writing logic.

The feature should:

1. sample a small number of triplets from the input video;
2. evaluate Practical-RIFE on these triplets in strict 2x mode;
3. compute aggregate PSNR and SSIM;
4. save sampled original triplets in a simple Vimeo-style directory structure;
5. expose aggregate metrics through `VideoInferenceResult` and serving responses;
6. allow benchmark workflows to measure the overhead of quality evaluation.

Do not implement MinIO upload/download, good/bad clip classification, retraining triggers, online quality history, or statistical drift logic in this task.

---

## Current context

Relevant existing modules:

- `src/video_interpolation/inference.py`
  - owns `VideoInferenceConfig`;
  - owns `VideoInferenceResult`;
  - owns `VideoInferenceTiming`;
  - owns `run_video_inference(...)`;
  - currently orchestrates full video inference.
- `src/video_interpolation/serving.py`
  - contains `PracticalRIFEVideoInferenceRunner`;
  - calls `run_video_inference(...)`.
- `src/video_interpolation/inference_benchmark.py`
  - runs video inference benchmarks;
  - records timing breakdowns;
  - logs benchmark runs to MLflow.
- `src/video_interpolation/mlflow.py`
  - contains benchmark logging helpers.
- `src/video_interpolation/metrics.py`
  - already implements PSNR and SSIM.
- `src/video_interpolation/data/preprocessing.py`
  - already contains PyAV metadata and selected-frame decoding logic that can guide this task.

Current production-facing inference target:

- model: Practical-RIFE v4.26;
- default serving backend: PyTorch runtime on `cuda`;
- alternative backend: ONNX Runtime with `CUDAExecutionProvider`;
- serving execution mode: `sequential`;
- interpolation mode: `arbitrary_nx`;
- service-level `interpolation_factor`: `2..4`;
- `scale` is a runtime/request parameter.

Quality evaluation target:

- Practical-RIFE only;
- sequential inference only for now;
- always 2x / `t = 0.5`;
- independent of the main inference `interpolation_factor`.

---

## High-level design decision

Do **not** make quality evaluation a BentoML-specific feature.

Instead:

```text
run_video_inference(...)
  -> optional quality evaluation step
  -> VideoInferenceResult contains quality metrics and timing
  -> serving runner/BentoML examples return these fields
```

BentoML services should call the project inference module and receive metrics from the result.

---

## Package structure requirement

Do not add another arbitrary root-level file under `src/video_interpolation/`.

The package root is already crowded. Prefer a small subpackage for this feature.

Recommended new structure:

```text
src/video_interpolation/video_quality/
  __init__.py
  config.py
  sampling.py
  triplets.py
  evaluation.py
```

Suggested responsibilities:

- `config.py`
  - quality-evaluation config/result dataclasses;
- `sampling.py`
  - video metadata, triplet index sampling, selected-frame decoding;
- `triplets.py`
  - Vimeo-style triplet writing;
- `evaluation.py`
  - quality evaluation orchestration.

If existing selected-frame decode helpers from `data/preprocessing.py` are useful, extract shared helpers carefully instead of importing private underscore-prefixed functions directly when possible.

Do not perform a broad package rewrite. Keep the change focused and backward compatible.

---

## Inference pipeline integration

Extend `VideoInferenceConfig` in `inference.py` with an optional quality-evaluation config.

Suggested conceptual shape:

```python
@dataclass(frozen=True)
class VideoQualityEvaluationConfig:
    enabled: bool = False
    triplet_output_dir: Path | None = None
    sample_count: int = 16
    random_seed: int | None = None
    scene_cut_ssim_threshold: float | None = 0.75
    fail_policy: Literal["raise", "warn"] = "raise"
```

Exact implementation can differ if it fits the project style better.

### Defaults

For generic local inference commands:

```text
quality evaluation disabled by default
```

For benchmark workflows:

```text
quality evaluation enabled by default
```

The benchmark CLI must provide a disabling flag, e.g.:

```text
--disable-quality-evaluation
```

For serving/BentoML runner:

```text
quality evaluation enabled by default
```

but it must be possible to disable it in code/config if it is too slow.

### Fail policy

If no valid triplets are sampled:

```text
psnr_mean = None
ssim_mean = None
quality_triplets_written = 0
```

This should not fail the whole inference run.

If quality evaluation raises a decode/model/runtime error:

- for local CLI/debug, `fail_policy="raise"` is acceptable;
- for serving/BentoML, use `fail_policy="warn"` so a successful video interpolation response is not lost only because quality evaluation failed.

Keep this implementation concise. Do not introduce a large, over-engineered error-handling system.

---

## Result and timing integration

Extend `VideoInferenceResult` with quality-related fields.

Suggested fields:

```python
quality_psnr_mean: float | None = None
quality_ssim_mean: float | None = None
quality_triplets_written: int = 0
quality_triplet_output_dir: Path | None = None
quality_error: str | None = None
```

Extend `VideoInferenceTiming` with:

```python
quality_evaluation_sec: float = 0.0
```

The BentoML response should include at least:

```python
{
    "psnr_mean": result.quality_psnr_mean,
    "ssim_mean": result.quality_ssim_mean,
}
```

Returning `triplet_output_dir` is allowed for service-worker convenience.

---

## Quality evaluation semantics

Quality evaluation must always use 2x interpolation.

For each sampled triplet:

```text
left   = frame_i
middle = frame_i+1
right  = frame_i+2
```

Model input:

```text
left, right, t = 0.5
```

Ground truth:

```text
middle
```

Metrics:

```text
prediction vs middle
```

This must remain true even if the main inference call used:

```text
interpolation_factor = 3
```

or:

```text
interpolation_factor = 4
```

Rationale:

- PSNR/SSIM require a real ground-truth frame;
- sampled triplets must remain compatible with Vimeo-style 2x training/fine-tuning;
- the project retraining strategy is based on 2x triplets.

Do not expose quality-evaluation interpolation factor as a user-facing parameter.

---

## Sampling behavior

Default:

```text
sample_count = 16
```

`sample_count` is an upper bound, not a required count.

If a video contains fewer valid triplets, evaluate fewer triplets.

If scene-cut filtering removes candidates, evaluate fewer triplets.

### Recommended sampling algorithm

Use the Stage 1 preprocessing approach as a reference.

Relevant existing ideas are in:

```text
src/video_interpolation/data/preprocessing.py
```

The quality evaluator should:

1. read video metadata with PyAV;
2. estimate/determine `frame_count`;
3. compute valid triplet start range:

   ```text
   0 <= start <= frame_count - 3
   ```

4. sample up to `sample_count` start indices uniformly/randomly;
5. sort sampled starts ascending;
6. convert each start to frame indices:

   ```text
   start, start + 1, start + 2
   ```

7. decode only the required frames, preferably using selected-frame / segment-seek PyAV logic;
8. build triplets;
9. apply scene-cut filter;
10. save accepted triplets;
11. run Practical-RIFE 2x prediction;
12. compute aggregate PSNR/SSIM.

Use deterministic sampling when `random_seed` is set.

Do not use PySceneDetect in this serving/inference path.

---

## Scene-cut filtering

Do not add static-triplet filtering.

Do not add `static_triplet_ssim_threshold`.

Implement only a lightweight scene-cut rejection filter.

Default:

```text
scene_cut_ssim_threshold = 0.75
```

For candidate triplet:

```text
left, middle, right
```

Reject if:

```text
SSIM(left, middle) < 0.75
or
SSIM(middle, right) < 0.75
```

This filter is not a good/bad clip classifier. It only prevents using likely scene-transition triplets for quality estimation and future training samples.

The threshold should be configurable. If set to `None`, skip scene-cut filtering.

---

## Triplet output format

Save all accepted sampled triplets.

Do not write:

- `summary.json`;
- `package_id.json`;
- `sequence_index.csv`;
- `metrics.csv`;
- per-sample metric files.

The downstream worker will upload files to MinIO and decide what to do with them.

Use simple Vimeo-style triplet naming:

```text
<triplet_output_dir>/<source_video_id>/<triplet_id>/im1.png
<triplet_output_dir>/<source_video_id>/<triplet_id>/im2.png
<triplet_output_dir>/<source_video_id>/<triplet_id>/im3.png
```

Example:

```text
/tmp/request_abc123/a7f31c9d4e2b/
  000000/
    im1.png
    im2.png
    im3.png
  000001/
    im1.png
    im2.png
    im3.png
```

Use a generated UUID-based source video id by default, e.g.:

```python
source_video_id = uuid.uuid4().hex[:12]
```

Do not use the original source filename as the directory name.

### Output directory rule

`VideoInferenceConfig.output_path` is a video file path, not a directory.

Therefore, when quality evaluation is enabled:

- if `triplet_output_dir` is provided, write triplets there;
- otherwise, default to:

  ```text
  output_path.parent / <source_video_id>
  ```

For BentoML/backend-worker usage, the worker will usually pass an output path inside a temporary request directory. The triplet directory should be placed in the same temporary directory.

---

## Metrics

Reuse existing metrics.

Use:

```text
src/video_interpolation/metrics.py
```

Specifically reuse:

- `compute_psnr(...)`;
- `compute_ssim(...)`.

Do not reimplement PSNR or SSIM.

Do not add LPIPS for this feature.

Do not write metric CSV/summary files for this online quality feature.

---

## Serving integration

Update `PracticalRIFEVideoInferenceRunner` so serving can enable quality evaluation through the normal inference config.

The runner should not implement sampling or metrics directly.

Expected serving flow:

```text
PracticalRIFEVideoInferenceRunner
  -> builds VideoInferenceConfig
  -> enables quality evaluation by default
  -> calls run_video_inference(...)
  -> returns output path + aggregate quality metrics
```

The BentoML examples should include quality metrics in returned dictionaries.

Quality evaluation should be easy to disable in the runner/example code.

---

## Benchmark integration

Update `inference_benchmark.py` so benchmarks can measure the overhead of quality evaluation.

Requirements:

- quality evaluation enabled by default in benchmark config/CLI;
- provide explicit disable flag, e.g. `--disable-quality-evaluation`;
- record quality timing and metric fields in benchmark records;
- log quality timing and aggregate metrics to MLflow via existing `mlflow.py` helpers;
- include quality fields in benchmark CSV/JSON reports.

Suggested benchmark fields:

```text
quality_evaluation_enabled
quality_evaluation_sec
quality_psnr_mean
quality_ssim_mean
quality_triplets_written
quality_triplet_output_dir
quality_error
```

This is needed to compare:

```text
inference without quality evaluation
vs
inference with sampled quality evaluation
```

Do not introduce a separate benchmark implementation. Extend the current one.

---

## CLI behavior

For existing local inference CLI commands:

- add an option to enable quality evaluation;
- keep quality evaluation disabled by default.

For benchmark CLI commands:

- quality evaluation enabled by default;
- add a disable flag.

Exact flag names can follow existing CLI conventions.

---

## Batch inference

Do not use batch inference for quality evaluation in this task.

Quality evaluation should use sequential Practical-RIFE inference for now.

Keep future batched evaluation possible, but do not implement it now.

Do not remove existing batch inference code.

---

## Tests

Add focused tests.

Tests should cover:

- sample-count as upper bound;
- too-short video behavior;
- deterministic sampling with seed;
- scene-cut SSIM filtering;
- Vimeo-style directory writing:
  - `<source_video_id>/<triplet_id>/im1.png`
  - `<source_video_id>/<triplet_id>/im2.png`
  - `<source_video_id>/<triplet_id>/im3.png`
- no metadata files written;
- reuse of existing metrics functions;
- aggregate `quality_psnr_mean` and `quality_ssim_mean`;
- `quality_triplets_written`;
- `quality_evaluation_sec` in timing;
- inference config validation;
- local inference default: quality disabled;
- benchmark default: quality enabled;
- benchmark disable flag behavior;
- serving runner integration;
- quality evaluation can be disabled;
- fail policy behavior:
  - no valid triplets -> metrics `None`, triplet count `0`;
  - warning policy does not fail successful inference.

Use fake/lightweight model/adapter where possible.

Do not require CUDA for unit tests.

If real-video smoke tests are added, keep them tiny.

---

## Documentation

Update human-facing documentation.

Document:

- quality evaluation is part of inference pipeline, not BentoML-specific logic;
- BentoML uses the existing inference interface;
- quality evaluation always uses 2x / `t=0.5`;
- main inference may use `interpolation_factor=2..4`;
- quality evaluation ignores the main interpolation factor;
- sampled triplet logic;
- `sample_count=16` default and upper-bound semantics;
- scene-cut SSIM threshold `0.75`;
- output triplet folder structure;
- no metadata files are written;
- PSNR/SSIM are returned in `VideoInferenceResult` and service responses;
- `quality_evaluation_sec` and benchmark overhead measurement;
- benchmark default enabled / disable flag;
- local inference default disabled / enable flag;
- MinIO upload is external to this project component;
- how to disable quality evaluation;
- why batch inference is not used for quality evaluation yet.

Update:

- `.agent/docs/PROJECT_MAP.md`;
- relevant Stage 2 docs;
- BentoML example docs if present;
- benchmark docs/config README if affected;
- active ExecPlan if this task belongs to a current milestone or follow-up.

---

## Validation

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

Run only safe smoke checks.

Do not run long videos.

---

## End-of-task summary

At the end, summarize:

- what module/package structure was added or changed;
- where quality evaluation logic lives;
- what inference config/result fields were added;
- how sampled triplets are selected;
- how triplets are saved;
- how PSNR/SSIM are computed and returned;
- how serving uses the feature;
- how benchmark overhead is measured;
- how quality evaluation is enabled/disabled in:
  - local inference;
  - benchmark;
  - serving;
- what tests were added;
- what validation commands were run;
- what docs/project-map/ExecPlan updates were made;
- limitations and follow-up work.
