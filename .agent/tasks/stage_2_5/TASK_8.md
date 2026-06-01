# Start Stage 2.5 Milestone 8 Implementation

Continue Stage 2.5 — ONNX Stabilization, Real-Image Equivalence, and Batched Inference.

Use `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md` as the main working document.

Briefly re-read:

- `.agent/AGENTS.md`
- `.agent/docs/PLANS.md`
- `.agent/docs/PROJECT_MAP.md`
- `.agent/stage_plans/02_5_inference_runtime_stabilization_stage_plan.md`
- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`

Implement only **Milestone 8 — Mini-Benchmarks and MLflow Logging**.

Do not start Milestone 9.

## Task

Add small, repeatable inference benchmark workflows for comparing inference backends, execution modes, interpolation modes, factors, scales and batch sizes.

The benchmark code should use the current Stage 2.5 runtime paths and should not introduce a new inference implementation.

## Scope

Support benchmark runs for the combinations that are currently viable in the active ExecPlan:

- PyTorch sequential inference;
- PyTorch batch inference;
- ONNX sequential inference where the model ONNX path is viable;
- ONNX batch inference where implemented and viable;
- fixed 2x;
- arbitrary Nx;
- EMA-VFI and Practical-RIFE, according to their current supported backends.

Use safe inputs only:

- synthetic tensors;
- real image pairs from `raw_data/pair_test`;
- very short videos from `raw_data/tmp_test` when video-level timing is needed.

Do not run long videos, full directories, or full datasets.

## Metrics

Record useful timing and throughput fields where practical:

- model name;
- backend;
- execution mode;
- interpolation mode;
- interpolation factor;
- batch size;
- input shape or sample id;
- provider and artifact path for ONNX;
- number of source frames / pairs / generated frames;
- decode time;
- preprocessing / tensor conversion time;
- model inference time;
- postprocessing time;
- encode time;
- total time;
- pairs per second;
- generated frames per second;
- peak VRAM if available.

## MLflow

Integrate benchmark logging with the existing project MLflow helpers.

Use `src/video_interpolation/mlflow.py` or add a small benchmark-specific helper there if needed.

Benchmark runs should be able to log:

- params;
- timing metrics;
- CSV/JSON summaries;
- visual/sample artifacts where generated.

Keep `--disable-mlflow` available for local smoke runs when the MLflow server is not running.

Do not require MLflow to be running for basic local benchmark smoke tests.

## Validation

Add focused tests for:

- benchmark config/argument validation;
- timing/metric aggregation;
- CSV/JSON report writing;
- MLflow-disabled behavior;
- backend/execution-mode selection logic using fake/lightweight runtimes where practical.

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

Run at least one small benchmark smoke with `--disable-mlflow`.

If MLflow is available, run one small MLflow logging smoke and record the run/artifacts.

## Updates

Update:

- `.agent/docs/exec-plans/active/02_5_inference_runtime_stabilization.execplan.md`
- `.agent/docs/PROJECT_MAP.md` if repository structure changes
- human-facing docs, especially Stage 2.5 docs and relevant config README files

Document:

- available benchmark commands;
- safe smoke examples;
- output locations;
- MLflow behavior;
- how to interpret timing fields;
- known limitations.

## End-of-task summary

Summarize:

- what benchmark files/commands were added;
- which model/backend/mode combinations are supported;
- what metrics are recorded;
- how MLflow logging was integrated;
- what tests were added;
- what smoke benchmarks were run;
- validation command results;
- docs/project-map/ExecPlan updates;
- what remains for Milestone 9.
