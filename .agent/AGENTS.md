---
name: "Video Frame Interpolation ML Service"
description: "University MLOps project for a video frame interpolation service with data preparation, model training/fine-tuning, experiment tracking, inference serving, and later web deployment."
category: "Lab work"
author: "Dmitry Vespin"
tags: ["CV", "VFI", "MLOps", "PyTorch", "MLflow", "BentoML", "Video Processing"]
lastUpdated: "2026-05-20"
---

# AGENTS.md

## Purpose of This File

This file defines stable repository-wide instructions for coding agents working on this project.

It is not a detailed implementation plan. Detailed technical requirements, algorithms, data formats, model-specific decisions, schemas, and per-stage task breakdowns belong in:

- `.agent/docs/general_plan.md` — global project plan and high-level architecture;
- `.agent/stage_plans/*.md` — detailed plans for specific development stages;
- `.agent/docs/exec-plans/*` — agent-authored execution plans for concrete implementation work.

---

## Project Overview

This repository implements a university MLOps project for **video frame interpolation**.

The final goal is a service where a user uploads a short video, the system inserts generated intermediate frames between original frames, and returns a processed video with increased frame rate.

The project combines:

- video data preprocessing;
- dataset versioning;
- training and fine-tuning of VFI models;
- experiment tracking and model registry;
- local and service-based inference;
- asynchronous backend processing;
- later monitoring, retraining triggers, and demonstration workflows.

The first development priority is to build a working ML core before implementing the full web service.

---

## Current Repository Context

At the beginning of implementation, the repository may already contain:

```text
datasets/
  anime/
  vimeo_triplet/
  ucf101_interp_ours/
  SNU_FILM/

model_repos/
  AMT/
  EMA-VFI/
  Practical-RIFE/

model_weights/
  AMT/
  EMA-VFI/
  Practical-RIFE/
```

The directories under `model_repos/` contain cloned external model implementations. They are part of the project workspace and may be adapted when needed, but they must be treated carefully as upstream model code.

The directories under `model_weights/` contain pretrained model weights and related model files for respective `model_repos/`'s models.

---

## Agent Instruction Files

The agent must understand and use the project instruction hierarchy.

### Repository-wide instruction files

```text
.agent/AGENTS.md
.agent/docs/PLANS.md
.agent/docs/PROJECT_MAP.md
.agent/docs/general_plan.md
```

- `AGENTS.md` — stable project-wide rules, workflow, tool stack, and agent behavior.
- `PLANS.md` — protocol for creating and maintaining ExecPlans.
- `PROJECT_MAP.md` — compact index of the repository structure.
- `general_plan.md` — high-level project plan, stages, architecture, major components, and global constraints.

### Stage-specific planning files

```text
.agent/stage_plans/
```

Stage plans are human-authored detailed specifications for individual development stages. They contain implementation decisions, technical requirements, algorithms, interfaces, schemas, and acceptance criteria for that stage.

### Agent-authored execution plans

```text
.agent/docs/exec-plans/active/
.agent/docs/exec-plans/completed/
```

ExecPlans are agent-authored working documents. They translate a human stage plan into a concrete implementation sequence with validation steps, progress logs, decision logs, and handoff notes.

---

## Source of Truth and Precedence

For project scope and acceptance, use this priority order:

1. direct user instruction in the current task;
2. current human-authored stage plan in `.agent/stage_plans/`;
3. `.agent/docs/general_plan.md`;
4. this `.agent/AGENTS.md`;
5. accepted existing code and artifacts.

For engineering workflow and execution discipline, use this priority order:

1. direct user instruction in the current task;
2. `.agent/AGENTS.md`;
3. `.agent/docs/PLANS.md`;
4. active ExecPlan.

If a human-authored plan conflicts with an ExecPlan, the human-authored plan wins unless the user explicitly approves a change.

If a user instruction conflicts with any file, the user instruction wins.

---

## Required Development Workflow

The project must be developed stage by stage.

For every numbered stage, the agent must work in three phases.

### Phase 1 — Planning Only

Before implementation, the agent must:

1. read `.agent/AGENTS.md`;
2. read `.agent/docs/PLANS.md`;
3. read `.agent/docs/PROJECT_MAP.md` if it exists;
4. read `.agent/docs/general_plan.md`;
5. read the current stage plan in `.agent/stage_plans/`;
6. inspect the repository state;
7. inspect relevant existing code, configs, datasets, and model repositories;
8. create an ExecPlan in `.agent/docs/exec-plans/active/`;
9. not begin implementation until the ExecPlan exists.

### Phase 2 — Implementation from the ExecPlan

During implementation, the agent must:

1. follow the current stage plan and active ExecPlan;
2. implement the smallest coherent unit that can be validated;
3. keep the ExecPlan current;
4. record progress, decisions, discoveries, and blockers;
5. run relevant checks and smoke tests;
6. update `.agent/docs/PROJECT_MAP.md` when important files or directories are added or changed;
7. update human-facing documentation when user-facing commands, configs, services, or workflows are added.

### Phase 3 — Closeout and Handoff

At the end of a stage or accepted milestone, the agent must:

1. summarize what was implemented;
2. summarize what was validated;
3. record unresolved issues and deferred work;
4. update `.agent/docs/PROJECT_MAP.md`;
5. update human-facing documentation if needed;
6. leave the repository restartable for the next session;
7. move the ExecPlan from `active/` to `completed/` only when the stage is done and accepted by the user.

Do not skip the planning phase. Do not silently jump to later stages.

---

## High-Level Development Stages

Detailed stage requirements belong in `.agent/stage_plans/`. This section only gives a high-level orientation.

### Stage 1 — ML Core

Build the local ML core:

- data preprocessing and dataset indexing;
- dataset version building;
- PyTorch dataset loading;
- baseline evaluation;
- model training/fine-tuning;
- candidate validation;
- MLflow experiment tracking;
- local video inference.

### Stage 2 — Backend and Inference Serving

Wrap the ML core into service components:

- BentoML-based inference service for models;
- FastAPI backend for non-inference application logic;
- asynchronous request processing;
- object storage and database integration;
- worker containers;
- Docker Compose runtime.

BentoML is the preferred tool for model serving, inference API, and model deployment packaging. FastAPI is used for application-level APIs outside model inference.

### Stage 3 — Retraining and Monitoring

Add model lifecycle automation:

- online quality history;
- online data package collection;
- retraining/fine-tuning request flow;
- trigger logic;
- candidate validation;
- model promotion/rollback;
- operational monitoring and alerts.

### Stage 4 — Frontend and Demonstration

Add the user-facing demonstration layer:

- minimal frontend;
- video upload;
- processing status;
- download link;
- controlled degradation/retraining demo.

---

## Global Architecture Direction

The full project is expected to evolve into a staged MLOps system with these major areas:

- access layer;
- orchestration and scheduling;
- compute layer with worker nodes;
- storage and model management;
- dataset sources and preprocessing;
- model training/fine-tuning;
- model serving;
- monitoring and retraining logic.

The high-level architecture and data flow are documented in `.agent/docs/general_plan.md`.

---

## Global Technical Stack

Use the following project-wide stack unless a user-approved plan changes it.

### Core language and tooling

- Python
- uv
- pytest
- ruff
- Typer for developer CLI commands
- Rich for developer-facing console output
- python-dotenv
- pydantic
- pydantic-settings

### ML and computer vision

- PyTorch
- torchvision
- Pillow
- OpenCV
- PyAV
- FFmpeg
- PySceneDetect
- LPIPS where needed
- numpy
- pandas
- tqdm

### Experiment tracking and model lifecycle

- MLflow for experiment tracking and model registry;
- BentoML for model serving, inference API, model store, and deployment packaging.

Training and experimentation artifacts are tracked in MLflow. Approved release models may be exported from MLflow and registered in the BentoML Model Store for inference service deployment.

### Backend, storage, and orchestration

- FastAPI for non-inference backend APIs;
- Celery for asynchronous task execution;
- Redis as the broker;
- PostgreSQL for relational metadata;
- SQLAlchemy;
- psycopg with binary extras;
- MinIO as local S3-compatible object storage;
- Docker Compose for local infrastructure and deployment-oriented services.

### Networking and deployment

- Docker Compose for MVP deployment;
- Kubernetes is not part of the MVP unless the user explicitly changes the plan.

---

## Dependency Policy

The agent may add dependencies only when they are required by the current stage plan or directly support the approved project stack.

Use `uv` for dependency management.

When adding dependencies, the agent must:

1. add them through `uv`;
2. document the reason in the active ExecPlan;
3. update human-facing documentation if the dependency changes setup or runtime commands.

The project should include the common dependencies required for the selected model repositories and project infrastructure, including:

```text
torch
torchvision
opencv-python
Pillow
av
scenedetect
mlflow
bentoml
lpips
numpy
pandas
tqdm
typer
rich
pydantic
pydantic-settings
python-dotenv
sqlalchemy
psycopg[binary]
```

The model repositories have their own dependency descriptions:

```text
model_repos/AMT/environment.yaml
model_repos/EMA-VFI/README.md
model_repos/Practical-RIFE/requirements.txt
```

Do not blindly install these files as-is. Instead:

1. inspect them;
2. compile a unified dependency list;
3. prefer current compatible package versions;
4. resolve the environment through `uv`;
5. only pin or downgrade packages when necessary.

Do not install `cudatoolkit` as a Python dependency. Use the CUDA/PyTorch build appropriate for the machine. If the project uses a custom PyTorch index in `pyproject.toml`, preserve that configuration unless the user changes it.

If a compatibility problem appears between a model repository and installed package versions, the agent must ask the user before choosing between:

- adapting the model code to the newer package version;
- downgrading/pinning the dependency.

The agent must not silently downgrade important packages.

---

## Model Repository Policy

Model repositories under `model_repos/` are external model implementations integrated into this project.

Rules:

- Treat each model repository as an external dependency module.
- Use the repository's existing training and inference logic where practical.
- Prefer adding project-level wrappers/adapters around model repositories.
- Adapt model repository code only when needed for integration.
- Keep modifications minimal and documented.
- Do not perform large unreviewed rewrites inside model repositories.
- Do not delete upstream files.
- Do not move model repository files unless the user explicitly approves.
- If model repository code is changed, record the reason and changed files in the active ExecPlan.

The goal is not to rewrite AMT, EMA-VFI, or Practical-RIFE from scratch. The goal is to integrate them into the project pipeline while preserving their working model logic.

---

## Runtime and Containerization Policy

Stage 1 development is primarily local Python development, but MLflow infrastructure should be containerized early.

For Stage 1, use Docker Compose for:

- MLflow tracking server;
- PostgreSQL for MLflow metadata;
- MinIO for MLflow artifacts and future object storage (Model Registry).

Later stages may reuse the same MinIO instance for training data, uploads, outputs, and service artifacts.

Do not require the full application stack to run before the ML core works.

For later stages, Docker Compose should manage:

- FastAPI backend;
- BentoML inference service;
- Celery workers;
- Redis;
- PostgreSQL;
- MinIO;
- MLflow;
- frontend when added.

When code runs inside Docker Compose, use service names for internal networking. When code runs locally on the host, use exposed localhost ports.

---

## Environment and Secrets

Use `.env` for local environment variables and secrets.

Expected environment variables may include:

```text
MLFLOW_TRACKING_URI=...
DATABASE_URL=...
MINIO_ENDPOINT=...
MINIO_ROOT_USER=...
MINIO_ROOT_PASSWORD=...
REDIS_URL=...
DATASET_ROOT=...
MODEL_WEIGHTS_ROOT=...
BENTOML_HOME=...
```

Rules:

- `.env` must not be committed.
- `.env.example` should contain placeholder keys.
- Secrets must not be printed in logs.
- Credentials must not be embedded in source code, configs committed to Git, Dockerfiles, stage plans, ExecPlans, or documentation.

---

## Project Structure Policy

The repository structure may evolve, but it must remain easy to navigate.

A typical structure may include:

```text
.agent/
configs/
datasets/
raw_data/anime/
docs/
model_repos/
model_weights/
scripts/
src/
tests/
docker-compose.yml
pyproject.toml
uv.lock
```

Reusable project code should live under `src/`.

Runnable developer entrypoints may live under `scripts/` or be exposed through package CLI commands.

Human-facing documentation belongs under `docs/`.

Agent instruction and planning files belong under `.agent/`.

Training/testing datasets should be stored inside `datasets/`. Here lives main training dataset - Vimeo-90K - as well as new user's training samples. Also this directory contains other well-known datasets for model evaluation.

Unprocessed raw video files to process into training triplets are stored in `raw_data/`. Currently only one domain specification exists inside - anime domain inside `raw_data/anime/` subdirectory.

Avoid putting core implementation in notebooks.

Do not force a large reorganization unless the active stage plan requires it or the user approves it.

---

## PROJECT_MAP.md Policy

`.agent/docs/PROJECT_MAP.md` is a compact index of the repository.

It should help future agents quickly find relevant files without reading the entire codebase.

The agent must update `PROJECT_MAP.md` when:

- new top-level directories are added;
- new important modules are added;
- a module responsibility changes;
- a new CLI entrypoint appears;
- a new config family appears;
- a stage or milestone is completed.

Keep `PROJECT_MAP.md` compact. It is an index, not full documentation.

Recommended style:

```markdown
## src/<package>/preprocessing/

Video preprocessing and sequence extraction.

- `reader.py` — video reading helpers.
- `sampler.py` — sequence sampling logic.
- `writer.py` — output writing and index creation.
```

---

## Implementation Principles

Before implementing a substantial change:

1. identify the current stage;
2. read the current stage plan;
3. read the active ExecPlan;
4. inspect existing code and configs;
5. implement the smallest coherent unit that can be validated;
6. run relevant checks or smoke tests;
7. update the ExecPlan;
8. update `PROJECT_MAP.md`;
9. update human-facing documentation if user-facing behavior changed.

General engineering rules:

- keep modules focused;
- separate pure logic from side effects;
- separate preprocessing, dataset management, training, evaluation, inference, storage, and service logic;
- keep CLI entrypoints thin and delegate real work to reusable functions/classes;
- prefer explicit configuration over hidden constants;
- avoid hidden global state except centralized settings;
- make file-writing operations safe to rerun where practical;
- keep logs readable;
- use Rich for developer-facing command output when useful;
- **do not add `from __future__ import annotations` automatically to every Python file**.

---

## Coding Style Principles

When writing code, follow general principals and conventions of the programming language the code is written on.

For python:

- follow PEP8;
- always write all imports, used inside code, before the code implementation. Do not hide imports inside function calls;

---

## Validation and Acceptance Policy

Use tests sparingly and meaningfully.

Do not create tests that only check hardcoded config values or trivial field equality. Such tests add maintenance burden without validating real behavior.

Write tests for critical functionality where failure would break the pipeline, such as:

- sequence sampling correctness;
- no sampling across invalid scene boundaries;
- static triplet filtering;
- deterministic split assignment when required;
- manifest generation behavior;
- dataset loading and tensor shape correctness;
- metric computation on known small inputs;
- model adapter smoke behavior when feasible.

Prefer smoke tests for full workflows:

- small preprocessing run;
- small dataset version build;
- DataLoader smoke run;
- baseline evaluation on a tiny manifest;
- local inference on a very short video;
- minimal MLflow logging run.

A stage is not complete just because code exists. It is complete when the intended run path works, artifacts are produced, validations are recorded, `PROJECT_MAP.md` is updated, and handoff documentation exists.

---

## Human-Facing Documentation

ExecPlans are for agent continuity. The repository must also contain concise human-facing operational documentation.

After any milestone that introduces or changes workflows, CLI commands, configs, services, or artifacts, update human-facing operational documentation.

Documentation must explain:

- implemented workflows;
- CLI commands;
- config files;
- important parameters;
- allowed values;
- defaults;
- practical effect of each parameter;
- required dependencies and services;
- setup steps;
- input formats;
- output formats;
- side effects of each command;
- where outputs are written;
- how to inspect results;
- how to rerun safely;
- known limitations and troubleshooting notes.

Documentation should be sufficient for the project owner to run, configure, inspect, and validate the implemented workflow without reverse-engineering the source code.

Human-facing documentation belongs primarily under `docs/`, but config documentation may live in `configs/README.md` and, when useful, in local README files inside config/module directories. Central docs may link to these README files instead of duplicating every field. The same local README pattern may be used for other codebase areas when a short directory-level explanation helps developers and future coding agents understand the implemented workflow.

For CLI commands, document:

- command syntax;
- purpose;
- required inputs;
- optional flags;
- allowed/default values;
- example invocations;
- side effects;
- generated outputs;
- how to inspect successful results;
- common failure cases.

For config files, document:

- purpose;
- fields;
- expected type;
- allowed values or expected format;
- default or example value;
- practical effect on behavior;
- whether changing the value requires rerunning preprocessing, rebuilding indexes, rebuilding dataset versions, retraining models, or restarting services.

---

## CLI Logging and Progress Policy

Developer-facing CLI output should be readable and informative.

Use Rich for CLI logs, tables, summaries, status messages, and progress bars.

Long-running operations must show progress when practical. Examples include:

- video preprocessing;
- frame extraction;
- sequence writing;
- source indexing;
- manifest building;
- metric evaluation;
- inference over frames or batches;
- training and evaluation loops.

If an exact progress total is known, use a progress bar. If an exact total is not known, print periodic status updates and a final summary.

CLI commands should print a concise final summary with key counts, output paths, skipped items, warnings, and success/failure status.

---

## What Not To Do

Do not:

- skip the ExecPlan step;
- silently ignore a stage plan requirement;
- implement future-stage services before the current stage works;
- hardcode secrets, absolute local dataset paths, model versions, or dataset versions;
- write large generated datasets, temporary video outputs, or model artifacts into Git unless explicitly requested;
- move core implementation into notebooks;
- introduce Kubernetes into the MVP;
- introduce RabbitMQ unless the user changes the broker decision;
- use WebDataset unless a later plan explicitly reintroduces it;
- silently downgrade dependencies;
- rewrite model repositories from scratch;
- leave MLflow runs, model artifacts, CLI workflows, or config behavior undocumented;
- ask the user to choose between options already fixed in the plans.

## When Blocked

If blocked:

1. identify the missing dependency, data, model weight, environment issue, or ambiguity;
2. inspect whether it should already exist from the current plan or repository state;
3. implement the minimal prerequisite only if it is clearly within scope;
4. otherwise document the blocker in the active ExecPlan and ask for direction.

When assumptions are unavoidable, record them in:

- config;
- code comments where relevant;
- active ExecPlan;
- human-facing docs if they affect operation.

---

## Final Goal

The final repository should allow the project owner or another developer to:

- prepare VFI training data from videos;
- build reproducible dataset versions;
- train and fine-tune VFI models;
- track experiments, metrics, configs, and artifacts with MLflow;
- validate candidate models before release;
- register release models for BentoML serving;
- run local and service-based video interpolation;
- process user video requests asynchronously;
- collect quality metrics and selected samples for future fine-tuning;
- demonstrate model degradation handling and retraining behavior;
- continue development across multiple agent sessions without hidden context.

The repository must remain restartable, inspectable, and reproducible throughout development.
