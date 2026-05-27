# PLANS.md

## Purpose

This document defines how coding agents must create, maintain, and execute **ExecPlans** for this repository.

An **ExecPlan** is an agent-authored engineering plan and execution log for one concrete project stage or substantial implementation task. It is not the human project plan itself. It translates the human-written stage plan into a concrete engineering plan for implementation, validation, progress tracking, decisions, and handoff.

In this repository:

- `.agent/AGENTS.md` defines stable project-wide behavior rules for agents;
- `.agent/global_plan.md` defines the high-level project vision, stages, architecture context, and global constraints;
- `.agent/stage_plans/*.md` define what each development stage must build;
- `.agent/docs/PROJECT_MAP.md` is the compact repository index;
- `.agent/docs/PLANS.md` defines how agents create and maintain ExecPlans;
- each ExecPlan defines how an agent will implement a specific stage or substantial task and records execution progress.

The human-authored project plans say **what** must be built.  
The ExecPlan says **how the agent will build, validate, and hand off that work**.

---

## Repository Workflow Model

This project uses three documentation layers.

### 1. Stable instruction layer

These files define general rules and navigation support:

```text
.agent/AGENTS.md
.agent/docs/PLANS.md
.agent/docs/PROJECT_MAP.md
```

They define:

- agent behavior rules;
- execution planning protocol;
- source-of-truth rules;
- repository navigation conventions;
- project map maintenance rules.

### 2. Human-authored project planning layer

These files define project intent and scope:

```text
.agent/global_plan.md
.agent/stage_plans/*.md
```

They define:

- final project goal;
- global architecture direction;
- major development stages;
- stage-specific requirements;
- non-negotiable technical decisions;
- stage acceptance expectations.

`global_plan.md` describes the whole project at a high level.  
A file in `.agent/stage_plans/` describes the detailed scope and constraints of one stage.

### 3. Agent-authored execution layer

These files are created and maintained by the coding agent:

```text
.agent/docs/exec-plans/active/*.md
.agent/docs/exec-plans/completed/*.md
```

They define:

- engineering decomposition;
- implementation sequence;
- module boundaries;
- validation strategy;
- progress and current state;
- decisions and discoveries;
- handoff notes.

ExecPlans must be restartable. A future agent session should be able to resume work using the repository, the relevant human-authored plans, `PROJECT_MAP.md`, and the active ExecPlan without relying on hidden memory.

---

## Priority and Source-of-Truth Rules

When creating or executing an ExecPlan, apply the following precedence rules.

### For project scope and acceptance

Use this order:

1. direct user instruction in the current task;
2. the current stage plan in `.agent/stage_plans/`;
3. `.agent/global_plan.md`;
4. `.agent/AGENTS.md`;
5. accepted repository code, artifacts, and documentation from completed prior work;
6. the active ExecPlan.

If an ExecPlan conflicts with a human-authored stage plan, the stage plan wins unless the user explicitly approves the change.

If a stage plan conflicts with `global_plan.md`, the stage plan wins for that stage because it is more specific.

### For engineering workflow and repository conventions

Use this order:

1. direct user instruction in the current task;
2. `.agent/AGENTS.md`;
3. `.agent/docs/PLANS.md`;
4. `.agent/docs/PROJECT_MAP.md`;
5. the active ExecPlan.

### For implementation details

Implementation details must come from the current stage plan and active ExecPlan. Do not move detailed stage-specific requirements into `AGENTS.md` or `PLANS.md`.

---

## When an ExecPlan Is Required

Create an ExecPlan for:

- every numbered project stage;
- any substantial feature spanning multiple files or modules;
- any non-trivial refactor;
- any task requiring validation, handoff, or multi-session continuity;
- any integration task involving external model repositories, MLflow, BentoML, Docker Compose, database/storage services, or model artifacts.

For the main project stages, recommended names are:

```text
.agent/docs/exec-plans/active/01_ml_core.execplan.md
.agent/docs/exec-plans/active/02_service_backend.execplan.md
.agent/docs/exec-plans/active/03_retraining_monitoring.execplan.md
.agent/docs/exec-plans/active/04_frontend_demo.execplan.md
```

When a stage is complete and accepted by the user, move its ExecPlan to:

```text
.agent/docs/exec-plans/completed/
```

There should normally be at most one active ExecPlan for the current stage unless the user explicitly requests subplans.

---

## Required Working Phases

Each stage or substantial implementation task must be handled in three phases.

### Phase 1 — Planning only

The agent must:

- read `.agent/AGENTS.md`;
- read `.agent/docs/PLANS.md`;
- read `.agent/docs/PROJECT_MAP.md` if it exists;
- read `.agent/global_plan.md`;
- read the relevant stage plan from `.agent/stage_plans/`;
- read the previous completed ExecPlan handoff if the current work depends on prior work;
- inspect relevant repository files, configs, scripts, model repositories, datasets, and artifacts;
- identify missing prerequisites and dependencies;
- create the active ExecPlan in `.agent/docs/exec-plans/active/`;
- not modify production code yet.

The output of this phase is a reviewable ExecPlan ready for implementation.

### Phase 2 — Implementation from plan

The agent must:

- re-read the active ExecPlan and current stage plan;
- implement according to both;
- update the ExecPlan continuously while working;
- update `.agent/docs/PROJECT_MAP.md` when important files, modules, commands, or configs are added or changed;
- run relevant validation checks and smoke runs;
- record decisions, discoveries, blockers, and partial outcomes.

### Phase 3 — Closeout and handoff

The agent must:

- finalize the `Outcomes & Handoff` section;
- record what was implemented and validated;
- record unresolved issues and deferred work;
- update human-facing documentation under `docs/` when commands, configs, services, or workflows were added;
- update `.agent/docs/PROJECT_MAP.md`;
- move the ExecPlan to `completed/` only when the stage or task is finished and accepted by the user.

Do not skip dependencies between stages. Do not silently invent missing outputs. If a required artifact is absent, document the blocker and implement the minimal prerequisite only if it is clearly within the current scope.

---

## What the Agent Must Read Before Creating an ExecPlan

When starting a stage or substantial task, the agent must read:

1. `.agent/AGENTS.md`;
2. `.agent/docs/PLANS.md`;
3. `.agent/docs/PROJECT_MAP.md`, if present;
4. `.agent/global_plan.md`;
5. the current `.agent/stage_plans/*.md` file;
6. the `Outcomes & Handoff` section of the immediately previous completed ExecPlan, if it exists and is relevant;
7. the `Decision Log` of the previous completed ExecPlan if the current work depends on earlier decisions;
8. relevant source files, configs, tests, Docker files, model repository files, dataset manifests, and artifacts.

The agent should not reread every prior ExecPlan in full unless a dependency is unclear or the current task explicitly builds on unfinished prior work.

---

## What an ExecPlan Must Achieve

An ExecPlan must be sufficient for a future agent session or developer to resume work using only:

- the current repository;
- the active ExecPlan;
- `.agent/AGENTS.md`;
- `.agent/docs/PLANS.md`;
- `.agent/docs/PROJECT_MAP.md`;
- `.agent/global_plan.md`;
- the relevant stage plan;
- completed prior handoff notes.

It must not rely on hidden memory.

An ExecPlan must:

- preserve the stage goal;
- preserve required functionality and interfaces;
- preserve artifact expectations;
- preserve validation expectations;
- translate the stage plan into a concrete implementation sequence;
- record progress, discoveries, decisions, and handoff.

It may add:

- module decomposition;
- file-level implementation order;
- smoke-run plan;
- useful behavioral tests;
- risk mitigation;
- fallback sequencing;
- implementation milestones.

It may not:

- silently remove or weaken stage requirements;
- change selected tools, models, data formats, or architecture decisions unless approved by the user;
- expand scope into later stages without explicit instruction;
- replace selected project tools with custom alternatives without recording and justifying the change.

---

## Required Sections in Every ExecPlan

Every ExecPlan must include all sections below.

# 1. Title and Metadata

Include:

- Stage:
- Status:
- Created:
- Updated:
- Stage plan:
- Global plan:
- Prior ExecPlan:
- Scope authority:

# 2. Stage Goal

Summarize the purpose of the stage in implementation language.

Explain:

- what this stage must produce;
- what this stage is not supposed to do yet;
- how it fits into the full project pipeline.

# 3. Source Documents and Authority

List the documents used to create the ExecPlan.

At minimum include:

- `.agent/AGENTS.md`;
- `.agent/docs/PLANS.md`;
- `.agent/docs/PROJECT_MAP.md`, if present;
- `.agent/global_plan.md`;
- the current `.agent/stage_plans/*.md` file;
- prior completed ExecPlan handoff if applicable;
- relevant source files already present.

State which document controls scope for this stage.

# 4. Context and Current Repository State

Describe the current repository situation as if the reader is new.

Include:

- relevant directories and modules;
- what already exists;
- what is missing;
- external model repositories and weights relevant to the stage;
- dataset files or manifests relevant to the stage;
- expected artifacts from previous stages;
- assumptions about the current tree.

Use full paths for important files.

# 5. Stage Requirements Restated

Restate the current stage requirements in execution-ready form.

Do not copy the entire stage plan. Extract the obligations that matter for implementation:

- required functionality;
- required modules/services;
- required configs;
- required artifacts;
- required interfaces;
- required validation behavior;
- mandatory constraints;
- non-negotiable user decisions.

# 6. Non-Goals and Deferred Work

State what must not be implemented in this stage.

Also state what may be postponed safely.

This section is mandatory to prevent scope creep.

# 7. Architecture and Implementation Strategy

Explain how the stage will be implemented.

Include, where relevant:

- modules and responsibilities;
- data flow;
- interface boundaries;
- file-level or module-level edit plan;
- code reuse strategy;
- model repository adaptation strategy;
- Docker/service integration strategy;
- MLflow/BentoML integration strategy;
- local vs service runtime expectations.

This must be concrete enough to guide implementation but should not become a vague research essay.

# 8. Milestones and Work Breakdown

Break the work into ordered milestones.

Each milestone should include:

- objective;
- likely files/modules involved;
- expected output;
- validation checkpoint.

Milestones should be small enough to support multi-session development.

# 9. Validation Strategy

Define how the stage will be validated.

The validation strategy must focus on meaningful behavior, not superficial tests. It must identify what should be checked by automated tests, what should be checked by smoke runs, and what should be checked manually.

The section must state what should pass when the stage is complete.

# 10. Expected Artifacts

List artifacts expected at completion.

Examples relevant to this project include:

- source modules;
- configs;
- CLI entrypoints;
- dataset indexes and manifests;
- dataset version directories;
- MLflow runs and artifacts;
- model checkpoints;
- candidate validation reports;
- BentoML model/service artifacts;
- Docker Compose files;
- API/service definitions;
- documentation.

Only include artifacts relevant to the current stage.

# 11. Risks, Assumptions, and Recovery

List:

- important assumptions;
- likely failure modes;
- fragile dependencies;
- external model repository risks;
- GPU/runtime risks;
- data or artifact risks;
- fallback plans;
- retry-safe operations;
- what to do if external services are unavailable.

### Dependency Decisions

When a milestone requires functionality normally provided by a selected project tool, the agent must check whether the corresponding dependency is already present.

If the dependency is missing, the agent must choose one of:

1. add the dependency if it is official/trusted and already implied by the stage plan or tech stack;
2. ask the user before adding it if it is heavy, obscure, GPU-sensitive, or architecture-changing;
3. implement a temporary local fallback only if the dependency cannot be installed or the fallback is explicitly documented as temporary.

The agent must not silently replace an intended external tool with a custom implementation unless the ExecPlan records:

- why the dependency was not added;
- what compatibility guarantees the custom implementation provides;
- how the implementation will be validated against the intended tool;
- whether the dependency should be added in a later milestone.

If dependency resolution creates a compatibility conflict with model repositories, the agent must ask the user before downgrading packages or rewriting model code for a newer package version.

# 12. Progress

Mandatory. Update during implementation.

Use timestamped or ordered entries.

Each entry should state:

- what was completed;
- what remains next;
- whether any plan change is required.

Do not leave this section stale.

# 13. Surprises & Discoveries

Mandatory. Update when implementation reveals something non-obvious.

Examples:

- model repository assumptions differ from README instructions;
- training script requires a different dataset layout than expected;
- checkpoint format differs from expected format;
- MLflow artifact layout differs from expected format;
- BentoML packaging requires a separate wrapper;
- a dependency version breaks a model repository import;
- GPU memory behavior changes the planned batch size.

Keep entries concise and evidence-oriented.

# 14. Decision Log

Mandatory.

Record every meaningful implementation decision affecting:

- architecture;
- sequencing;
- test strategy;
- defaults;
- fallback behavior;
- scope interpretation;
- model repository adaptation;
- dependency selection;
- config/interface behavior.

Each entry should say:

- what decision was made;
- why;
- alternatives considered when relevant;
- downstream implication.

# 15. Outcomes & Handoff

Mandatory.

At completion or pause, summarize:

- what was delivered;
- what was validated;
- what remains deferred;
- what the next stage/session should know;
- important files, configs, commands, tests, and artifacts.

This is the main continuity bridge to the next stage or next agent run.

---

## Validation Policy

Validation must be useful and proportional to the project stage.

The agent must not create large numbers of shallow tests that only check hardcoded config values, trivial field presence, or implementation details with no meaningful behavior. Tests should not exist merely to increase test count.

Automated tests should be written only for critical or failure-prone logic whose correctness affects the project pipeline.

Examples of useful tests in this project:

- scene interval filtering does not sample across scene boundaries;
- sequence sampler respects sequence length and frame-step constraints;
- static triplet filtering rejects near-identical frames when required;
- split logic does not place the same source video in multiple splits;
- dataset version builder creates valid manifests from realistic small inputs;
- `UniversalTripletDataset` loads real/synthetic PNG triplets with correct shapes;
- baseline metric computation gives sane results on controlled synthetic samples;
- candidate validation applies configured acceptance thresholds correctly;
- service-level smoke tests confirm API/task/inference paths work after those stages exist.

Examples of tests to avoid:

- tests that only assert a config key equals a hardcoded default;
- tests that only check that a dataclass field exists;
- tests that duplicate implementation logic without validating behavior;
- tests that require large datasets, full model training, or long GPU jobs;
- tests that mock everything so heavily that no real workflow behavior is checked.

Validation should combine:

- small targeted tests for critical logic;
- smoke runs for end-to-end workflows;
- artifact checks for generated manifests, reports, checkpoints, and MLflow/BentoML artifacts;
- manual inspection when visual/video quality is relevant.

Each ExecPlan must state which validation type is appropriate for each milestone.

A stage is not complete simply because code exists. It is complete when the required workflow can be run, expected artifacts are produced, validations are recorded, documentation is updated, and handoff notes are current.

---

## ExecPlan Writing Rules

### Write for a fresh reader

Assume the next agent has no memory beyond the repository and the ExecPlan.

### Be concrete

Name files, modules, services, configs, and commands explicitly where possible.

Avoid vague statements such as:

- “improve the pipeline”;
- “add tests”;
- “make deployment work”;
- “integrate the model”.

Instead, state which module, command, adapter, config, artifact, or service will be changed.

### Preserve stage intent

Do not reinterpret the stage into a different task because it is easier.

### Separate mandatory and optional work

If a stage plan contains optional enhancements, label them clearly. The required route must remain executable without optional work.

### Keep it restartable

At any stopping point, the ExecPlan must say:

- what has been done;
- what is next;
- what changed from the original plan;
- what to inspect first to resume.

### Avoid hidden assumptions

If an implementation depends on a local path, GPU assumption, model weight, dataset file, or service endpoint, record it explicitly in the ExecPlan and, where appropriate, in config/docs.

---

## ExecPlan Update Rules During Implementation

Update the active ExecPlan whenever:

- a milestone is completed;
- a blocker appears;
- an assumption changes;
- a validation result changes the implementation path;
- a service integration detail is discovered;
- a model repository compatibility detail is discovered;
- a dependency decision is made;
- a session ends before the stage is complete.

Every implementation session must leave `Progress` current.

Every meaningful course correction must appear in `Decision Log`.

Every non-obvious discovery must appear in `Surprises & Discoveries`.

Never silently drift away from the plan.

---

## Planning-Only Prompt Behavior

When asked to create or update an ExecPlan in planning mode, the agent must:

- read the required documents;
- inspect the repository;
- create or update the ExecPlan;
- not modify production code;
- not create speculative implementation artifacts.

Planning mode output should be a reviewable ExecPlan.

---

## Implementation Prompt Behavior

When asked to implement from an ExecPlan, the agent must:

- read `.agent/AGENTS.md`;
- read `.agent/docs/PLANS.md`;
- read `.agent/docs/PROJECT_MAP.md`, if present;
- read `.agent/global_plan.md`;
- read the current stage plan;
- read the active ExecPlan;
- implement the requested milestone or subset;
- update the ExecPlan while working;
- update `PROJECT_MAP.md` when important structure changes;
- run relevant validation;
- not jump to the next stage unless explicitly instructed.

If ambiguity appears, resolve it conservatively in favor of:

1. preserving the current user instruction;
2. preserving the stage plan;
3. preserving reproducibility;
4. preserving compatibility with later stages.

If a true blocker remains, document it in the ExecPlan and ask for direction.

---

## Human-Facing Documentation Expectations

ExecPlans are not a substitute for human-facing documentation.

When a stage or milestone adds commands, configs, services, workflows, or artifacts, update documentation under `docs/`.

ExecPlans for milestones that add user-facing CLI workflows should explicitly account for human-facing documentation, readable CLI logging, and progress indicators for long-running commands.

Documentation must explain:

- what was added;
- required dependencies/services;
- how to run the workflow;
- required inputs and their expected formats;
- generated outputs and where they are stored;
- CLI parameters and their allowed values;
- how each important parameter affects behavior;
- side effects of commands;
- success criteria;
- known limitations;
- troubleshooting notes where relevant.

For CLI interfaces, document:

- command name;
- purpose;
- required arguments;
- optional arguments;
- allowed values;
- defaults;
- examples;
- expected output files/artifacts.

For config files, document:

- purpose;
- expected types;
- allowed values;
- defaults;
- whether changing the field requires rebuilding datasets, retraining models, repackaging BentoML services, or restarting services.

Documentation should be sufficient for the project owner to run, configure, inspect, and validate the delivered workflow without reverse-engineering commands from source code.

---

## Completion Criteria for an ExecPlan

An ExecPlan is complete only when:

- the stage or substantial task is implemented or intentionally paused in a well-defined state;
- `Progress` is current;
- `Surprises & Discoveries` records non-obvious findings;
- `Decision Log` explains major course corrections;
- `Outcomes & Handoff` explains what the next stage/session needs to know;
- validations and artifacts are recorded;
- `PROJECT_MAP.md` is updated when relevant;
- human-facing documentation is updated where applicable.

A stale or unfinished ExecPlan is a process failure.

---

## Template for New ExecPlans

Use this skeleton.

# Title and Metadata

- Stage:
- Status:
- Created:
- Updated:
- Stage plan:
- Global plan:
- Prior ExecPlan:
- Scope authority:

## Stage Goal

## Source Documents and Authority

## Context and Current Repository State

## Stage Requirements Restated

## Non-Goals and Deferred Work

## Architecture and Implementation Strategy

## Milestones and Work Breakdown

### Milestone 1

### Milestone 2

### Milestone 3

## Validation Strategy

## Expected Artifacts

## Risks, Assumptions, and Recovery

## Progress

- Not started yet.

## Surprises & Discoveries

- None yet.

## Decision Log

- No decisions recorded yet.

## Outcomes & Handoff

- Not completed yet.

---

## Expected Task Invocation Modes

This section is an instruction for agents, not a required prompt for the user to copy verbatim.

### Planning Mode

When a task is phrased as planning-only work, the agent must:

- read the required documents;
- inspect relevant repository files;
- create or update the active ExecPlan;
- not modify production code.

### Implementation Mode

When a task is phrased as implementation work, the agent must:

- read the required documents;
- read the active ExecPlan;
- implement only the requested scope or ExecPlan milestone;
- update the ExecPlan while working;
- run relevant validation;
- stop with a clear handoff if the stage is not complete.

---

## Final Principle

The human defines project intent through `global_plan.md` and stage plans.  
The agent defines the engineering execution plan through ExecPlans.  
The agent then executes that plan while keeping it current.

This separation preserves human control over scope while allowing multi-session implementation.
