# QuantApprentice_clean Pipeline Mapping

This document maps the original `QuantApprentice_clean` reproducibility package to the studio-side
multi-agent system design.

## Why This Mapping Exists

`QuantApprentice_clean` contains the core research code for the paper:

- outer-loop teacher construction
- teacher explainability report generation
- inner-loop warmup and lesson revision
- scope alignment evaluation
- long-range market backtests

`QuantApprentice_studio` should not rewrite those algorithms. Instead, it should call them through
wrapper-friendly interfaces while keeping new logs, metadata, caches, and rerun outputs inside the
studio workspace.

## Stage Mapping

### 1. Outer Loop Teacher Construction

- Shell wrapper in clean:
  - `scripts/run_outer_loop.sh`
- Canonical module entrypoint:
  - `python -m quant_toolkit.teacher_loop.run_autonomous_teacher_loop`
- Direct callable hint:
  - `quant_toolkit.teacher_loop.loop.launch_until_target(max_new_rounds=...)`
- What it does:
  - hypothesis proposal
  - factor registration
  - teacher training
  - walk-forward validation
  - explainability generation
  - teacher-zoo indexing

### 2. Explainability Report v2 Rebuild

- Script in clean:
  - `scripts/rebuild_teacher_factor_reports_v2.py`
- What it does:
  - rebuild branch-oriented explainability reports from existing teacher explainability artifacts
- Safe use case:
  - upgrade or regenerate report assets without retraining teachers

### 3. Inner Loop Warmup + Alignment Suite

- Shell wrapper in clean:
  - `scripts/run_inner_loop_v7_bestguard_explore_longbatch.sh`
- Canonical module entrypoint:
  - `python -m quant_toolkit.examples.run_scorefit_scope_experiment_suite`
- Lower-level callable hints:
  - `quant_toolkit.apprentice_loop.run_multi_teacher_scoped_warmup(config)`
  - `quant_toolkit.apprentice_loop.run_multi_teacher_replay(config)`
- What it does:
  - warmup batch preparation
  - lesson evolution
  - checkpoint selection
  - alignment evaluation

### 4. Scope Alignment Only

- Shell wrapper in clean:
  - `scripts/run_scope_alignment.sh`
- Canonical module entrypoint:
  - `python -m quant_toolkit.examples.run_scope_alignment_test`
- What it does:
  - evaluate an existing teacher selection and optional final lesson state without rerunning the full warmup suite

### 5. Long-Range Market Backtest

- Shell wrapper in clean:
  - `scripts/run_market_backtest.sh`
- Canonical module entrypoint:
  - `python -m quant_toolkit.examples.run_market_scorefit_backtest`
- What it does:
  - run market-wide scoring from an existing final lesson state
  - generate signal-level scoring outputs and NAV summaries

## Studio Wrapper Policy

The studio wrapper layer should:

- keep `QuantApprentice_clean` as the algorithm source of truth
- default new outputs into `QuantApprentice_studio/runs/clean_pipeline/...`
- avoid writing new metadata back into old experiment folders
- expose stable wrapper calls for later agents such as:
  - `PlannerAgent`
  - `TeacherTrainingAgent`
  - `ExplainabilityAgent`
  - `ApprenticeAgent`
  - `EvaluationAgent`

## Implemented Wrapper Entry

The first wrapper implementation lives at:

- [`src/quant_apprentice_studio/tools/clean_pipeline.py`](../src/quant_apprentice_studio/tools/clean_pipeline.py)

It provides:

- a stage catalog
- safe environment construction
- command planning
- subprocess execution with provenance logs
- studio-owned rerun roots
