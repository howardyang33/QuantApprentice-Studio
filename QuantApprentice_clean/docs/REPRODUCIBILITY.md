# Reproducibility Notes

This package intentionally excludes private datasets, large cached feature matrices, model checkpoints, and generated reports.

## Expected External Inputs

At minimum, the runnable pipelines expect:

- daily stock data under `TEACHER_LOOP_DATA_DIR`
- a writable `QUANT_MEMORY_DIR`
- teacher-loop artifacts under `TEACHER_LOOP_ARTIFACT_ROOT` for inner-loop and backtest stages
- teacher reports under `TEACHER_LOOP_REPORT_ROOT` when using explainability-driven lesson initialization

If the environment variables are not set, the code falls back to repository-relative paths such as:

- `./day_klines`
- `./research_memory`
- `./reports/teacher_loop`
- `./reports/apprentice_loop`

## Recommended Environment Variables

```bash
export QUANT_PROJECT_ROOT=/path/to/QuantApprentice
export TEACHER_LOOP_DATA_DIR=/path/to/day_klines
export QUANT_MEMORY_DIR=/path/to/research_memory
export TEACHER_LOOP_REPORT_ROOT=/path/to/reports/teacher_loop
export APPRENTICE_REPORT_ROOT=/path/to/reports/apprentice_loop
export TEACHER_LOOP_ARTIFACT_ROOT=/path/to/research_memory/artifacts/teacher_loop
```

## Core Workflows

### 1. Outer Loop Teacher Construction

```bash
bash scripts/run_outer_loop.sh
```

This launches the autonomous teacher construction loop through:

- hypothesis proposal
- factor registration
- model training
- yearly walk-forward validation
- teacher-zoo indexing

### 2. Rebuild Explainability Report v2

```bash
python scripts/rebuild_teacher_factor_reports_v2.py --help
```

### 3. Inner Loop Warmup and Alignment

```bash
bash scripts/run_inner_loop_v7_bestguard_explore_longbatch.sh --help
```

This is the packaged entrypoint for the `v7_bestguard_explore_longbatch` lesson-evolution workflow.

### 4. Scope Alignment Only

```bash
bash scripts/run_scope_alignment.sh --help
```

### 5. Long-Range Market Backtest

```bash
bash scripts/run_market_backtest.sh --help
```

## Example JSON Inputs

- `templates/teacher_selection.example.json`
- `templates/final_lesson_state.example.json`

These are schema placeholders only; they do not contain private artifacts.
