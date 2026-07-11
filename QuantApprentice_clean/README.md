# QuantApprentice

Clean reproducibility package for the QuantApprentice project.

This repository is a curated export of the code used for:

- outer-loop teacher construction
- teacher explainability report generation
- inner-loop teacher-standard internalization
- scope alignment evaluation
- long-range market backtests

Data artifacts, trained model artifacts, and large experiment outputs are intentionally excluded.

## Repository Layout

- `quant_toolkit/teacher_loop/`: outer-loop teacher discovery and validation
- `quant_toolkit/apprentice_loop/`: inner-loop warmup and lesson revision logic
- `quant_toolkit/backtest/`: NAV and market backtest utilities
- `quant_toolkit/examples/`: runnable experiment entrypoints
- `scripts/`: lightweight shell wrappers and report rebuild utilities
- `templates/`: example JSON inputs for teacher selection and final lesson state
- `docs/REPRODUCIBILITY.md`: setup and run notes

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set data and artifact roots if they differ from the repository-relative defaults:

```bash
export QUANT_PROJECT_ROOT=/path/to/QuantApprentice
export TEACHER_LOOP_DATA_DIR=/path/to/day_klines
export QUANT_MEMORY_DIR=/path/to/research_memory
```

## Main Entry Points

- Outer loop:
  - `bash scripts/run_outer_loop.sh`
- Inner loop warmup + alignment (`v7_bestguard_explore_longbatch`):
  - `bash scripts/run_inner_loop_v7_bestguard_explore_longbatch.sh --help`
- Standalone scope alignment:
  - `bash scripts/run_scope_alignment.sh --help`
- Long-range market backtest:
  - `bash scripts/run_market_backtest.sh --help`

## Notes

- The package defaults to repository-relative paths instead of machine-specific absolute paths.
- The included examples expect the same artifact schema used in the original experiments.
- See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for details.
