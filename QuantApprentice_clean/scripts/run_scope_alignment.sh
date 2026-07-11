#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QUANT_PROJECT_ROOT="${QUANT_PROJECT_ROOT:-$ROOT_DIR}"
export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR}"

python -m quant_toolkit.examples.run_scope_alignment_test "$@"
