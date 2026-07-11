#!/usr/bin/env python3
"""Canonical entrypoint for Pilot 2.2 within-gate walk-forward execution."""

from __future__ import annotations

import traceback

from quant_toolkit.examples.run_pilot2_2_within_gate import append_error, main


if __name__ == "__main__":
    try:
        main()
    except Exception:
        append_error(traceback.format_exc())
        raise
