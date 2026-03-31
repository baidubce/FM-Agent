#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="$ROOT_DIR/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

export XDG_CACHE_HOME="$ROOT_DIR/.cache"
mkdir -p "$XDG_CACHE_HOME"

python baselines/fdm/model.py
python baselines/fem/model.py
python baselines/pinn/model.py
python baselines/truncated/model.py

python scripts/analysis/run_unified_eval.py
python scripts/analysis/plot_comparison.py
python scripts/analysis/convergence_analysis.py
python scripts/analysis/plot_flux_field.py
python scripts/analysis/gen_fig_framework.py
python scripts/analysis/gen_fig_motivation.py
python scripts/analysis/gen_fig_evolution.py
