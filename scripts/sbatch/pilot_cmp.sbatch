#!/bin/bash
# Pilot model-comparison: 3 parallel PE jobs + gather (afterok).
# Run with: bash scripts/sbatch/pilot_cmp.sbatch
# Do not sbatch this file directly — it only submits child jobs.
set -euo pipefail
cd /home/yzhan629/bilby
mkdir -p logs outdir_pilot

NLIVE=100 \
DURATION=16 \
OUTDIR=outdir_pilot \
LABEL=n100_dz1 \
SAMPLE=acceptance-walk \
DLOGZ=1 \
EXTRA_FLAGS='--no-check-point-plot' \
bash scripts/sbatch/cmp_submit.sbatch
