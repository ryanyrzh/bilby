#!/bin/bash
# Tier-4 smoke model-comparison: 3 parallel PE jobs + gather (afterok).
# Run with: bash scripts/sbatch/tier4_cmp_smoke.sbatch
# Do not sbatch this file directly — it only submits child jobs.
set -euo pipefail
cd /home/yzhan629/bilby
mkdir -p logs outdir_test

NLIVE=10 \
DURATION=4 \
OUTDIR=outdir_test \
LABEL=cmp_smoke \
SAMPLE=unif \
DLOGZ=10 \
MAXCALL=5000 \
EXTRA_FLAGS='--no-check-point' \
VALIDATE=1 \
bash scripts/sbatch/cmp_submit.sbatch
