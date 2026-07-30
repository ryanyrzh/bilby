#!/bin/bash
# Submit Tier 4 (parallel) and Tier 5 (after Tier 4) lensing smoke tests.
set -euo pipefail

cd /home/yzhan629/bilby
mkdir -p logs outdir_test output_test plots_test

echo "Submitting Tier 4 jobs in parallel..."
JOB_AGN=$(sbatch --parsable scripts/sbatch/tier4_agn_smoke.sbatch)
JOB_GEN=$(sbatch --parsable scripts/sbatch/tier4_generic_smoke.sbatch)
JOB_CMP=$(bash scripts/sbatch/tier4_cmp_smoke.sbatch)

echo "  tier4_agn_smoke     -> ${JOB_AGN}"
echo "  tier4_generic_smoke -> ${JOB_GEN}"
echo "  tier4_cmp_gather    -> ${JOB_CMP}"

echo "Submitting Tier 5 (depends on all Tier 4 jobs)..."
JOB_T5=$(sbatch --parsable --dependency=afterok:${JOB_AGN}:${JOB_GEN}:${JOB_CMP} \
    scripts/sbatch/tier5_grid_smoke.sbatch)
echo "  tier5_grid_smoke    -> ${JOB_T5}"

echo ""
echo "Monitor: squeue -u $USER"
echo "Logs:    /home/yzhan629/bilby/logs/"
