#!/bin/bash
# Submit pilot PE jobs (nlive=100, dlogz=1, acceptance-walk).
set -euo pipefail

cd /home/yzhan629/bilby
mkdir -p logs outdir_pilot

echo "Submitting pilot PE jobs in parallel..."
JOB_AGN=$(sbatch --parsable scripts/sbatch/pilot_agn_pe.sbatch)
JOB_GEN=$(sbatch --parsable scripts/sbatch/pilot_generic_pe.sbatch)
JOB_CMP=$(bash scripts/sbatch/pilot_cmp.sbatch)

echo "  pilot_agn_pe     -> ${JOB_AGN}"
echo "  pilot_generic_pe -> ${JOB_GEN}"
echo "  pilot_cmp_gather -> ${JOB_CMP}"
