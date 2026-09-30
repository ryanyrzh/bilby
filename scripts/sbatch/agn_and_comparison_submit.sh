#!/bin/bash
# Submit AGN PE job, then gather with afterok dependency.

set -euo pipefail

cd /home/yzhan629/bilby
RUN_DIR=outdir_lensing/n1000_dz0.2_dur32

export OUTDIR=outdir_lensing
export LABEL=n1000_dz0.2_dur32

JOB_AGN=$(MODEL=agn EXTRA_FLAGS="--no-check-point-plot --resume" \
  sbatch --parsable \
    --job-name=agn_pe \
    --output="${RUN_DIR}/%x_%j.out" \
    --error="${RUN_DIR}/%x_%j.err" \
    --export=ALL,MODEL,NLIVE,DURATION,OUTDIR,LABEL,SAMPLE,DLOGZ,MAXCALL,EXTRA_FLAGS \
    scripts/sbatch/lensing_pe.sbatch)

sbatch --dependency="afterok:${JOB_AGN}" \
  --output="${RUN_DIR}/%x_%j.out" \
  --error="${RUN_DIR}/%x_%j.err" \
  --export=ALL,OUTDIR,LABEL \
  scripts/sbatch/lensing_comparison.sbatch