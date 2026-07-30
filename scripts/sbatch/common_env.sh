#!/bin/bash
# Shared environment for bilby lensing smoke tests.
set -euo pipefail

echo "Job ${SLURM_JOB_ID:-local} on $(hostname); CPUs=${SLURM_CPUS_PER_TASK:-1}"

module load texlive/20220321
source /home/yzhan629/bilby/.venv/bin/activate
cd /home/yzhan629/bilby
mkdir -p logs outdir_test outdir_pilot output_test plots_test

unset PYTHONPATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
