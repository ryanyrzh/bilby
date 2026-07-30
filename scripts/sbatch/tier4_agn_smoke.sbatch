#!/bin/bash
#SBATCH --job-name=tier4_agn_smoke
#SBATCH --partition=parallel
#SBATCH --account=mkamion1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yzhan629@jh.edu

set -euo pipefail
source /home/yzhan629/bilby/scripts/sbatch/common_env.sh

NPOOL=${SLURM_CPUS_PER_TASK:-1}

python examples/gw_examples/lensing_examples/agn_lensed_pe.py \
    --nlive 10 \
    --npool "${NPOOL}" \
    --duration 4 \
    --outdir outdir_test \
    --label agn_smoke \
    --sample unif \
    --dlogz 10 \
    --maxcall 5000 \
    --no-check-point \
    --no-plot-corner

python scripts/validate_tier4.py outdir_test pe:agn_smoke
