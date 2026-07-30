#!/bin/bash
#SBATCH --job-name=pilot_agn_pe
#SBATCH --partition=parallel
#SBATCH --account=mkamion1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yzhan629@jh.edu

set -euo pipefail
source /home/yzhan629/bilby/scripts/sbatch/common_env.sh

NPOOL=${SLURM_CPUS_PER_TASK:-1}

python examples/gw_examples/lensing_examples/agn_lensed_pe.py \
    --nlive 1000 \
    --npool "${NPOOL}" \
    --duration 16 \
    --outdir outdir_pilot \
    --label agn_pilot_n1000_dz0.1 \
    --sample acceptance-walk \
    --dlogz 0.1 \
    --no-check-point-plot

python scripts/validate_tier4.py outdir_pilot pe:agn_pilot_n1000_dz0.1
