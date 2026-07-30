#!/bin/bash
#SBATCH --job-name=tier5_grid_smoke
#SBATCH --partition=parallel
#SBATCH --account=mkamion1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yzhan629@jh.edu

set -euo pipefail
source /home/yzhan629/bilby/scripts/sbatch/common_env.sh

python scripts/validate_tier4.py outdir_test pe:agn_smoke pe:generic_smoke cmp:cmp_smoke

python examples/gw_examples/lensing_examples/compute_SNR_threshold_Mc.py \
    --nMc 2 \
    --nR 2 \
    --cores 1 \
    --nlive 50 \
    --steps 1 \
    --duration 4 \
    --model agn \
    --logB 2.0 \
    --outdir output_test \
    --plotdir plots_test \
    --label smoke

python scripts/validate_tier5.py output_test plots_test
