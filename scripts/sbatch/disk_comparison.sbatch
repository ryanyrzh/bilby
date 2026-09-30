#!/bin/bash
#SBATCH --job-name=disk_comparison
#SBATCH --partition=parallel
#SBATCH --account=mkamion1
#SBATCH --output=outdir_lensing/%x_%j.out
#SBATCH --error=outdir_lensing/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yzhan629@jh.edu

# Gather fixed-binary agn/generic/simple PE results and write LABEL_disk_summary.json.
# Expects {LABEL}_{agn,generic,simple}_disk_result.json under OUTDIR/LABEL/.
# Optional overrides: OUTDIR LABEL
# Outputs go under OUTDIR/LABEL/ (overridden by a submit script for SLURM logs).
set -euo pipefail
source /home/yzhan629/bilby/scripts/sbatch/common_env.sh

OUTDIR=${OUTDIR:-outdir_lensing}
LABEL=${LABEL:-n100_dz1}
RUN_DIR="${OUTDIR}/${LABEL}"
mkdir -p "${RUN_DIR}"

python examples/gw_examples/lensing_examples/disk_comparison.py \
    --outdir "${RUN_DIR}" \
    --label "${LABEL}"
