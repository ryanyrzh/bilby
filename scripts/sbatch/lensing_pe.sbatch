#!/bin/bash
#SBATCH --job-name=lensing_pe
#SBATCH --partition=parallel
#SBATCH --account=mkamion1
#SBATCH --output=outdir_lensing/%x_%j.out
#SBATCH --error=outdir_lensing/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yzhan629@jh.edu

# One model PE for the comparison suite.
# Required: MODEL=agn|generic|simple
# Optional overrides: NLIVE DURATION OUTDIR LABEL SAMPLE DLOGZ MAXCALL EXTRA_FLAGS
# Outputs go under OUTDIR/LABEL/ (overridden by lensing_submit.sh for SLURM logs).
set -euo pipefail
source /home/yzhan629/bilby/scripts/sbatch/common_env.sh

MODEL=${MODEL:?Set MODEL to agn, generic, or simple}
NLIVE=${NLIVE:-100}
DURATION=${DURATION:-16}
OUTDIR=${OUTDIR:-outdir_lensing}
LABEL=${LABEL:-n100_dz1}
SAMPLE=${SAMPLE:-acceptance-walk}
DLOGZ=${DLOGZ:-1}
MAXCALL=${MAXCALL:-}
EXTRA_FLAGS=${EXTRA_FLAGS:---no-check-point-plot}

RUN_DIR="${OUTDIR}/${LABEL}"
mkdir -p "${RUN_DIR}"

NPOOL=${SLURM_CPUS_PER_TASK:-1}
SCRIPT="examples/gw_examples/lensing_examples/${MODEL}_pe.py"

CMD=(
    python "${SCRIPT}"
    --nlive "${NLIVE}"
    --npool "${NPOOL}"
    --duration "${DURATION}"
    --outdir "${RUN_DIR}"
    --label "${LABEL}"
    --sample "${SAMPLE}"
    --dlogz "${DLOGZ}"
)
if [[ -n "${MAXCALL}" ]]; then
    CMD+=(--maxcall "${MAXCALL}")
fi
# shellcheck disable=SC2206
CMD+=(${EXTRA_FLAGS})

echo "Running: ${CMD[*]}"
"${CMD[@]}"
