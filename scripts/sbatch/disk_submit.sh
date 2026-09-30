#!/bin/bash
# Submit fixed-binary {agn, generic, simple} PE jobs in parallel, then gather
# with an afterok dependency.
# All PE results and SLURM logs go under OUTDIR/LABEL/.
#
# Env overrides (exported into PE/comparison jobs):
#   NLIVE DURATION OUTDIR LABEL SAMPLE DLOGZ MAXCALL EXTRA_FLAGS
#
set -euo pipefail

cd /home/yzhan629/bilby

export NLIVE=${NLIVE:-1000}
export DURATION=${DURATION:-32}
export DLOGZ=${DLOGZ:-0.1}
export SAMPLE=${SAMPLE:-acceptance-walk}
export OUTDIR=${OUTDIR:-outdir_disk}
export LABEL=${LABEL:-n1000_dz0.1_dur32}
export EXTRA_FLAGS=${EXTRA_FLAGS:---no-check-point-plot}
export MAXCALL=${MAXCALL:-}
RUN_DIR="${OUTDIR}/${LABEL}"
mkdir -p "${RUN_DIR}"

EXPORT_VARS=ALL,MODEL,NLIVE,DURATION,OUTDIR,LABEL,SAMPLE,DLOGZ,MAXCALL,EXTRA_FLAGS

submit_pe() {
    local model=$1
    MODEL="${model}" sbatch --parsable \
        --job-name="${model}_pe" \
        --output="${RUN_DIR}/%x_%j.out" \
        --error="${RUN_DIR}/%x_%j.err" \
        --export="${EXPORT_VARS}" \
        scripts/sbatch/lensing_pe.sbatch
}

JOB_AGN=$(submit_pe agn_disk)
JOB_GEN=$(submit_pe generic_disk)
JOB_SIM=$(submit_pe simple_disk)

JOB_COMPARISON=$(sbatch --parsable \
    --dependency="afterok:${JOB_AGN}:${JOB_GEN}:${JOB_SIM}" \
    --output="${RUN_DIR}/%x_%j.out" \
    --error="${RUN_DIR}/%x_%j.err" \
    --export="${EXPORT_VARS}" \
    scripts/sbatch/disk_comparison.sbatch)

echo "  agn_disk_pe      -> ${JOB_AGN}" >&2
echo "  generic_disk_pe  -> ${JOB_GEN}" >&2
echo "  simple_disk_pe   -> ${JOB_SIM}" >&2
echo "  disk_comparison  -> ${JOB_COMPARISON} (afterok: ${JOB_AGN}, ${JOB_GEN}, ${JOB_SIM})" >&2
