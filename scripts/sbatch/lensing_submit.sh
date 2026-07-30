#!/bin/bash
# Submit {agn, generic, simple} PE jobs in parallel, then gather with afterok dependency.
# All PE results and SLURM logs go under OUTDIR/LABEL/ (default: outdir_lensing/nolabel/).
#
# Env overrides (exported into PE/comparison jobs):
#   NLIVE DURATION OUTDIR LABEL SAMPLE DLOGZ MAXCALL EXTRA_FLAGS
#
set -euo pipefail

cd /home/yzhan629/bilby

OUTDIR=${OUTDIR:-outdir_lensing}
LABEL=${LABEL:-nolabel}
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

JOB_AGN=$(submit_pe agn)
JOB_GEN=$(submit_pe generic)
JOB_SIM=$(submit_pe simple)

JOB_COMPARISON=$(sbatch --parsable \
    --dependency="afterok:${JOB_AGN}:${JOB_GEN}:${JOB_SIM}" \
    --output="${RUN_DIR}/%x_%j.out" \
    --error="${RUN_DIR}/%x_%j.err" \
    --export="${EXPORT_VARS}" \
    scripts/sbatch/lensing_comparison.sbatch)

echo "  agn_pe           -> ${JOB_AGN}" >&2
echo "  generic_pe       -> ${JOB_GEN}" >&2
echo "  simple_pe        -> ${JOB_SIM}" >&2
echo "  comparison       -> ${JOB_COMPARISON} (afterok:${JOB_AGN}:${JOB_GEN}:${JOB_SIM})" >&2

echo "${JOB_COMPARISON}"
