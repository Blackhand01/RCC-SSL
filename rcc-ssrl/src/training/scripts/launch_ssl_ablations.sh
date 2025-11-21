#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Launch all SSL ablation jobs for a given model (dino_v3 | moco_v3 | ibot | i_jepa)
#
# Usage:
#   ./launch_ssl_ablations.sh <ssl_model>
#
# Example:
#   ./launch_ssl_ablations.sh dino_v3
#
# What it does:
#   - Scans configs/ablations/<ssl_model>/ for ablation YAMLs (exp_*_<ssl_model>_abl*.yaml)
#   - Creates an experiment group directory:
#       /beegfs-scratch/.../outputs/mlruns/experiments/exp_{DATETIME}_{ssl_model}/
#   - Creates a subfolder per ablation inside that group
#   - Submits one Slurm job per YAML via train_single_node.sbatch
#   - Writes Slurm logs to /home/mla_group_01/rcc-ssrl/src/logs/<ssl_model>/
#   - Saves a jobs manifest (job id, yaml, subdir) in both the logs dir and the group dir
# -----------------------------------------------------------------------------

if [[ $# -lt 1 ]]; then
  echo "ERROR: Missing <ssl_model> (dino_v3 | moco_v3 | ibot | i_jepa)" >&2
  exit 2
fi

MODEL="$1"
case "$MODEL" in
  dino_v3|moco_v3|ibot|i_jepa) ;;
  *) echo "ERROR: Unsupported model '$MODEL'"; exit 2 ;;
esac

# Map model -> code letter for job_name
case "$MODEL" in
  dino_v3) CODE="D" ;;
  moco_v3) CODE="M" ;;
  ibot)    CODE="B" ;;
  i_jepa)  CODE="J" ;;
esac

ROOT="/home/mla_group_01/rcc-ssrl"
TRAIN_DIR="$ROOT/src/training"
CFG_DIR="$TRAIN_DIR/configs/ablations/${MODEL}"
SBATCH_SCRIPT="$TRAIN_DIR/slurm/train_single_node.sbatch"

OUT_BASE="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments"
DATETIME="$(date +%Y%m%d_%H%M%S)"
EXP_GROUP="exp_${DATETIME}_${MODEL}"
EXP_ROOT="${OUT_BASE}/${EXP_GROUP}"

LOGDIR="$ROOT/src/logs/${MODEL}"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found"; exit 3; }
[[ -f "$SBATCH_SCRIPT" ]] || { echo "ERROR: sbatch script not found: $SBATCH_SCRIPT"; exit 3; }
[[ -d "$CFG_DIR" ]] || { echo "ERROR: config dir not found: $CFG_DIR"; exit 3; }

# Sort YAMLs deterministically (natural sort on ablNN)
mapfile -t CFGS < <(find "$CFG_DIR" -maxdepth 1 -type f -name "exp_${MODEL}_abl*.yaml" | sort -V)
[[ ${#CFGS[@]} -gt 0 ]] || { echo "ERROR: No ablation YAMLs in $CFG_DIR"; exit 4; }

mkdir -p "$EXP_ROOT" "$LOGDIR"

MANIFEST_LOG="${LOGDIR}/jobs_manifest_${EXP_GROUP}.tsv"
MANIFEST_EXP="${EXP_ROOT}/jobs_manifest.tsv"
echo -e "job_id\tjob_name\tmodel\tyaml_path\trun_name\texp_group\tslurm_log" | tee "$MANIFEST_LOG" > "$MANIFEST_EXP"

echo "==============================================================="
echo " Model           : $MODEL"
echo " Ablations       : ${#CFGS[@]}"
echo " Group (MLflow)  : ${EXP_GROUP}"
echo " Group dir       : ${EXP_ROOT}"
echo " Logs dir        : ${LOGDIR}"
echo " Sbatch script   : ${SBATCH_SCRIPT}"
echo "==============================================================="

for CFG in "${CFGS[@]}"; do
  BASE="$(basename "$CFG" .yaml)"                  # e.g., exp_moco_v3_abl01
  SUBDIR="${BASE}"
  # Extract ablation number (NN) robustly (works with/without .yaml)
  FNAME="$(basename "$CFG")"
  ABL_NUM="$(sed -nE 's/.*_abl([0-9]+)\.yaml$/\1/p' <<< "${FNAME}")"
  [[ -n "$ABL_NUM" ]] || ABL_NUM="$(sed -nE 's/.*_abl([0-9]+)$/\1/p' <<< "${BASE}")"
  [[ -n "$ABL_NUM" ]] || ABL_NUM="00"
  # Force decimal interpretation in case YAML uses leading zeros (08, 09, ...)
  ABL_NUM_FMT="$(printf "%02d" "$((10#${ABL_NUM}))")"

  JOB_NAME="tr${CODE}abl${ABL_NUM_FMT}"            # e.g., trMabl01

  # Create per-ablation folder inside the experiment group
  mkdir -p "${EXP_ROOT}/${SUBDIR}"

  # Slurm log files (coded)
  SLURM_LOG_OUT="${LOGDIR}/${JOB_NAME}_%j.out"
  SLURM_LOG_ERR="${LOGDIR}/${JOB_NAME}_%j.err"

  # Export ablation + naming hints for Python (consumed by launch_training/orchestrator)
  EXPORTS="ALL,MLFLOW_EXPERIMENT_NAME=${EXP_GROUP},EXP_GROUP=${EXP_GROUP},EXP_SUBDIR=${SUBDIR},OUTPUTS_GROUP_DIR=${EXP_ROOT},RUN_NAME=${JOB_NAME},ABLATION_ID=${ABL_NUM_FMT},MODEL_KEY=${MODEL}"

  # Submit and capture job id, passing the YAML path as first argument
  JOBID=$(sbatch \
    --job-name="${JOB_NAME}" \
    --output="${SLURM_LOG_OUT}" \
    --error="${SLURM_LOG_ERR}" \
    --export="${EXPORTS}" \
    --parsable \
    "${SBATCH_SCRIPT}" "${CFG}")

  # Concrete stdout log path for the manifest
  SLURM_LOG_CONCRETE="${SLURM_LOG_OUT//%j/${JOBID}}"
  echo -e "${JOBID}\t${JOB_NAME}\t${MODEL}\t${CFG}\t${JOB_NAME}\t${EXP_GROUP}\t${SLURM_LOG_CONCRETE}" | tee -a "$MANIFEST_LOG" >> "$MANIFEST_EXP"

  echo "Submitted: ${SUBDIR} as ${JOB_NAME} -> JobID ${JOBID}"
done

echo "---------------------------------------------------------------"
echo "All jobs submitted."
echo "Manifest:"
echo "  - ${MANIFEST_LOG}"
echo "  - ${MANIFEST_EXP}"
echo
echo "Monitor with:  watch -n 5 'squeue -u $USER -o \"%.18i %.12j %.8T %.10M %.6D %R\" | grep -E \"tr[DMJB]abl\"'"
echo "Tail a log:     tail -n 200 -f ${LOGDIR}/<JOB_NAME>_<JOBID>.out"
