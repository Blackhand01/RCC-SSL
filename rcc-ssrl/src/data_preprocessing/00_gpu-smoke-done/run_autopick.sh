#!/usr/bin/env bash
set -euo pipefail

SBATCH_FILE="${SBATCH_FILE:-gpu_smoke_autolog.sbatch}"
USE_GRES="${USE_GRES:-0}"         # 0 => --gpus=1 ; 1 => --gres=gpu:1
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"

pick_partition() {
  # Preference: A40 -> V100 -> A100 with idleness
  local p
  for p in gpu_a40 gpu_v100 gpu_a100; do
    if sinfo -p "$p" -N -h -o "%t" | grep -Eq "idle|mix"; then
      echo "$p"; return 0
    fi
  done
  # Fallback: first available GPU partition
  sinfo -h -o "%P" | grep -E "^gpu" | head -n1
}

PARTITION="$(pick_partition)"
if [[ -z "${PARTITION}" ]]; then
  echo "[ERR] No GPU partition available"; exit 1
fi

echo "[INFO] Selected partition: ${PARTITION}"

GPU_FLAG=("--gpus=${GPUS_PER_JOB}")
if [[ "${USE_GRES}" == "1" ]]; then
  GPU_FLAG=("--gres=gpu:${GPUS_PER_JOB}")
fi

# Submit
JOBID=$(sbatch -p "${PARTITION}" "${GPU_FLAG[@]}" "${SBATCH_FILE}" | awk '{print $NF}')
echo "[INFO] JOBID=${JOBID}"

# Simple monitor: waits for RUNNING, then shows node
last_state=""
while true; do
  line=$(squeue -j "${JOBID}" -h -o "%T %R %V")
  [[ -z "$line" ]] && { echo "[INFO] Job no longer in queue (finished or cancelled)"; break; }
  state=$(awk '{print $1}' <<<"$line")
  reason=$(cut -d' ' -f2- <<<"$line")
  if [[ "$state" != "$last_state" ]]; then
    echo "[QUEUE] $(date -Is)  STATE=${state}  INFO=${reason}"
    last_state="$state"
  fi
  if [[ "$state" == "RUNNING" ]]; then
    node=$(squeue -j "${JOBID}" -h -o "%N")
    echo "[RUN] Node: ${node}"
    break
  fi
  sleep 5
done

# Log paths
OUT="gpu-smoke.${JOBID}.out"
ERR="gpu-smoke.${JOBID}.err"
echo "[LOG] tail -f ${OUT}  # and/or ${ERR}"
