#!/usr/bin/env bash
set -euo pipefail

SBATCH_FILE="${SBATCH_FILE:-gpu_smoke_autolog.sbatch}"
USE_GRES="${USE_GRES:-0}"         # 0 => --gpus=1 ; 1 => --gres=gpu:1
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"

pick_partition() {
  # Preferenza: A40 -> V100 -> A100 con idleness
  local p
  for p in gpu_a40 gpu_v100 gpu_a100; do
    if sinfo -p "$p" -N -h -o "%t" | grep -Eq "idle|mix"; then
      echo "$p"; return 0
    fi
  done
  # Fallback: prima GPU partition disponibile
  sinfo -h -o "%P" | grep -E "^gpu" | head -n1
}

PARTITION="$(pick_partition)"
if [[ -z "${PARTITION}" ]]; then
  echo "[ERR] Nessuna partizione GPU disponibile"; exit 1
fi

echo "[INFO] Partizione scelta: ${PARTITION}"

GPU_FLAG=("--gpus=${GPUS_PER_JOB}")
if [[ "${USE_GRES}" == "1" ]]; then
  GPU_FLAG=("--gres=gpu:${GPUS_PER_JOB}")
fi

# Submit
JOBID=$(sbatch -p "${PARTITION}" "${GPU_FLAG[@]}" "${SBATCH_FILE}" | awk '{print $NF}')
echo "[INFO] JOBID=${JOBID}"

# Monitor semplice: attende RUNNING, poi mostra nodo
last_state=""
while true; do
  line=$(squeue -j "${JOBID}" -h -o "%T %R %V")
  [[ -z "$line" ]] && { echo "[INFO] Job non più in coda (finito o cancellato)"; break; }
  state=$(awk '{print $1}' <<<"$line")
  reason=$(cut -d' ' -f2- <<<"$line")
  if [[ "$state" != "$last_state" ]]; then
    echo "[QUEUE] $(date -Is)  STATE=${state}  INFO=${reason}"
    last_state="$state"
  fi
  if [[ "$state" == "RUNNING" ]]; then
    node=$(squeue -j "${JOBID}" -h -o "%N")
    echo "[RUN] Nodo: ${node}"
    break
  fi
  sleep 5
done

# Percorsi di log
OUT="gpu-smoke.${JOBID}.out"
ERR="gpu-smoke.${JOBID}.err"
echo "[LOG] tail -f ${OUT}  # e/o ${ERR}"
