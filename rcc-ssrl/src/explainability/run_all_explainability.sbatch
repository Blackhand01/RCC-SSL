#!/usr/bin/env bash
#SBATCH -J xai_all
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/xai/run_all.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/xai/run_all.%j.err
#SBATCH -p gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=compute-5-14,compute-5-11,compute-3-12

# This job wraps the run_full_xai.sh orchestrator and runs it inside an allocation.

set -euo pipefail

echo "[INFO] Host: $(hostname)"
echo "[INFO] xai_all wrapper: delegating configuration to run_full_xai.sh"

# Orchestrator reale (esiste già)
SCRIPT="/home/mla_group_01/rcc-ssrl/src/explainability/run_full_xai.sh"

if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERROR] Orchestrator script not found: $SCRIPT" >&2
  exit 1
fi

chmod +x "$SCRIPT"

# srun runs the job step inside the allocation provided by sbatch.
SRUN_ARGS=( srun --ntasks=1 )

echo "[INFO] Command: ${SRUN_ARGS[*]} $SCRIPT"
"${SRUN_ARGS[@]}" "$SCRIPT"

echo "[OK] xai_all job completed (run_full_xai.sh exited with code $?)"
