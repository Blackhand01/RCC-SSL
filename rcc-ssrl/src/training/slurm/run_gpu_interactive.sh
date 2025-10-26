#!/usr/bin/env bash
set -euo pipefail
srun -p gpu_a40 -A mla_group_01 --gpus=1 --cpus-per-task=8 --mem=48G --time=01:00:00 --pty bash -l <<'EOF'
module load miniconda3/3.13.25
eval "$(conda shell.bash hook)"; conda activate train
nvidia-smi -L
cd "$HOME/rcc-ssrl/src/training"
SKIP_VENV=1 PYTHONUNBUFFERED=1 python -u launch_training.py
EOF
