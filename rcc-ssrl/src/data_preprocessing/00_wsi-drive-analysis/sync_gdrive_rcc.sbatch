#!/bin/bash
#SBATCH --job-name=sync_gdrive_rcc
#SBATCH --partition=cpu_sapphire
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --chdir=/home/mla_group_01
#SBATCH --output=/home/mla_group_01/sync_gdrive_%j.log
#SBATCH --error=/home/mla_group_01/sync_gdrive_%j.err

# --- Fix SCRATCH ---
export SCRATCH="${SCRATCH:-/beegfs-scratch/$USER}"

# Parametri
GDRIVE_FOLDER_ID="${GDRIVE_FOLDER_ID:-1Vr1oH1irxY5UUXdrAgvX4dPpgb8-01Ma}"
DEST_DIR="${DEST_DIR:-$SCRATCH/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw}"
BW_LIMIT="${BW_LIMIT:-0}"

set -euo pipefail
echo "[SBATCH] FOLDER_ID=$GDRIVE_FOLDER_ID  DEST_DIR=$DEST_DIR  BW=$BW_LIMIT"
echo "[SBATCH] SCRATCH=$SCRATCH"
bash "$HOME/jobs/rclone_sync_body.sh" "$GDRIVE_FOLDER_ID" "$DEST_DIR" "$BW_LIMIT"
