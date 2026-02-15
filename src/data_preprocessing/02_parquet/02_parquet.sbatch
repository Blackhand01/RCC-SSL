#!/bin/bash
#SBATCH --job-name=02_parquet
#SBATCH --partition=cpu_sapphire
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/mla_group_01/logs/02_parquet/%x_%j.log
#SBATCH --error=/home/mla_group_01/logs/02_parquet/%x_%j.err
#SBATCH --chdir=/home/mla_group_01/rcc-ssrl

set -euo pipefail

METADATA_CSV="${METADATA_CSV:-/home/mla_group_01/configs/rcc_metadata.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/mla_group_01/reports/02_parquet}"
INVENTORY_CSV="${INVENTORY_CSV:-/home/mla_group_01/reports/0_phase/wsi_inventory.csv}"

echo "[SBATCH] METADATA_CSV=${METADATA_CSV}"
echo "[SBATCH] INVENTORY_CSV=${INVENTORY_CSV}"
echo "[SBATCH] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[SBATCH] LOGS in /home/mla_group_01/logs/02_parquet/"

bash "/home/mla_group_01/scripts/02_parquet/02_parquet_body.sh" \
  "${METADATA_CSV}" "${OUTPUT_DIR}" "${INVENTORY_CSV}"
