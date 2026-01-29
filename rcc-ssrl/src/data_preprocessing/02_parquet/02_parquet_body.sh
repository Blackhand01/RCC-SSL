#!/usr/bin/env bash
# Usage: bash 02_parquet_body.sh [METADATA_CSV] [OUTPUT_DIR] [INVENTORY_CSV]
set -euo pipefail
METADATA_CSV="${1:-/home/mla_group_01/rcc-ssrl/configs/rcc_metadata.csv}"
OUTPUT_DIR="${2:-/home/mla_group_01/rcc-ssrl/reports/02_parquet}"
INVENTORY_CSV="${3:-/home/mla_group_01/rcc-ssrl/reports/0_phase/wsi_inventory.csv}"

echo "[INFO] JobID : ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host  : $(hostname)"
echo "[INFO] Start : $(date -Is)"
echo "[INFO] METADATA  = ${METADATA_CSV}"
echo "[INFO] INVENTORY = ${INVENTORY_CSV}"
echo "[INFO] OUTPUT    = ${OUTPUT_DIR}"

module purge >/dev/null 2>&1 || true
module load python/3.10 >/dev/null 2>&1 || true

VENV_DIR="/home/mla_group_01/rcc-ssrl/.venvs/rcc"
ACT="${VENV_DIR}/bin/activate"
if [[ -f "$ACT" ]]; then
  # shellcheck disable=SC1090
  source "$ACT"
else
  python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1090
  source "$ACT"
  python -m pip install -U pip setuptools wheel
fi

# Minimum requirements
REQ="$(dirname "$0")/requirements.txt"
if [[ -f "$REQ" ]]; then
  python -m pip install -r "$REQ"
else
  python -m pip install pandas pyarrow
fi

python -V; pip -V || true
mkdir -p "$OUTPUT_DIR"

python "$(dirname "$0")/parquet_build.py" \
  --metadata "$METADATA_CSV" \
  --inventory "$INVENTORY_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --csv-also

echo "[INFO] End   : $(date -Is)"
