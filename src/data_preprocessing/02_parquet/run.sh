#!/usr/bin/env bash
set -euo pipefail
jid=$(sbatch "$(dirname "$0")/02_parquet.sbatch" | awk '{print $4}')
echo "[INFO] Submitted batch job ID: ${jid}"
echo "========== LOG OUTPUT =========="
echo "[INFO] Expecting: /home/mla_group_01/logs/02_parquet/02_parquet_${jid}.log"
