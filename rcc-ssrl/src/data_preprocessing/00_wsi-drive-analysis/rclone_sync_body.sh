#!/usr/bin/env bash
# Usage: bash rclone_sync_body.sh <GDRIVE_FOLDER_ID> <DEST_DIR> [BANDWIDTH=0]
set -euo pipefail
FOLDER_ID="${1:?Missing GDRIVE_FOLDER_ID}"
DEST_DIR="${2:?Missing DEST_DIR}"
BW_LIMIT="${3:-0}"

echo "[INFO] JobID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Nodo : $(hostname)"
echo "[INFO] Inizio: $(date -Is)"
echo "[INFO] DEST_DIR = ${DEST_DIR}"

export RCLONE_CONFIG_DIR="${RCLONE_CONFIG_DIR:-$HOME/.config/rclone}"
CONF="$RCLONE_CONFIG_DIR/rclone.conf"
[[ -f "$CONF" ]] || { echo "[ERROR] rclone.conf non trovato in $CONF"; exit 1; }

mkdir -p "$DEST_DIR"

# Usa rclone di sistema o scarica al volo
if ! command -v rclone >/dev/null 2>&1; then
  TMPBIN="${TMPDIR:-/tmp}/rclone"
  curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o "$TMPBIN.zip"
  unzip -q "$TMPBIN.zip" -d "${TMPDIR:-/tmp}/rclone_unpack"
  RCLONE_BIN=$(find "${TMPDIR:-/tmp}/rclone_unpack" -type f -name rclone | head -n1)
  chmod +x "$RCLONE_BIN"
else
  RCLONE_BIN="$(command -v rclone)"
fi

echo "[INFO] Uso rclone: $RCLONE_BIN"
echo "[INFO] Sync Google Drive (folder-id=$FOLDER_ID) → $DEST_DIR"

"$RCLONE_BIN" copy "gdrive:" "$DEST_DIR" \
  --drive-root-folder-id "$FOLDER_ID" \
  --create-empty-src-dirs --checkers=8 --transfers=8 \
  --retries=5 --low-level-retries=20 --fast-list --progress



echo "[INFO] Validazione…"
FILE_COUNT=$(find "$DEST_DIR" -type f | wc -l | tr -d ' ')
DIR_COUNT=$(find "$DEST_DIR" -type d | wc -l | tr -d ' ')
TOTAL_SIZE=$(du -sh "$DEST_DIR" | awk '{print $1}')
echo "[RESULT] File: $FILE_COUNT  Cartelle: $DIR_COUNT  Size: $TOTAL_SIZE"
echo "[INFO] Fine: $(date -Is)"
