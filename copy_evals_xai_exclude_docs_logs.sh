#!/usr/bin/env bash
# Script compatto: genera trace per src/xai, src/evaluation, src/explainability
# Esclude file .md, directory '__pycache__', file '*.pyc', '*.err', '*.out', '*.json', '*.png'.
set -euo pipefail


EXCLUDE_DIRS=("__pycache__" ".venv" "logs" "docs" "output")
EXCLUDE_FILES=(
  "*.md" "*.pt"  "*.tex" "*.txt" "*.pyc""*.log" "*.err" "*.out" "*.json" "*.jsonl" "*.png" "*.csv" "*.npy" "*.npz" "*.tar" "*.jpg" "*.jpeg" "*.tif" "*.tiff" "*.bmp" "*.gif" "*.pdf"
   "*.pyc"
)

SRC_DIRS=(
  "/home/mla_group_01/rcc-ssrl/src/explainability"
)

should_exclude() {
  local path="$1"
  for dir in "${EXCLUDE_DIRS[@]}"; do
    [[ "$path" =~ ^$dir/ ]] && return 0
  done
  for pat in "${EXCLUDE_FILES[@]}"; do
    [[ "$path" == $pat ]] && return 0
    [[ "$path" == *${pat#\*} ]] && return 0
  done
  return 1
}

for SRC in "${SRC_DIRS[@]}"; do
  TRACE_FILE="$SRC/trace_$(basename "$SRC").md"
  echo "[INFO] Genero $TRACE_FILE"
  TMP_LIST="$(mktemp)"
  find "$SRC" -type f ! -name "$(basename "$TRACE_FILE")" -print | sed "s|$SRC/||" | sort > "$TMP_LIST"
  {
    while IFS= read -r relpath; do
      should_exclude "$relpath" && continue
      # Salta file >2MB per evitare rallentamenti
      abs="$SRC/$relpath"
      if [[ -f "$abs" && $(stat -c%s "$abs") -gt 2097152 ]]; then
        echo "$relpath [SKIPPED: >2MB]"
        continue
      fi
      abs="$SRC/$relpath"
      echo "$relpath codice <<"
      cat "$abs"
      echo ">>"
      echo
    done < "$TMP_LIST"
  } > "$TRACE_FILE"
  rm -f "$TMP_LIST"
  echo "[OK] Generato $TRACE_FILE"
done