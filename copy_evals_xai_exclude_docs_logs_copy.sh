#!/usr/bin/env bash
# Genera trace markdown solo per un sottoinsieme di file chiave nella explainability
set -euo pipefail

TRACE_FILE="/home/mla_group_01/rcc-ssrl/src/explainability/trace_explainability_selected.md"
echo "[INFO] Scrittura in: $TRACE_FILE"

FILES=(
  "/home/mla_group_01/rcc-ssrl/src/explainability/utils/class_utils.py"
  "/home/mla_group_01/rcc-ssrl/src/explainability/utils/roi_utils.py"
  "/home/mla_group_01/rcc-ssrl/src/explainability/run_spatial-concept.py"
  "/home/mla_group_01/rcc-ssrl/src/explainability/run_comparision.py"
  "/home/mla_group_01/rcc-ssrl/src/explainability/paths.py"
)

EXCLUDE_PATTERNS=("*.pyc" "*.png" "*.json" "*.npy" "*.npz" "*.tar" "*.jpg" "*.jpeg" "*.tif" "*.bmp" "*.gif" "*.pdf")

should_exclude() {
  local path="$1"
  for pat in "${EXCLUDE_PATTERNS[@]}"; do
    [[ "$path" == $pat ]] && return 0
    [[ "$path" == *${pat#\*} ]] && return 0
  done
  return 1
}

{
  for f in "${FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
      echo "$f [MISSING]"
      continue
    fi
    fname=$(basename "$f")
    should_exclude "$fname" && echo "$fname [SKIPPED: pattern]" && continue

    # Salta file > 2MB
    if [[ $(stat -c%s "$f") -gt 2097152 ]]; then
      echo "$fname [SKIPPED: >2MB]"
      continue
    fi

    echo "$fname codice <<"
    cat "$f"
    echo ">>"
    echo
  done
} > "$TRACE_FILE"

echo "[OK] Generato $TRACE_FILE"
