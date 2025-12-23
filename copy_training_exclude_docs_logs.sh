#!/usr/bin/env bash
# -------------------------------------------------------------------
# Script: copy_training_exclude_docs_logs.sh
# Scopo : Copiare ricorsivamente 04_training/ escludendo 'docs/', 'logs/',
#         directory '__pycache__' e file '*.pyc', e generare trace.md con
#         "nomefile codice <<contenuto>>" per revisione.
# -------------------------------------------------------------------

set -euo pipefail
# === Costanti di inclusione/esclusione ===
EXCLUDE_DIRS=("configs" "tools" "reporting" "scripts" "__pycache__" ".venv" "docs" "logs")
EXCLUDE_FILES=("__init__.py" "*.pyc" "*.err" "*.out" "*.json" "*.png" "utils/viz.py" "viz.py" "utils/reproducibility.py")
# File extra da includere sempre
INCLUDE_EXTRA_FILES=(
  # "/home/mla_group_01/rcc-ssrl/scripts/06_xai/config_xai.yaml"
  # "/home/mla_group_01/rcc-ssrl/scripts/06_xai/run_xai.sh"
  # "/home/mla_group_01/rcc-ssrl/scripts/06_xai/ssl_linear_loader.py"
  # "/home/mla_group_01/rcc-ssrl/scripts/06_xai/attention_rollout.py"
  # "/home/mla_group_01/rcc-ssrl/scripts/06_xai/xai_generate.sbatch"
  # "/home/mla_group_01/rcc-ssrl/scripts/05_evaluation/eval_test_only.py"
  "/home/mla_group_01/rcc-ssrl/src/training/scripts/launch_ssl_ablations.sh"
  "/home/mla_group_01/rcc-ssrl/src/training/scripts/generate_ssl_ablation_configs.py"
)

# === Percorsi sorgente e destinazione ===
SRC_DIR="/home/mla_group_01/rcc-ssrl/src/training"
TRACE_FILE="/home/mla_group_01/rcc-ssrl/src/training/trace.md"

# === Creazione destinazione ===
:

# === Copia con esclusioni ===
EXTRA_XAI_FILES=(
  # "/home/mla_group_01/rcc-ssrl/scripts/06_xai/config_xai.yaml"
  # "/home/mla_group_01/rcc-ssrl/scripts/06_xai/run_xai.sh"
  # "/home/mla_group_01/rcc-ssrl/scripts/06_xai/ssl_linear_loader.py"
  # "/home/mla_group_01/rcc-ssrl/scripts/06_xai/attention_rollout.py"
  # "/home/mla_group_01/rcc-ssrl/scripts/06_xai/xai_generate.sbatch"
  # "/home/mla_group_01/rcc-ssrl/scripts/05_evaluation/eval_test_only.py"
  "/home/mla_group_01/rcc-ssrl/src/training/scripts/launch_ssl_ablations.sh"
  "/home/mla_group_01/rcc-ssrl/src/training/scripts/generate_ssl_ablation_configs.py"
)

# === Generazione trace.md ===
echo "[INFO] Genero $TRACE_FILE"

# Genera lista file da tracciare
TMP_LIST="$(mktemp)"
find "$SRC_DIR" -type f ! -name 'trace.md' -print | sed "s|$SRC_DIR/||" | sort > "$TMP_LIST"

# Funzione di esclusione
should_exclude() {
  local path="$1"
  for dir in "${EXCLUDE_DIRS[@]}"; do
    if [[ "$path" =~ ^$dir/ ]]; then
      return 0
    fi
  done
  for pat in "${EXCLUDE_FILES[@]}"; do
    if [[ "$path" == $pat ]]; then
      return 0
    fi
    if [[ "$path" == *${pat#*\*} ]]; then
      return 0
    fi
  done
  return 1
}

# Scrive trace.md
{
  while IFS= read -r relpath; do
    if should_exclude "$relpath"; then
      continue
    fi
    abs="$SRC_DIR/$relpath"
    echo "$relpath codice <<"
    cat "$abs"
    echo ">>"
    echo
  done < "$TMP_LIST"

  # Includi sempre i file extra richiesti
  for f in "${INCLUDE_EXTRA_FILES[@]}"; do
    fname="${f##/home/mla_group_01/rcc-ssrl/src/training/}"
    if [[ -f "$f" ]]; then
      echo "$fname codice <<"
      cat "$f"
      echo ">>"
      echo
    fi
  done
} > "$TRACE_FILE"

rm -f "$TMP_LIST"
echo "[OK] Generato $TRACE_FILE"
