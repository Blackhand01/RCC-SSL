#!/usr/bin/env bash
# -------------------------------------------------------------------
# Script: copy_training_exclude_docs_logs.sh
# Scopo : Copiare ricorsivamente 04_training/ escludendo 'docs/', 'logs/',
#         directory '__pycache__' e file '*.pyc', e generare trace.md con
#         "nomefile codice <<contenuto>>" per revisione.
# -------------------------------------------------------------------

set -euo pipefail

# === Percorsi sorgente e destinazione ===
SRC_DIR="/home/mla_group_01/rcc-ssrl/src/training_copy"
DST_DIR="/home/mla_group_01/rcc-ssrl/src/training_tmp"

# === Creazione destinazione ===
mkdir -p "$DST_DIR"

# === Copia con esclusioni ===
echo "[INFO] Copia da: $SRC_DIR"
echo "[INFO] A:        $DST_DIR"
echo "[INFO] Escludendo: docs/, __pycache__/, *.pyc, *.err, *.out, .venv/"

rsync -av \
  --exclude 'docs/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.err' \
  --exclude '*.out' \
  --exclude '.venv/' \
  --exclude '/training/utils/reproducibility.py' \
  --exclude '/training/slurm/train_multi_node.sbatch' \
  --exclude '/training/reporting/' \
  "$SRC_DIR"/ "$DST_DIR"/

echo "[OK] Copia completata."

# === Generazione trace.md ===
TRACE_FILE="$DST_DIR/trace.md"
echo "[INFO] Genero $TRACE_FILE"

# Nota: ordiniamo i percorsi; escludiamo di nuovo per sicurezza.
# Includiamo file testuali tipici; se vuoi TUTTO, rimuovi il grep -E nel while.
# Qui includiamo: .py, .sh, .sbatch, .yaml, .yml, .md, .txt
TMP_LIST="$(mktemp)"
find "$DST_DIR" -type f ! -name 'trace.md' \
  ! -path "*/__pycache__/*" ! -name "*.pyc" \
  ! -name "*.err" ! -name "*.out" ! -path "*/.venv/*" \
  -print | sed "s|$DST_DIR/||" | sort > "$TMP_LIST"

# Se vuoi filtrare per estensioni testuali, lascia questa variabile;
# per includere tutti i file, imposta FILE_FILTER_REGEX=".*"
FILE_FILTER_REGEX='(\.py|\.sh|\.sbatch|\.yaml|\.yml|\.md|\.txt)$'

# Scrive trace.md
{
  while IFS= read -r relpath; do
    if [[ "$relpath" =~ $FILE_FILTER_REGEX ]]; then
      abs="$DST_DIR/$relpath"
      echo "$relpath codice <<"
      # stampa contenuto così com'è (senza alterare i caratteri)
      cat "$abs"
      echo ">>"
      echo
    fi
  done < "$TMP_LIST"
} > "$TRACE_FILE"

rm -f "$TMP_LIST"
echo "[OK] Generato $TRACE_FILE"
