#!/usr/bin/env bash
# Genera trace markdown solo per un sottoinsieme di file chiave nella explainability
set -euo pipefail

TRACE_FILE="/home/mla_group_01/rcc-ssrl/src/explainability/trace_explainability_selected.md"
echo "[INFO] Scrittura in: $TRACE_FILE"

FILES=(
  # quelli che hai già elencato
  "/home/mla_group_01/rcc-ssrl/src/explainability/utils/class_utils.py"
  "/home/mla_group_01/rcc-ssrl/src/explainability/utils/roi_utils.py"
  "/home/mla_group_01/rcc-ssrl/src/explainability/run_spatial-concept.py"
  # "/home/mla_group_01/rcc-ssrl/src/explainability/run_comparision.py"
  "/home/mla_group_01/rcc-ssrl/src/explainability/paths.py"
  "/home/mla_group_01/rcc-ssrl/src/explainability/run_xai_pipeline.sh"
  "/home/mla_group_01/rcc-ssrl/src/explainability/run_xai_pipeline.sbatch"

  # # FONDAMENTALI: qui avviene il load del backbone e viene stampato missing/unexpected
  "/home/mla_group_01/rcc-ssrl/src/explainability/spatial/attention_rollout.py"

  # # dove costruisci backbone/modello per XAI (resolver / factory)
  "/home/mla_group_01/rcc-ssrl/src/training/trainer/backbones.py"
  # "/home/mla_group_01/rcc-ssrl/src/training/trainer/features.py"
  "/home/mla_group_01/rcc-ssrl/src/training/trainer/heads.py"
  # "/home/mla_group_01/rcc-ssrl/src/training/trainer/loops.py"

  # se esiste: utilità di checkpointing / load_state_dict nel training
  "/home/mla_group_01/rcc-ssrl/src/training/orchestrator.py"
  "/home/mla_group_01/rcc-ssrl/src/explainability/spatial/ssl_linear_loader.py"
  "/home/mla_group_01/rcc-ssrl/src/evaluation/ssl_linear_loader.py"

  # # definizioni dei modelli SSL (almeno quelli che stanno dando mismatch)
  "/home/mla_group_01/rcc-ssrl/src/training/models/jepa.py"
  "/home/mla_group_01/rcc-ssrl/src/training/models/dino_v3*.py"
  "/home/mla_group_01/rcc-ssrl/src/training/models/moco_v3*.py"
  "/home/mla_group_01/rcc-ssrl/src/training/models/ibot*.py"

  # "/home/mla_group_01/rcc-ssrl/src/evaluation/eval.py"
  # "/home/mla_group_01/rcc-ssrl/src/evaluation/tools/auto_eval.py"
  # "/home/mla_group_01/rcc-ssrl/src/explainability/concept/calibration/utils.py"

  "/home/mla_group_01/rcc-ssrl/src/explainability/configs/spatial.yaml"
  "/home/mla_group_01/rcc-ssrl/src/explainability/configs/roi.yaml"
  "/home/mla_group_01/rcc-ssrl/src/explainability/configs/config_concept_plip.yaml"
  "/home/mla_group_01/rcc-ssrl/src/explainability/configs/no_roi.yaml"

  # config YAML generato (uno) per verificare che backbone_name/params siano coerenti
  "/home/mla_group_01/rcc-ssrl/src/explainability/output/spatial/_pipeline_cfgs/1086716/exp_i_jepa_abl01.yaml"
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

SRC_ROOT="/home/mla_group_01/rcc-ssrl/src"

{
  for f in "${FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
      echo "$f [MISSING]"
      continue
    fi

    relpath="${f#${SRC_ROOT}/}"
    fname=$(basename "$f")

    should_exclude "$fname" && echo "$relpath [SKIPPED: pattern]" && continue

    if [[ $(stat -c%s "$f") -gt 2097152 ]]; then
      echo "$relpath [SKIPPED: >2MB]"
      continue
    fi

    echo "$relpath codice <<"
    cat "$f"
    echo ">>"
    echo
  done
} > "$TRACE_FILE"

echo "[OK] Generato $TRACE_FILE"
