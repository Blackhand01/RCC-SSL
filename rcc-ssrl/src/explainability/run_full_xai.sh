#!/usr/bin/env bash
# Orchestrate full explainability pipeline:
# - Stage 0: global concept bank (dataset-level, only if missing)
# - Stage 1/2: spatial + concept XAI for all ablations in an experiment
#
# VLM: ONLY local HF model via VLMClientHF (no HTTP server).

set -euo pipefail

# -----------------------------------------------------------------------------
# GLOBAL CONFIG (single source of truth for explainability)
# -----------------------------------------------------------------------------
REPO_ROOT="/home/mla_group_01/rcc-ssrl"
SRC_DIR="${REPO_ROOT}/src"
# --- Experiment-level config (EDIT HERE) ---
EXP_ROOT="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3"
MODEL_NAME="moco_v3"                    # base SSL backbone name (non *_ssl_linear_best)
BACKBONE_NAME="vit_small_patch16_224"

# Python environment (usato per Stage 0 + Stage 1/2)
VENV_PATH="/home/mla_group_01/rcc-ssrl/.venvs/xai"

# Dataset-level config (concept bank)
TRAIN_WDS_DIR="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/train"
TRAIN_WDS_PATTERN="shard-*.tar"
CANDIDATES_CSV="${SRC_DIR}/explainability/concept/ontology/concept_candidates_rcc.csv"
CANDIDATES_IMG_ROOT="${SRC_DIR}/explainability/concept/ontology/concept_candidates_images"

# Ontologia + concept bank usati per Stage 0 e Stage 2 
VERS="debug"  # ontology + concept bank version (EDIT HERE)
ONTOLOGY_YAML="${SRC_DIR}/explainability/concept/ontology/ontology_rcc_${VERS}.yaml"
CONCEPT_BANK_CSV="${SRC_DIR}/explainability/concept/ontology/concepts_rcc_${VERS}.csv"

# VLM (LLaVA-Med) – HF ONLY
VLM_MODEL_PATH="Eren-Senoglu/llava-med-v1.5-mistral-7b-hf"

# Flags opzionali (per futura estensione; al momento solo log)
ONLY_SPATIAL="${ONLY_SPATIAL:-0}"
ONLY_CONCEPT="${ONLY_CONCEPT:-0}"

# Export env necessari downstream (sbatch XAI + run_explainability.py)
export VENV_PATH
export CONCEPT_BANK_CSV

# -----------------------------------------------------------------------------
# LOGGING: tutti i log per modello vanno sotto logs/xai/${MODEL_NAME}
# -----------------------------------------------------------------------------
LOG_ROOT="${SRC_DIR}/logs/xai"
LOG_DIR="${LOG_ROOT}/${MODEL_NAME}"
mkdir -p "${LOG_DIR}"

LOG_SUFFIX="${SLURM_JOB_ID:-local_$$}"

echo "[INFO] run_full_xai.sh starting"
echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] EXP_ROOT=${EXP_ROOT}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] BACKBONE_NAME=${BACKBONE_NAME}"
echo "[INFO] VLM_MODEL_PATH=${VLM_MODEL_PATH}"
echo "[INFO] VENV_PATH=${VENV_PATH}"
echo "[INFO] ONTOLOGY_YAML=${ONTOLOGY_YAML}"
echo "[INFO] CONCEPT_BANK_CSV=${CONCEPT_BANK_CSV}"
echo "[INFO] TRAIN_WDS_DIR=${TRAIN_WDS_DIR}"
echo "[INFO] TRAIN_WDS_PATTERN=${TRAIN_WDS_PATTERN}"

# ------------------- env -------------------

export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

if [[ -n "${VENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

# ------------------- STAGE 0: concept bank (solo se NON esiste o è vuota) -------------------

# Conta le righe se il file esiste
if [[ -f "${CONCEPT_BANK_CSV}" ]]; then
  num_lines=$(wc -l < "${CONCEPT_BANK_CSV}")
else
  num_lines=0
fi

if [[ "${num_lines}" -le 1 ]]; then
  echo "[WARN] Concept bank missing or empty (lines=${num_lines}); rebuilding Stage 0."

  # prima di lanciare Python, controlla che esistano i tar attesi
  if ! compgen -G "${TRAIN_WDS_DIR}/${TRAIN_WDS_PATTERN}" > /dev/null; then
    echo "[ERROR] No shards found under ${TRAIN_WDS_DIR}/${TRAIN_WDS_PATTERN}" >&2
    echo "[ERROR] - Controlla che TRAIN_WDS_DIR punti alla cartella corretta" >&2
    echo "[ERROR] - Controlla che il pattern TRAIN_WDS_PATTERN (default shard-*.tar) sia corretto" >&2
    exit 1
  fi

  # 0a) concept_candidates_rcc.csv (train WDS -> PNG + CSV)
  echo "[INFO] Stage 0a: building concept_candidates_rcc.csv"
  python3 -m explainability.concept.ontology.build_concept_candidates \
    --train-dir "${TRAIN_WDS_DIR}" \
    --pattern "${TRAIN_WDS_PATTERN}" \
    --image-key "img.jpg;jpg;jpeg;png" \
    --meta-key "meta.json;json" \
    --out-csv "${CANDIDATES_CSV}" \
    --images-root "${CANDIDATES_IMG_ROOT}"

  # 0b) concepts_rcc_*.csv (VLM HF su candidates)
  echo "[INFO] Stage 0b: building concept bank via local HF VLM"
  export VLM_DEBUG="${VLM_DEBUG:-1}"

  python3 -m explainability.concept.ontology.build_concept_bank \
    --ontology "${ONTOLOGY_YAML}" \
    --images-csv "${CANDIDATES_CSV}" \
    --model-name "${VLM_MODEL_PATH}" \
    --out-csv "${CONCEPT_BANK_CSV}" \
    --max-images 0

  # hard check: concept bank deve avere almeno header + 1 riga
  lines_after=$(wc -l < "${CONCEPT_BANK_CSV}")
  if [[ "${lines_after}" -le 1 ]]; then
    echo "[ERROR] Concept bank ${CONCEPT_BANK_CSV} still empty after Stage 0 (lines=${lines_after}). Aborting." >&2
    exit 1
  fi
else
  echo "[INFO] Concept bank found at ${CONCEPT_BANK_CSV} with ${num_lines} lines – skipping Stage 0."
fi


# ------------------- STAGE 1/2: spatial + concept XAI per esperimento -------------------

ORCH_CMD=( python3 -m explainability.run_explainability
  --experiment-root "${EXP_ROOT}"
  --model-name "${MODEL_NAME}"
  --spatial-config-template "${SRC_DIR}/explainability/spatial/config_xai.yaml"
  --concept-config-template "${SRC_DIR}/explainability/concept/config_concept.yaml"
)

echo "[INFO] Stage 1/2: running experiment-level explainability:"
echo "[INFO]   ${ORCH_CMD[*]}"
"${ORCH_CMD[@]}"

echo "[OK] run_full_xai.sh completed."
