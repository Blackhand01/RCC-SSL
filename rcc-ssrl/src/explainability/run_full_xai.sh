#!/usr/bin/env bash
# Orchestrate full explainability pipeline:
# - Stage 0: global concept bank (dataset-level, only if missing)
# - Stage 1/2: spatial + concept XAI for all ablations in an experiment

# NOTE:
#  - opzionalmente può lanciare un server LLaVA-Med locale (controller + worker)
#    per la Stage 0b (build_concept_bank).
#  - abilita questo comportamento esportando:
#        START_LOCAL_VLM=1
#        VLM_MODEL_PATH=/path/or/hf/id/of/microsoft/llava-med-v1.5-mistral-7b  # opzionale
#  - va eseguito su un nodo con GPU (srun/sbatch), NON sul login node.

set -euo pipefail

# ------------------- defaults (override via env or args if vuoi, ma qui li teniamo fissi) -------------------

REPO_ROOT="/home/mla_group_01/rcc-ssrl"
SRC_DIR="${REPO_ROOT}/src"

# Dataset-level (Stage 0)
TRAIN_WDS_DIR="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/train"
CANDIDATES_CSV="${SRC_DIR}/explainability/concept/ontology/concept_candidates_rcc.csv"
CANDIDATES_IMG_ROOT="${SRC_DIR}/explainability/concept/ontology/concept_candidates_images"

# default: file di debug a 4 concetti
ONTOLOGY_YAML_DEFAULT="${SRC_DIR}/explainability/concept/ontology/ontology_rcc_debug.yaml"
ONTOLOGY_YAML="${ONTOLOGY_YAML:-$ONTOLOGY_YAML_DEFAULT}"

CONCEPT_BANK_CSV_DEFAULT="${SRC_DIR}/explainability/concept/ontology/concepts_rcc_debug.csv"
CONCEPT_BANK_CSV="${CONCEPT_BANK_CSV:-$CONCEPT_BANK_CSV_DEFAULT}"

# Experiment-level (Stage 1/2)
EXP_ROOT_DEFAULT="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3"
MODEL_NAME_DEFAULT="moco_v3_ssl_linear_best"
BACKBONE_NAME_DEFAULT="vit_small_patch16_224"

# VLM config for concept bank
VLM_CONTROLLER_DEFAULT="http://localhost:10000"
VLM_MODEL_DEFAULT="llava-med-v1.5-mistral-7b"
PRESENCE_THRESHOLD_DEFAULT="0.3"

# Se START_LOCAL_VLM=1, run_full_xai lancerà un server LLaVA-Med locale
# (controller + model_worker) prima di build_concept_bank e lo killerà alla fine.
START_LOCAL_VLM="${START_LOCAL_VLM:-0}"

# Path o HF id del modello LLaVA-Med
VLM_MODEL_PATH_DEFAULT="microsoft/llava-med-v1.5-mistral-7b"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-$VLM_MODEL_PATH_DEFAULT}"
VLM_WARMUP_SECONDS="${VLM_WARMUP_SECONDS:-120}"

# Path al repo e al python di LLaVA-Med (override via env se necessario)
LLAVA_REPO_ROOT_DEFAULT="/home/mla_group_01/LLaVA-Med"
LLAVA_REPO_ROOT="${LLAVA_REPO_ROOT:-$LLAVA_REPO_ROOT_DEFAULT}"

LLAVA_PYTHON_BIN_DEFAULT="/home/mla_group_01/llava-med-venv/bin/python"
LLAVA_PYTHON_BIN="${LLAVA_PYTHON_BIN:-$LLAVA_PYTHON_BIN_DEFAULT}"

# ------------------- helper: LLaVA-Med server locale -------------------
start_local_vlm() {
  if [[ "${START_LOCAL_VLM}" != "1" ]]; then
    return 0
  fi

  echo "[INFO] Starting local LLaVA-Med controller on ${VLM_CONTROLLER}"
  echo "[INFO]   LLAVA_REPO_ROOT=${LLAVA_REPO_ROOT}"
  echo "[INFO]   LLAVA_PYTHON_BIN=${LLAVA_PYTHON_BIN}"

  if [[ ! -x "${LLAVA_PYTHON_BIN}" ]]; then
    echo "[ERROR] LLAVA_PYTHON_BIN='${LLAVA_PYTHON_BIN}' non eseguibile; controlla il venv LLaVA-Med." >&2
    return 1
  fi

  if [[ ! -d "${LLAVA_REPO_ROOT}" ]]; then
    echo "[ERROR] LLAVA_REPO_ROOT='${LLAVA_REPO_ROOT}' non esiste; clona il repo LLaVA-Med lì o override via env." >&2
    return 1
  fi

  pushd "${LLAVA_REPO_ROOT}" >/dev/null

  "${LLAVA_PYTHON_BIN}" -m llava.serve.controller \
    --host "0.0.0.0" \
    --port 10000 \
    > /tmp/llava_controller.log 2>&1 &
  VLM_CTRL_PID=$!
  sleep 5

  "${LLAVA_PYTHON_BIN}" -m llava.serve.model_worker \
    --host "0.0.0.0" \
    --controller "${VLM_CONTROLLER}" \
    --port 40000 \
    --worker "http://127.0.0.1:40000" \
    --model-path "${VLM_MODEL_PATH}" \
    --multi-modal \
    > /tmp/llava_worker.log 2>&1 &
  VLM_WORKER_PID=$!

  popd >/dev/null

  echo "[INFO] Waiting ${VLM_WARMUP_SECONDS}s for VLM to load weights..."
  sleep "${VLM_WARMUP_SECONDS}"
}

stop_local_vlm() {
  if [[ "${START_LOCAL_VLM}" != "1" ]]; then
    return 0
  fi
  echo "[INFO] Stopping local LLaVA-Med server"
  if [[ -n "${VLM_WORKER_PID:-}" ]]; then
    kill "${VLM_WORKER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${VLM_CTRL_PID:-}" ]]; then
    kill "${VLM_CTRL_PID}" 2>/dev/null || true
  fi
}

VENV_PATH="${VENV_PATH:-}"   # opzionale: export VENV_PATH=/path/to/venv
EXP_ROOT="${EXP_ROOT:-$EXP_ROOT_DEFAULT}"
MODEL_NAME="${MODEL_NAME:-$MODEL_NAME_DEFAULT}"
BACKBONE_NAME="${BACKBONE_NAME:-$BACKBONE_NAME_DEFAULT}"
VLM_CONTROLLER="${VLM_CONTROLLER:-$VLM_CONTROLLER_DEFAULT}"
VLM_MODEL="${VLM_MODEL:-$VLM_MODEL_DEFAULT}"
PRESENCE_THRESHOLD="${PRESENCE_THRESHOLD:-$PRESENCE_THRESHOLD_DEFAULT}"

# Flags esperimento (override con export ONLY_SPATIAL=1 etc se ti serve)
ONLY_SPATIAL="${ONLY_SPATIAL:-0}"
ONLY_CONCEPT="${ONLY_CONCEPT:-0}"

# ------------------- logging -------------------

echo "[INFO] run_full_xai.sh starting"
echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] EXP_ROOT=${EXP_ROOT}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] BACKBONE_NAME=${BACKBONE_NAME}"
echo "[INFO] VLM_CONTROLLER=${VLM_CONTROLLER}"
echo "[INFO] VLM_MODEL=${VLM_MODEL}"
echo "[INFO] PRESENCE_THRESHOLD=${PRESENCE_THRESHOLD}"

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

  # 0a) concept_candidates_rcc.csv (train WDS -> PNG + CSV)
  echo "[INFO] Stage 0a: building concept_candidates_rcc.csv"
  python3 -m explainability.concept.ontology.build_concept_candidates \
    --train-dir "${TRAIN_WDS_DIR}" \
    --pattern "shard-*.tar" \
    --image-key "img.jpg;jpg;jpeg;png" \
    --meta-key "meta.json;json" \
    --out-csv "${CANDIDATES_CSV}" \
    --images-root "${CANDIDATES_IMG_ROOT}"

  # 0b) concepts_rcc_debug.csv (VLM su candidates)
  echo "[INFO] Stage 0b: building concepts_rcc_debug.csv via VLM"
  start_local_vlm
  python3 -m explainability.concept.ontology.build_concept_bank \
    --ontology "${ONTOLOGY_YAML}" \
    --images-csv "${CANDIDATES_CSV}" \
    --controller "${VLM_CONTROLLER}" \
    --model-name "${VLM_MODEL}" \
    --out-csv "${CONCEPT_BANK_CSV}" \
    --presence-threshold "${PRESENCE_THRESHOLD}" \
    --max-images 100
  stop_local_vlm

  # hard check: concept bank deve avere almeno header + 1 riga
  lines_after=$(wc -l < "${CONCEPT_BANK_CSV}")
  if [[ "${lines_after}" -le 1 ]]; then
    echo "[ERROR] Concept bank ${CONCEPT_BANK_CSV} still empty after Stage 0 (lines=${lines_after}). Aborting."
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

# flags optionali: se vuoi supportare solo spatial/solo concept, estendi run_explainability.py di conseguenza.
# Al momento il tuo run_explainability.py non ha --only-spatial / --only-concept, quindi li ignoriamo.
if [[ "${ONLY_SPATIAL}" == "1" ]]; then
  echo "[WARN] ONLY_SPATIAL=1 set, but run_explainability.py non supporta ancora il flag; eseguo comunque full pipeline."
fi
if [[ "${ONLY_CONCEPT}" == "1" ]]; then
  echo "[WARN] ONLY_CONCEPT=1 set, ma run_explainability.py non supporta ancora il flag; eseguo comunque full pipeline."
fi

echo "[INFO] Stage 1/2: running experiment-level explainability:"
echo "[INFO]   ${ORCH_CMD[*]}"
"${ORCH_CMD[@]}"

echo "[OK] run_full_xai.sh completed."
