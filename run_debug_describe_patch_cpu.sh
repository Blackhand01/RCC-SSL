#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/mla_group_01/rcc-ssrl"
VENV_PATH="${REPO_ROOT}/.venvs/xai-concept_pa-llava"
IMAGE_PATH="${REPO_ROOT}/src/explainability/concept/ontology/concept_candidate_patches/CHROMO/HP20.2506_13_27492_3419.png"
MODEL_ROOT="${REPO_ROOT}/Pathology-LLaVA-raw"

echo "[INFO] REPO_ROOT  = ${REPO_ROOT}"
echo "[INFO] VENV_PATH  = ${VENV_PATH}"
echo "[INFO] IMAGE_PATH = ${IMAGE_PATH}"
echo "[INFO] MODEL_ROOT = ${MODEL_ROOT}"

# 1) Attiva venv
if [[ ! -d "${VENV_PATH}" ]]; then
  echo "[ERRORE] VENV non trovato: ${VENV_PATH}" >&2
  exit 1
fi
source "${VENV_PATH}/bin/activate"

# 2) Esporta PYTHONPATH corretto
export PYTHONPATH="${REPO_ROOT}/src:\
${REPO_ROOT}/Pathology-LLaVA-raw:\
${REPO_ROOT}/Pathology-LLaVA-raw/xtuner_add:\
${REPO_ROOT}/PA-LLaVA:\
${REPO_ROOT}/PA-LLaVA/xtuner_add:${PYTHONPATH:-}"

echo "[INFO] PYTHONPATH = ${PYTHONPATH}"

# 3) Vai nella root del repo (così i path relativi funzionano)
cd "${REPO_ROOT}"

# 4) Lancia il debug
python -m explainability.concept.ontology.debug_describe_patch \
  --image "${IMAGE_PATH}" \
  --model-root "${MODEL_ROOT}" \
  --device cpu \
  --runs 3
