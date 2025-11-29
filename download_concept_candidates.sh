#!/usr/bin/env bash
set -euo pipefail

# === PARAMETRI REMOTI (modifica SOLO se necessario) ===
REMOTE_USER="mla_group_01"
REMOTE_HOST="login1.cluster.lan"

REMOTE_BASE="/home/mla_group_01/rcc-ssrl/src/explainability/concept/ontology"
REMOTE_SUBDIR="concept_candidates_images"

# === DOVE SALVARE IN LOCALE ===
# Usa il primo argomento come cartella di destinazione, altrimenti la directory corrente
LOCAL_BASE="${1:-.}"

echo "[INFO] Scarico ${REMOTE_BASE}/${REMOTE_SUBDIR} da ${REMOTE_USER}@${REMOTE_HOST} in ${LOCAL_BASE}"
mkdir -p "${LOCAL_BASE}"

# Comando:
# - lato remoto: cd nella dir base, crea un tar.gz dello subdir e lo manda su stdout
# - lato locale: legge dallo stdin e lo estrae dentro LOCAL_BASE
ssh "${REMOTE_USER}@${REMOTE_HOST}" \
  "cd '${REMOTE_BASE}' && tar -czf - '${REMOTE_SUBDIR}'" \
  | tar -xzf - -C "${LOCAL_BASE}"

echo "[OK] Download completato."
echo "[OK] Contenuto in: ${LOCAL_BASE}/${REMOTE_SUBDIR}"
