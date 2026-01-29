#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# XAI pipeline orchestrator (robust paths via explainability.paths)
#
# Phases:
#   (1/5) calibration + shortlist (skip if outputs already exist)
#   (2/5) concept NO-ROI (model-independent)
#   (3/5) spatial attention rollout (attention_rollout.py) on ALL ablations
#   (4/5) ROI mask + concept ROI on ALL ablations (run_spatial-concept.py)
#         (reuses precomputed rollout from 3/5 when available)
#   (5/5) ROI vs NO-ROI comparison (per ablation)
# ------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJ:-$(cd "${SCRIPT_DIR}/../.." && pwd)}}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venvs/xai/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[FATAL] python not found (set PYTHON_BIN or load a python module)." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

LOG_LEVEL="${LOG_LEVEL:-INFO}"
RUN_ID="${RUN_ID:-latest}"
EXP_PREFIX="${EXP_PREFIX:-exp_20251109_}"

# Resolve canonical paths from explainability.paths (respects XAI_ROOT / MODELS_ROOT env overrides).
MODELS_ROOT="${MODELS_ROOT:-$("${PYTHON_BIN}" -c 'from explainability.paths import resolve_models_root; print(resolve_models_root())')}"
XAI_ROOT="$("${PYTHON_BIN}" -c 'from explainability.paths import XAI_ROOT; print(XAI_ROOT)')"
CALIB_META_DIR="$("${PYTHON_BIN}" -c 'from explainability.paths import CALIBRATION_PATHS; print(CALIBRATION_PATHS.metadata_dir)')"
CALIB_ANALYSIS_DIR="$("${PYTHON_BIN}" -c 'from explainability.paths import CALIBRATION_PATHS; print(CALIBRATION_PATHS.analysis_dir)')"
SHORTLIST_YAML="$("${PYTHON_BIN}" -c 'from explainability.paths import CALIBRATION_PATHS; print(CALIBRATION_PATHS.shortlist_yaml)')"
SPATIAL_OUTPUT_ROOT="$("${PYTHON_BIN}" -c 'import pathlib; from explainability.paths import XAI_ROOT; print(pathlib.Path(XAI_ROOT) / "spatial")')"
PIPELINE_CFG_ROOT="$("${PYTHON_BIN}" -c 'import pathlib; from explainability.paths import XAI_ROOT; print(pathlib.Path(XAI_ROOT) / "spatial" / "_pipeline_cfgs")')"

# Canonical calibration outputs expected by concept NO-ROI (and downstream ROI).
export CALIBRATION_METADATA_DIR="${CALIB_META_DIR}"
export CONCEPT_SHORTLIST_YAML="${SHORTLIST_YAML}"

# Deterministic default for dataset (override via env).
export WDS_TEST_DIR="${WDS_TEST_DIR:-/mnt/beegfs-compat/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test}"

echo "[PRE] Host=$(hostname)"
echo "[PRE] Project=${PROJECT_ROOT}"
echo "[PRE] Python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "[PRE] MODELS_ROOT=${MODELS_ROOT}"
echo "[PRE] XAI_ROOT=${XAI_ROOT}"
echo "[PRE] RUN_ID=${RUN_ID}"
echo "[PRE] EXP_PREFIX=${EXP_PREFIX}"
echo "[PRE] WDS_TEST_DIR(raw)=${WDS_TEST_DIR}"
printf "[PRE] WDS_TEST_DIR(%%q)=%q\n" "${WDS_TEST_DIR}"

# Hard fail early in the same environment the job runs in.
if [[ ! -d "${WDS_TEST_DIR}" ]]; then
  echo "[FATAL] WDS_TEST_DIR not visible on this node: ${WDS_TEST_DIR}" >&2
  echo "[FATAL] Debug:" >&2
  ls -ld /mnt/beegfs-compat >&2 || true
  df -hT /mnt/beegfs-compat >&2 || true
  ls -ld "${WDS_TEST_DIR}" >&2 || true
  exit 2
fi

CALIB_TEXT_FEATS="${CALIB_META_DIR}/text_features.pt"
CALIB_CONCEPTS_JSON="${CALIB_META_DIR}/concepts.json"
CALIB_SCORES_NPY="${CALIB_META_DIR}/scores_fp32.npy"
CALIB_LABELS_NPY="${CALIB_META_DIR}/labels.npy"
CALIB_METRICS_CSV="${CALIB_ANALYSIS_DIR}/metrics_per_class.csv"

CALIB_CFG="${CALIB_CFG:-$("${PYTHON_BIN}" -c 'from explainability.paths import CALIBRATION_CONFIG_YAML; print(CALIBRATION_CONFIG_YAML)')}"
NO_ROI_CFG="${NO_ROI_CFG:-$("${PYTHON_BIN}" -c 'from explainability.paths import NO_ROI_CONFIG_YAML; print(NO_ROI_CONFIG_YAML)')}"
ROI_CFG="${ROI_CFG:-$("${PYTHON_BIN}" -c 'from explainability.paths import SPATIAL_CONCEPT_CONFIG_YAML; print(SPATIAL_CONCEPT_CONFIG_YAML)')}"
COMP_CFG="${COMP_CFG:-$("${PYTHON_BIN}" - <<'PY'
from explainability.paths import CONFIG_DIR
from pathlib import Path
p_new = Path(CONFIG_DIR) / "comparison.yaml"
p_old = Path(CONFIG_DIR) / "comparision.yaml"
print(p_new if p_new.exists() else p_old)
PY
)}"
SPATIAL_CFG="${SPATIAL_CFG:-$("${PYTHON_BIN}" -c 'from explainability.paths import SPATIAL_CONFIG_YAML; print(SPATIAL_CONFIG_YAML)')}"
if [[ ! -f "${SPATIAL_CFG}" ]]; then
  # Fallback: reuse ROI config if spatial.yaml is not present in this branch.
  SPATIAL_CFG="${ROI_CFG}"
fi

CALIB_PY="${CALIB_PY:-${PROJECT_ROOT}/src/explainability/concept/calibration/calibration.py}"
SHORTLIST_PY="${SHORTLIST_PY:-${PROJECT_ROOT}/src/explainability/concept/calibration/build_shortlist.py}"
NO_ROI_PY="${NO_ROI_PY:-${PROJECT_ROOT}/src/explainability/concept/run_no_roi.py}"
ATTN_ROLLOUT_PY="${ATTN_ROLLOUT_PY:-${PROJECT_ROOT}/src/explainability/spatial/attention_rollout.py}"
SPATIAL_CONCEPT_PY="${SPATIAL_CONCEPT_PY:-${PROJECT_ROOT}/src/explainability/run_spatial-concept.py}"
COMP_PY="${COMP_PY:-${PROJECT_ROOT}/src/explainability/run_comparision.py}"

mkdir -p "${PROJECT_ROOT}/src/logs/xai"
mkdir -p "${SPATIAL_OUTPUT_ROOT}"
mkdir -p "${PIPELINE_CFG_ROOT}/${RUN_ID}"

log_skip() {
  # usage: log_skip "msg" ["details..."]
  local msg="$1"; shift || true
  echo "  [SKIP] ${msg}"
  if [[ "$#" -gt 0 ]]; then
    printf "         %b\n" "$*"
  fi
}

log_warn() {
  local msg="$1"; shift || true
  echo "  [WARN] ${msg}" >&2
  if [[ "$#" -gt 0 ]]; then
    printf "         %b\n" "$*" >&2
  fi
}

echo "[1/5] Calibration + deep validation + shortlist (skip if already present)"
need_calib=0
need_deep=0
need_shortlist=0
if [[ ! -f "${CALIB_TEXT_FEATS}" || ! -f "${CALIB_CONCEPTS_JSON}" || ! -f "${CALIB_SCORES_NPY}" || ! -f "${CALIB_LABELS_NPY}" ]]; then
  need_calib=1
fi
if [[ ! -f "${CALIB_METRICS_CSV}" ]]; then
  need_deep=1
fi
if [[ ! -f "${SHORTLIST_YAML}" ]]; then
  need_shortlist=1
fi

if [[ "${need_calib}" -eq 0 && "${need_deep}" -eq 0 && "${need_shortlist}" -eq 0 ]]; then
  echo "  [SKIP] Calibration outputs already present:"
  echo "         - ${CALIB_TEXT_FEATS}"
  echo "         - ${CALIB_CONCEPTS_JSON}"
  echo "         - ${CALIB_SCORES_NPY}"
  echo "         - ${CALIB_LABELS_NPY}"
  echo "         - ${CALIB_METRICS_CSV}"
  echo "         - ${SHORTLIST_YAML}"
else
  if [[ "${need_calib}" -eq 1 ]]; then
    echo "  [RUN] calibration.py calibrate --config ${CALIB_CFG}"
    "${PYTHON_BIN}" "${CALIB_PY}" calibrate --config "${CALIB_CFG}" --log-level "${LOG_LEVEL}"
  fi
  if [[ "${need_deep}" -eq 1 ]]; then
    echo "  [RUN] calibration.py deep-validate --metadata-dir ${CALIB_META_DIR} --out-dir ${CALIB_ANALYSIS_DIR}"
    "${PYTHON_BIN}" "${CALIB_PY}" deep-validate \
      --metadata-dir "${CALIB_META_DIR}" \
      --out-dir "${CALIB_ANALYSIS_DIR}" \
      --log-level "${LOG_LEVEL}"
  fi
  if [[ "${need_shortlist}" -eq 1 ]]; then
    if [[ ! -f "${CALIB_METRICS_CSV}" ]]; then
      echo "[FATAL] Missing metrics_per_class.csv after deep validation: ${CALIB_METRICS_CSV}" >&2
      exit 2
    fi
    echo "  [RUN] build_shortlist.py (uses calibration analysis to write shortlist)"
    "${PYTHON_BIN}" "${SHORTLIST_PY}"
  fi
fi

echo "[2/5] Concept NO-ROI (model-independent)"
if ! "${PYTHON_BIN}" "${NO_ROI_PY}" \
  --config "${NO_ROI_CFG}" \
  --test-dir "${WDS_TEST_DIR}" \
  --overwrite \
  --log-level "${LOG_LEVEL}"; then
  echo "[FATAL] NO-ROI failed (this is model-independent; fix upstream before continuing)." >&2
  exit 2
fi

echo "[3/5] Spatial attention rollout on ALL ablations (attention_rollout.py)"
shopt -s nullglob
for EXP_DIR in "${MODELS_ROOT}"/"${EXP_PREFIX}"*; do
  [[ -d "${EXP_DIR}" ]] || continue
  for ABL_DIR in "${EXP_DIR}"/exp_*_abl*; do
    [[ -d "${ABL_DIR}" ]] || continue

    MODEL_ID="$(basename "${ABL_DIR}")"
    OUT_DIR="${SPATIAL_OUTPUT_ROOT}/${MODEL_ID}/${RUN_ID}"
    CFG_PATH="${PIPELINE_CFG_ROOT}/${RUN_ID}/${MODEL_ID}.yaml"

    if [[ -f "${OUT_DIR}/index.csv" ]]; then
      echo "  [SKIP] ${MODEL_ID}: spatial already exists -> ${OUT_DIR}"
      continue
    fi

    CKPT_DIR="${ABL_DIR}/checkpoints"
    if [[ ! -d "${CKPT_DIR}" ]]; then
      log_skip "${MODEL_ID}: missing checkpoints/ dir" \
        "Debug:\n  ls -lah \"${ABL_DIR}\"\n  find \"${ABL_DIR}\" -maxdepth 2 -type d -print | sort"
      continue
    fi

    echo "  [CFG]  ${MODEL_ID} -> ${CFG_PATH}"

    # Generate per-model rollout config. Missing expected files => print [SKIP] and exit(10).
    set +e
    CFG_PATH="${CFG_PATH}" \
      ABL_DIR="${ABL_DIR}" \
      MODEL_ID="${MODEL_ID}" \
      RUN_ID="${RUN_ID}" \
      SPATIAL_OUTPUT_ROOT="${SPATIAL_OUTPUT_ROOT}" \
      SPATIAL_CFG="${SPATIAL_CFG}" \
      WDS_TEST_DIR="${WDS_TEST_DIR}" \
      "${PYTHON_BIN}" - <<'PY'
import os
import sys
from pathlib import Path
from typing import Optional
import yaml

abl_dir = Path(os.environ["ABL_DIR"]).expanduser().resolve()
model_id = str(os.environ["MODEL_ID"])
run_id = str(os.environ["RUN_ID"])
spatial_output_root = Path(os.environ["SPATIAL_OUTPUT_ROOT"]).expanduser().resolve()
spatial_cfg_path = Path(os.environ["SPATIAL_CFG"]).expanduser().resolve()
cfg_path = Path(os.environ["CFG_PATH"]).expanduser().resolve()
wds_test_dir_env = str(os.getenv("WDS_TEST_DIR", "")).strip()

def deep_get(cfg, keys, default=None):
    cur = cfg
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def skip(reason: str):
    print(f"[SKIP] [{model_id}] {reason}", file=sys.stderr)
    raise SystemExit(10)

def crash(reason: str):
    print(f"[CRASH] [{model_id}] {reason}", file=sys.stderr)
    raise SystemExit(1)

# -------------------- load SPATIAL config (selection + discard_ratio) --------------------
sp_cfg = {}
if spatial_cfg_path.exists():
    try:
        obj = yaml.safe_load(spatial_cfg_path.read_text())
        sp_cfg = obj if isinstance(obj, dict) else {}
    except Exception as e:
        # not fatal; use defaults
        sp_cfg = {}

sel_cfg = sp_cfg.get("selection", {}) if isinstance(sp_cfg.get("selection", {}), dict) else {}
per_cfg = sel_cfg.get("per_class", {}) if isinstance(sel_cfg.get("per_class", {}), dict) else {}
gl_cfg = sel_cfg.get("global_low_conf", {}) if isinstance(sel_cfg.get("global_low_conf", {}), dict) else {}

selection = {
    "per_class": {
        "topk_tp": int(per_cfg.get("topk_tp", 2) or 2),
        "topk_fp": int(per_cfg.get("topk_fp", 2) or 2),
        "topk_fn": int(per_cfg.get("topk_fn", 2) or 2),
    }
}
try:
    gl_topk = int(gl_cfg.get("topk", 0) or 0)
    if gl_topk > 0:
        selection["global_low_conf"] = {"topk": gl_topk}
except Exception:
    pass

discard_ratio = 0.90
try:
    xai_cfg = sp_cfg.get("xai", {}) if isinstance(sp_cfg.get("xai", {}), dict) else {}
    ar = xai_cfg.get("attn_rollout", {}) if isinstance(xai_cfg.get("attn_rollout", {}), dict) else {}
    discard_ratio = float(ar.get("discard_ratio", discard_ratio))
except Exception:
    discard_ratio = 0.90

# -------------------- checkpoints: if missing, SKIP --------------------
ckpt_dir = abl_dir / "checkpoints"
if not ckpt_dir.exists():
    skip("missing checkpoints/ dir")

# prefer explicit env overrides
env_bb = os.getenv("XAI_SSL_BACKBONE_CKPT", "").strip()
env_hd = os.getenv("XAI_SSL_HEAD_CKPT", "").strip()

def one_existing(p: str) -> Optional[Path]:
    if not p:
        return None
    q = Path(p).expanduser().resolve()
    return q if q.exists() else None

def find_backbone() -> Optional[Path]:
    if env_bb:
        return one_existing(env_bb)
    bb = sorted(list(ckpt_dir.glob("*_ssl_best.pt")) + list(ckpt_dir.glob("*_ssl_best.pth")))
    bb = [p for p in bb if "linear" not in p.name]
    return bb[0].resolve() if len(bb) >= 1 else None

def find_head() -> Optional[Path]:
    if env_hd:
        return one_existing(env_hd)
    hd = sorted(list(ckpt_dir.glob("*_ssl_linear_best.pt")) + list(ckpt_dir.glob("*_ssl_linear_best.pth")))
    return hd[0].resolve() if len(hd) >= 1 else None

ssl_backbone_ckpt = find_backbone()
if ssl_backbone_ckpt is None:
    skip(f"missing backbone checkpoint (*_ssl_best.(pt|pth) excluding '*linear*') in {ckpt_dir}")

ssl_head_ckpt = find_head()
if ssl_head_ckpt is None:
    skip(f"missing head checkpoint (*_ssl_linear_best.(pt|pth)') in {ckpt_dir}")

# -------------------- eval dir: if missing/ambiguous, SKIP --------------------
eval_root = abl_dir / "eval"
if not eval_root.exists():
    skip("missing eval/ dir")

eval_override = os.getenv("XAI_EVAL_DIR", "").strip()
if eval_override:
    eval_dir = Path(eval_override).expanduser().resolve()
    if not eval_dir.exists():
        skip(f"XAI_EVAL_DIR points to non-existing path: {eval_dir}")
else:
    eval_dirs = sorted([p for p in eval_root.iterdir() if p.is_dir()])
    if len(eval_dirs) != 1:
        skip(f"eval/ is ambiguous (found {len(eval_dirs)} dirs); set XAI_EVAL_DIR if you want to force one")
    eval_dir = eval_dirs[0].resolve()

# -------------------- eval config: if missing, SKIP --------------------
eval_cfg_path = eval_dir / "config_resolved.yaml"
eval_cfg_override = os.getenv("XAI_EVAL_CONFIG", "").strip()
if eval_cfg_override:
    eval_cfg_path = Path(eval_cfg_override).expanduser().resolve()

if not eval_cfg_path.exists():
    skip("missing eval config_resolved.yaml (or set XAI_EVAL_CONFIG)")

try:
    obj = yaml.safe_load(eval_cfg_path.read_text())
    eval_cfg = obj if isinstance(obj, dict) else {}
except Exception as e:
    skip(f"cannot parse eval config YAML: {eval_cfg_path} ({e})")

class_order = deep_get(eval_cfg, ("labels", "class_order"), None)
if not (isinstance(class_order, list) and all(isinstance(x, str) for x in class_order) and len(class_order) >= 2):
    skip("missing/invalid labels.class_order in eval config")

backbone_name = deep_get(eval_cfg, ("model", "backbone_name"), None)
if not (isinstance(backbone_name, str) and backbone_name.strip()):
    skip("missing/invalid model.backbone_name in eval config")
backbone_name = backbone_name.strip()

backend = str(deep_get(eval_cfg, ("data", "backend"), "") or "").strip().lower()
if backend not in ("webdataset", "imagefolder"):
    skip("missing/invalid data.backend in eval config (expected 'webdataset' or 'imagefolder')")

img_size = deep_get(eval_cfg, ("data", "img_size"), None)
imagenet_norm = deep_get(eval_cfg, ("data", "imagenet_norm"), None)
if not isinstance(img_size, int):
    skip("missing/invalid data.img_size in eval config")
if not isinstance(imagenet_norm, bool):
    skip("missing/invalid data.imagenet_norm in eval config")

num_workers = int(deep_get(eval_cfg, ("data", "num_workers"), 8) or 8)
batch_size = int(deep_get(eval_cfg, ("data", "batch_size"), 1) or 1)

data_cfg = {
    "backend": backend,
    "img_size": int(img_size),
    "imagenet_norm": bool(imagenet_norm),
    "num_workers": num_workers,
    "batch_size": batch_size,
}

# Prefer eval-config test_dir, but allow env WDS_TEST_DIR as fallback for webdataset.
if data_cfg["backend"] == "webdataset":
    w = deep_get(eval_cfg, ("data", "webdataset"), None)
    if not isinstance(w, dict):
        skip("missing data.webdataset block in eval config")
    test_dir = str(w.get("test_dir") or "").strip()
    if not test_dir:
        if wds_test_dir_env:
            test_dir = wds_test_dir_env
        else:
            skip("missing data.webdataset.test_dir in eval config (and WDS_TEST_DIR env not set)")
    data_cfg["webdataset"] = {
        "test_dir": test_dir,
        "pattern": str(w.get("pattern", "shard-*.tar")),
        "image_key": str(w.get("image_key", "img.jpg;jpg;jpeg;png")),
        "meta_key": str(w.get("meta_key", "meta.json;json")),
    }
elif data_cfg["backend"] == "imagefolder":
    ifd = deep_get(eval_cfg, ("data", "imagefolder"), None)
    if not isinstance(ifd, dict):
        skip("missing data.imagefolder block in eval config")
    test_dir = str(ifd.get("test_dir") or "").strip()
    if not test_dir:
        skip("missing data.imagefolder.test_dir in eval config")
    data_cfg["imagefolder"] = {"test_dir": test_dir}

# -------------------- required eval artifacts for attention_rollout: if missing, SKIP --------------------
pred_csv = eval_dir / "predictions.csv"
logits_npy = eval_dir / "logits_test.npy"
if not pred_csv.exists():
    skip(f"missing eval predictions: {pred_csv}")
if not logits_npy.exists():
    skip(f"missing eval logits: {logits_npy}")

cfg = {
    "experiment": {
        "seed": int(deep_get(eval_cfg, ("experiment", "seed"), 0) or 0),
        "run_id": run_id,
        "outputs_root": str(spatial_output_root),
    },
    "runtime": {"device": "cuda"},
    "model": {
        "name": model_id,
        "arch_hint": "ssl_linear",
        "backbone_name": backbone_name,
        "ssl_backbone_ckpt": str(ssl_backbone_ckpt),
        "ssl_head_ckpt": str(ssl_head_ckpt),
    },
    "evaluation_inputs": {
        "eval_run_dir": str(eval_dir),
        "predictions_csv": "predictions.csv",
        "logits_npy": "logits_test.npy",
    },
    "labels": {"class_order": class_order},
    "data": data_cfg,
    "selection": selection,
    "xai": {
        "methods": ["attn_rollout"],
        "attn_rollout": {"discard_ratio": float(discard_ratio)},
    },
}

cfg_path.parent.mkdir(parents=True, exist_ok=True)
cfg_path.write_text(
    yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=4096).rstrip() + "\n",
    encoding="utf-8",
)
PY
    rc=$?
    set -e

    if [[ "${rc}" -eq 10 ]]; then
      # Python already printed the specific reason.
      continue
    elif [[ "${rc}" -ne 0 ]]; then
      log_skip "${MODEL_ID}: config generation crashed (rc=${rc})."
      continue
    fi

    if [[ ! -f "${CFG_PATH}" ]]; then
      log_skip "${MODEL_ID}: config not generated (unexpected: CFG_PATH missing)" \
        "Debug:\n  ls -lah \"$(dirname "${CFG_PATH}")\""
      continue
    fi

    echo "  [RUN] attention_rollout.py --config ${CFG_PATH}"
    set +e
    "${PYTHON_BIN}" "${ATTN_ROLLOUT_PY}" --config "${CFG_PATH}"
    rc=$?
    set -e
    if [[ "${rc}" -ne 0 ]]; then
      log_skip "${MODEL_ID}: attention rollout failed (rc=${rc})" \
        "Debug:\n  ls -lah \"${CKPT_DIR}\"\n  ls -lah \"${OUT_DIR}\" || true\n  head -n 40 \"${CFG_PATH}\" || true"
      continue
    fi

  done
done
shopt -u nullglob

echo "[4/5] ROI + concept on ALL ablations (reuse rollout from 3/5 when available)"
set +e
"${PYTHON_BIN}" "${SPATIAL_CONCEPT_PY}" \
  --models-root "${MODELS_ROOT}" \
  --exp-prefix "${EXP_PREFIX}" \
  --config "${ROI_CFG}" \
  --run-id "${RUN_ID}" \
  --reuse-attn-rollout \
  --attn-rollout-outputs-root "${SPATIAL_OUTPUT_ROOT}" \
  --attn-rollout-run-id "${RUN_ID}" \
  --log-level "${LOG_LEVEL}"
rc=$?
set -e
if [[ "${rc}" -ne 0 ]]; then
  log_warn "ROI+concept step returned non-zero (rc=${rc}). Continuing to comparisons anyway."
fi

echo "[5/5] ROI vs NO-ROI comparison (PER ABLATION)"
shopt -s nullglob
for EXP_DIR in "${MODELS_ROOT}"/"${EXP_PREFIX}"*; do
  [[ -d "${EXP_DIR}" ]] || continue
  for ABL_DIR in "${EXP_DIR}"/exp_*_abl*; do
    [[ -d "${ABL_DIR}" ]] || continue
    MODEL_ID="$(basename "${ABL_DIR}")"
    echo "  [CMP] ${MODEL_ID}"
    set +e
    "${PYTHON_BIN}" "${COMP_PY}" --model-root "${ABL_DIR}" --config "${COMP_CFG}" --log-level "${LOG_LEVEL}"
    rc=$?
    set -e
    if [[ "${rc}" -ne 0 ]]; then
      log_skip "${MODEL_ID}: comparison failed (rc=${rc})"
      continue
    fi
  done
done
shopt -u nullglob

echo "DONE run_id=${RUN_ID}"
