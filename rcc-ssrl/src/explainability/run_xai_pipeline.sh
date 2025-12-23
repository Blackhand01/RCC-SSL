#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# XAI pipeline orchestrator (hard paths, deterministic default RUN_ID)
#
# Phases:
#   (1/5) calibration + shortlist (skip if outputs already exist)
#   (2/5) concept NO-ROI (model-independent)
#   (3/5) spatial attention rollout (attention_rollout.py) on ALL ablations
#   (4/5) ROI mask + concept ROI on ALL ablations (run_spatial-concept.py)
#         (reuses precomputed rollout from 3/5 when available)
#   (5/5) ROI vs NO-ROI comparison (per ablation)
# ------------------------------------------------------------------

PROJECT_ROOT="/home/mla_group_01/rcc-ssrl"
PYTHON_BIN="/home/mla_group_01/rcc-ssrl/.venvs/xai/bin/python"

MODELS_ROOT="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/models"
EXP_PREFIX="exp_20251109_1815"

EXP_DIRS=(
  "/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/models/exp_20251109_181534_i_jepa"
  "/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/models/exp_20251109_181538_dino_v3"
  "/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/models/exp_20251109_181540_moco_v3"
  "/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/models/exp_20251109_181551_ibot"
)

CALIB_CFG="/home/mla_group_01/rcc-ssrl/src/explainability/configs/calibration.yaml"
NO_ROI_CFG="/home/mla_group_01/rcc-ssrl/src/explainability/configs/no_roi.yaml"
ROI_CFG="/home/mla_group_01/rcc-ssrl/src/explainability/configs/roi.yaml"
COMP_CFG="/home/mla_group_01/rcc-ssrl/src/explainability/configs/comparision.yaml"

CALIB_PY="/home/mla_group_01/rcc-ssrl/src/explainability/concept/calibration/calibration.py"
SHORTLIST_PY="/home/mla_group_01/rcc-ssrl/src/explainability/concept/calibration/build_shortlist.py"

NO_ROI_PY="/home/mla_group_01/rcc-ssrl/src/explainability/concept/run_no_roi.py"
ATTN_ROLLOUT_PY="/home/mla_group_01/rcc-ssrl/src/explainability/spatial/attention_rollout.py"
SPATIAL_CONCEPT_PY="/home/mla_group_01/rcc-ssrl/src/explainability/run_spatial-concept.py"
COMP_PY="/home/mla_group_01/rcc-ssrl/src/explainability/run_comparision.py"

OUTPUT_ROOT="/home/mla_group_01/rcc-ssrl/src/explainability/output"
SPATIAL_OUTPUT_ROOT="${OUTPUT_ROOT}/spatial"
PIPELINE_CFG_ROOT="${SPATIAL_OUTPUT_ROOT}/_pipeline_cfgs"

LOG_LEVEL="${LOG_LEVEL:-INFO}"
RUN_ID="${RUN_ID:-latest}"

mkdir -p /home/mla_group_01/rcc-ssrl/src/logs/xai
mkdir -p "${SPATIAL_OUTPUT_ROOT}"
mkdir -p "${PIPELINE_CFG_ROOT}/${RUN_ID}"

cd "${PROJECT_ROOT}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# Canonical calibration outputs expected by concept NO-ROI (and downstream ROI).
export CALIBRATION_METADATA_DIR="${OUTPUT_ROOT}/calibration/metadata"
export CONCEPT_SHORTLIST_YAML="${PROJECT_ROOT}/src/explainability/configs/concepts_shortlist.yaml"

# Deterministic default (override if you want).
export WDS_TEST_DIR="${WDS_TEST_DIR:-/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test}"

echo "[PRE] Host=$(hostname)"
echo "[PRE] Python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "[PRE] WDS_TEST_DIR(raw)=${WDS_TEST_DIR}"
printf "[PRE] WDS_TEST_DIR(%%q)=%q\n" "${WDS_TEST_DIR}"

# Hard fail early in the same environment the job runs in.
if [[ ! -d "${WDS_TEST_DIR}" ]]; then
  echo "[FATAL] WDS_TEST_DIR not visible on this node: ${WDS_TEST_DIR}"
  echo "[FATAL] Debug:"
  ls -ld /beegfs-scratch || true
  df -hT /beegfs-scratch || true
  ls -ld "${WDS_TEST_DIR}" || true
  exit 2
fi

CALIB_META_DIR="${OUTPUT_ROOT}/calibration/metadata"
CALIB_TEXT_FEATS="${CALIB_META_DIR}/text_features.pt"
CALIB_CONCEPTS_JSON="${CALIB_META_DIR}/concepts.json"
SHORTLIST_YAML="${PROJECT_ROOT}/src/explainability/configs/concepts_shortlist.yaml"

echo "[1/5] Calibration + shortlist (skip if already present)"
if [[ -f "${CALIB_TEXT_FEATS}" && -f "${CALIB_CONCEPTS_JSON}" && -f "${SHORTLIST_YAML}" ]]; then
  echo "  [SKIP] Calibration outputs already present:"
  echo "         - ${CALIB_TEXT_FEATS}"
  echo "         - ${CALIB_CONCEPTS_JSON}"
  echo "         - ${SHORTLIST_YAML}"
else
  echo "  [RUN] calibration.py --config ${CALIB_CFG}"
  "${PYTHON_BIN}" "${CALIB_PY}" --config "${CALIB_CFG}"
  echo "  [RUN] build_shortlist.py (uses calibration analysis to write shortlist)"
  "${PYTHON_BIN}" "${SHORTLIST_PY}"
fi

echo "[2/5] Concept NO-ROI (model-independent)"
"${PYTHON_BIN}" "${NO_ROI_PY}" \
  --config "${NO_ROI_CFG}" \
  --test-dir "${WDS_TEST_DIR}" \
  --overwrite \
  --log-level "${LOG_LEVEL}"

echo "[3/5] Spatial attention rollout on ALL ablations (attention_rollout.py)"
for EXP_DIR in "${EXP_DIRS[@]}"; do
  if [[ ! -d "${EXP_DIR}" ]]; then
    echo "  [WARN] Missing EXP_DIR: ${EXP_DIR} (skipping)"
    continue
  fi

  # Only ablation dirs, ignore "reporting" and other folders.
  shopt -s nullglob
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
      echo "  [SKIP] ${MODEL_ID}: missing checkpoints/ dir"
      continue
    fi

    BACKBONE_CAND=( "${CKPT_DIR}"/*_ssl_best.pt "${CKPT_DIR}"/*_ssl_best.pth )
    HEAD_CAND=( "${CKPT_DIR}"/*_ssl_linear_best.pt "${CKPT_DIR}"/*_ssl_linear_best.pth )

    BACKBONE_OK=()
    for p in "${BACKBONE_CAND[@]}"; do
      [[ -f "${p}" ]] || continue
      if [[ "${p,,}" == *linear* ]]; then
        continue
      fi
      BACKBONE_OK+=( "${p}" )
    done

    HEAD_OK=()
    for p in "${HEAD_CAND[@]}"; do
      [[ -f "${p}" ]] || continue
      HEAD_OK+=( "${p}" )
    done

    if [[ "${#BACKBONE_OK[@]}" -lt 1 || "${#HEAD_OK[@]}" -lt 1 ]]; then
      echo "  [SKIP] ${MODEL_ID}: missing *_ssl_best.(pt|pth) or *_ssl_linear_best.(pt|pth) in ${CKPT_DIR}"
      continue
    fi

    echo "  [CFG]  ${MODEL_ID} -> ${CFG_PATH}"
    CFG_PATH="${CFG_PATH}" \
    ABL_DIR="${ABL_DIR}" \
    MODEL_ID="${MODEL_ID}" \
    RUN_ID="${RUN_ID}" \
    SPATIAL_OUTPUT_ROOT="${SPATIAL_OUTPUT_ROOT}" \
    ROI_CFG="${ROI_CFG}" \
    "${PYTHON_BIN}" - <<'PY'
import fnmatch
import os
from pathlib import Path

import yaml
import torch

abl_dir = Path(os.environ["ABL_DIR"]).expanduser().resolve()
model_id = str(os.environ["MODEL_ID"])
run_id = str(os.environ["RUN_ID"])
spatial_output_root = Path(os.environ["SPATIAL_OUTPUT_ROOT"]).expanduser().resolve()
roi_cfg_path = Path(os.environ["ROI_CFG"]).expanduser().resolve()
cfg_path = Path(os.environ["CFG_PATH"]).expanduser().resolve()

def _deep_get(cfg, keys, default=None):
    cur = cfg
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

# -------------------- load ROI config (selection + rollout discard_ratio) --------------------
roi_cfg = {}
if roi_cfg_path.exists():
    try:
        obj = yaml.safe_load(roi_cfg_path.read_text())
        roi_cfg = obj if isinstance(obj, dict) else {}
    except Exception:
        roi_cfg = {}

sel_cfg = roi_cfg.get("selection", {}) if isinstance(roi_cfg.get("selection", {}), dict) else {}
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
    xai_cfg = roi_cfg.get("xai", {}) if isinstance(roi_cfg.get("xai", {}), dict) else {}
    ar = xai_cfg.get("attn_rollout", {}) if isinstance(xai_cfg.get("attn_rollout", {}), dict) else {}
    discard_ratio = float(ar.get("discard_ratio", discard_ratio))
except Exception:
    discard_ratio = 0.90

# -------------------- resolve checkpoints --------------------
ckpt_dir = abl_dir / "checkpoints"
if not ckpt_dir.exists():
    raise SystemExit(f"[{model_id}] checkpoints/ not found: {ckpt_dir}")

backbone = sorted([p for p in ckpt_dir.glob("*_ssl_best.pt") if "linear" not in p.name.lower()], key=lambda p: p.name)
head = sorted(list(ckpt_dir.glob("*_ssl_linear_best.pt")), key=lambda p: p.name)
if not backbone or not head:
    raise SystemExit(f"[{model_id}] missing *_ssl_best.pt or *_ssl_linear_best.pt in {ckpt_dir}")

ssl_backbone_ckpt = backbone[-1]
ssl_head_ckpt = head[-1]

# -------------------- resolve latest eval dir --------------------
eval_root = abl_dir / "eval"
if not eval_root.exists():
    raise SystemExit(f"[{model_id}] eval/ not found: {eval_root}")

parents = sorted(
    [p for p in eval_root.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, "*_ssl_linear_best*")],
    key=lambda p: p.name,
)
candidates = []
for par in parents:
    ts_dirs = sorted([d for d in par.iterdir() if d.is_dir()], key=lambda p: p.name)
    if ts_dirs:
        ts = ts_dirs[-1]
        candidates.append((par.name, ts.name, ts))
if not candidates:
    raise SystemExit(f"[{model_id}] no eval dirs under: {eval_root}")
eval_dir = sorted(candidates, key=lambda t: (t[0], t[1]))[-1][2]

# -------------------- load eval config (class_order + dataset spec + backbone_name) --------------------
eval_cfg = {}
for name in ("config_resolved.yaml", "config_eval.yaml", "config.yaml"):
    p = eval_dir / name
    if not p.exists():
        continue
    try:
        obj = yaml.safe_load(p.read_text())
        if isinstance(obj, dict):
            eval_cfg = obj
            break
    except Exception:
        continue

class_order = _deep_get(eval_cfg, ("labels", "class_order"), None)
if not (isinstance(class_order, list) and all(isinstance(x, str) for x in class_order) and len(class_order) >= 2):
    for kp in (("data", "class_names"), ("dataset", "class_names"), ("data", "classes"), ("dataset", "classes")):
        v = _deep_get(eval_cfg, kp, None)
        if isinstance(v, list) and all(isinstance(x, str) for x in v) and len(v) >= 2:
            class_order = list(v)
            break
if not class_order:
    # fallback (still runnable even if eval config is missing)
    class_order = ["ccRCC", "pRCC", "CHROMO", "ONCO", "NOT_TUMOR"]

def _infer_in_dim(head_ckpt: Path, n_classes: int) -> int:
    obj = torch.load(str(head_ckpt), map_location="cpu")
    state = None
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "net", "model"):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
        if state is None:
            state = obj
    else:
        state = {}
    cands = []
    for k, v in state.items():
        if torch.is_tensor(v) and v.ndim == 2:
            if int(v.shape[0]) == int(n_classes):
                cands.append((k, v))
    if not cands:
        for k, v in state.items():
            if torch.is_tensor(v) and v.ndim == 2:
                cands.append((k, v))
    if not cands:
        raise RuntimeError(f"Cannot infer in_dim from head ckpt (no 2D tensors): {head_ckpt}")
    weight = max(cands, key=lambda kv: int(kv[1].shape[1]))[1]
    return int(weight.shape[1])

embed_dim_inferred = _infer_in_dim(ssl_head_ckpt, n_classes=len(class_order))
dim_to_backbone = {
    384: "vit_small_patch16_224",
    768: "vit_base_patch16_224",
    1024: "vit_large_patch16_224",
}
backbone_name = dim_to_backbone.get(embed_dim_inferred)
if not backbone_name:
    for kp in (("model", "backbone_name"), ("model", "backbone"), ("model", "arch"), ("model", "name"), ("backbone_name",)):
        v = _deep_get(eval_cfg, kp, None)
        if isinstance(v, str) and v.strip():
            backbone_name = v.strip()
            break
if not backbone_name:
    backbone_name = "vit_base_patch16_224"

backend = str(_deep_get(eval_cfg, ("data", "backend"), "") or "").strip().lower()
if not backend:
    if isinstance(_deep_get(eval_cfg, ("data", "webdataset"), None), dict):
        backend = "webdataset"
    elif isinstance(_deep_get(eval_cfg, ("data", "imagefolder"), None), dict):
        backend = "imagefolder"

img_size = int(_deep_get(eval_cfg, ("data", "img_size"), 224) or 224)
imagenet_norm = bool(_deep_get(eval_cfg, ("data", "imagenet_norm"), False))
num_workers = int(_deep_get(eval_cfg, ("data", "num_workers"), 8) or 8)
batch_size = int(_deep_get(eval_cfg, ("data", "batch_size"), 1) or 1)

data_cfg = {
    "backend": backend or "webdataset",
    "img_size": img_size,
    "imagenet_norm": imagenet_norm,
    "num_workers": num_workers,
    "batch_size": batch_size,
}

if data_cfg["backend"] == "webdataset":
    w = _deep_get(eval_cfg, ("data", "webdataset"), {}) if isinstance(_deep_get(eval_cfg, ("data", "webdataset"), None), dict) else {}
    test_dir = str(w.get("test_dir") or "").strip() or os.getenv("WDS_TEST_DIR", "").strip()
    if not test_dir:
        raise SystemExit(
            f"[{model_id}] Missing WebDataset test_dir. "
            f"Set data.webdataset.test_dir in eval config or export WDS_TEST_DIR."
        )
    data_cfg["webdataset"] = {
        "test_dir": test_dir,
        "pattern": str(w.get("pattern", "shard-*.tar")),
        "image_key": str(w.get("image_key", "img.jpg;jpg;jpeg;png")),
        "meta_key": str(w.get("meta_key", "meta.json;json")),
    }
elif data_cfg["backend"] == "imagefolder":
    ifd = _deep_get(eval_cfg, ("data", "imagefolder"), {}) if isinstance(_deep_get(eval_cfg, ("data", "imagefolder"), None), dict) else {}
    test_dir = str(ifd.get("test_dir") or "").strip()
    if not test_dir:
        raise SystemExit(f"[{model_id}] Missing ImageFolder test_dir in eval config.")
    data_cfg["imagefolder"] = {"test_dir": test_dir}

cfg = {
    "experiment": {
        "seed": int(_deep_get(eval_cfg, ("experiment", "seed"), 0) or 0),
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
        "embed_dim_inferred": int(embed_dim_inferred),
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
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

    if [[ -f "${CFG_PATH}" ]]; then
      echo "  [RUN] attention_rollout.py --config ${CFG_PATH}"
      "${PYTHON_BIN}" "${ATTN_ROLLOUT_PY}" --config "${CFG_PATH}"
    else
      echo "  [SKIP] ${MODEL_ID}: config not generated (previous step skipped)"
    fi
  done
  shopt -u nullglob
done

echo "[4/5] ROI + concept on ALL ablations (reuse rollout from 3/5 when available)"
"${PYTHON_BIN}" "${SPATIAL_CONCEPT_PY}" \
  --models-root "${MODELS_ROOT}" \
  --exp-prefix "${EXP_PREFIX}" \
  --config "${ROI_CFG}" \
  --run-id "${RUN_ID}" \
  --reuse-attn-rollout \
  --attn-rollout-outputs-root "${SPATIAL_OUTPUT_ROOT}" \
  --attn-rollout-run-id "${RUN_ID}" \
  --log-level "${LOG_LEVEL}"

echo "[5/5] ROI vs NO-ROI comparison (PER ABLATION)"
for EXP_DIR in "${EXP_DIRS[@]}"; do
  if [[ ! -d "${EXP_DIR}" ]]; then
    continue
  fi
  shopt -s nullglob
  for ABL_DIR in "${EXP_DIR}"/exp_*_abl*; do
    [[ -d "${ABL_DIR}" ]] || continue
    echo "  [CMP] $(basename "${ABL_DIR}")"
    "${PYTHON_BIN}" "${COMP_PY}" --model-root "${ABL_DIR}" --config "${COMP_CFG}" --log-level "${LOG_LEVEL}"
  done
  shopt -u nullglob
done

echo "DONE run_id=${RUN_ID}"
