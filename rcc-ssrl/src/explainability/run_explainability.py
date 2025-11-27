#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main orchestrator to run spatial and concept explainability.

Itera sulle ablation di un esperimento e per ognuna:
- costruisce config_xai.yaml (spatial) e config_concept.yaml (concept)
- lancia gli sbatch relativi.

Comportamento:
- Se per una certa ablation NON esiste la cartella di eval per il modello
  (eval/<model_name>/...), quella ablation viene SKIPPATA con un warning.
- Se esiste ma non contiene nessun run (sottocartella timestamp), viene skippata.
- Se mancano i checkpoint SSL backbone/head, l'ablation viene skippata.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

log = logging.getLogger("run_explainability")


def run_sbatch(batch_file: Path, config_path: Path, log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    os.environ["CONFIG_PATH"] = str(config_path)
    os.environ["LOG_DIR"] = log_dir
    log.info(
        f"[SBATCH] submitting {batch_file} "
        f"with CONFIG_PATH={config_path}, LOG_DIR={log_dir}"
    )
    subprocess.run(["sbatch", str(batch_file)], check=True)


def _derive_backbone_basename(model_name: str) -> str:
    suffix = "_ssl_linear_best"
    if model_name.endswith(suffix):
        return model_name[: -len(suffix)]
    return model_name


def _find_latest_eval_run(ablation_dir: Path, model_name: str) -> Optional[Path]:
    base = ablation_dir / "eval" / model_name
    if not base.is_dir():
        log.warning(f"[SKIP] Eval directory not found for {ablation_dir.name}: {base}")
        return None

    run_dirs = [d for d in base.iterdir() if d.is_dir()]
    if not run_dirs:
        log.warning(f"[SKIP] No eval runs found under {base}")
        return None

    latest = sorted(run_dirs)[-1]
    log.info(f"[EVAL] Using eval run {latest} for ablation {ablation_dir.name}")
    return latest


def _check_checkpoint(path: Path, label: str, ablation_name: str) -> bool:
    if path.is_file():
        log.info(f"[CKPT] {label} FOUND for {ablation_name}: {path}")
        return True
    log.warning(f"[CKPT] {label} MISSING for {ablation_name}: {path}")
    return False


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run explainability pipeline.")
    parser.add_argument(
        "--experiment-root",
        required=True,
        type=Path,
        help="Path to root dir with ablations folders",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        type=str,
        help="Folder/model name for XAI outputs (es. moco_v3_ssl_linear_best)",
    )
    parser.add_argument(
        "--spatial-config-template",
        required=True,
        type=Path,
        help="Path to spatial/config_xai.yaml template",
    )
    parser.add_argument(
        "--concept-config-template",
        required=True,
        type=Path,
        help="Path to concept/config_concept.yaml template",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    if not args.experiment_root.is_dir():
        raise FileNotFoundError(f"Experiment root not found: {args.experiment_root}")

    ablation_folders = sorted(
        [
            p
            for p in args.experiment_root.iterdir()
            if p.is_dir() and p.name.startswith("exp_")
        ]
    )

    if not ablation_folders:
        log.error(f"No ablation folders found under {args.experiment_root}")
        return

    backbone_base = _derive_backbone_basename(args.model_name)
    log.info(f"[INFO] MODEL_NAME={args.model_name}  BACKBONE_BASE={backbone_base}")
    log.info(f"[INFO] Found {len(ablation_folders)} ablations.")

    spatial_template = yaml.safe_load(open(args.spatial_config_template))
    concept_template = yaml.safe_load(open(args.concept_config_template))

    for ablation in ablation_folders:
        log.info("=" * 80)
        log.info(f"[ABLATION] Processing: {ablation}")

        eval_run = _find_latest_eval_run(ablation, args.model_name)
        if eval_run is None:
            log.warning(f"[ABLATION] Skipping {ablation.name}: no eval run available.")
            continue

        backbone_ckpt = ablation / "checkpoints" / f"{backbone_base}__ssl_best.pt"
        head_ckpt = ablation / "checkpoints" / f"{backbone_base}__ssl_linear_best.pt"

        has_backbone = _check_checkpoint(backbone_ckpt, "ssl_backbone_ckpt", ablation.name)
        has_head = _check_checkpoint(head_ckpt, "ssl_head_ckpt", ablation.name)

        if not (has_backbone and has_head):
            log.warning(
                f"[ABLATION] Skipping {ablation.name}: missing required checkpoints."
            )
            continue

        # ---------------------- SPATIAL CONFIG ----------------------
        spatial_config = yaml.safe_load(open(args.spatial_config_template))
        spatial_config["experiment"]["outputs_root"] = str(ablation / "06_xai")
        spatial_config["model"]["name"] = args.model_name
        spatial_config["model"]["ssl_backbone_ckpt"] = str(backbone_ckpt)
        spatial_config["model"]["ssl_head_ckpt"] = str(head_ckpt)
        spatial_config["evaluation_inputs"]["eval_run_dir"] = str(eval_run)

        spatial_config_path = ablation / "06_xai" / "config_xai.yaml"
        spatial_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(spatial_config_path, "w") as f:
            yaml.dump(spatial_config, f)
        log.info(f"[CFG] Spatial config written to {spatial_config_path}")

        # ---------------------- CONCEPT CONFIG ----------------------
        concept_config = yaml.safe_load(open(args.concept_config_template))
        concept_config["experiment"]["outputs_root"] = str(ablation / "06_xai")
        concept_config["model"]["name"] = args.model_name
        concept_config["model"]["ssl_backbone_ckpt"] = str(backbone_ckpt)
        concept_config["model"]["ssl_head_ckpt"] = str(head_ckpt)
        concept_config["evaluation_inputs"]["eval_run_dir"] = str(eval_run)

        concept_config_path = ablation / "06_xai" / "config_concept.yaml"
        with open(concept_config_path, "w") as f:
            yaml.dump(concept_config, f)
        log.info(f"[CFG] Concept config written to {concept_config_path}")

        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"/home/mla_group_01/rcc-ssrl/src/logs/xai/{backbone_base}/{ablation.name}/{datetime_str}"
        spatial_sbatch = Path(__file__).parent / "spatial" / "xai_generate.sbatch"
        concept_sbatch = Path(__file__).parent / "concept" / "xai_concept.sbatch"

        log.info(f"[LAUNCH] Spatial XAI sbatch for {ablation.name}")
        run_sbatch(spatial_sbatch, spatial_config_path, log_dir)

        log.info(f"[LAUNCH] Concept XAI sbatch for {ablation.name}")
        run_sbatch(concept_sbatch, concept_config_path, log_dir)

    log.info("[DONE] run_explainability completed.")
    log.info("       Check Slurm logs under /home/mla_group_01/rcc-ssrl/src/logs/xai/...")


if __name__ == "__main__":
    main()
