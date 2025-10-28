#!/usr/bin/env bash
set -euo pipefail
module load miniconda3/3.13.25
eval "$(conda shell.bash hook)"
# crea env CUDA-enabled; pin versioni chiave se serve
conda create -y -n train python=3.10
conda activate train
# torch/cu121 + torchvision compatibile + dipendenze tipiche
pip install --upgrade pip
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install webdataset pandas Pillow matplotlib seaborn pyyaml
# opzionale: mlflow, scikit-learn, ecc.
pip install scikit-learn==1.5.2
