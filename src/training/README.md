# Training Module

This directory contains the core logic, configurations, and scripts required to train and evaluate various Self-Supervised Learning (SSL) architectures and supervised baselines for the project. 

For a general overview of the project, please refer to the [main repository README](../../README.md).

## Directory Structure

Here is an overview of the folders and their contents to help you navigate the training pipeline:

* **`configs/`**
  Contains all the configuration files (YAML and JSON) used to define training parameters, hyperparameters, and dataset paths.
  * `ablations/`: Configurations for specific ablation studies (e.g., DINO v3, i-JEPA, iBOT, MoCo v3, supervised, and transfer learning).
  * `includes/`: Shared configuration snippets, such as `data_paths.yaml`.
  * `templates/`: Base templates used to generate specific experiment configurations.

* **`data/` & `datasets/`**
  Handles data loading and preprocessing.
  * `data/webdataset.py`: Logic for reading and parsing data using the WebDataset format.
  * `datasets/`: Contains scripts for building datasets (`builders.py`), defining augmentations/transformations (`transforms.py`), and mapping labels (`labels.py`).

* **`loss/`**
  Contains custom loss function implementations specific to the SSL methods used.
  * Features implementations such as `dino_clstoken_loss.py`, `ibot_patch_loss.py`, `koleo_loss.py`, and `gram_loss.py`.

* **`models/`**
  PyTorch implementations and wrappers for the different architectures supported by this project:
  * SSL Models: `dino_v3.py`, `i_jepa.py`, `ibot.py`, `moco_v3.py`.
  * Baselines: `supervised.py`, `transfer.py`.

* **`slurm/`**
  Contains `.sbatch` scripts designed for executing training jobs on High-Performance Computing (HPC) clusters managed by SLURM. 
  * Examples include `launch_train_job.sbatch` and scripts for single-node training setups.

* **`trainer/`**
  The core components that orchestrate the training process.
  * `loops.py`: Contains the main training and validation loops.
  * `backbones.py` & `heads.py`: Logic for building the model architectures (feature extractors and projection heads).
  * `features.py`: Utilities for handling and extracting features during the training or evaluation process.

* **`utils/`**
  A collection of helper modules for various underlying tasks:
  * `distributed.py`: Utilities for distributed data parallel (DDP) training.
  * `reproducibility.py`: Seeds and deterministic settings.
  * `io.py`, `paths.py`: File system and I/O helpers.
  * `torch_ops.py`, `viz.py`: Tensor operations and visualization tools.

## Key Entry Points

* **`launch_training.py`**: The primary entry point script used to start the training process from the command line.
* **`orchestrator.py`**: Orchestrates the setup of the model, datasets, and training loops based on the provided configurations.
* **`requirements.txt` / `pip-freeze.txt`**: List of Python dependencies required specifically to run the training pipeline.
