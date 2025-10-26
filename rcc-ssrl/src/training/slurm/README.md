# SLURM Job Scripts – Training RCC-SSRL

Questa cartella contiene script per il lancio del training su cluster SLURM, sia su singolo nodo che in modalità distribuita o in container.

## Script disponibili

- **train_single_node.sbatch**  
  Lancia il training su **un singolo nodo GPU**.  
  Uso tipico:  
  ```bash
  sbatch train_single_node.sbatch
  ```

- **train_multi_node_ddp.sbatch**  
  Lancia il training in **modalità distribuita multi-nodo** (DDP) usando tutte le GPU disponibili su più nodi.  
  Uso tipico:  
  ```bash
  sbatch train_multi_node_ddp.sbatch
  ```

- **train_in_apptainer.sbatch**  
  Lancia il training **dentro un container Apptainer/Singularity**.  
  Uso tipico:  
  ```bash
  sbatch train_in_apptainer.sbatch
  ```

- **run_gpu_interactive.sh**  
  Ottiene una **sessione interattiva** su un nodo GPU per debug o test rapido.  
  Uso tipico:  
  ```bash
  bash run_gpu_interactive.sh
  ```

## Differenze principali

- Gli script `.sbatch` sono per job batch (non interattivi), mentre `run_gpu_interactive.sh` è per sessioni interattive.
- `train_single_node.sbatch` e `train_multi_node_ddp.sbatch` usano Conda direttamente sul nodo, mentre `train_in_apptainer.sbatch` esegue tutto dentro un container.
- `train_multi_node_ddp.sbatch` abilita il training distribuito su più nodi e più GPU per nodo tramite `torchrun`.
- Tutti gli script scrivono log e errori nella cartella `src/training/logs/`.

## Note

- Modifica i path e i parametri SLURM secondo le risorse del tuo cluster.
- Per il training distribuito, assicurati che il dataset sia accessibile da tutti i nodi.
- Per Apptainer/Singularity, specifica il path corretto all’immagine `.sif` tramite la variabile `$SIF`.

Batch:
  sbatch src/training/slurm/train_single_node.sbatch
Monitor:
  squeue -u $USER
  tail -f rcc-train.<JOBID>.out
Interattivo:
  bash src/training/slurm/run_gpu_interactive.sh
