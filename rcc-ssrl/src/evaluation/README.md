# Evaluation pipeline (SSL RCC project)

Questa directory contiene la pipeline di evaluation per i modelli SSL (MoCo v3, DINOv3, iBOT, i-JEPA, ecc.) e gli script per l'aggregazione a livello di paziente.

Struttura principale:

- `eval.py`  
  Script di evaluation “test-only” per un singolo modello (encoder + classificatore lineare).
  Legge una config YAML generata da `tools/auto_eval.py`, esegue l'eval sulle patch e salva:
  - `metrics_<model>.json`
  - `report_per_class.json`
  - `cm_<model>.png`
  - `logits_test.npy`
  - `predictions.csv` (arricchito con `wds_key` + metadati per allineamento XAI)

- `tools/auto_eval.py`  
  Auto-discovery dei run in MLflow (`mlruns`), generazione degli YAML di eval (`auto_configs/`) e, opzionalmente, submit dei job SLURM che chiamano `eval.py` tramite `eval_models.sbatch`.

- `tools/batch_patient_aggregation.py`  
  Aggregazione patch → paziente, usando tutti i patch e **escludendo sempre** `NOT_TUMOR` dalla decisione finale.  
  Produce metriche per paziente, confusion matrix a livello paziente e un CSV di riepilogo.

- `eval_models.sbatch`  
  Script SLURM che:
  - crea/aggiorna il venv dedicato (`.venvs/eval`),
  - installa `requirements_eval.txt`,
  - lancia `eval.py --config "$CFG_PATH"` dove `CFG_PATH` viene esportato da `tools/auto_eval.py`.

- `ssl_linear_loader.py`  
  Wrapper per caricare backbone SSL (ResNet o ViT) + head lineare dai checkpoint MoCo/DINO/iBOT/iJEPA, con logica di auto-swap ResNet↔ViT quando necessario.

- `auto_configs/`  
  YAML di evaluation generati automaticamente da `tools/auto_eval.py`.  
  Ogni file corrisponde a un singolo run MLflow (una specifica ablation / modello).


## Ambiente e requisiti

Gli script SLURM (`eval_models.sbatch`) gestiscono automaticamente:

- creazione venv: `.venvs/eval`
- installazione dipendenze: `pip install -r src/evaluation/requirements_eval.txt`

Se vuoi lanciare localmente (senza SLURM), puoi fare:

```bash
cd /home/mla_group_01/rcc-ssrl
python3 -m venv .venvs/eval
source .venvs/eval/bin/activate
pip install --upgrade pip
pip install -r src/evaluation/requirements_eval.txt


Esecuzione: evaluation automatica
1. Generazione config di eval + submit su SLURM

Esempio: lanciare l’evaluation automatica su tutti i run dentro un esperimento MLflow (exp_20251118_105221_dino_v3):

python /home/mla_group_01/rcc-ssrl/src/evaluation/tools/auto_eval.py \
  --mlruns-root "/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251213_121650_i_jepa" \
  --submit


Esecuzione: aggregazione per paziente

Una volta che eval.py ha prodotto predictions.csv e (opzionalmente) logits_test.npy per ogni run, puoi aggregare a livello di paziente.

1. Aggregazione su un intero esperimento (tutte le ablation/run)

Esempio:

python /home/mla_group_01/rcc-ssrl/src/evaluation/tools/batch_patient_aggregation.py \
  --mlruns-root /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/ablation_final/exp_20251213_121650_i_jepa \
  --method prob_sum


genera Reliability Diagram + ECE e Risk–Coverage (patch e/o patient)

source /home/mla_group_01/rcc-ssrl/.venvs/eval/bin/activate  # consigliato se esiste
python /home/mla_group_01/rcc-ssrl/src/evaluation/tools/calibration_and_coverage.py \
  --run-dir /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/ablation_final/exp_20251209_234736_dino_v3/exp_dino_v3_abl03/eval/dino_v3_ssl_linear_best/20251211_104905 \
  --out-dir /home/mla_group_01/rcc-ssrl/src/evaluation/results/dino_v3_best_calibration \
  --n-bins 15 \
  --title "Dino v3 (patch-level)"
