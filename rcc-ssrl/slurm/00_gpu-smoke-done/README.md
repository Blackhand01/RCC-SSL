# GPU Smoke Test (HPC@PoliTo, SLURM)

## Requisiti
- Account su HPC@PoliTo e accesso a una partition GPU.
- Moduli `cuda`, `python/pytorch` **oppure** immagine Apptainer (`--nv`).

## Esecuzione
```bash
# Test batch semplice
bash run.sh batch

# Test interattivo (debug su nodo GPU)
bash run.sh interactive
# quindi:
module load cuda/12.2 anaconda/2024
python3 gpu_smoke.py

# Test avanzato con autopick partizione e monitor
bash run_autopick.sh

# Se la queue richiede GRES
USE_GRES=1 bash run_autopick.sh
```

## Output atteso

* `CUDA available: True`, `GPU count >= 1`, nome GPU (`A100`, `V100`, ecc.)
* Tempo forward < 5 s, nessun errore nel .out/.err
* Log mostra partizione scelta, JOBID, nodo assegnato, dettagli GPU/driver
