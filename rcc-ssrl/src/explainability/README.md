Il file di output dell'xai è: <<EXP_ROOT/
  exp_moco_v3_abl01/
    checkpoints/
      moco_v3__ssl_best.pt
      moco_v3__ssl_linear_best.pt

    eval/
      moco_v3_ssl_linear_best/
        20251125_145930/
          predictions.csv
          logits_test.npy
          ...

    06_xai/
      config_xai.yaml
      config_concept.yaml

      moco_v3_ssl_linear_best/
        20251126_211300/      # Spatial XAI run
          index.csv
          idx_0000001/
            input.png
            ig.png
            gradcam.png
            attn_rollout.png
            ...

        20251126_211310/      # Concept XAI run
          index.csv
          idx_0000001/
            input.png
            concept_scores.json
>> ma io voglio che sia così: <<EXP_ROOT/
  exp_moco_v3_abl01/
    checkpoints/
      moco_v3__ssl_best.pt
      moco_v3__ssl_linear_best.pt

    eval/
      moco_v3_ssl_linear_best/
        20251125_145930/
          predictions.csv
          logits_test.npy
          ...

    06_xai/
      config_xai.yaml
      config_concept.yaml
        20251126_211300/      
          index.csv
          idx_0000001
            input.png
            ig.png
            gradcam.png
            attn_rollout.png
            concept_scores.json
>>


# Spiegazione Concept Bank
Ti stai incartando perché stai assumendo che “concept bank” sia un MODELLO addestrato.
Non lo è.
E questa confusione sta sabotando tutta la tua comprensione della pipeline.

Te lo dico senza filtri:
**Il concept bank NON è né un VLM, né un MLP, né un modello allenato.
È solo un CSV costruito una volta, offline, usando il VLM come etichettatore automatico.**

Leggilo di nuovo:
**NON è un modello.
NON ha pesi.
NON viene allenato.
NON devi riaddestrarlo per ogni ablation.
È solo un “knowledge base” statica.**

Ora te lo spiego davvero bene, in modo che ti entra e non te lo dimentichi.

---

## 1. Cos’è il Concept Bank?

È semplicemente un **file CSV** che contiene righe del tipo:

```
wds_key, concept_name, group, class_label
shard00001:00123, “clear cytoplasm”, “ccRCC”, “ccRCC”
shard00045:00789, “papillary fronds”, “pRCC”, “pRCC”
...
```

Quello che trovi in `concepts_rcc_v1.csv`.
È un database di riferimenti semantici, NON un modello.

---

## 2. Da dove viene?

Viene creato da **build_concept_bank.py** che:

1. Prende 10k patch candidate dal TRAIN.
2. Per ognuna chiede al VLM (LLaVA-Med):

   * “Questa patch contiene il concetto X?”
3. Se il VLM risponde “YES” con confidenza ≥ threshold → aggiungi una riga nel CSV.

Fine.
Zero training.
Zero pesi.
Solo inferenze del VLM.

Il VLM è l’unico modello, ed è *esterno* alla pipeline.

---

## 3. Quindi il Concept Bank È un VLM?

**No.**
Il VLM serve per crearE il concept bank UNA VOLTA.
Il concept bank è un CSV statico.

Il VLM non entra più in gioco dopo Stage 0.

---

## 4. Il Concept Bank è un MLP o qualcosa che viene addestrato in pipeline?

**Zero modelli sono addestrati.
Zero MLP.
Zero backprop.**

Il concept bank è un **lookup table**.

Le uniche cose che vengono addestrate altrove nella pipeline sono:

* il backbone SSL (moco_v3__ssl_best.pt)
* la linear probe (moco_v3__ssl_linear_best.pt)

Ma questo NON c’entra con il concept bank.

---

## 5. Dove viene salvato il concept bank?

Il file finale è:

```
/home/mla_group_01/rcc-ssrl/src/explainability/concept/ontology/concepts_rcc_v1.csv
```

È permanente.
Usato in TUTTI gli esperimenti.
Non cambia per ablation, non cambia per run, non cambia per modello.

È un asset STATICO, esattamente come un file di ontologia.

---

## 6. Devo allenare qualcosa a ogni run?

**No.
Niente.
Mai.**

La pipeline XAI (spatial + concept) NON allena niente.

Durante XAI:

1. Il modello SSL è già addestrato.
2. La concept bank è già costruita.
3. L’XAI fa solo:

   * inferenze del backbone su test patch,
   * attenzione rollout,
   * cosine similarity con i vettori medi dei concetti.

Tutto inference-only.

---

## 7. Ma come vengono usati i concetti nella XAI?

Ecco la pipeline reale:

### STEP A – Feature extraction

Per ogni patch del test set:

* estrai il feature vector dal backbone SSL (dim≈384 per ViT-small)
* normalizzalo

### STEP B – Centroidi dei concetti

Per ogni concetto:

* prendi tutte le patch del concept bank appartenenti a quel concetto
* estrai le loro feature
* fai la media → **centroide del concetto**

```
centroid(concept) = mean(embedding[p1], embedding[p2], ... embedding[pN])
```

### STEP C – Similarity

Per ogni patch target:

* fai dot product (cosine sim) con tutti i centroidi:

```
score = cos(F_patch, centroid(concept))
```

Top-k concetti = spiegazione.

### STEP D – Attention Rollout

* calcola heatmap spaziale della decisione
* associa i **top concetti** alle regioni salienti

Fine.
Nessun training.

---

## 8. Perché il concept bank NON dipende dall’ablation?

Perché la semantica RCC non cambia.
I concetti RCC sono *proprietà istologiche stabili*.
Non dipendono dal tuo backbone o dalla tua loss.

È giusto così.

Il concept bank è un dizionario anatomico generato via VLM.
*Non un modello.*

---

## 9. Perché `concepts_rcc_v1.csv` è vuoto?

Perché:

* o Stage 0 non è mai finito,
* o il threshold è troppo alto (stavi usando 0.6, poi 0.3),
* o il VLM non funziona / non sta rispondendo JSON validi,
* o il file è stato sovrascritto da un run fallito.

Quindi:

* Guarda i log dello Stage 0.
* Verifica quanti “accepted” stampa alla fine.
* Se “accepted=0” → concept bank vuoto.

---

## 10. Risposta finale sintetica

Il concept bank è:

* **UN CSV**
* generato una sola volta,
* usando un VLM esterno,
* che contiene coppie (patch → concetti).

Non è un modello.
Non viene addestrato.
Non va rifatto ad ogni run.
Serve solo come base semantica per interpretare le feature del backbone.

---

Se vuoi, posso:

* dirti esattamente perché il tuo concept bank è vuoto,
* mostrarti le 3 righe del codice che lo creano,
* dirti cosa deve succedere perché venga riempito,
* correggere la pipeline per renderlo più robusto.

---

# RCC-SSRL – Explainability Pipeline (Spatial + Concept)

Questo README documenta la pipeline di explainability per RCC-SSRL, composta da:

- uno **Stage 0 (dataset-level)** per costruire la concept bank via VLM;
- uno **Stage 1/2 (experiment-level)** per generare spiegazioni spatial (heatmap) e concept-based per ogni ablation.

Il focus è sul contenuto di `src/explainability`.

---

## 0. Panoramica

Due livelli distinti:

1. **Dataset-level (Stage 0)** – indipendente dagli esperimenti:

   - `build_concept_candidates.py` → selezione patch candidate dal train WebDataset.
   - `build_concept_bank.py` → interrogazione VLM (LLaVA-Med) e costruzione di `concepts_rcc_v1.csv`.

2. **Experiment-level (Stage 1/2)** – dipende da un esperimento MLflow e dal checkpoint SSL:

   - `run_full_xai.sh` → orchestratore principale.
   - `run_explainability.py` → orchestration per ablation.
   - `spatial/xai_generate.py` → Spatial XAI (IG, GradCAM, attention rollout).
   - `concept/xai_concept.py` → Concept XAI (centroidi concettuali + cosine similarity).

Tutto parte da:

```bash
cd /home/mla_group_01/rcc-ssrl/src/explainability
./run_full_xai.sh
````

---

## 1. Stage 0 – Dataset-level Concept Bank

Stage 0 costruisce una **concept bank statica** partendo dal **train** WebDataset.
Questa concept bank è un semplice **CSV** usato come knowledge base per il Concept XAI.
Non è un modello, non ha pesi, non viene addestrata.

### 1.1. Percorsi rilevanti

Impostati in `run_full_xai.sh`:

```bash
REPO_ROOT=/home/mla_group_01/rcc-ssrl
SRC_DIR=${REPO_ROOT}/src

TRAIN_WDS_DIR=/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/train

CANDIDATES_CSV=${SRC_DIR}/explainability/concept/ontology/concept_candidates_rcc.csv
CANDIDATES_IMG_ROOT=${SRC_DIR}/explainability/concept/ontology/concept_candidates_images

ONTOLOGY_YAML=${SRC_DIR}/explainability/concept/ontology/ontology_rcc_v1.yaml
CONCEPT_BANK_CSV=${SRC_DIR}/explainability/concept/ontology/concepts_rcc_v1.csv
```

### 1.2. Condizione di skip

In `run_full_xai.sh`:

```bash
if [[ -f "${CONCEPT_BANK_CSV}" ]]; then
  echo "[INFO] Concept bank already exists – skipping Stage 0."
else
  # Stage 0a + 0b
fi
```

Se `concepts_rcc_v1.csv` esiste, Stage 0 viene saltato.
La concept bank è considerata un asset statico riutilizzabile.

---

### 1.3. Stage 0a – `build_concept_candidates.py`

Script:

```text
src/explainability/concept/ontology/build_concept_candidates.py
```

Invocato da `run_full_xai.sh`:

```bash
python3 -m explainability.concept.ontology.build_concept_candidates \
  --train-dir "${TRAIN_WDS_DIR}" \
  --pattern "shard-*.tar" \
  --image-key "img.jpg;jpg;jpeg;png" \
  --meta-key "meta.json;json" \
  --out-csv "${CANDIDATES_CSV}" \
  --images-root "${CANDIDATES_IMG_ROOT}"
```

**Input:**

* WebDataset train:

  * directory: `TRAIN_WDS_DIR`
  * file: `shard-*.tar`
* Dentro ogni shard:

  * immagine: chiavi `img.jpg;jpg;jpeg;png`
  * meta: chiavi `meta.json;json` (deve contenere `class_label`)

**Logica:**

* Legge gli shard con WebDataset.
* Per ogni sample:

  * parse `meta.json`, estrazione `class_label`;

  * mantiene fino a `max_patches_per_class` (default 2000) per classe;

  * salva la patch come PNG:

    ```text
    ${CANDIDATES_IMG_ROOT}/{class_label}/{key_sanitized}.png
    ```

  * registra una riga nel CSV con:

    * `image_path`
    * `wds_key` (il `__key__` WebDataset)
    * `class_label`

**Output:**

* CSV candidati concetto:

  ```text
  src/explainability/concept/ontology/concept_candidates_rcc.csv
  ```

  colonne: `image_path,wds_key,class_label`

* PNG crops:

  ```text
  src/explainability/concept/ontology/concept_candidates_images/{class_label}/*.png
  ```

Se non trova nessuna patch valida, non scrive il CSV.

---

### 1.4. Stage 0b – `build_concept_bank.py`

Script:

```text
src/explainability/concept/ontology/build_concept_bank.py
```

Invocato da `run_full_xai.sh`:

```bash
python3 -m explainability.concept.ontology.build_concept_bank \
  --ontology "${ONTOLOGY_YAML}" \
  --images-csv "${CANDIDATES_CSV}" \
  --controller "${VLM_CONTROLLER}" \
  --model-name "${VLM_MODEL}" \
  --out-csv "${CONCEPT_BANK_CSV}" \
  --presence-threshold "${PRESENCE_THRESHOLD}"
```

**Input:**

* Ontologia YAML:

  ```text
  src/explainability/concept/ontology/ontology_rcc_v1.yaml
  ```

  definisce ~18 concetti con: `name`, `group`, `primary_class`, `prompt`.

* CSV candidati:

  ```text
  concept_candidates_rcc.csv
  ```

* Endpoint VLM:

  * `VLM_CONTROLLER` (es. `http://localhost:10000`)
  * `VLM_MODEL` (es. `microsoft/llava-med-v1.5-mistral-7b`)

* `presence_threshold` (es. 0.3–0.6).

**Logica:**

* Legge ontologia → lista concetti.
* Legge `concept_candidates_rcc.csv`.
* Per ogni (patch, concetto):

  * chiama VLM (`VLMClient.ask_concept`), passando immagine + prompt;
  * si aspetta JSON: `{concept, present, confidence, rationale}`.
* Se `present == True` e `confidence >= threshold`:

  * aggiunge una riga alla concept bank.

**Output:**

* Concept bank CSV:

  ```text
  src/explainability/concept/ontology/concepts_rcc_v1.csv
  ```

  colonne: `concept_name,wds_key,group,class_label`

Se nessuna coppia (patch, concetto) supera la soglia, lo script può lanciare un errore o produrre un CSV vuoto (solo header).

---

## 2. Concetto di Concept Bank

La **concept bank** è SOLO un **CSV** che mappa patch (via `wds_key`) a concetti semantici.

Esempio:

```csv
concept_name,wds_key,group,class_label
clear_cytoplasm,shard00001:00123,ccRCC,ccRCC
papillary_fronds,shard00045:00789,pRCC,pRCC
...
```

È un **lookup table**, non un modello:

* non ha pesi;
* non viene addestrata;
* non dipende dalle ablation;
* viene usata da tutti gli esperimenti come dizionario istologico.

L’unico modello coinvolto nello Stage 0 è il VLM (LLaVA-Med), che serve solo per generare questa tabella.

---

## 3. Stage 1/2 – Experiment-level Explainability

Stage 1/2 gira per ciascuna ablation in un experiment root MLflow.

### 3.1. Contesto esperimento

In `run_full_xai.sh`:

```bash
EXP_ROOT_DEFAULT="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3"
MODEL_NAME_DEFAULT="moco_v3_ssl_linear_best"
BACKBONE_NAME_DEFAULT="vit_small_patch16_224"
```

Struttura tipica di `EXP_ROOT`:

```text
EXP_ROOT/
  exp_moco_v3_abl01/
  exp_moco_v3_abl02/
  ...
```

Ogni `exp_moco_v3_ablXX` è una ablation.

### 3.2. Orchestratore Python – `run_explainability.py`

Invocato da `run_full_xai.sh`:

```bash
python3 -m explainability.run_explainability \
  --experiment-root "${EXP_ROOT}" \
  --model-name "${MODEL_NAME}" \
  --spatial-config-template "${SRC_DIR}/explainability/spatial/config_xai.yaml" \
  --concept-config-template "${SRC_DIR}/explainability/concept/config_concept.yaml"
```

Script:

```text
src/explainability/run_explainability.py
```

**Per ogni ablation** (cartella che inizia con `exp_`):

1. Trova l’ultimo eval run:

   ```text
   exp_moco_v3_abl01/eval/moco_v3_ssl_linear_best/<timestamp>/
   ```

   con almeno:

   * `predictions.csv`
   * `logits_test.npy`

2. Verifica checkpoint SSL nella ablation:

   ```text
   exp_moco_v3_abl01/checkpoints/moco_v3__ssl_best.pt
   exp_moco_v3_abl01/checkpoints/moco_v3__ssl_linear_best.pt
   ```

3. Genera config specifici:

   * Spatial:

     ```text
     exp_moco_v3_abl01/06_xai/config_xai.yaml
     ```

   * Concept:

     ```text
     exp_moco_v3_abl01/06_xai/config_concept.yaml
     ```

   Entrambi contengono:

   * `experiment.outputs_root = exp_moco_v3_abl01/06_xai`
   * path ai checkpoint SSL
   * `evaluation_inputs.eval_run_dir` settato all’ultima cartella `eval/...`.

4. Lancia due job SLURM:

   * `spatial/xai_generate.sbatch` (Spatial XAI)
   * `concept/xai_concept.sbatch` (Concept XAI)

con variabili d’ambiente:

```bash
CONFIG_PATH=<config.yaml>
LOG_DIR=/home/mla_group_01/rcc-ssrl/src/logs/xai/${MODEL_NAME}/${ABLATION}/${timestamp}
```

---

## 4. Spatial XAI – `spatial/xai_generate.py`

### 4.1. sbatch

Script SLURM:

```text
src/explainability/spatial/xai_generate.sbatch
```

Esegue:

```bash
cd /home/mla_group_01/rcc-ssrl/src/explainability/spatial
srun python3 xai_generate.py --config "$CONFIG_PATH"
```

### 4.2. Config

Template:

```text
src/explainability/spatial/config_xai.yaml
```

Parametri chiave (modificati dall’orchestratore per ablation):

* `experiment.outputs_root`: `EXP_ROOT/exp_*/06_xai`.
* `evaluation_inputs.eval_run_dir`: directory eval con `predictions.csv` + `logits_test.npy`.
* `data.webdataset.test_dir`:
  `/.../data/processed/rcc_webdataset_final/test`
* `data.labels.class_order`:
  `["ccRCC", "pRCC", "CHROMO", "ONCO", "NOT_TUMOR"]`
* `model`:

  * `backbone_name: vit_small_patch16_224`
  * `ssl_backbone_ckpt`, `ssl_head_ckpt`: path ai checkpoint.
* `selection`: logica TP/FP/FN per classe e low-confidence.
* `xai.methods`: `["attn_rollout", "gradcam", "ig", ...]`.

### 4.3. Flusso

1. Carica artefatti eval:

   * `predictions.csv` (deve contenere `y_true`, `y_pred`, `wds_key`).
   * `logits_test.npy`.

2. Costruisce il modello SSL (backbone + head).

3. Costruisce DataLoader WebDataset sul **test set**.

4. Seleziona le patch da spiegare (`select_items`):
   target = subset di `wds_key` (TP/FP/FN, etc.).

5. Per ogni patch target:

   * esegue forward;
   * genera heatmap:

     * Integrated Gradients (IG, se Captum disponibile),
     * GradCAM (se torchcam e layer disponibili),
     * Attention rollout (per ViT).

6. Salva per ogni patch:

   ```text
   06_xai/moco_v3_ssl_linear_best/<timestamp>/idx_0000001/
     input.png
     ig.png
     gradcam.png
     attn_rollout.png
     ...
   ```

7. Indice globale:

   ```text
   06_xai/moco_v3_ssl_linear_best/<timestamp>/index.csv
   ```

   con:

   * `global_idx, wds_key, true, pred, conf, methods, png_paths, selection_reason`.

---

## 5. Concept XAI – `concept/xai_concept.py`

### 5.1. sbatch

Script SLURM:

```text
src/explainability/concept/xai_concept.sbatch
```

Esegue:

```bash
cd /home/mla_group_01/rcc-ssrl/src/explainability/concept
srun python3 xai_concept.py --config "$CONFIG_PATH"
```

### 5.2. Config

Template:

```text
src/explainability/concept/config_concept.yaml
```

Parametri chiave:

* `experiment.outputs_root`: `EXP_ROOT/exp_*/06_xai`.
* `evaluation_inputs.eval_run_dir`: stessa eval run dello Spatial.
* `data.webdataset.test_dir`:
  `/.../rcc_webdataset_final/test`
* `model`: stesso backbone e checkpoint dello Spatial.
* `selection`: stessa logica (TP/FP/FN, low-conf).
* `concepts`:

  ```yaml
  concepts:
    meta_csv: /home/mla_group_01/rcc-ssrl/src/explainability/concept/ontology/concepts_rcc_v1.csv
    concept_name_col: "concept_name"
    key_col: "wds_key"
    group_col: "group"
    class_col: "class_label"
    similarity: "cosine"
    topk_per_patch: 5
    min_patches_per_concept: 5
  ```

> Nota: nel template possono esserci parametri duplicati (`similarity`, `topk_per_patch`, `min_patches_per_concept`); vale l’ultima definizione.

### 5.3. Flusso

1. Carica artefatti eval (`predictions.csv`, `logits_test.npy`).

2. Costruisce il modello SSL.

3. Seleziona targets da spiegare (`select_items` → lista di `wds_key`).

4. Carica concept bank:

   * `concepts_rcc_v1.csv` →
     `concept_to_keys`, `concept_meta`, `concept_keys`.

5. Determina tutte le chiavi per cui servono feature:

   ```python
   all_needed_keys = target_set ∪ concept_keys
   ```

6. Costruisce DataLoader WebDataset sul **test set**:

   * e qui sta il problema: si cercano features anche per chiavi che provengono dal **train** (concept bank), ma i loader leggono solo dal **test**; molte `wds_key` per i concetti non verranno mai viste.

7. Estrae feature:

   ```python
   for img_t, meta, key in loader:
       if key not in all_needed_keys:
           continue
       feat_by_key[key] = backbone.forward_global(...)
       if key in target_set:
           input_tensor_by_key[key] = img_t
   ```

   Logga eventuali chiavi mancanti:

   ```text
   Missing features for N keys (will be ignored).
   ```

8. Costruisce centroidi per concetto:

   ```python
   feats = [feat_by_key[k] for k in ckeys if k in feat_by_key]
   if len(feats) < min_patches_per_concept:
       skip concept
   centroid = mean(feats)
   ```

   Se nessun concetto ha abbastanza patch (`>= min_patches_per_concept`):

   ```python
   raise RuntimeError("No valid concept centroids could be built.")
   ```

9. Similarity patch–concetto:

   * forma matrice centroidi `[C, D]`;
   * normalizza per cosine similarity;
   * per ogni patch target:

     ```python
     scores = centroid_matrix @ feat_vec
     ```

     → ordina, prende top-k concetti.

10. Output per patch:

    ```text
    06_xai/moco_v3_ssl_linear_best/<timestamp>/idx_0000001/
      input.png
      concept_scores.json
    ```

    `concept_scores.json` contiene:

    * `wds_key, true_label, pred_label, conf, selection_reason`
    * `similarity: "cosine"`
    * `concept_scores`: lista di concetti con score e meta.

11. Indice globale:

    ```text
    06_xai/moco_v3_ssl_linear_best/<timestamp>/index.csv
    ```

    con colonne tipo:

    * `global_idx, wds_key, true_label, pred_label, conf, selection_reason, top_concepts`.

---

## 6. Incoerenze e punti critici

### 6.1. Train vs Test per i concetti

Fatto attuale:

* Stage 0 (concept bank) usa **TRAIN** WebDataset.
* Concept XAI (`xai_concept.py`) estrae feature solo dal **TEST** (`data.webdataset.test_dir`).

Conseguenza:

* Molte `wds_key` nella concept bank (train) non compaiono nei shard test.
* `feat_by_key` non ha feature per quei concetti → i relativi centroidi non vengono costruiti.
* Se tutti i concetti finiscono sotto `min_patches_per_concept`, Concept XAI fallisce con:

  ```text
  No valid concept centroids could be built.
  ```

Questa è una contraddizione strutturale, non un dettaglio.

Per riparare devi scegliere una strategia coerente, ad esempio:

* **Opzione A:**
  Stage 0 sui **train shards**, Concept XAI che legge **train** (non test).
* **Opzione B:**
  rifare Stage 0 sui **test shards**, se vuoi concetti definiti sul test.
* **Opzione C:**
  usare **train+test** per estrarre feature nei centroidi (loader su entrambe le sorgenti).

Finché concept bank e sorgente di feature non sono allineati, Concept XAI è strutturalmente fragile.

---

### 6.2. Requisiti forti non negoziabili

* `predictions.csv` deve contenere:

  * `y_true`, `y_pred`, `wds_key`
  * `wds_key` deve matchare `__key__` nel WebDataset (train/test).

* `logits_test.npy` deve avere:

  * righe = righe di `predictions.csv`
  * colonne = numero classi (`len(labels.class_order)`).

* `concepts_rcc_v1.csv` non deve essere vuoto:

  ```bash
  wc -l src/explainability/concept/ontology/concepts_rcc_v1.csv
  ```

  Se è `1` (solo header), Stage 0 non ha prodotto concetti → Concept XAI non è utilizzabile.

---

### 6.3. Dettagli minori

* Duplicazioni nel `config_concept.yaml` (`similarity`, `topk_per_patch`, ecc.):
  tecnicamente innocue ma inutilmente confondenti; va pulito.

* `imagenet_norm: false` nei config XAI:
  se il training è stato fatto con normalizzazione ImageNet, c’è un mismatch tra training e XAI.

---

## 7. Struttura degli output per ablation

Per una ablation `exp_moco_v3_abl01`:

```text
EXP_ROOT/
  exp_moco_v3_abl01/
    checkpoints/
      moco_v3__ssl_best.pt
      moco_v3__ssl_linear_best.pt

    eval/
      moco_v3_ssl_linear_best/
        20251125_145930/
          predictions.csv
          logits_test.npy
          ...

    06_xai/
      config_xai.yaml
      config_concept.yaml

      moco_v3_ssl_linear_best/
        20251126_211300/      # Spatial XAI run
          index.csv
          idx_0000001/
            input.png
            ig.png
            gradcam.png
            attn_rollout.png
            ...

        20251126_211310/      # Concept XAI run
          index.csv
          idx_0000001/
            input.png
            concept_scores.json
```

---

## 8. Esecuzione rapida

1. Assicurarsi che:

   * Train WebDataset sia al path corretto.
   * Ontologia YAML sia consistente.
   * VLM (LLaVA-Med) sia avviato all’endpoint previsto.

2. Creare concept bank (prima esecuzione):

   ```bash
   cd src/explainability
   ./run_full_xai.sh
   ```

   Se `concepts_rcc_v1.csv` non esiste, verranno eseguiti Stage 0a + 0b.

3. Lanciare explainability per un experiment:

   * Settare `EXP_ROOT`, `MODEL_NAME` in `run_full_xai.sh` (se diverso).
   * Rieseguire:

     ```bash
     ./run_full_xai.sh
     ```

   Stage 0 verrà saltato se la concept bank esiste già; verrà eseguito solo Stage 1/2.

4. Verificare gli output in:

   ```text
   EXP_ROOT/exp_*/06_xai/moco_v3_ssl_linear_best/<timestamp>/
   ```

---

## 9. Sintesi concettuale

* La **concept bank** (`concepts_rcc_v1.csv`) è un **CSV statico**, generato una sola volta con il VLM, che associa patch del train a concetti istologici.
* La pipeline di XAI **non addestra nulla**: usa solo il backbone SSL + head già addestrati.
* Lo **Spatial XAI** produce heatmap (IG, GradCAM, attention rollout) sulle patch selezionate del test.
* Il **Concept XAI** collega le feature delle patch a centroidi di concetto costruiti a partire dalla concept bank.

Se questi tre oggetti non sono coerenti fra loro (concept bank vs train/test, wds_key coerente tra eval e WebDataset, concept bank non vuota), il sistema si rompe per design.
