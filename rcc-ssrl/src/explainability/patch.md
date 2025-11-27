Bene, adesso il quadro è chiaro:

1. La concept bank è vuota (solo header).
2. `run_full_xai.sh` vede il file esistente e **salta** Stage 0 → rimani bloccato con un CSV inutile.
3. Anche se sistemiamo questo, c’è il problema strutturale: concept bank fatta su **train**, ma Concept XAI estrae feature solo da **test**, quindi non troverà mai le patch dei concetti.

Ti do una correzione completa e concreta della pipeline, con patch precise.

---

## 1. Problema 1: concept bank vuota ma considerata “valida”

Fatto attuale in `run_full_xai.sh`:

```bash
if [[ -f "${CONCEPT_BANK_CSV}" ]]; then
  echo "[INFO] Concept bank already exists at ${CONCEPT_BANK_CSV} – skipping Stage 0."
else
  # Stage 0a + 0b
fi
```

Se `concepts_rcc_v1.csv` ha solo l’header (1 riga), il check passa e Stage 0 viene saltato.
È palesemente sbagliato.

### Fix 1: controlla che la concept bank sia **non vuota**

Modifica il blocco Stage 0 in `run_full_xai.sh` così:

```bash
# ------------------- STAGE 0: concept bank (solo se NON esiste o è vuota) -------------------

# Conta le righe se il file esiste
if [[ -f "${CONCEPT_BANK_CSV}" ]]; then
  num_lines=$(wc -l < "${CONCEPT_BANK_CSV}")
else
  num_lines=0
fi

if [[ "${num_lines}" -le 1 ]]; then
  echo "[WARN] Concept bank missing or empty (lines=${num_lines}); rebuilding Stage 0."

  # 0a) concept_candidates_rcc.csv (train WDS -> PNG + CSV)
  echo "[INFO] Stage 0a: building concept_candidates_rcc.csv"
  python3 -m explainability.concept.ontology.build_concept_candidates \
    --train-dir "${TRAIN_WDS_DIR}" \
    --pattern "shard-*.tar" \
    --image-key "img.jpg;jpg;jpeg;png" \
    --meta-key "meta.json;json" \
    --out-csv "${CANDIDATES_CSV}" \
    --images-root "${CANDIDATES_IMG_ROOT}"

  # 0b) concepts_rcc_v1.csv (VLM su candidates)
  echo "[INFO] Stage 0b: building concepts_rcc_v1.csv via VLM"
  start_local_vlm
  python3 -m explainability.concept.ontology.build_concept_bank \
    --ontology "${ONTOLOGY_YAML}" \
    --images-csv "${CANDIDATES_CSV}" \
    --controller "${VLM_CONTROLLER}" \
    --model-name "${VLM_MODEL}" \
    --out-csv "${CONCEPT_BANK_CSV}" \
    --presence-threshold "${PRESENCE_THRESHOLD}"
  stop_local_vlm
else
  echo "[INFO] Concept bank found at ${CONCEPT_BANK_CSV} with ${num_lines} lines – skipping Stage 0."
fi
```

In più, **adesso**:

```bash
rm /home/mla_group_01/rcc-ssrl/src/explainability/concept/ontology/concepts_rcc_v1.csv
```

altrimenti non rigenera niente.

### Nota fondamentale: perché la concept bank è vuota?

Non è solo un bug di controllo file:

* `build_concept_bank.py` scrive l’header, poi:

  * se nessuna patch supera `present==True` e `confidence>=threshold`, `accepted=0` e alla fine fa:

    ```python
    if accepted == 0:
        raise RuntimeError("Concept bank is empty (...)")
    ```

* Con `set -e` nello shell, questo manda in errore Stage 0, ma il file CSV con solo header resta sul disco.

Quindi:

* Hai **due problemi**:

  1. File vuoto considerato “ok” (fixato sopra).
  2. Il VLM non returna niente di accettabile (o non viene parse-ato correttamente).

Per il secondo, la colpa non è nella pipeline, ma in:

* VLM non avviato o endpoint sbagliato.
* Risposta del modello non in JSON nel formato atteso.
* `presence_threshold` troppo alto per i tuoi prompt.

Per debug pratico:

```bash
START_LOCAL_VLM=1 PRESENCE_THRESHOLD=0.3 ./run_full_xai.sh
```

e controlla i log:

* `/tmp/llava_controller.log`
* `/tmp/llava_worker.log`
* eventuali log di `build_concept_bank` in stdout.

---

## 2. Problema 2: concept bank su TRAIN, Concept XAI che guarda solo TEST

Attuale:

* Stage 0a/b usa `TRAIN_WDS_DIR`:

  * `/.../rcc_webdataset_final/train`

* `concept/config_concept.yaml` ha:

  ```yaml
  data:
    backend: "webdataset"
    img_size: 224
    imagenet_norm: false
    num_workers: 4
    batch_size: 1
    webdataset:
      test_dir: "/.../rcc_webdataset_final/test"
      pattern: "shard-*.tar"
      image_key: "img.jpg;jpg;jpeg;png"
      meta_key: "meta.json;json"
  ```

* `xai_concept.py` fa:

  ```python
  w = cfg["data"]["webdataset"]
  loader = make_wds_loader_with_keys(
      w["test_dir"], ..., preprocess_fn, num_workers
  )
  ```

Quindi:

* concept bank → `wds_key` delle patch di **train**.
* Concept XAI → cerca quelle key **solo nei shard test**.
* Risultato: zero feature per la quasi totalità dei concetti → `No valid concept centroids` appena la concept bank smette di essere vuota.

Se vuoi un pipeline sensato, devi fare in modo che Concept XAI possa:

* estrarre feature per:

  * patch target (test) → per spiegare le predizioni
  * patch concetti (train) → per i centroidi dei concetti

Quindi il loader dev’essere su **train + test**.

### Fix 2a: aggiungi `train_dir` al config concept

Modifica `concept/config_concept.yaml` in questo modo:

```yaml
data:
  backend: "webdataset"
  img_size: 224
  imagenet_norm: false
  num_workers: 4
  batch_size: 1
  webdataset:
    # NUOVO
    train_dir: "/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/train"
    # esistente
    test_dir: "/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test"
    pattern: "shard-*.tar"
    image_key: "img.jpg;jpg;jpeg;png"
    meta_key: "meta.json;json"
```

Elimina le duplicazioni finali:

```yaml
  similarity: "cosine"
  topk_per_patch: 5
  min_patches_per_concept: 5

  # rimuovi il secondo blocco duplicato
  # similarity: "cosine"
  # topk_per_patch: 5
  # min_patches_per_concept: 5
```

### Fix 2b: fai leggere train+test in `xai_concept.py`

Ora adatti `xai_concept.py` per scorrere **sia train che test**.
Dentro `main`, sostituisci il blocco:

```python
w = cfg["data"]["webdataset"]
loader = make_wds_loader_with_keys(
    w["test_dir"],
    w["pattern"],
    w["image_key"],
    w["meta_key"],
    preprocess_fn,
    int(cfg["data"]["num_workers"]),
)
```

con qualcosa del genere:

```python
w = cfg["data"]["webdataset"]
num_workers = int(cfg["data"]["num_workers"])

train_dir = w.get("train_dir")
test_dir = w.get("test_dir")

dirs = []
if train_dir:
    dirs.append(train_dir)
if test_dir:
    dirs.append(test_dir)

if not dirs:
    raise RuntimeError("Concept XAI: no webdataset.train_dir or test_dir specified in config.")

# Map keys to eval indices (per conf) come prima
idx_by_key = {k: i for i, k in enumerate(keys)}

feat_by_key: Dict[str, np.ndarray] = {}
input_tensor_by_key: Dict[str, torch.Tensor] = {}

with torch.no_grad():
    for d in dirs:
        try:
            loader = make_wds_loader_with_keys(
                d,
                w["pattern"],
                w["image_key"],
                w["meta_key"],
                preprocess_fn,
                num_workers,
            )
        except FileNotFoundError as e:
            log.warning(f"[Concept XAI] No shards found in {d}: {e}")
            continue

        log.info(f"[Concept XAI] Scanning WebDataset shards in {d}")
        for batch in loader:
            if batch is None:
                continue
            img_t, meta_any, key = batch

            # Se il key non serve né per concetti né per target, skip
            if key not in all_needed_keys:
                continue

            x = img_t.to(device)
            if x.ndim == 3:
                x = x.unsqueeze(0)

            feats = model.backbone.forward_global(x)  # [1, D]
            feat_by_key[key] = feats.squeeze(0).cpu().numpy()

            # Solo per target, salviamo anche il tensor per salvare input.png
            if key in target_set and key not in input_tensor_by_key:
                input_tensor_by_key[key] = img_t  # tensor già preprocessato
```

Poi immediatamente sotto **lascia invariato**:

```python
missing = all_needed_keys - set(feat_by_key.keys())
if missing:
    log.warning(f"Missing features for {len(missing)} keys (will be ignored).")
```

Così:

* le feature per i concetti verranno prese dal **train** (dove i wds_key esistono).
* le feature per i target test verranno prese dal **test** (perché keys di eval sono di lì).
* `input_tensor_by_key` conterrà solo le patch test da salvare come `input.png`.

Questa è la correzione strutturale che ti mancava.

---

## 3. Il resto dei log: cosa è davvero un problema e cosa no

Dal log di `run_full_xai.sh`:

* Stage 1/2 ha lanciato sbatch per varie ablations.
  Per molte ablations hai:

  ```text
  [SKIP] Eval directory not found for exp_moco_v3_abl0X ...
  [ABLATION] Skipping ... no eval run available.
  ```

  Questo non è un bug della pipeline XAI: significa che **non hai mai fatto eval** per quelle ablations (mancano `eval/moco_v3_ssl_linear_best/...`). Se vuoi XAI anche lì, devi prima salvare `predictions.csv` + `logits_test.npy` per quelle ablations.

* `squeue -u mla_group_01` vuoto subito dopo:

  * vuol dire che i job XAI submitatti (spatial+concept) si sono chiusi molto rapidamente (successo o errore).
  * i log sono in:

    * Spatial:
      `/home/mla_group_01/rcc-ssrl/src/logs/xai/moco_v3_ssl_linear_best/<ablation>/<timestamp>/xai_spatial.*.err/out`
    * Concept:
      `/home/mla_group_01/rcc-ssrl/src/logs/xai/moco_v3_ssl_linear_best/<ablation>/<timestamp>/xai_concept.*.err/out`

Con la concept bank vuota, Concept XAI ti darà inevitabilmente:

* o RuntimeError da `build_concept_bank` (Stage 0)
* o, se bypassi Stage 0, `No valid concept centroids could be built.` in `xai_concept.py`

Finché non risolvi i due fix sopra, è normale che la parte “concept” imploda.

---

## 4. Piano concreto e ordinato

Se vuoi smettere di girare in tondo:

1. **Fixa Stage 0 in `run_full_xai.sh`** come ti ho scritto, in modo che controlli `wc -l` e rigeneri la concept bank se è vuota.

2. **Cancella il CSV rotto**:

   ```bash
   rm /home/mla_group_01/rcc-ssrl/src/explainability/concept/ontology/concepts_rcc_v1.csv
   ```

3. **Abbassa per debug la soglia**:

   * esporta, per una run di test:

     ```bash
     export PRESENCE_THRESHOLD=0.3
     export START_LOCAL_VLM=1
     ```

4. **Sistema `config_concept.yaml` + `xai_concept.py` per usare train+test** come sopra. Senza questo, anche con una concept bank valida, Concept XAI resterà concettualmente zoppa.

5. Rilancia:

   ```bash
   cd /home/mla_group_01/rcc-ssrl/src/explainability
   ./run_full_xai.sh
   ```

6. Se dopo tutto questo `concepts_rcc_v1.csv` è ancora con 1 riga:

   * il problema non è la pipeline XAI, è il VLM:

     * endpoint sbagliato
     * risposta non parse-abile
     * modello che non mette mai `present=true`

In sintesi: adesso la pipeline così com’è non è semplicemente “pignola”, è incoerente.
Finché non allinei:

* controllo di esistenza concept bank ↔ contenuto reale
* split usati per generare i concetti ↔ split da cui estrai feature

non avrai un sistema di explainability minimamente credibile.
