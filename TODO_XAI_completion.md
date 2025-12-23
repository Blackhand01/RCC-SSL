# Piano di Completamento Documento XAI LaTeX

## Analisi della Situazione Attuale

### Documento LaTeX Esistente
Il documento `paper_xai_latex.md` è una bozza che descrive la metodologia XAI ma manca di:
- Dati quantitativi concreti
- Tabelle con risultati
- Figure con esempi
- Metriche di valutazione

### Sistema XAI Disponibile
Il sistema include:
1. **XAI Concettuale con PLIP**: 75 concetti istologici definiti
2. **XAI Spaziale**: Attention rollout su ViT backbone  
3. **Pipeline Unificata**: Spatial + Concept XAI con ROI
4. **Calibrazione**: Shortlist di concetti per classe
5. **Confronto**: ROI vs NO-ROI per validazione

## Piano di Esecuzione

### Fase 1: Preparazione e Analisi Dati Esistenti
1. **Analizzare dati di calibrazione esistenti**
   - Esaminare shortlist concetti per classe
   - Analizzare metriche AUC/AP per concetti
   - Identificare concetti diagnostici vs confondenti

2. **Verificare pipeline e configurazioni**
   - Controllare configurazioni YAML
   - Verificare setup PLIP e modelli ViT
   - Validare percorsi dati e output

### Fase 2: Generazione Risultati Concreti
1. **Eseguire calibrazione concetti**
   - Calcolare metriche per tutti i 75 concetti
   - Costruire shortlist ottimizzata per classe
   - Generare statistiche di stabilità cross-split

2. **Eseguire pipeline XAI completa**
   - Calcolare attention rollout su subset test
   - Estrarre ROI dalle heatmap
   - Scoring concetti su ROI vs immagine completa
   - Generare confronti quantitativi

3. **Creare visualizzazioni**
   - Heatmap esempi attention rollout
   - Barplot concetti ROI vs full
   - Tabelle sintetiche ROI vs NO-ROI

### Fase 3: Completamento Documento LaTeX
1. **Compilare tabelle quantitative**
   - Tabella shortlist concettuale per classe
   - Tabella sintesi ROI vs full
   - Metriche aggregate per classe

2. **Integrare risultati qualitativi**
   - Esempi TP/FP/FN con overlay heatmap
   - Casi studio concept XAI
   - Analisi qualitativa concetti emergenti

3. **Aggiungere discussione risultati**
   - Interpretazione concetti diagnostici
   - Analisi riduzione confondenti in ROI
   - Limitazioni e validazione metodologia

### Fase 4: Validazione e Finalizzazione
1. **Validare coerenza dati**
   - Verificare consistenza metriche
   - Controllare allineamento figure/tabelle
   - Validare riproducibilità risultati

2. **Ottimizzare presentazione**
   - Migliorare formattazione tabelle
   - Ottimizzare qualità figure
   - Finalizzare riferimenti bibliografici

## File da Modificare

### File Principale
- `rcc-ssrl/src/explainability/docs/paper_xai_latex.md` - Documento LaTeX principale

### File di Supporto da Generare
- Tabelle CSV con risultati quantitativi
- Figure PNG/PDF per esempi
- Script di generazione automatica contenuti

## Dipendenze Tecniche

### Dati Richiesti
- Modelli PLIP pre-calibrati
- Checkpoints modelli ViT backbone  
- Dataset WebDataset test set
- Artefatti calibrazione esistenti

### Software Richiesto
- Python con librerie XAI (captum, torchcam)
- PLIP model e tokenizer
- Matplotlib/Seaborn per figure
- LaTeX per compilazione finale

## Timeline Stimata
- **Fase 1**: 30 min - Analisi e preparazione
- **Fase 2**: 2-3 ore - Generazione risultati  
- **Fase 3**: 1-2 ore - Completamento documento
- **Fase 4**: 30 min - Validazione finale

**Totale**: 4-6 ore di lavoro

## Note Importanti
- Documento deve rimanere in formato Markdown compatibile LaTeX
- Risultati devono essere riproducibili e consistenti
- Figurine e tabelle devono essere di qualità pubblicazione
- Mantenere coerenza terminologica con resto del paper

## Prossimi Passi
1. Conferma piano con utente
2. Avvio Fase 1: Analisi dati esistenti
3. Esecuzione sequenziale delle fasi
4. Validazione risultati finali
