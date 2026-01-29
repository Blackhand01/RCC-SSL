import os
import shutil
import glob
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURAZIONE ---
# Percorso sorgente (BeeGFS dove stanno i dati pesanti)
SOURCE_ROOT = Path("/mnt/beegfs-compat/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/ablation_final")

# Percorso destinazione (Login node -> Docs per GitHub)
DEST_ROOT = Path("/home/mla_group_01/rcc-ssrl/docs/ablation_showcase")

# Cartelle/File specifici da copiare (relativi alla cartella dell'ablazione)
TARGETS = [
    # Training Plots & Metrics
    "plots/*.png",
    "metrics/*.json",
    "metrics/*.csv",
    
    # Configuration
    "configuration/*.yaml",
    
    # Evaluation (solo immagini e report json)
    "eval/**/*.png",
    "eval/**/report_per_class.json",
    "eval/**/metrics_*.json",
    
    # XAI Summary
    "attention_rollout_concept/run_latest/xai_summary.csv",
    "attention_rollout_concept/run_latest/xai_summary.json",
    "attention_rollout_concept/run_latest/montage.png", 
]

def copy_all_xai_items(abl_path, dest_abl_path):
    """Copia TUTTE le cartelle items per l'XAI."""
    items_src = abl_path / "attention_rollout_concept/run_latest/items"
    items_dest = dest_abl_path / "attention_rollout_concept/run_latest/items"
    
    if not items_src.exists():
        return

    # Trova tutte le cartelle idx_XXXXXX
    all_items = sorted(list(items_src.glob("idx_*")))
    
    if all_items:
        print(f"    -> Copiando {len(all_items)} item XAI (TUTTI)...")
        # Creiamo la cartella items di destinazione se non esiste
        items_dest.mkdir(parents=True, exist_ok=True)
        
        for item in tqdm(all_items, desc="    Progress", unit="item", leave=False):
            dest_item = items_dest / item.name
            
            # Se la cartella esiste già, la rimuoviamo per sovrascriverla pulita
            if dest_item.exists():
                shutil.rmtree(dest_item)
                
            # Copia ricorsiva
            shutil.copytree(item, dest_item)

def main():
    if not SOURCE_ROOT.exists():
        print(f"ERRORE: La cartella sorgente non esiste: {SOURCE_ROOT}")
        return

    # Trova tutti gli esperimenti (es. exp_2025..._ibot)
    experiments = sorted(list(SOURCE_ROOT.glob("exp_*")))
    
    print(f"Trovati {len(experiments)} esperimenti in {SOURCE_ROOT}")
    print(f"Destinazione: {DEST_ROOT}")
    
    for exp_path in experiments:
        exp_name = exp_path.name
        print(f"\nProcessando Esperimento: {exp_name}")
        
        # Trova le ablazioni dentro l'esperimento (es. exp_ibot_abl01)
        ablations = sorted(list(exp_path.glob("exp_*_abl*")))
        
        for abl_path in ablations:
            abl_name = abl_path.name
            print(f"  > Ablazione: {abl_name}")
            
            # Crea cartella destinazione: docs/ablation_showcase/<exp>/<abl>
            dest_abl_path = DEST_ROOT / exp_name / abl_name
            dest_abl_path.mkdir(parents=True, exist_ok=True)
            
            # 1. Copia i file target (Plots, Metrics, Configs)
            files_copied = 0
            for pattern in TARGETS:
                # Usa glob per trovare i file che matchano il pattern
                found_files = list(abl_path.glob(pattern))
                for f in found_files:
                    # Calcola il percorso relativo per mantenere la struttura
                    rel_path = f.relative_to(abl_path)
                    dest_file = dest_abl_path / rel_path
                    
                    # Crea la cartella genitore se non esiste
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(f, dest_file)
                    files_copied += 1
            
            print(f"    -> Copiati {files_copied} file di reportistica.")
            
            # 2. Copia TUTTI gli Items XAI
            copy_all_xai_items(abl_path, dest_abl_path)

    print(f"\n--- Finito! ---")
    print(f"Tutti i dati sono stati copiati in: {DEST_ROOT}")

if __name__ == "__main__":
    main()