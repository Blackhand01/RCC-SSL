import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import os

# --- PATHS CONFIGURATION ---
METADATA_PATH = "/home/mla_group_01/rcc-ssrl/src/data_preprocessing/reports/01_rcc_metadata/rcc_metadata.csv"
# Point to the statistics of the balanced train
BALANCED_STATS_PATH = "/mnt/beegfs-compat/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_balanced_v2/stats_train.json"
FOLDS_JSON_PATH = "/mnt/beegfs-compat/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_splits/holdout_70_15_15/folds.json"
OUTPUT_IMAGE = "training_set_distribution_balanced.png"

# --- PROFESSIONAL PALETTE FROM PAPER ---
COLOR_TUMOR = "#d62728"    # Brick red (Tumor)
COLOR_NONTUMOR = "#1f77b4" # Blu acciaio (Non-Tumore)
COLOR_PATIENTS = "#333333" # Grigio scuro (Pazienti)

def generate_figure_train_only():
    # 1. Load Metadata for patient mapping
    meta_df = pd.read_csv(METADATA_PATH)
    p_to_st = meta_df.groupby('patient_id')['subtype'].first().to_dict()

    # 2. Load Folds and Stats
    with open(FOLDS_JSON_PATH, 'r') as f:
        folds = json.load(f)
    with open(BALANCED_STATS_PATH, 'r') as f:
        stats = json.load(f)

    subtypes = ["ccRCC", "pRCC", "CHROMO", "ONCO"]
    data = []

    # 3. Aggregation for train split only
    for st in subtypes:
        # Count of UNIQUE patients only in train
        num_patients = folds["by_class_counts"]["train"].get(st, 0)
        
        # Tumor patches selected for train
        tumor_patches = stats["selected"]["train"].get(st, 0)
        
        # Non-tumor patches associated with this specific subtype in train
        # The balancing logic assigns NOT_TUMOR based on present patients
        nt_patches = 0
        nt_train_map = stats["patients_train"].get("NOT_TUMOR", {})
        
        # Sum NOT_TUMOR patches only of patients belonging to this subtype in train
        train_patients = folds["patients"]["train"]
        nt_patches = sum(nt_train_map.get(p, 0) for p in train_patients if p_to_st.get(p) == st)

        data.append({
            "Subtype": st,
            "Patients": num_patients,
            "Tumor Patches": int(tumor_patches),
            "Non-Tumor Patches": int(nt_patches)
        })

    df = pd.DataFrame(data)

    # --- RENDERING GRAPH ---
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Stacked bars (Y-axis 1 - Patches)
    # Visually show balance between tumor classes and non-tumor component
    df.set_index("Subtype")[["Non-Tumor Patches", "Tumor Patches"]].plot(
        kind="bar", stacked=True, ax=ax1, 
        color=[COLOR_NONTUMOR, COLOR_TUMOR], alpha=0.85,
        edgecolor='black', linewidth=0.5, width=0.6, rot=0
    )

    ax1.set_ylabel("Number of Training Patches", fontsize=12, fontweight='bold', labelpad=15)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1000)) # More granular for train only
    ax1.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax1.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

    # Stratification line (Y-axis 2 - Patients in Train)
    ax2 = ax1.twinx()
    ax2.plot(df["Subtype"], df["Patients"], color=COLOR_PATIENTS, marker="s", 
             markersize=10, linewidth=2.5, label="Patients (Train)", markerfacecolor='white', markeredgewidth=2, zorder=10)

    ax2.set_ylabel("Number of Patients in Train Set", color=COLOR_PATIENTS, fontsize=12, fontweight='bold', labelpad=15)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(5)) 
    ax2.set_ylim(0, df["Patients"].max() + 15) 
    ax2.tick_params(axis='y', labelcolor=COLOR_PATIENTS)
    ax2.grid(False)

    # --- LEGEND AND TITLE ADJUSTMENT ---
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    ax1.legend(handles1 + handles2, ["Non-Tumor Patches", "Tumor Patches", "Unique Patients"], 
               loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False, fontsize=11)

    for i, p in enumerate(df["Patients"]):
        ax2.annotate(f"N={p}", (i, p), textcoords="offset points", xytext=(0,15), 
                     ha='center', fontweight='bold', color=COLOR_PATIENTS, fontsize=10)

    plt.title("Balanced Training Set Distribution (Subtype Stratification)", 
              fontsize=14, fontweight='bold', pad=35)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"[OK] Graph saved: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    generate_figure_train_only()