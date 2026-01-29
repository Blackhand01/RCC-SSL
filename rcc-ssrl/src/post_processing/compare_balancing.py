import pandas as pd
import matplotlib.pyplot as plt
import json

# --- CONFIGURATION ---
METADATA_PATH = "/home/mla_group_01/rcc-ssrl/src/data_preprocessing/reports/01_rcc_metadata/rcc_metadata.csv"
BALANCED_STATS_PATH = "/mnt/beegfs-compat/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_balanced_v2/stats_train.json"
OUTPUT_IMAGE = "balancing_comparison_fig.png"

def generate_balancing_comparison():
    # 1. Load Original Metadata (Imbalanced)
    meta_df = pd.read_csv(METADATA_PATH)
    # We assume that each row represents a patch or proportional estimate
    raw_counts = meta_df['subtype'].value_counts()

    # 2. Load Balanced Statistics (Train Set)
    with open(BALANCED_STATS_PATH, 'r') as f:
        stats = json.load(f)
    
    subtypes = ["ccRCC", "pRCC", "CHROMO", "ONCO"]
    balanced_counts = {st: stats["selected"]["train"].get(st, 0) for st in subtypes}

    # Create DataFrame for the plot
    comparison_df = pd.DataFrame({
        "Original Dataset": [raw_counts.get(st, 0) for st in subtypes],
        "Balanced Train Set": [balanced_counts[st] for st in subtypes]
    }, index=subtypes)

    # --- RENDERING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=False)

    # Plot 1: Original Imbalanced Distribution
    comparison_df["Original Dataset"].plot(kind="bar", ax=ax1, color="#7f7f7f", alpha=0.7, edgecolor='black')
    ax1.set_title("1. Original Imbalanced Distribution", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Number of Patches")
    
    # Plot 2: Balanced Training Set
    comparison_df["Balanced Train Set"].plot(kind="bar", ax=ax2, color="#d62728", alpha=0.8, edgecolor='black')
    ax2.set_title("2. Balanced Training Set (Final)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Number of Patches")

    # Add labels on values for clarity
    for ax in [ax1, ax2]:
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}", (p.get_x() * 1.005, p.get_height() * 1.005), 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle("Data Balancing Strategy: From Raw WSI to Stratified Training", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"[OK] Comparison graph saved: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    generate_balancing_comparison()