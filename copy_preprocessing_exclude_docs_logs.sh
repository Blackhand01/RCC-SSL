#!/bin/bash

# Script to dump preprocessing-related files into a single Markdown file
# Usage: ./dump_preprocessing_snapshot_to_md.sh <output_markdown_file>
#
# It will collect:
# - SLURM scripts for preprocessing stages
# - Reports (JSON/CSV/MD) for drive analysis, metadata, parquet
#
# NOTE:
# - CSV files are truncated to the first N lines for readability.
# - Parquet files are NOT inlined (only referenced), since they are binary.

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <output_markdown_file>"
    exit 1
fi

OUTPUT_MD=$1
PROJECT_ROOT="/home/mla_group_01/rcc-ssrl"
CSV_HEAD_LINES=80  # adjust if needed

# Directories whose structure we want to list
DIRS_TO_LIST=(
  "slurm/00_gpu-smoke-done"
  "slurm/00_wsi-drive-analysis"
  "slurm/01_rcc_metadata"
  "slurm/02_parquet"
  "reports/00_wsi-drive-analysis"
  "reports/01_rcc_metadata"
  "reports/02_parquet"
)

# Files whose content we want to inline (when text-like)
FILES_TO_INLINE=(
  "reports/00_wsi-drive-analysis/ccRCC_mapping.json"
  "reports/00_wsi-drive-analysis/CHROMO_patient_mapping.json"
  "reports/00_wsi-drive-analysis/ONCO_patient_mapping.json"
  "reports/00_wsi-drive-analysis/pRCC_mapping.json"
  "reports/00_wsi-drive-analysis/metadata.csv"
  "reports/00_wsi-drive-analysis/rcc_dataset_stats.json"
  "reports/00_wsi-drive-analysis/wsi_drive_analysis.md"
  "reports/00_wsi-drive-analysis/wsi_inventory.csv"
  "reports/01_rcc_metadata/rcc_metadata.csv"
  "reports/02_parquet/slides.csv"
  "reports/02_parquet/slides.parquet"
)

# Helper: detect a "language" for markdown fences based on extension
detect_language() {
    local path="$1"
    case "$path" in
        *.sh)    echo "bash" ;;
        *.sbatch) echo "bash" ;;
        *.py)    echo "python" ;;
        *.json)  echo "json" ;;
        *.csv)   echo "csv" ;;
        *.md)    echo "markdown" ;;
        *)       echo "" ;;
    esac
}

# Start fresh
mkdir -p "$(dirname "$OUTPUT_MD")"
: > "$OUTPUT_MD"

# Global header
{
    echo "# RCC-SSRL Preprocessing Snapshot"
    echo
    echo "_Generated on $(date) from project root: \`$PROJECT_ROOT\`_"
    echo
} >> "$OUTPUT_MD"

############################################
# 1. Directory listings
############################################

echo "## Directory structure" >> "$OUTPUT_MD"
echo >> "$OUTPUT_MD"

for rel_dir in "${DIRS_TO_LIST[@]}"; do
    abs_dir="$PROJECT_ROOT/$rel_dir"
    if [ ! -d "$abs_dir" ]; then
        echo "Warning: directory not found: $abs_dir" >&2
        echo "### Directory: \`$rel_dir\` (NOT FOUND)" >> "$OUTPUT_MD"
        echo >> "$OUTPUT_MD"
        continue
    fi

    echo "Processing directory: $abs_dir"

    echo "### Directory: \`$rel_dir\`" >> "$OUTPUT_MD"
    echo >> "$OUTPUT_MD"
    echo '```text' >> "$OUTPUT_MD"
    # Shallow listing (depth 2 is usually enough for structure)
    (cd "$abs_dir" && find . -maxdepth 2 -type f | sort) >> "$OUTPUT_MD"
    echo '```' >> "$OUTPUT_MD"
    echo >> "$OUTPUT_MD"
done

############################################
# 2. File contents
############################################

echo "## File contents" >> "$OUTPUT_MD"
echo >> "$OUTPUT_MD"

for rel_path in "${FILES_TO_INLINE[@]}"; do
    abs_path="$PROJECT_ROOT/$rel_path"

    if [ ! -e "$abs_path" ]; then
        echo "Warning: file not found: $abs_path" >&2
        echo "### File: \`$rel_path\` (NOT FOUND)" >> "$OUTPUT_MD"
        echo >> "$OUTPUT_MD"
        continue
    fi

    echo "Inlining: $abs_path"

    ext="${rel_path##*.}"
    lang=$(detect_language "$rel_path")

    echo "### File: \`$rel_path\`" >> "$OUTPUT_MD"
    echo >> "$OUTPUT_MD"

    # Handle parquet specially (do not inline binary)
    if [ "$ext" = "parquet" ]; then
        echo "_Parquet file (binary) – not inlined. Path: \`$rel_path\`_" >> "$OUTPUT_MD"
        echo >> "$OUTPUT_MD"
        continue
    fi

    # CSV: truncate for readability
    if [ "$ext" = "csv" ]; then
        echo "_Showing first $CSV_HEAD_LINES lines for brevity._" >> "$OUTPUT_MD"
        echo >> "$OUTPUT_MD"
        echo '```'"$lang" >> "$OUTPUT_MD"
        head -n "$CSV_HEAD_LINES" "$abs_path" >> "$OUTPUT_MD"
        echo '```' >> "$OUTPUT_MD"
        echo >> "$OUTPUT_MD"
        continue
    fi

    # Default: dump full content (for JSON, MD, scripts, etc.)
    echo '```'"$lang" >> "$OUTPUT_MD"
    cat "$abs_path" >> "$OUTPUT_MD"
    echo '```' >> "$OUTPUT_MD"
    echo >> "$OUTPUT_MD"
done

echo "Done. Markdown written to: $OUTPUT_MD"
