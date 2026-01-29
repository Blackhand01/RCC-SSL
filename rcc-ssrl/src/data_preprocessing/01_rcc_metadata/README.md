# rcc_metadata — Enrichment di metadata.csv

## Input
- `metadata.csv` (base map WSI↔XML/ROI) — default: `$REPORT_DIR/metadata.csv`
- RAW root (`$RAW_DATA_DIR`): folder with WSI/XML/ROI.

## Output (in `$REPORT_DIR`)
- `rcc_metadata.csv` — **without path** e **without slide_id**, it include only:
  `subtype,patient_id,wsi_filename,annotation_xml,roi_files,num_rois,source_dir,
   wsi_size_bytes,vendor,width0,height0,level_count,mpp_x,mpp_y,objective_power,
   xml_roi_tumor,xml_roi_not_tumor,xml_roi_total`
- NB: `wsi_size_bytes` it is refered **only on WSI** ( XML are ingnored).

## Rapid execution
```bash
export RAW_DATA_DIR=/beegfs-scratch/.../data/raw
export REPORT_DIR=/home/mla_group_01/rcc-ssrl/reports/0_phase
# opzionale: export METADATA_CSV=$REPORT_DIR/metadata.csv
$HOME_ROOT/scripts/rcc_metadata/run.sh
```
