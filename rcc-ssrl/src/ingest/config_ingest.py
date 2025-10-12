from pathlib import Path

HOME_ROOT    = Path("/home/mla_group_01/rcc-ssrl")
SCRATCH_ROOT = Path("/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project")

DATA_ROOT    = SCRATCH_ROOT / "data"
META_ROOT    = SCRATCH_ROOT / "configs"
OUT_MANIFEST = DATA_ROOT / "processed" / "rcc_manifest.parquet"
OUT_DATACARD = DATA_ROOT / "processed" / "rcc_datacard.yaml"
LOGS_DIR     = SCRATCH_ROOT / "outputs" / "logs"

# I 4 mapping JSON devono stare qui (vedi step 2 sotto)
CC_MAP = META_ROOT / "ccRCC_mapping.json"
PR_MAP = META_ROOT / "pRCC_mapping.json"
CH_MAP = META_ROOT / "CHROMO_patient_mapping.json"
ON_MAP = META_ROOT / "ONCO_patient_mapping.json"

# directory da scansionare per file reali
SCAN_ROOTS = [ DATA_ROOT / "raw" ]

FILE_EXT    = {".svs", ".scn", ".tif", ".tiff"}
HASH_ALGO   = "none"
MAX_WORKERS = 8
DRYRUN      = False

WSI_EXT = {".svs", ".scn"}
ROI_EXT = {".tif", ".tiff", ".svs"}
XML_EXT = {".xml"}
