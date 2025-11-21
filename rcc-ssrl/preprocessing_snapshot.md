# RCC-SSRL Preprocessing Snapshot

_Generated on Tue Nov 18 12:41:01 PM UTC 2025 from project root: `/home/mla_group_01/rcc-ssrl`_

## Directory structure

### Directory: `slurm/00_gpu-smoke-done`

```text
./gpu_smoke_autolog.sbatch
./gpu_smoke.py
./gpu_smoke.sbatch
./README.md
./run_autopick.sh
./run.sh
./TODO.md
```

### Directory: `slurm/00_wsi-drive-analysis`

```text
./rclone_sync_body.sh
./sync_gdrive_rcc.sbatch
```

### Directory: `slurm/01_rcc_metadata`

```text
./rcc_metadata_enrich.py
./README.md
./requirements.txt
./run.sh
```

### Directory: `slurm/02_parquet`

```text
./02_parquet_body.sh
./02_parquet.sbatch
./parquet_build.py
./README.md
./requirements.txt
./run.sh
```

### Directory: `reports/00_wsi-drive-analysis`

```text
./ccRCC_mapping.json
./CHROMO_patient_mapping.json
./metadata.csv
./ONCO_patient_mapping.json
./pRCC_mapping.json
./rcc_dataset_stats.json
./wsi_drive_analysis.md
./wsi_inventory.csv
```

### Directory: `reports/01_rcc_metadata`

```text
./rcc_metadata.csv
```

### Directory: `reports/02_parquet`

```text
./slides.csv
./slides.parquet
```

## File contents

### File: `reports/00_wsi-drive-analysis/ccRCC_mapping.json`

```json
{
  "HP19.754.A5.ccRCC.scn": [
    "HP19.754.A5.ccRCC.xml"
  ],
  "HP19.754.A8.ccRCC.scn": [
    "HP19.754.A8.ccRCC.xml"
  ],
  "HP19.10064.A13.ccRCC.scn": [
    "HP19.10064.A13.ccRCC.xml"
  ],
  "HP19.4372.A6.ccRCC.scn": [
    "HP19.4372.A6.ccRCC.xml"
  ],
  "HP19.8394.A3.ccRCC.scn": [
    "HP19.8394.A3.ccRCC.xml"
  ],
  "HP19.5524.A4.ccRCC.scn": [
    "HP19.5524.A4.ccRCC.xml"
  ],
  "HP19.7949.2A1.ccRCC.scn": [
    "HP19.7949.2A1.ccRCC.xml"
  ],
  "HP19.7840.A6.ccRCC.scn": [
    "HP19.7840.A6.ccRCC.xml"
  ],
  "HP19.754.A3.ccRCC.scn": [
    "HP19.754.A3.ccRCC.xml"
  ],
  "HP19.9347.2A.ccRCC.scn": [
    "HP19.9347.2A.ccRCC.xml"
  ],
  "HP19.5524.A2.ccRCC.scn": [
    "HP19.5524.A2.ccRCC.xml"
  ],
  "HP19.4075.A1.ccRCC.scn": [
    "HP19.4075.A1.ccRCC.xml"
  ],
  "HP19.754.A4.ccRCC.scn": [
    "HP19.754.A4.ccRCC.xml"
  ],
  "HP19.7840.A1.ccRCC.scn": [
    "HP19.7840.A1.ccRCC.xml"
  ],
  "HP19.3695.2A3.ccRCC.scn": [
    "HP19.3695.2A3.ccRCC.xml"
  ],
  "HP19.754.A9.ccRCC.scn": [
    "HP19.754.A9.ccRCC.xml"
  ],
  "HP19.4372.A7.ccRCC.scn": [
    "HP19.4372.A7.ccRCC.xml"
  ],
  "HP19.9421.A2.ccRCC.scn": [
    "HP19.9421.A2.ccRCC.xml"
  ],
  "HP19.7840.A7.ccRCC.scn": [
    "HP19.7840.A7.ccRCC.xml"
  ],
  "HP19.999.A.ccRCC.scn": [
    "HP19.999.A.ccRCC.xml"
  ],
  "HP19.5524.A3.ccRCC.scn": [
    "HP19.5524.A3.ccRCC.xml"
  ],
  "HP19.7715.2A4.ccRCC.scn": [
    "HP19.7715.2A4.ccRCC.xml"
  ],
  "HP19.10064.A14.ccRCC.scn": [
    "HP19.10064.A14.ccRCC.xml"
  ],
  "HP19.2434.A6.ccRCC.scn": [
    "HP19.2434.A6.ccRCC.xml"
  ],
  "HP19.7840.A9.ccRCC-1.scn": [
    "HP19.7840.A9.ccRCC-1.xml"
  ],
  "HP19.7421.A6.ccRCC.scn": [
    "HP19.7421.A6.ccRCC.xml"
  ],
  "HP19.10064.A1.ccRCC.scn": [
    "HP19.10064.A1.ccRCC.xml"
  ],
  "HP19.754.A7.ccRCC.scn": [
    "HP19.754.A7.ccRCC.xml"
  ],
  "HP19.7840.A2.ccRCC.scn": [
    "HP19.7840.A2.ccRCC.xml"
  ],
  "HP19.7840.A7.ccRCC-2.scn": [
    "HP19.7840.A7.ccRCC-2.xml"
  ],
  "HP19.4075.A5.ccRCC.scn": [
    "HP19.4075.A5.ccRCC.xml"
  ],
  "HP19.8394.A1.ccRCC.scn": [
    "HP19.8394.A1.ccRCC.xml"
  ],
  "HP19.2434.A3.ccRCC.scn": [
    "HP19.2434.A3.ccRCC.xml"
  ],
  "HP19.4372.A4.ccRCC.scn": [
    "HP19.4372.A4.ccRCC.xml"
  ],
  "HP19.7840.A7.ccRCC-1.scn": [
    "HP19.7840.A7.ccRCC-1.xml"
  ],
  "HP19.7840.A4.ccRCC.scn": [
    "HP19.7840.A4.ccRCC.xml"
  ],
  "HP19.754.A1.ccRCC.scn": [
    "HP19.754.A1.ccRCC.xml"
  ],
  "HP19.10064.A7.ccRCC.scn": [
    "HP19.10064.A7.ccRCC.xml"
  ],
  "HP19.2434.A5.ccRCC.scn": [
    "HP19.2434.A5.ccRCC.xml"
  ],
  "HP19.7840.A9.ccRCC.scn": [
    "HP19.7840.A9.ccRCC.xml"
  ],
  "HP19.4075.A3.ccRCC.scn": [
    "HP19.4075.A3.ccRCC.xml"
  ],
  "HP19.4075.A9.ccRCC.scn": [
    "HP19.4075.A9.ccRCC.xml"
  ],
  "HP19.3695.2A1.ccRCC.scn": [
    "HP19.3695.2A1.ccRCC.xml"
  ],
  "H19.754.IHC.ccRCC.scn": [
    "H19.754.IHC.ccRCC.xml"
  ],
  "HP19.754.A6.ccRCC.scn": [
    "HP19.754.A6.ccRCC.xml"
  ],
  "HP19.4372.A8.ccRCC.scn": [
    "HP19.4372.A8.ccRCC.xml"
  ],
  "HP19.7840.A1.ccRCC-1.scn": [
    "HP19.7840.A1.ccRCC-1.xml"
  ],
  "HP19.10064.A10.ccRCC.scn": [
    "HP19.10064.A10.ccRCC.xml"
  ],
  "HP19.2434.A2.ccRCC.scn": [
    "HP19.2434.A2.ccRCC.xml"
  ],
  "HP19.7840.A5.ccRCC.scn": [
    "HP19.7840.A5.ccRCC.xml"
  ],
  "HP19.7840.A12.ccRCC.scn": [
    "HP19.7840.A12.ccRCC.xml"
  ],
  "HP19.10064.A1-1.ccRCC.scn": [
    "HP19.10064.A1-1.ccRCC.xml"
  ],
  "HP19.7949.2A2.ccRCC.scn": [
    "HP19.7949.2A2.ccRCC.xml"
  ],
  "HP19.10064.A6.ccRCC.scn": [
    "HP19.10064.A6.ccRCC.xml"
  ],
  "HP19.5254.A.ccRCC.scn": [
    "HP19.5254.A.ccRCC.xml"
  ],
  "HP19.4372.A3.ccRCC.scn": [
    "HP19.4372.A3.ccRCC.xml"
  ],
  "HP19.10064.A16.ccRCC.scn": [
    "HP19.10064.A16.ccRCC.xml"
  ],
  "HP19.2434.A4.ccRCC.scn": [
    "HP19.2434.A4.ccRCC.xml"
  ],
  "HP19.7864.1A.ccRCC.scn": [
    "HP19.7864.1A.ccRCC.xml"
  ],
  "HP19.4075.A2.ccRCC.scn": [
    "HP19.4075.A2.ccRCC.xml"
  ],
  "HP15.12550.A7.ccRCC.scn": [
    "HP15.12550.A7.ccRCC.xml"
  ],
  "HP12.8355.A8.ccRCC.scn": [
    "HP12.8355.A8.ccRCC.xml"
  ],
  "HP12.9282.A10.ccRCC.scn": [
    "HP12.9282.A10.ccRCC.xml"
  ],
  "HP12.9282.A5.ccRCC.scn": [
    "HP12.9282.A5.ccRCC.xml"
  ],
  "HP10.2986.A4.ccRCC.scn": [],
  "HP14.1749.A6.ccRCC.scn": [
    "HP14.1749.A6.ccRCC.xml"
  ],
  "HP13.7465.A5.ccRCC.scn": [
    "HP13.7465.A5.ccRCC.xml"
  ],
  "HP12.6691.T1.ccRCC.scn": [
    "HP12.6691.T1.ccRCC.xml"
  ],
  "HP14.7813.A8.ccRCC.scn": [
    "HP14.7813.A8.ccRCC.xml"
  ],
  "HP14.11034.A.ccRCC.scn": [
    "HP14.11034.A.ccRCC.xml"
  ],
  "HP11.12318.A1.ccRCC.scn": [
    "HP11.12318.A1.ccRCC.xml"
  ],
  "HP10.2695.A4.ccRCC.scn": [
    "HP10.2695.A4.ccRCC.xml"
  ],
  "HP12.7225.A.ccRCC.scn": [
    "HP12.7225.A.ccRCC.xml"
  ],
  "HP16.819.A2.ccRCC.scn": [
    "HP16.819.A2.ccRCC.xml"
  ],
  "HP12.390.A5.ccRCC.scn": [
    "HP12.390.A5.ccRCC.xml"
  ],
  "HP10.2986_A4_ccRCC.scn": [
    "HP10.2986_A4_ccRCC.xml"
  ],
  "HP15.12550.A1.ccRCC.scn": [
    "HP15.12550.A1.ccRCC.xml"
  ],
  "HP13.1799.B2.ccRCC.scn": [
    "HP13.1799.B2.ccRCC.xml"
  ],
  "HP12.6073.A5-1.ccRCC.scn": [
    "HP12.6073.A5-1.ccRCC.xml"
  ],
  "HP15.12550.A6.ccRCC.scn": [
    "HP15.12550.A6.ccRCC.xml"
  ],
  "HP02.10180.1A2.ccRCC.scn": [
    "HP02.10180.1A2.ccRCC.xml"
  ],
  "HP15.11259.A6.ccRCC.scn": [
    "HP15.11259.A6.ccRCC.xml"
  ],
  "HP12.3187.A5.ccRCC.scn": [
    "HP12.3187.A5.ccRCC.xml"
  ],
  "HP12.9282.A4.ccRCC.scn": [
    "HP12.9282.A4.ccRCC.xml"
  ],
  "HP14.5347.A6.ccRCC.scn": [
    "HP14.5347.A6.ccRCC.xml"
  ],
  "HP15.2902.A4.ccRCC.scn": [
    "HP15.2902.A4.ccRCC.xml"
  ],
  "HP16.6209.2B4.ccRCC.scn": [
    "HP16.6209.2B4.ccRCC.xml"
  ],
  "HP14.1993.A1.ccRCC.scn": [
    "HP14.1993.A1.ccRCC.xml"
  ],
  "HP14.1749.A1.ccRCC.scn": [
    "HP14.1749.A1.ccRCC.xml"
  ],
  "HP12.6073.1A6.ccRCC.scn": [
    "HP12.6073.1A6.ccRCC.xml"
  ],
  "HP12.390.A10.ccRCC.scn": [
    "HP12.390.A10.ccRCC.xml"
  ],
  "HP16.6209.2B1.ccRCC.scn": [
    "HP16.6209.2B1.ccRCC.xml"
  ],
  "HP15.2902.A1.ccRCC.scn": [
    "HP15.2902.A1.ccRCC.xml"
  ],
  "HP12.4271.A8.ccRCC.scn": [
    "HP12.4271.A8.ccRCC.xml"
  ],
  "HP15.1480.A3.ccRCC.scn": [
    "HP15.1480.A3.ccRCC.xml"
  ],
  "HP12.9282.A7.ccRCC.scn": [
    "HP12.9282.A7.ccRCC.xml"
  ],
  "HP14.9097.2F.ccRCC.scn": [
    "HP14.9097.2F.ccRCC.xml"
  ],
  "HP12.8793.B1.ccRCC.scn": [
    "HP12.8793.B1.ccRCC.xml"
  ],
  "HP12.13358.1A3.ccRCC.scn": [
    "HP12.13358.1A3.ccRCC.xml"
  ],
  "HP10.5813.A2.ccRCC.scn": [
    "HP10.5813.A2.ccRCC.xml"
  ],
  "HP13.6992.A3.ccRCC.scn": [
    "HP13.6992.A3.ccRCC.xml"
  ],
  "HP12.13358.1A8.ccRCC.scn": [
    "HP12.13358.1A8.ccRCC.xml"
  ],
  "HP11.12277.A4.ccRCC.scn": [
    "HP11.12277.A4.ccRCC.xml"
  ],
  "HP12.13358.1A5.ccRCC.scn": [
    "HP12.13358.1A5.ccRCC.xml"
  ],
  "HP12.13588.A5.ccRCC.scn": [
    "HP12.13588.A5.ccRCC.xml"
  ],
  "HP15.11259.A3.ccRCC.scn": [
    "HP15.11259.A3.ccRCC.xml"
  ],
  "HP14.9685.A7.ccRCC.scn": [
    "HP14.9685.A7.ccRCC.xml"
  ],
  "HP12.4271.A11.ccRCC.scn": [
    "HP12.4271.A11.ccRCC.xml"
  ],
  "HP12.6073.A5.ccRCC.scn": [
    "HP12.6073.A5.ccRCC.xml"
  ],
  "HP14.1749.A8.ccRCC.scn": [
    "HP14.1749.A8.ccRCC.xml"
  ],
  "HP14.69.2A2.ccRCC.scn": [
    "HP14.69.2A2.ccRCC.xml"
  ],
  "HP16.6211.E.ccRCC.scn": [
    "HP16.6211.E.ccRCC.xml"
  ],
  "HP12.5998.B.ccRCC.scn": [
    "HP12.5998.B.ccRCC.xml"
  ],
  "HP15.11259.A4.ccRCC.scn": [
    "HP15.11259.A4.ccRCC.xml"
  ],
  "HP12.13588.A2.ccRCC.scn": [
    "HP12.13588.A2.ccRCC.xml"
  ],
  "HP12.9282.A6.ccRCC.scn": [
    "HP12.9282.A6.ccRCC.xml"
  ],
  "HP14.13101.E.ccRCC.scn": [
    "HP14.13101.E.ccRCC.xml"
  ],
  "HP16.6211.C.ccRCC.scn": [
    "HP16.6211.C.ccRCC.xml"
  ],
  "HP16.819.A1.ccRCC.scn": [
    "HP16.819.A1.ccRCC.xml"
  ],
  "HP12.6073_A5_ccRCC.scn": [
    "HP12.6073_A5_ccRCC.xml"
  ],
  "HP14.5590.A3.ccRCC.scn": [
    "HP14.5590.A3.ccRCC.xml"
  ],
  "HP12.5998.I.ccRCC.scn": [
    "HP12.5998.I.ccRCC.xml"
  ],
  "HP14.69.2A9.ccRCC.scn": [
    "HP14.69.2A9.ccRCC.xml"
  ],
  "HP14.13101.C.ccRCC.scn": [
    "HP14.13101.C.ccRCC.xml"
  ],
  "HP12.13358.1A4.ccRCC.scn": [
    "HP12.13358.1A4.ccRCC.xml"
  ]
}```

### File: `reports/00_wsi-drive-analysis/CHROMO_patient_mapping.json`

```json
{
  "HP20.2506": {
    "roi_files": [
      "13.tif",
      "12.svs"
    ],
    "wsi_files": [
      "HP20.2506_1339.svs",
      "HP20.2506_1338 .svs",
      "HP20.2506_1342.svs"
    ]
  },
  "HP19012316": {
    "roi_files": [
      "11.svs",
      "10.svs",
      "9.svs"
    ],
    "wsi_files": [
      "HP19012316-0-C-HE_2195.svs"
    ]
  },
  "HP18014084": {
    "roi_files": [
      "8.svs"
    ],
    "wsi_files": [
      "HP18014084-0-A1-HE3,4_2198.svs",
      "HP18014084-0-A3-HE_2199.svs"
    ]
  },
  "HP17008718": {
    "roi_files": [
      "2.svs",
      "1.svs"
    ],
    "wsi_files": [
      "HP17008718-0-C2-HE_2180.svs",
      "HP17008718-0-BM-HE_2181.svs"
    ]
  },
  "HP20002300": {
    "roi_files": [
      "3.svs",
      "4.svs",
      "5.svs",
      "7.svs",
      "6.svs"
    ],
    "wsi_files": [
      "HP20002300-0-A10-HE_2210.svs",
      "HP20002300-0-A8-HE_2209.svs",
      "HP20002300-0-A4-HE_2208.svs"
    ]
  }
}```

### File: `reports/00_wsi-drive-analysis/ONCO_patient_mapping.json`

```json
{
  "HP18005453": {
    "roi_files": [
      "13.tif",
      "12.tif",
      "10.tif",
      "11.tif",
      "9.tif",
      "8.tif"
    ],
    "wsi_files": [
      "HP18005453-0-A3-HE_1970.svs",
      "HP18005453-0-A1-HE_1968.svs",
      "HP18005453-0-A2-HE_1969.svs"
    ]
  },
  "HP20.5602": {
    "roi_files": [
      "15.tif",
      "14.tif",
      "16.tif",
      "17.tif",
      "19.tif",
      "18.tif",
      "20.tif",
      "21.tif"
    ],
    "wsi_files": [
      "HP20.5602_1351.svs",
      "HP20.5602_1355.svs",
      "HP20.5602_1349.svs"
    ]
  },
  "HP20002450": {
    "roi_files": [
      "2.svs",
      "3.svs",
      "1.svs"
    ],
    "wsi_files": [
      "HP20002450-0-B-HE3,4_2174.svs",
      "HP20002450-0-D-3,4_2175.svs"
    ]
  },
  "HP20001530": {
    "roi_files": [
      "4.svs",
      "5.svs",
      "7.svs",
      "6.svs"
    ],
    "wsi_files": [
      "HP20001530-0-F-HE_2213.svs",
      "HP20001530-0-A-HE_2211.svs",
      "HP20001530-0-B-HE_2212.svs"
    ]
  },
  "HP18009209": {
    "roi_files": [
      "25.tif",
      "24.tif",
      "23.tif",
      "22.tif"
    ],
    "wsi_files": [
      "HP18009209-0-A3-HE_2176.svs",
      "HP18009209-0-A9-HE_2177.svs"
    ]
  }
}```

### File: `reports/00_wsi-drive-analysis/pRCC_mapping.json`

```json
{
  "HP17.11714.A6.pRCC.scn": [
    "HP17.11714.A6.pRCC.xml"
  ],
  "HP18.5818.A5.pRCC.scn": [
    "HP18.5818.A5.pRCC.xml"
  ],
  "HP19.1773.A10.pRCC.scn": [
    "HP19.1773.A10.pRCC.xml"
  ],
  "HP17.7980.B2.pRCC.scn": [
    "HP17.7980.B2.pRCC.xml"
  ],
  "HP18.11474.A5.pRCC.scn": [
    "HP18.11474.A5.pRCC.xml"
  ],
  "HP19.1277.A4.pRCC.scn": [
    "HP19.1277.A4.pRCC.xml"
  ],
  "HP18.5818.A3.pRCC.scn": [
    "HP18.5818.A3.pRCC.xml"
  ],
  "HP18.11474.A9.pRCC.scn": [
    "HP18.11474.A9.pRCC.xml"
  ],
  "HP18.11474.A8.pRCC.scn": [
    "HP18.11474.A8.pRCC.xml"
  ],
  "HP18.13618.A1.pRCC.scn": [
    "HP18.13618.A1.pRCC.xml"
  ],
  "HP17.7980.B.pRCC.scn": [
    "HP17.7980.B.pRCC.xml"
  ],
  "HP19.1277.A3.pRCC.scn": [
    "HP19.1277.A3.pRCC.xml"
  ],
  "HP19.1277.A2.pRCC.scn": [
    "HP19.1277.A2.pRCC.xml"
  ],
  "HP18.11474.A11.pRCC.scn": [
    "HP18.11474.A11.pRCC.xml"
  ],
  "HP18.11474.A10.pRCC.scn": [
    "HP18.11474.A10.pRCC.xml"
  ],
  "HP18.11474.A7.pRCC.scn": [
    "HP18.11474.A7.pRCC.xml"
  ],
  "HP18.11474.A6.pRCC.scn": [
    "HP18.11474.A6.pRCC.xml"
  ],
  "HP17.11714.A6-1.pRCC.scn": [
    "HP17.11714.A6-1.pRCC.xml"
  ],
  "HP18.13618.A5.pRCC.scn": [
    "HP18.13618.A5.pRCC.xml"
  ],
  "HP18.13618.A4.pRCC.scn": [
    "HP18.13618.A4.pRCC.xml"
  ],
  "HP18.11474.A14.pRCC.scn": [
    "HP18.11474.A14.pRCC.xml"
  ],
  "HP18.11474.A13.pRCC.scn": [
    "HP18.11474.A13.pRCC.xml"
  ],
  "HP17.11714.A8.pRCC.scn": [
    "HP17.11714.A8.pRCC.xml"
  ],
  "HP17.7980.A2.pRCC.scn": [
    "HP17.7980.A2.pRCC.xml"
  ],
  "HP19.1277.A1.pRCC.scn": [
    "HP19.1277.A1.pRCC.xml"
  ],
  "HP18.13618.A2.pRCC.scn": [
    "HP18.13618.A2.pRCC.xml"
  ],
  "HP12.6710.A5.pRCC.scn": [
    "HP12.6710.A5.pRCC.xml"
  ],
  "HP14.10122.B2.pRCC.scn": [
    "HP14.10122.B2.pRCC.xml"
  ],
  "HP12.7726.A.pRCC.scn": [
    "HP12.7726.A.pRCC.xml"
  ],
  "HP14.5971.A5.pRCC.scn": [
    "HP14.5971.A5.pRCC.xml"
  ],
  "HP10.9650.A7.pRCC.scn": [
    "HP10.9650.A7.pRCC.xml"
  ],
  "HP14.2377.pRCC.scn": [
    "HP14.2377.pRCC.xml"
  ],
  "HP12.5904.A10.pRCC.scn": [
    "HP12.5904.A10.pRCC.xml"
  ],
  "HP12.7601.A9.pRCC.scn": [
    "HP12.7601.A9.pRCC.xml"
  ],
  "HP09.5392.A2.pRCC.scn": [
    "HP09.5392.A2.pRCC.xml"
  ],
  "HP13.3201.A5.pRCC.scn": [
    "HP13.3201.A5.pRCC.xml"
  ],
  "HP13.3201.D1.pRCC.scn": [
    "HP13.3201.D1.pRCC.xml"
  ],
  "HP14.5971.A3.pRCC.scn": [
    "HP14.5971.A3.pRCC.xml"
  ],
  "HP12.7726.F.pRCC.scn": [
    "HP12.7726.F.pRCC.xml"
  ],
  "HP13.3311.1B.pRCC.scn": [
    "HP13.3311.1B.pRCC.xml"
  ],
  "HP14.8231.A6.pRCC.scn": [
    "HP14.8231.A6.pRCC.xml"
  ],
  "HP11.6090.E.pRCC.scn": [
    "HP11.6090.E.pRCC.xml"
  ],
  "HP15.9102.H.pRCC.scn": [
    "HP15.9102.H.pRCC.xml"
  ],
  "HP12.7726.C.pRCC.scn": [
    "HP12.7726.C.pRCC.xml"
  ],
  "HP14.10122.B1.pRCC.scn": [
    "HP14.10122.B1.pRCC.xml"
  ],
  "HP10.9650.A8.pRCC.scn": [
    "HP10.9650.A8.pRCC.xml"
  ],
  "HP12.7726.E.pRCC.scn": [
    "HP12.7726.E.pRCC.xml"
  ],
  "HP14.4279.R1.pRCC.scn": [
    "HP14.4279.R1.pRCC.xml"
  ]
}```

### File: `reports/00_wsi-drive-analysis/metadata.csv`

_Showing first 80 lines for brevity._

```csv
subtype,patient_id,wsi_filename,annotation_xml,num_annotations,roi_files,num_rois,source_dir
ccRCC,HP19.754,HP19.754.A5.ccRCC.scn,HP19.754.A5.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.754,HP19.754.A8.ccRCC.scn,HP19.754.A8.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.10064,HP19.10064.A13.ccRCC.scn,HP19.10064.A13.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.4372,HP19.4372.A6.ccRCC.scn,HP19.4372.A6.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.8394,HP19.8394.A3.ccRCC.scn,HP19.8394.A3.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.5524,HP19.5524.A4.ccRCC.scn,HP19.5524.A4.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.7949,HP19.7949.2A1.ccRCC.scn,HP19.7949.2A1.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.7840,HP19.7840.A6.ccRCC.scn,HP19.7840.A6.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.754,HP19.754.A3.ccRCC.scn,HP19.754.A3.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.9347,HP19.9347.2A.ccRCC.scn,HP19.9347.2A.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.5524,HP19.5524.A2.ccRCC.scn,HP19.5524.A2.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.4075,HP19.4075.A1.ccRCC.scn,HP19.4075.A1.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.754,HP19.754.A4.ccRCC.scn,HP19.754.A4.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.7840,HP19.7840.A1.ccRCC.scn,HP19.7840.A1.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.3695,HP19.3695.2A3.ccRCC.scn,HP19.3695.2A3.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.754,HP19.754.A9.ccRCC.scn,HP19.754.A9.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.4372,HP19.4372.A7.ccRCC.scn,HP19.4372.A7.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.9421,HP19.9421.A2.ccRCC.scn,HP19.9421.A2.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.7840,HP19.7840.A7.ccRCC.scn,HP19.7840.A7.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.999,HP19.999.A.ccRCC.scn,HP19.999.A.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.5524,HP19.5524.A3.ccRCC.scn,HP19.5524.A3.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.7715,HP19.7715.2A4.ccRCC.scn,HP19.7715.2A4.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.10064,HP19.10064.A14.ccRCC.scn,HP19.10064.A14.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.2434,HP19.2434.A6.ccRCC.scn,HP19.2434.A6.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.7840,HP19.7840.A9.ccRCC-1.scn,HP19.7840.A9.ccRCC-1.xml,1,,0,ccRCC
ccRCC,HP19.7421,HP19.7421.A6.ccRCC.scn,HP19.7421.A6.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.10064,HP19.10064.A1.ccRCC.scn,HP19.10064.A1.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.754,HP19.754.A7.ccRCC.scn,HP19.754.A7.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.7840,HP19.7840.A2.ccRCC.scn,HP19.7840.A2.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.7840,HP19.7840.A7.ccRCC-2.scn,HP19.7840.A7.ccRCC-2.xml,1,,0,ccRCC
ccRCC,HP19.4075,HP19.4075.A5.ccRCC.scn,HP19.4075.A5.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.8394,HP19.8394.A1.ccRCC.scn,HP19.8394.A1.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.2434,HP19.2434.A3.ccRCC.scn,HP19.2434.A3.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.4372,HP19.4372.A4.ccRCC.scn,HP19.4372.A4.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.7840,HP19.7840.A7.ccRCC-1.scn,HP19.7840.A7.ccRCC-1.xml,1,,0,ccRCC
ccRCC,HP19.7840,HP19.7840.A4.ccRCC.scn,HP19.7840.A4.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.754,HP19.754.A1.ccRCC.scn,HP19.754.A1.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.10064,HP19.10064.A7.ccRCC.scn,HP19.10064.A7.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.2434,HP19.2434.A5.ccRCC.scn,HP19.2434.A5.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.7840,HP19.7840.A9.ccRCC.scn,HP19.7840.A9.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.4075,HP19.4075.A3.ccRCC.scn,HP19.4075.A3.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.4075,HP19.4075.A9.ccRCC.scn,HP19.4075.A9.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.3695,HP19.3695.2A1.ccRCC.scn,HP19.3695.2A1.ccRCC.xml,1,,0,ccRCC
ccRCC,H19.754,H19.754.IHC.ccRCC.scn,H19.754.IHC.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.754,HP19.754.A6.ccRCC.scn,HP19.754.A6.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.4372,HP19.4372.A8.ccRCC.scn,HP19.4372.A8.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.7840,HP19.7840.A1.ccRCC-1.scn,HP19.7840.A1.ccRCC-1.xml,1,,0,ccRCC
ccRCC,HP19.10064,HP19.10064.A10.ccRCC.scn,HP19.10064.A10.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.2434,HP19.2434.A2.ccRCC.scn,HP19.2434.A2.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.7840,HP19.7840.A5.ccRCC.scn,HP19.7840.A5.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.7840,HP19.7840.A12.ccRCC.scn,HP19.7840.A12.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.10064,HP19.10064.A1-1.ccRCC.scn,HP19.10064.A1-1.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.7949,HP19.7949.2A2.ccRCC.scn,HP19.7949.2A2.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.10064,HP19.10064.A6.ccRCC.scn,HP19.10064.A6.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.5254,HP19.5254.A.ccRCC.scn,HP19.5254.A.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.4372,HP19.4372.A3.ccRCC.scn,HP19.4372.A3.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.10064,HP19.10064.A16.ccRCC.scn,HP19.10064.A16.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.2434,HP19.2434.A4.ccRCC.scn,HP19.2434.A4.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.7864,HP19.7864.1A.ccRCC.scn,HP19.7864.1A.ccRCC.xml,1,,0,ccRCC
ccRCC,HP19.4075,HP19.4075.A2.ccRCC.scn,HP19.4075.A2.ccRCC.xml,1,,0,ccRCC
ccRCC,HP15.12550,HP15.12550.A7.ccRCC.scn,HP15.12550.A7.ccRCC.xml,1,,0,pre/ccRCC
ccRCC,HP12.8355,HP12.8355.A8.ccRCC.scn,HP12.8355.A8.ccRCC.xml,1,,0,pre/ccRCC
ccRCC,HP12.9282,HP12.9282.A10.ccRCC.scn,HP12.9282.A10.ccRCC.xml,1,,0,pre/ccRCC
ccRCC,HP12.9282,HP12.9282.A5.ccRCC.scn,HP12.9282.A5.ccRCC.xml,1,,0,pre/ccRCC
ccRCC,HP14.1749,HP14.1749.A6.ccRCC.scn,HP14.1749.A6.ccRCC.xml,1,,0,pre/ccRCC
ccRCC,HP13.7465,HP13.7465.A5.ccRCC.scn,HP13.7465.A5.ccRCC.xml,1,,0,pre/ccRCC
ccRCC,HP12.6691,HP12.6691.T1.ccRCC.scn,HP12.6691.T1.ccRCC.xml,1,,0,pre/ccRCC
ccRCC,HP14.7813,HP14.7813.A8.ccRCC.scn,HP14.7813.A8.ccRCC.xml,1,,0,pre/ccRCC
ccRCC,HP14.11034,HP14.11034.A.ccRCC.scn,HP14.11034.A.ccRCC.xml,1,,0,pre/ccRCC
ccRCC,HP11.12318,HP11.12318.A1.ccRCC.scn,HP11.12318.A1.ccRCC.xml,1,,0,pre/ccRCC
ccRCC,HP10.2695,HP10.2695.A4.ccRCC.scn,HP10.2695.A4.ccRCC.xml,1,,0,pre/ccRCC
ccRCC,HP12.7225,HP12.7225.A.ccRCC.scn,HP12.7225.A.ccRCC.xml,1,,0,pre/ccRCC
ccRCC,HP16.819,HP16.819.A2.ccRCC.scn,HP16.819.A2.ccRCC.xml,1,,0,pre/ccRCC
ccRCC,HP12.390,HP12.390.A5.ccRCC.scn,HP12.390.A5.ccRCC.xml,1,,0,pre/ccRCC
ccRCC,HP10.2986,HP10.2986_A4_ccRCC.scn,HP10.2986_A4_ccRCC.xml,1,,0,pre/ccRCC
ccRCC,HP15.12550,HP15.12550.A1.ccRCC.scn,HP15.12550.A1.ccRCC.xml,1,,0,pre/ccRCC
ccRCC,HP13.1799,HP13.1799.B2.ccRCC.scn,HP13.1799.B2.ccRCC.xml,1,,0,pre/ccRCC
ccRCC,HP12.6073,HP12.6073.A5-1.ccRCC.scn,HP12.6073.A5-1.ccRCC.xml,1,,0,pre/ccRCC
ccRCC,HP15.12550,HP15.12550.A6.ccRCC.scn,HP15.12550.A6.ccRCC.xml,1,,0,pre/ccRCC
```

### File: `reports/00_wsi-drive-analysis/rcc_dataset_stats.json`

```json
{
  "ccRCC": {
    "n_wsis": 125,
    "n_xml_annotations": 124,
    "n_patients": 10
  },
  "pRCC": {
    "n_wsis": 48,
    "n_xml_annotations": 48,
    "n_patients": 10
  },
  "CHROMO": {
    "n_patients": 5,
    "n_wsis": 11,
    "n_roi_files": 13
  },
  "ONCO": {
    "n_patients": 5,
    "n_wsis": 13,
    "n_roi_files": 25
  },
  "ALL": {
    "total_wsis": 197,
    "total_xmls": 172,
    "total_roi_files": 38,
    "total_patients": 30
  }
}```

### File: `reports/00_wsi-drive-analysis/wsi_drive_analysis.md`

```markdown
# # WSI Inventory Summary

- Generated: 2025-10-13T10:48:31.775597
- RAW root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw
- Total slides: 240
- With errors: 0
## By extension
- .scn: 173
- .svs: 47
- .tif: 20
## By class hint
- CHROMO: 25
- ONCO: 41
- ccRCC: 126
- pRCC: 48


## Patients (240)
- Annotations_chromo/1
- Annotations_chromo/10
- Annotations_chromo/11
- Annotations_chromo/12
- Annotations_chromo/13
- Annotations_chromo/2
- Annotations_chromo/3
- Annotations_chromo/4
- Annotations_chromo/5
- Annotations_chromo/6
- Annotations_chromo/7
- Annotations_chromo/8
- Annotations_chromo/9

- Annotations_onco/1
- Annotations_onco/10
- Annotations_onco/11
- Annotations_onco/12
- Annotations_onco/13
- Annotations_onco/14
- Annotations_onco/15
- Annotations_onco/16
- Annotations_onco/17
- Annotations_onco/18
- Annotations_onco/19
- Annotations_onco/2
- Annotations_onco/20
- Annotations_onco/21
- Annotations_onco/22
- Annotations_onco/23
- Annotations_onco/24
- Annotations_onco/25
- Annotations_onco/3
- Annotations_onco/4
- Annotations_onco/5
- Annotations_onco/6
- Annotations_onco/7
- Annotations_onco/8
- Annotations_onco/9

- CHROMO/HP17008718-0-BM-HE_2181
- CHROMO/HP17008718-0-C2-HE_2180
- CHROMO/HP18014084-0-A1-HE3,4_2198
- CHROMO/HP18014084-0-A3-HE_2199
- CHROMO/HP19012316-0-C-HE_2195
- CHROMO/HP20.2506_1338 
- CHROMO/HP20.2506_1339
- CHROMO/HP20.2506_1342
- CHROMO/HP20002300-0-A10-HE_2210
- CHROMO/HP20002300-0-A4-HE_2208
- CHROMO/HP20002300-0-A8-HE_2209
- CHROMO/HP70605

- ONCOCYTOMA/HP18005453-0-A1-HE_1968
- ONCOCYTOMA/HP18005453-0-A2-HE_1969
- ONCOCYTOMA/HP18005453-0-A3-HE_1970
- ONCOCYTOMA/HP18009209-0-A3-HE_2176
- ONCOCYTOMA/HP18009209-0-A9-HE_2177
- ONCOCYTOMA/HP19008963-2-A-HE_1971
- ONCOCYTOMA/HP19008963-2-B-HE_1972
- ONCOCYTOMA/HP20.5602_1349
- ONCOCYTOMA/HP20.5602_1351
- ONCOCYTOMA/HP20.5602_1355
- ONCOCYTOMA/HP20001530-0-A-HE_2211
- ONCOCYTOMA/HP20001530-0-B-HE_2212
- ONCOCYTOMA/HP20001530-0-F-HE_2213
- ONCOCYTOMA/HP20002450-0-B-HE3,4_2174
- ONCOCYTOMA/HP20002450-0-D-3,4_2175
- ONCOCYTOMA/HP51171

- ccRCC/H19.754.IHC.ccRCC
- ccRCC/HP19.10064.A1-1.ccRCC
- ccRCC/HP19.10064.A1.ccRCC
- ccRCC/HP19.10064.A10.ccRCC
- ccRCC/HP19.10064.A13.ccRCC
- ccRCC/HP19.10064.A14.ccRCC
- ccRCC/HP19.10064.A16.ccRCC
- ccRCC/HP19.10064.A6.ccRCC
- ccRCC/HP19.10064.A7.ccRCC
- ccRCC/HP19.2434.A2.ccRCC
- ccRCC/HP19.2434.A3.ccRCC
- ccRCC/HP19.2434.A4.ccRCC
- ccRCC/HP19.2434.A5.ccRCC
- ccRCC/HP19.2434.A6.ccRCC
- ccRCC/HP19.3695.2A1.ccRCC
- ccRCC/HP19.3695.2A3.ccRCC
- ccRCC/HP19.4075.A1.ccRCC
- ccRCC/HP19.4075.A2.ccRCC
- ccRCC/HP19.4075.A3.ccRCC
- ccRCC/HP19.4075.A5.ccRCC
- ccRCC/HP19.4075.A9.ccRCC
- ccRCC/HP19.4372.A3.ccRCC
- ccRCC/HP19.4372.A4.ccRCC
- ccRCC/HP19.4372.A6.ccRCC
- ccRCC/HP19.4372.A7.ccRCC
- ccRCC/HP19.4372.A8.ccRCC
- ccRCC/HP19.5254.A.ccRCC
- ccRCC/HP19.5524.A2.ccRCC
- ccRCC/HP19.5524.A3.ccRCC
- ccRCC/HP19.5524.A4.ccRCC
- ccRCC/HP19.7421.A6.ccRCC
- ccRCC/HP19.754.A1.ccRCC
- ccRCC/HP19.754.A3.ccRCC
- ccRCC/HP19.754.A4.ccRCC
- ccRCC/HP19.754.A5.ccRCC
- ccRCC/HP19.754.A6.ccRCC
- ccRCC/HP19.754.A7.ccRCC
- ccRCC/HP19.754.A8.ccRCC
- ccRCC/HP19.754.A9.ccRCC
- ccRCC/HP19.7715.2A4.ccRCC
- ccRCC/HP19.7840.A1.ccRCC
- ccRCC/HP19.7840.A1.ccRCC-1
- ccRCC/HP19.7840.A12.ccRCC
- ccRCC/HP19.7840.A2.ccRCC
- ccRCC/HP19.7840.A4.ccRCC
- ccRCC/HP19.7840.A5.ccRCC
- ccRCC/HP19.7840.A6.ccRCC
- ccRCC/HP19.7840.A7.ccRCC
- ccRCC/HP19.7840.A7.ccRCC-1
- ccRCC/HP19.7840.A7.ccRCC-2
- ccRCC/HP19.7840.A9.ccRCC
- ccRCC/HP19.7840.A9.ccRCC-1
- ccRCC/HP19.7864.1A.ccRCC
- ccRCC/HP19.7949.2A1.ccRCC
- ccRCC/HP19.7949.2A2.ccRCC
- ccRCC/HP19.8394.A1.ccRCC
- ccRCC/HP19.8394.A3.ccRCC
- ccRCC/HP19.9347.2A.ccRCC
- ccRCC/HP19.9421.A2.ccRCC
- ccRCC/HP19.999.A.ccRCC

- pRCC/HP17.11714.A6-1.pRCC
- pRCC/HP17.11714.A6.pRCC
- pRCC/HP17.11714.A8.pRCC
- pRCC/HP17.7980.A2.pRCC
- pRCC/HP17.7980.B.pRCC
- pRCC/HP17.7980.B2.pRCC
- pRCC/HP18.11474.A10.pRCC
- pRCC/HP18.11474.A11.pRCC
- pRCC/HP18.11474.A13.pRCC
- pRCC/HP18.11474.A14.pRCC
- pRCC/HP18.11474.A5.pRCC
- pRCC/HP18.11474.A6.pRCC
- pRCC/HP18.11474.A7.pRCC
- pRCC/HP18.11474.A8.pRCC
- pRCC/HP18.11474.A9.pRCC
- pRCC/HP18.13618.A1.pRCC
- pRCC/HP18.13618.A2.pRCC
- pRCC/HP18.13618.A4.pRCC
- pRCC/HP18.13618.A5.pRCC
- pRCC/HP18.5818.A3.pRCC
- pRCC/HP18.5818.A5.pRCC
- pRCC/HP19.1277.A1.pRCC
- pRCC/HP19.1277.A2.pRCC
- pRCC/HP19.1277.A3.pRCC
- pRCC/HP19.1277.A4.pRCC
- pRCC/HP19.1773.A10.pRCC

- pre/ccRCC/HP02.10180.1A2.ccRCC
- pre/ccRCC/HP10.2695.A4.ccRCC
- pre/ccRCC/HP10.2986.A4.ccRCC
- pre/ccRCC/HP10.2986_A4_ccRCC
- pre/ccRCC/HP10.5813.A2.ccRCC
- pre/ccRCC/HP11.12277.A4.ccRCC
- pre/ccRCC/HP11.12318.A1.ccRCC
- pre/ccRCC/HP12.13358.1A3.ccRCC
- pre/ccRCC/HP12.13358.1A4.ccRCC
- pre/ccRCC/HP12.13358.1A5.ccRCC
- pre/ccRCC/HP12.13358.1A8.ccRCC
- pre/ccRCC/HP12.13588.A2.ccRCC
- pre/ccRCC/HP12.13588.A5.ccRCC
- pre/ccRCC/HP12.3187.A5.ccRCC
- pre/ccRCC/HP12.390.A10.ccRCC
- pre/ccRCC/HP12.390.A5.ccRCC
- pre/ccRCC/HP12.4271.A11.ccRCC
- pre/ccRCC/HP12.4271.A8.ccRCC
- pre/ccRCC/HP12.5998.B.ccRCC
- pre/ccRCC/HP12.5998.I.ccRCC
- pre/ccRCC/HP12.6073.1A6.ccRCC
- pre/ccRCC/HP12.6073.A5-1.ccRCC
- pre/ccRCC/HP12.6073.A5.ccRCC
- pre/ccRCC/HP12.6073_A5_ccRCC
- pre/ccRCC/HP12.6691.T1.ccRCC
- pre/ccRCC/HP12.7225.A.ccRCC
- pre/ccRCC/HP12.8355.A8.ccRCC
- pre/ccRCC/HP12.8793.B1.ccRCC
- pre/ccRCC/HP12.9282.A10.ccRCC
- pre/ccRCC/HP12.9282.A4.ccRCC
- pre/ccRCC/HP12.9282.A5.ccRCC
- pre/ccRCC/HP12.9282.A6.ccRCC
- pre/ccRCC/HP12.9282.A7.ccRCC
- pre/ccRCC/HP13.1799.B2.ccRCC
- pre/ccRCC/HP13.6992.A3.ccRCC
- pre/ccRCC/HP13.7465.A5.ccRCC
- pre/ccRCC/HP14.11034.A.ccRCC
- pre/ccRCC/HP14.13101.C.ccRCC
- pre/ccRCC/HP14.13101.E.ccRCC
- pre/ccRCC/HP14.1749.A1.ccRCC
- pre/ccRCC/HP14.1749.A6.ccRCC
- pre/ccRCC/HP14.1749.A8.ccRCC
- pre/ccRCC/HP14.1993.A1.ccRCC
- pre/ccRCC/HP14.5347.A6.ccRCC
- pre/ccRCC/HP14.5590.A3.ccRCC
- pre/ccRCC/HP14.69.2A2.ccRCC
- pre/ccRCC/HP14.69.2A9.ccRCC
- pre/ccRCC/HP14.7813.A8.ccRCC
- pre/ccRCC/HP14.9097.2F.ccRCC
- pre/ccRCC/HP14.9685.A7.ccRCC
- pre/ccRCC/HP15.11259.A3.ccRCC
- pre/ccRCC/HP15.11259.A4.ccRCC
- pre/ccRCC/HP15.11259.A6.ccRCC
- pre/ccRCC/HP15.12550.A1.ccRCC
- pre/ccRCC/HP15.12550.A6.ccRCC
- pre/ccRCC/HP15.12550.A7.ccRCC
- pre/ccRCC/HP15.1480.A3.ccRCC
- pre/ccRCC/HP15.2902.A1.ccRCC
- pre/ccRCC/HP15.2902.A4.ccRCC
- pre/ccRCC/HP16.6209.2B1.ccRCC
- pre/ccRCC/HP16.6209.2B4.ccRCC
- pre/ccRCC/HP16.6211.C.ccRCC
- pre/ccRCC/HP16.6211.E.ccRCC
- pre/ccRCC/HP16.819.A1.ccRCC
- pre/ccRCC/HP16.819.A2.ccRCC

- pre/pRCC/HP09.5392.A2.pRCC
- pre/pRCC/HP10.9650.A7.pRCC
- pre/pRCC/HP10.9650.A8.pRCC
- pre/pRCC/HP11.6090.E.pRCC
- pre/pRCC/HP12.5904.A10.pRCC
- pre/pRCC/HP12.6710.A5.pRCC
- pre/pRCC/HP12.7601.A9.pRCC
- pre/pRCC/HP12.7726.A.pRCC
- pre/pRCC/HP12.7726.C.pRCC
- pre/pRCC/HP12.7726.E.pRCC
- pre/pRCC/HP12.7726.F.pRCC
- pre/pRCC/HP13.3201.A5.pRCC
- pre/pRCC/HP13.3201.D1.pRCC
- pre/pRCC/HP13.3311.1B.pRCC
- pre/pRCC/HP14.10122.B1.pRCC
- pre/pRCC/HP14.10122.B2.pRCC
- pre/pRCC/HP14.2377.pRCC
- pre/pRCC/HP14.4279.R1.pRCC
- pre/pRCC/HP14.5971.A3.pRCC
- pre/pRCC/HP14.5971.A5.pRCC
- pre/pRCC/HP14.8231.A6.pRCC
- pre/pRCC/HP15.9102.H.pRCC

Nota: pre/pRCC e pRCC/ contengono annotazioni per la stessa classe tumorale e vanno considerate insieme, lo stesso vale per pre/ccRCC e ccRCC/.

## XML files (173)
- ccRCC/ccRCC_xml/H19.754.IHC.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.10064.A1-1.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.10064.A1.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.10064.A10.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.10064.A13.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.10064.A14.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.10064.A16.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.10064.A6.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.10064.A7.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.2434.A2.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.2434.A3.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.2434.A4.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.2434.A5.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.2434.A6.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.3695.2A1.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.3695.2A3.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.4075.A1.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.4075.A2.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.4075.A3.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.4075.A5.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.4075.A9.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.4372.A3.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.4372.A4.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.4372.A6.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.4372.A7.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.4372.A8.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.5254.A.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.5524.A2.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.5524.A3.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.5524.A4.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.7421.A6.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.754.A1.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.754.A3.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.754.A4.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.754.A5.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.754.A6.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.754.A7.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.754.A8.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.754.A9.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.7715.2A4.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.7840.A1.ccRCC-1.xml
- ccRCC/ccRCC_xml/HP19.7840.A1.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.7840.A12.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.7840.A2.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.7840.A4.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.7840.A5.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.7840.A6.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.7840.A7.ccRCC-1.xml
- ccRCC/ccRCC_xml/HP19.7840.A7.ccRCC-2.xml
- ccRCC/ccRCC_xml/HP19.7840.A7.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.7840.A9.ccRCC-1.xml
- ccRCC/ccRCC_xml/HP19.7840.A9.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.7864.1A.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.7949.2A1.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.7949.2A2.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.8394.A1.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.8394.A3.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.9347.2A.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.9421.A2.ccRCC.xml
- ccRCC/ccRCC_xml/HP19.999.A.ccRCC.xml
- pRCC/pRCC_xml/HP17.11714.A6-1.pRCC.xml
- pRCC/pRCC_xml/HP17.11714.A6.pRCC.xml
- pRCC/pRCC_xml/HP17.11714.A8.pRCC.xml
- pRCC/pRCC_xml/HP17.7980.A2.pRCC.xml
- pRCC/pRCC_xml/HP17.7980.B.pRCC.xml
- pRCC/pRCC_xml/HP17.7980.B2.pRCC.xml
- pRCC/pRCC_xml/HP18.11474.A10.pRCC.xml
- pRCC/pRCC_xml/HP18.11474.A11.pRCC.xml
- pRCC/pRCC_xml/HP18.11474.A13.pRCC.xml
- pRCC/pRCC_xml/HP18.11474.A14.pRCC.xml
- pRCC/pRCC_xml/HP18.11474.A5.pRCC.xml
- pRCC/pRCC_xml/HP18.11474.A6.pRCC.xml
- pRCC/pRCC_xml/HP18.11474.A7.pRCC.xml
- pRCC/pRCC_xml/HP18.11474.A8.pRCC.xml
- pRCC/pRCC_xml/HP18.11474.A9.pRCC.xml
- pRCC/pRCC_xml/HP18.13618.A1.pRCC.xml
- pRCC/pRCC_xml/HP18.13618.A2.pRCC.xml
- pRCC/pRCC_xml/HP18.13618.A4.pRCC.xml
- pRCC/pRCC_xml/HP18.13618.A5.pRCC.xml
- pRCC/pRCC_xml/HP18.5818.A3.pRCC.xml
- pRCC/pRCC_xml/HP18.5818.A5.pRCC.xml
- pRCC/pRCC_xml/HP19.1277.A1.pRCC.xml
- pRCC/pRCC_xml/HP19.1277.A2.pRCC.xml
- pRCC/pRCC_xml/HP19.1277.A3.pRCC.xml
- pRCC/pRCC_xml/HP19.1277.A4.pRCC.xml
- pRCC/pRCC_xml/HP19.1773.A10.pRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP02.10180.1A2.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP10.2695.A4.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP10.2986_A4_ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP10.5813.A2.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP11.12277.A4.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP11.12318.A1.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.13358.1A3.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.13358.1A4.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.13358.1A5.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.13358.1A8.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.13588.A2.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.13588.A5.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.3187.A5.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.390.A10.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.390.A5.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.4271.A11.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.4271.A8.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.5998.B.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.5998.I.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.6073.1A6.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.6073.A5-1.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.6073.A5.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.6073_A5_ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.6691.T1.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.7225.A.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.8355.A8.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.8793.B1.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.9282.A10.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.9282.A4.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.9282.A5.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.9282.A6.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP12.9282.A7.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP13.1799.B2.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP13.6992.A3.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP13.7465.A5.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP14.11034.A.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP14.13101.C.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP14.13101.E.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP14.1749.A1.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP14.1749.A6.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP14.1749.A8.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP14.1993.A1.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP14.5347.A6.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP14.5590.A3.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP14.69.2A2.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP14.69.2A9.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP14.7183.A8.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP14.7813.A8.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP14.9097.2F.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP14.9685.A7.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP15.11259.A3.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP15.11259.A4.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP15.11259.A6.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP15.12550.A1.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP15.12550.A6.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP15.12550.A7.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP15.1480.A3.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP15.2902.A1.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP15.2902.A4.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP16.6209.2B1.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP16.6209.2B4.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP16.6211.C.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP16.6211.E.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP16.819.A1.ccRCC.xml
- pre/ccRCC/pre_ccRCC_xml/HP16.819.A2.ccRCC.xml
- pre/pRCC/pre_pRCC_xml/HP09.5392.A2.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP10.9650.A7.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP10.9650.A8.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP11.6090.E.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP12.5904.A10.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP12.6710.A5.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP12.7601.A9.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP12.7726.A.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP12.7726.C.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP12.7726.E.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP12.7726.F.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP13.3201.A5.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP13.3201.D1.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP13.3311.1B.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP14.10122.B1.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP14.10122.B2.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP14.2377.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP14.4279.R1.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP14.5971.A3.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP14.5971.A5.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP14.8231.A6.pRCC.xml
- pre/pRCC/pre_pRCC_xml/HP15.9102.H.pRCC.xml


## Annotation Files (`*.xml`)

Le annotazioni XML, prodotte con software come Aperio o QuPath, descrivono **regioni d’interesse (ROI)** su ciascuna slide.
Struttura tipica:

```xml
<Annotations>
  <Annotation Id="1" Name="tumor" ReadOnly="0">
    <Regions>
      <Region Id="1" Type="0" Text="Tumor region" GeoShape="Points">
        <Vertices>
          <Vertex X="37279.31" Y="4662.18"/>
          <Vertex X="37319.55" Y="4588.89"/>
          ...
        </Vertices>
      </Region>
    </Regions>
  </Annotation>
  <Annotation Id="2" Name="non_tumor">
    <Regions>...</Regions>
  </Annotation>
</Annotations>
```

---

## XLSX files (3)
- Annotations_chromo/CHROMO_patients_correspondence.xlsx
- Annotations_onco/ONCO_patients_correspondence.xlsx
- pRCC/pRCC.xlsx

Tutti gli Excel hanno lo stesso schema:

| ID (slide range) | Patient_ID |
| ---------------- | ---------- |
| 1-3              | HPXXXXXXX  |

#### CHROMO_patients_correspondence.xlsx

| ID    | Patient_ID |
| ----- | ---------- |
| 1-2   | HP17008718 |
| 3-7   | HP20002300 |
| 8     | HP18014084 |
| 9-11  | HP19012316 |
| 12-13 | HP20.2506  |

#### ONCO_patients_correspondence.xlsx

| ID    | Patient_ID |
| ----- | ---------- |
| 1-3   | HP20002450 |
| 4-7   | HP20001530 |
| 8-13  | HP18005453 |
| 14-21 | HP20.5602  |
| 22-25 | HP18090209 |

```

### File: `reports/00_wsi-drive-analysis/wsi_inventory.csv`

_Showing first 80 lines for brevity._

```csv
slide_path,rel_path,ext,size_bytes,vendor,width0,height0,level_count,level_downsamples,mpp_x,mpp_y,objective_power,compression,xml_present,xml_roi_total,xml_roi_tumor,xml_roi_non_tumor,read_backend,error,class_hint
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/1.svs,Annotations_onco/1.svs,.svs,58001209,aperio,32647,7468,3,"1.0,4.000183801004779,16.014591222755197",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
109559x92026 [63381,40020 32647x7",0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/10.tif,Annotations_onco/10.tif,.tif,238131476,generic-tiff,6465,11585,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/12.tif,Annotations_onco/12.tif,.tif,884814074,generic-tiff,15297,18241,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/14.tif,Annotations_onco/14.tif,.tif,484283822,generic-tiff,15937,11201,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/11.tif,Annotations_onco/11.tif,.tif,1603160078,generic-tiff,29185,18817,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/16.tif,Annotations_onco/16.tif,.tif,550864772,generic-tiff,13537,14049,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/15.tif,Annotations_onco/15.tif,.tif,976670998,generic-tiff,20865,16897,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/19.tif,Annotations_onco/19.tif,.tif,37789944,generic-tiff,5057,3233,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/17.tif,Annotations_onco/17.tif,.tif,429241860,generic-tiff,15425,9921,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/13.tif,Annotations_onco/13.tif,.tif,1696309000,generic-tiff,27649,22145,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/18.tif,Annotations_onco/18.tif,.tif,500341840,generic-tiff,15489,12161,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/20.tif,Annotations_onco/20.tif,.tif,105170050,generic-tiff,6993,5217,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/23.tif,Annotations_onco/23.tif,.tif,182812158,generic-tiff,13313,6081,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/2.svs,Annotations_onco/2.svs,.svs,84337987,aperio,13212,27478,3,"1.0,4.0001455815984865,16.009019960819614",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
109559x92026 [58115,42605 13212x2",0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/24.tif,Annotations_onco/24.tif,.tif,296717406,generic-tiff,12097,9601,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/3.svs,Annotations_onco/3.svs,.svs,75840837,aperio,23441,13771,3,"1.0,4.0005211173756035,8.001042234751207",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
107567x92026 [52494,40406 23441x1",0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/21.tif,Annotations_onco/21.tif,.tif,1660490772,generic-tiff,27777,20609,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/5.svs,Annotations_onco/5.svs,.svs,138345291,aperio,24701,24126,3,"1.0,4.00024678164168,16.008857562956365",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
175295x92026 [7659,40786 24701x24",0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/22.tif,Annotations_onco/22.tif,.tif,1928638662,generic-tiff,28353,26561,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/4.svs,Annotations_onco/4.svs,.svs,794360227,aperio,52944,66253,4,"1.0,4.0000301877679165,16.001570048309176,32.00797685598958",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
109559x92026 [35999,8234 52944x66",0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/6.svs,Annotations_onco/6.svs,.svs,448695211,aperio,45190,43371,3,"1.0,4.000226869933636,16.00309184324138",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
175295x92026 [28722,19818 45190x4",0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/7.svs,Annotations_onco/7.svs,.svs,784913157,aperio,61657,56009,4,"1.0,4.00006814722774,16.002453635386154,32.009061563566235",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
105575x92026 [26903,23169 61657x5",0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/8.tif,Annotations_onco/8.tif,.tif,629668778,generic-tiff,12161,15809,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/9.tif,Annotations_onco/9.tif,.tif,571266798,generic-tiff,18369,10561,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_onco/25.tif,Annotations_onco/25.tif,.tif,1870632842,generic-tiff,23170,30465,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,ONCO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_chromo/10.svs,Annotations_chromo/10.svs,.svs,26222857,aperio,10077,10713,2,"1.0,4.000385197962253",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
125495x92026 [37153,55302 10077x1",0,0,0,0,openslide,,CHROMO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_chromo/1.svs,Annotations_chromo/1.svs,.svs,50097753,aperio,12446,17329,3,"1.0,4.0004368601807005,8.002160100708702",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
119519x92026 [77550,10053 12446x1",0,0,0,0,openslide,,CHROMO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_chromo/12.svs,Annotations_chromo/12.svs,.svs,84417583,aperio,23559,15318,3,"1.0,4.000515876970228,8.002756256814774",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
109559x80017 [20313,23643 23559x1",0,0,0,0,openslide,,CHROMO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_chromo/2.svs,Annotations_chromo/2.svs,.svs,169195059,aperio,35328,20776,3,"1.0,4.0,16.003081664098612",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
119519x92026 [37339,19818 35328x2",0,0,0,0,openslide,,CHROMO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_chromo/11.svs,Annotations_chromo/11.svs,.svs,57090429,aperio,17860,13332,3,"1.0,4.0,8.002096537539748",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
125495x92026 [57700,10015 17860x1",0,0,0,0,openslide,,CHROMO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_chromo/4.svs,Annotations_chromo/4.svs,.svs,249113749,aperio,52562,20584,3,"1.0,4.000076103500761,16.00341483390973",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
157367x92026 [13883,1628 52562x20",0,0,0,0,openslide,,CHROMO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_chromo/3.svs,Annotations_chromo/3.svs,.svs,113519483,aperio,29775,16276,3,"1.0,4.000201531640467,8.001923891532737",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
157367x92026 [5841,25563 29775x16",0,0,0,0,openslide,,CHROMO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_chromo/6.svs,Annotations_chromo/6.svs,.svs,83697023,aperio,8425,42892,3,"1.0,4.000237416904083,16.010793939049996",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
123503x92026 [96124,31595 8425x42",0,0,0,0,openslide,,CHROMO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_chromo/5.svs,Annotations_chromo/5.svs,.svs,621314787,aperio,50457,53845,4,"1.0,4.000076782839825,16.002170154229397,32.014174040452204",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
175295x78207 [13095,7601 50457x53",0,0,0,0,openslide,,CHROMO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_chromo/8.svs,Annotations_chromo/8.svs,.svs,92801901,aperio,20117,19685,3,"1.0,4.000201028709364,8.002010691347962",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
119519x92026 [51500,57976 20117x1",0,0,0,0,openslide,,CHROMO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_chromo/7.svs,Annotations_chromo/7.svs,.svs,100994735,aperio,11297,38392,3,"1.0,4.000177053824363,16.002375576698036",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
123503x92026 [76976,16276 11297x3",0,0,0,0,openslide,,CHROMO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_chromo/9.svs,Annotations_chromo/9.svs,.svs,47728205,aperio,19115,10375,3,"1.0,4.000892419411052,8.003328495057078",0.25190000000000001,0.25190000000000001,40,"Aperio Image Library v12.2.2 
107567x88242 [68958,5507 19115x10",0,0,0,0,openslide,,CHROMO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/Annotations_chromo/13.tif,Annotations_chromo/13.tif,.tif,1916155240,generic-tiff,30849,22209,1,1.0,0.25189947342777147,0.25189947342777147,,,0,0,0,0,openslide,,CHROMO
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.754.A3.ccRCC.scn,ccRCC/HP19.754.A3.ccRCC.scn,.scn,1450516040,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.98593267568448,255.60639423076924,1022.425576923077",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.754.A6.ccRCC.scn,ccRCC/HP19.754.A6.ccRCC.scn,.scn,1412185688,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.98593267568448,255.4999069285211,1020.7260254280602",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.7840.A7.ccRCC-1.scn,ccRCC/HP19.7840.A7.ccRCC-1.scn,.scn,715480236,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.98593267568448,255.39359680980417,1014.1723904131188",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.754.A7.ccRCC.scn,ccRCC/HP19.754.A7.ccRCC.scn,.scn,1142213292,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.98593267568448,255.4999069285211,1020.7260254280602",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/H19.754.IHC.ccRCC.scn,ccRCC/H19.754.IHC.ccRCC.scn,.scn,567371250,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.98593267568448,255.71305915987682,1012.4952380952382",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.754.A8.ccRCC.scn,ccRCC/HP19.754.A8.ccRCC.scn,.scn,1173686952,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.98593267568448,255.4999069285211,1022.425576923077",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.10064.A10.ccRCC.scn,ccRCC/HP19.10064.A10.ccRCC.scn,.scn,2003084070,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.98593267568448,255.71305915987682,1022.425576923077",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.754.A9.ccRCC.scn,ccRCC/HP19.754.A9.ccRCC.scn,.scn,1045740184,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.98593267568448,256.1276494961482,1022.425576923077",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.10064.A1-1.ccRCC.scn,ccRCC/HP19.10064.A1-1.ccRCC.scn,.scn,2426658640,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.97926194841071,255.60639423076924,1015.8606866002215",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.7840.A4.ccRCC.scn,ccRCC/HP19.7840.A4.ccRCC.scn,.scn,982678652,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.98593267568448,255.08732548071362,1030.8187104971657",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.10064.A13.ccRCC.scn,ccRCC/HP19.10064.A13.ccRCC.scn,.scn,1747198590,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.98593267568448,255.71305915987682,1022.425576923077",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.7864.1A.ccRCC.scn,ccRCC/HP19.7864.1A.ccRCC.scn,.scn,254027346,leica,106259,306939,6,"1.0,4.000102080108492,16.000095552667176,63.98593267568448,255.71305915987682,1022.425576923077",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.10064.A14.ccRCC.scn,ccRCC/HP19.10064.A14.ccRCC.scn,.scn,355647246,leica,106259,306939,6,"1.0,4.000102080108492,16.000095552667176,63.98593267568448,254.98119210367366,1024.1364966555184",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.7840.A6.ccRCC.scn,ccRCC/HP19.7840.A6.ccRCC.scn,.scn,1009951182,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.98593267568448,255.71305915987682,1022.425576923077",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.10064.A1.ccRCC.scn,ccRCC/HP19.10064.A1.ccRCC.scn,.scn,2344675570,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.98593267568448,255.4999069285211,1024.1364966555184",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.7840.A5.ccRCC.scn,ccRCC/HP19.7840.A5.ccRCC.scn,.scn,943259724,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.98593267568448,255.71305915987682,1012.4952380952382",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.10064.A16.ccRCC.scn,ccRCC/HP19.10064.A16.ccRCC.scn,.scn,1404478636,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.98593267568448,255.39359680980417,1020.7260254280602",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.7840.A12.ccRCC.scn,ccRCC/HP19.7840.A12.ccRCC.scn,.scn,798412652,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.98593267568448,255.4999069285211,1014.1723904131188",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.9421.A2.ccRCC.scn,ccRCC/HP19.9421.A2.ccRCC.scn,.scn,393061440,leica,106259,306939,6,"1.0,4.000102080108492,16.000095552667176,63.98593267568448,255.4999069285211,1024.1364966555184",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.2434.A2.ccRCC.scn,ccRCC/HP19.2434.A2.ccRCC.scn,.scn,937979584,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.98593267568448,255.39359680980417,1014.1723904131188",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.7840.A7.ccRCC.scn,ccRCC/HP19.7840.A7.ccRCC.scn,.scn,891074832,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.98593267568448,256.1276494961482,1014.1723904131188",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/raw/ccRCC/HP19.2434.A4.ccRCC.scn,ccRCC/HP19.2434.A4.ccRCC.scn,.scn,1084773056,leica,106259,306939,6,"1.0,4.0000007262911526,15.998891062171904,63.98593267568448,255.71305915987682,1022.425576923077",0.25,0.25,40,,0,0,0,0,openslide,,ccRCC
```

### File: `reports/01_rcc_metadata/rcc_metadata.csv`

_Showing first 80 lines for brevity._

```csv
subtype,patient_id,wsi_filename,annotation_xml,roi_files,num_rois,source_dir,wsi_size_bytes,vendor,width0,height0,level_count,mpp_x,mpp_y,objective_power,xml_roi_tumor,xml_roi_not_tumor,xml_roi_total
ccRCC,HP19.754,HP19.754.A5.ccRCC.scn,HP19.754.A5.ccRCC.xml,,0,ccRCC,1689627456,leica,106259,306939,6,0.25,0.25,40,5,3,8
ccRCC,HP19.754,HP19.754.A8.ccRCC.scn,HP19.754.A8.ccRCC.xml,,0,ccRCC,1173686952,leica,106259,306939,6,0.25,0.25,40,1,3,4
ccRCC,HP19.10064,HP19.10064.A13.ccRCC.scn,HP19.10064.A13.ccRCC.xml,,0,ccRCC,1747198590,leica,106259,306939,6,0.25,0.25,40,8,8,16
ccRCC,HP19.4372,HP19.4372.A6.ccRCC.scn,HP19.4372.A6.ccRCC.xml,,0,ccRCC,1216327640,leica,106259,306939,6,0.25,0.25,40,9,2,11
ccRCC,HP19.8394,HP19.8394.A3.ccRCC.scn,HP19.8394.A3.ccRCC.xml,,0,ccRCC,982990540,leica,106259,306939,6,0.25,0.25,40,2,0,2
ccRCC,HP19.5524,HP19.5524.A4.ccRCC.scn,HP19.5524.A4.ccRCC.xml,,0,ccRCC,1545210746,leica,106259,306939,6,0.25,0.25,40,2,4,6
ccRCC,HP19.7949,HP19.7949.2A1.ccRCC.scn,HP19.7949.2A1.ccRCC.xml,,0,ccRCC,1344509338,leica,106259,306939,6,0.25,0.25,40,0,8,8
ccRCC,HP19.7840,HP19.7840.A6.ccRCC.scn,HP19.7840.A6.ccRCC.xml,,0,ccRCC,1009951182,leica,106259,306939,6,0.25,0.25,40,3,0,3
ccRCC,HP19.754,HP19.754.A3.ccRCC.scn,HP19.754.A3.ccRCC.xml,,0,ccRCC,1450516040,leica,106259,306939,6,0.25,0.25,40,2,3,5
ccRCC,HP19.9347,HP19.9347.2A.ccRCC.scn,HP19.9347.2A.ccRCC.xml,,0,ccRCC,1315974208,leica,106259,306939,6,0.25,0.25,40,6,2,8
ccRCC,HP19.5524,HP19.5524.A2.ccRCC.scn,HP19.5524.A2.ccRCC.xml,,0,ccRCC,1146430324,leica,106259,306939,6,0.25,0.25,40,2,2,4
ccRCC,HP19.4075,HP19.4075.A1.ccRCC.scn,HP19.4075.A1.ccRCC.xml,,0,ccRCC,1580373822,leica,106259,306939,6,0.25,0.25,40,3,4,7
ccRCC,HP19.754,HP19.754.A4.ccRCC.scn,HP19.754.A4.ccRCC.xml,,0,ccRCC,1028482616,leica,106259,306939,6,0.25,0.25,40,1,2,3
ccRCC,HP19.7840,HP19.7840.A1.ccRCC.scn,HP19.7840.A1.ccRCC.xml,,0,ccRCC,786989730,leica,106259,306939,6,0.25,0.25,40,1,0,1
ccRCC,HP19.3695,HP19.3695.2A3.ccRCC.scn,HP19.3695.2A3.ccRCC.xml,,0,ccRCC,1334337876,leica,106259,306939,6,0.25,0.25,40,0,4,4
ccRCC,HP19.754,HP19.754.A9.ccRCC.scn,HP19.754.A9.ccRCC.xml,,0,ccRCC,1045740184,leica,106259,306939,6,0.25,0.25,40,1,3,4
ccRCC,HP19.4372,HP19.4372.A7.ccRCC.scn,HP19.4372.A7.ccRCC.xml,,0,ccRCC,1132280774,leica,106259,306939,6,0.25,0.25,40,6,2,8
ccRCC,HP19.9421,HP19.9421.A2.ccRCC.scn,HP19.9421.A2.ccRCC.xml,,0,ccRCC,393061440,leica,106259,306939,6,0.25,0.25,40,3,4,7
ccRCC,HP19.7840,HP19.7840.A7.ccRCC.scn,HP19.7840.A7.ccRCC.xml,,0,ccRCC,891074832,leica,106259,306939,6,0.25,0.25,40,5,0,5
ccRCC,HP19.999,HP19.999.A.ccRCC.scn,HP19.999.A.ccRCC.xml,,0,ccRCC,1413223708,leica,106259,306939,6,0.25,0.25,40,3,2,5
ccRCC,HP19.5524,HP19.5524.A3.ccRCC.scn,HP19.5524.A3.ccRCC.xml,,0,ccRCC,1319936988,leica,106259,306939,6,0.25,0.25,40,4,3,7
ccRCC,HP19.7715,HP19.7715.2A4.ccRCC.scn,HP19.7715.2A4.ccRCC.xml,,0,ccRCC,1205324006,leica,106259,306939,6,0.25,0.25,40,2,3,5
ccRCC,HP19.10064,HP19.10064.A14.ccRCC.scn,HP19.10064.A14.ccRCC.xml,,0,ccRCC,355647246,leica,106259,306939,6,0.25,0.25,40,1,0,1
ccRCC,HP19.2434,HP19.2434.A6.ccRCC.scn,HP19.2434.A6.ccRCC.xml,,0,ccRCC,952202036,leica,106259,306939,6,0.25,0.25,40,6,2,8
ccRCC,HP19.7840,HP19.7840.A9.ccRCC-1.scn,HP19.7840.A9.ccRCC-1.xml,,0,ccRCC,797684844,leica,106259,306939,6,0.25,0.25,40,3,0,3
ccRCC,HP19.7421,HP19.7421.A6.ccRCC.scn,HP19.7421.A6.ccRCC.xml,,0,ccRCC,757777460,leica,106259,306939,6,0.25,0.25,40,3,5,8
ccRCC,HP19.10064,HP19.10064.A1.ccRCC.scn,HP19.10064.A1.ccRCC.xml,,0,ccRCC,2344675570,leica,106259,306939,6,0.25,0.25,40,4,2,6
ccRCC,HP19.754,HP19.754.A7.ccRCC.scn,HP19.754.A7.ccRCC.xml,,0,ccRCC,1142213292,leica,106259,306939,6,0.25,0.25,40,4,5,9
ccRCC,HP19.7840,HP19.7840.A2.ccRCC.scn,HP19.7840.A2.ccRCC.xml,,0,ccRCC,867502582,leica,106259,306939,6,0.25,0.25,40,1,0,1
ccRCC,HP19.7840,HP19.7840.A7.ccRCC-2.scn,HP19.7840.A7.ccRCC-2.xml,,0,ccRCC,792711090,leica,106259,306939,6,0.25,0.25,40,5,0,5
ccRCC,HP19.4075,HP19.4075.A5.ccRCC.scn,HP19.4075.A5.ccRCC.xml,,0,ccRCC,1445701748,leica,106259,306939,6,0.25,0.25,40,3,2,5
ccRCC,HP19.8394,HP19.8394.A1.ccRCC.scn,HP19.8394.A1.ccRCC.xml,,0,ccRCC,987972966,leica,106259,306939,6,0.25,0.25,40,4,4,8
ccRCC,HP19.2434,HP19.2434.A3.ccRCC.scn,HP19.2434.A3.ccRCC.xml,,0,ccRCC,1204778242,leica,106259,306939,6,0.25,0.25,40,1,2,3
ccRCC,HP19.4372,HP19.4372.A4.ccRCC.scn,HP19.4372.A4.ccRCC.xml,,0,ccRCC,1230482890,leica,106259,306939,6,0.25,0.25,40,3,3,6
ccRCC,HP19.7840,HP19.7840.A7.ccRCC-1.scn,HP19.7840.A7.ccRCC-1.xml,,0,ccRCC,715480236,leica,106259,306939,6,0.25,0.25,40,7,0,7
ccRCC,HP19.7840,HP19.7840.A4.ccRCC.scn,HP19.7840.A4.ccRCC.xml,,0,ccRCC,982678652,leica,106259,306939,6,0.25,0.25,40,1,0,1
ccRCC,HP19.754,HP19.754.A1.ccRCC.scn,HP19.754.A1.ccRCC.xml,,0,ccRCC,783811206,leica,106259,306939,6,0.25,0.25,40,2,1,3
ccRCC,HP19.10064,HP19.10064.A7.ccRCC.scn,HP19.10064.A7.ccRCC.xml,,0,ccRCC,2177317658,leica,106259,306939,6,0.25,0.25,40,6,5,11
ccRCC,HP19.2434,HP19.2434.A5.ccRCC.scn,HP19.2434.A5.ccRCC.xml,,0,ccRCC,1553820284,leica,106259,306939,6,0.25,0.25,40,4,1,5
ccRCC,HP19.7840,HP19.7840.A9.ccRCC.scn,HP19.7840.A9.ccRCC.xml,,0,ccRCC,868679938,leica,106259,306939,6,0.25,0.25,40,2,1,3
ccRCC,HP19.4075,HP19.4075.A3.ccRCC.scn,HP19.4075.A3.ccRCC.xml,,0,ccRCC,1180389218,leica,106259,306939,6,0.25,0.25,40,4,1,5
ccRCC,HP19.4075,HP19.4075.A9.ccRCC.scn,HP19.4075.A9.ccRCC.xml,,0,ccRCC,1131750352,leica,106259,306939,6,0.25,0.25,40,4,2,6
ccRCC,HP19.3695,HP19.3695.2A1.ccRCC.scn,HP19.3695.2A1.ccRCC.xml,,0,ccRCC,2161134484,leica,106259,306939,6,0.25,0.25,40,0,15,15
ccRCC,H19.754,H19.754.IHC.ccRCC.scn,H19.754.IHC.ccRCC.xml,,0,ccRCC,567371250,leica,106259,306939,6,0.25,0.25,40,2,5,7
ccRCC,HP19.754,HP19.754.A6.ccRCC.scn,HP19.754.A6.ccRCC.xml,,0,ccRCC,1412185688,leica,106259,306939,6,0.25,0.25,40,9,8,17
ccRCC,HP19.4372,HP19.4372.A8.ccRCC.scn,HP19.4372.A8.ccRCC.xml,,0,ccRCC,1101100858,leica,106259,306939,6,0.25,0.25,40,4,0,4
ccRCC,HP19.7840,HP19.7840.A1.ccRCC-1.scn,HP19.7840.A1.ccRCC-1.xml,,0,ccRCC,852836242,leica,106259,306939,6,0.25,0.25,40,1,0,1
ccRCC,HP19.10064,HP19.10064.A10.ccRCC.scn,HP19.10064.A10.ccRCC.xml,,0,ccRCC,2003084070,leica,106259,306939,6,0.25,0.25,40,7,2,9
ccRCC,HP19.2434,HP19.2434.A2.ccRCC.scn,HP19.2434.A2.ccRCC.xml,,0,ccRCC,937979584,leica,106259,306939,6,0.25,0.25,40,4,0,4
ccRCC,HP19.7840,HP19.7840.A5.ccRCC.scn,HP19.7840.A5.ccRCC.xml,,0,ccRCC,943259724,leica,106259,306939,6,0.25,0.25,40,3,0,3
ccRCC,HP19.7840,HP19.7840.A12.ccRCC.scn,HP19.7840.A12.ccRCC.xml,,0,ccRCC,798412652,leica,106259,306939,6,0.25,0.25,40,1,0,1
ccRCC,HP19.10064,HP19.10064.A1-1.ccRCC.scn,HP19.10064.A1-1.ccRCC.xml,,0,ccRCC,2426658640,leica,106259,306939,6,0.25,0.25,40,4,2,6
ccRCC,HP19.7949,HP19.7949.2A2.ccRCC.scn,HP19.7949.2A2.ccRCC.xml,,0,ccRCC,1483004042,leica,106259,306939,6,0.25,0.25,40,0,6,6
ccRCC,HP19.10064,HP19.10064.A6.ccRCC.scn,HP19.10064.A6.ccRCC.xml,,0,ccRCC,2610779244,leica,106259,306939,6,0.25,0.25,40,8,5,13
ccRCC,HP19.5254,HP19.5254.A.ccRCC.scn,HP19.5254.A.ccRCC.xml,,0,ccRCC,1649902336,leica,106259,306939,6,0.25,0.25,40,6,1,7
ccRCC,HP19.4372,HP19.4372.A3.ccRCC.scn,HP19.4372.A3.ccRCC.xml,,0,ccRCC,1137569160,leica,106259,306939,6,0.25,0.25,40,12,7,19
ccRCC,HP19.10064,HP19.10064.A16.ccRCC.scn,HP19.10064.A16.ccRCC.xml,,0,ccRCC,1404478636,leica,106259,306939,6,0.25,0.25,40,8,6,14
ccRCC,HP19.2434,HP19.2434.A4.ccRCC.scn,HP19.2434.A4.ccRCC.xml,,0,ccRCC,1084773056,leica,106259,306939,6,0.25,0.25,40,1,2,3
ccRCC,HP19.7864,HP19.7864.1A.ccRCC.scn,HP19.7864.1A.ccRCC.xml,,0,ccRCC,254027346,leica,106259,306939,6,0.25,0.25,40,4,4,8
ccRCC,HP19.4075,HP19.4075.A2.ccRCC.scn,HP19.4075.A2.ccRCC.xml,,0,ccRCC,1527461720,leica,106259,306939,6,0.25,0.25,40,4,4,8
ccRCC,HP15.12550,HP15.12550.A7.ccRCC.scn,HP15.12550.A7.ccRCC.xml,,0,pre/ccRCC,767723142,leica,106259,306939,6,0.25,0.25,40,6,6,12
ccRCC,HP12.8355,HP12.8355.A8.ccRCC.scn,HP12.8355.A8.ccRCC.xml,,0,pre/ccRCC,863823010,leica,106259,306939,6,0.25,0.25,40,10,26,36
ccRCC,HP12.9282,HP12.9282.A10.ccRCC.scn,HP12.9282.A10.ccRCC.xml,,0,pre/ccRCC,982577414,leica,106259,306939,6,0.25,0.25,40,2,0,2
ccRCC,HP12.9282,HP12.9282.A5.ccRCC.scn,HP12.9282.A5.ccRCC.xml,,0,pre/ccRCC,1164863608,leica,106259,306939,6,0.25,0.25,40,7,7,14
ccRCC,HP14.1749,HP14.1749.A6.ccRCC.scn,HP14.1749.A6.ccRCC.xml,,0,pre/ccRCC,622227788,leica,106259,306939,6,0.25,0.25,40,14,2,16
ccRCC,HP13.7465,HP13.7465.A5.ccRCC.scn,HP13.7465.A5.ccRCC.xml,,0,pre/ccRCC,474568420,leica,106259,306939,6,0.25,0.25,40,3,2,5
ccRCC,HP12.6691,HP12.6691.T1.ccRCC.scn,HP12.6691.T1.ccRCC.xml,,0,pre/ccRCC,770613564,leica,106259,306939,6,0.25,0.25,40,3,7,10
ccRCC,HP14.7813,HP14.7813.A8.ccRCC.scn,HP14.7813.A8.ccRCC.xml,,0,pre/ccRCC,1194545462,leica,106259,306939,6,0.25,0.25,40,1,3,4
ccRCC,HP14.11034,HP14.11034.A.ccRCC.scn,HP14.11034.A.ccRCC.xml,,0,pre/ccRCC,616673724,leica,106259,306939,6,0.25,0.25,40,15,4,19
ccRCC,HP11.12318,HP11.12318.A1.ccRCC.scn,HP11.12318.A1.ccRCC.xml,,0,pre/ccRCC,710682134,leica,106259,306939,6,0.25,0.25,40,19,8,27
ccRCC,HP10.2695,HP10.2695.A4.ccRCC.scn,HP10.2695.A4.ccRCC.xml,,0,pre/ccRCC,1013859340,leica,106259,306939,6,0.25,0.25,40,10,15,25
ccRCC,HP12.7225,HP12.7225.A.ccRCC.scn,HP12.7225.A.ccRCC.xml,,0,pre/ccRCC,727979688,leica,106259,306939,6,0.25,0.25,40,7,6,13
ccRCC,HP16.819,HP16.819.A2.ccRCC.scn,HP16.819.A2.ccRCC.xml,,0,pre/ccRCC,763773246,leica,106259,306939,6,0.25,0.25,40,2,3,5
ccRCC,HP12.390,HP12.390.A5.ccRCC.scn,HP12.390.A5.ccRCC.xml,,0,pre/ccRCC,1106790912,leica,106259,306939,6,0.25,0.25,40,3,18,21
ccRCC,HP10.2986,HP10.2986_A4_ccRCC.scn,HP10.2986_A4_ccRCC.xml,,0,pre/ccRCC,833342700,leica,106259,306939,6,0.25,0.25,40,21,0,21
ccRCC,HP15.12550,HP15.12550.A1.ccRCC.scn,HP15.12550.A1.ccRCC.xml,,0,pre/ccRCC,746920644,leica,106259,306939,6,0.25,0.25,40,3,4,7
ccRCC,HP13.1799,HP13.1799.B2.ccRCC.scn,HP13.1799.B2.ccRCC.xml,,0,pre/ccRCC,1181953034,leica,106259,306939,6,0.25,0.25,40,0,4,4
ccRCC,HP12.6073,HP12.6073.A5-1.ccRCC.scn,HP12.6073.A5-1.ccRCC.xml,,0,pre/ccRCC,889956856,leica,106259,306939,6,0.25,0.25,40,8,7,15
ccRCC,HP15.12550,HP15.12550.A6.ccRCC.scn,HP15.12550.A6.ccRCC.xml,,0,pre/ccRCC,771117104,leica,106259,306939,6,0.25,0.25,40,8,3,11
```

### File: `reports/02_parquet/slides.csv`

_Showing first 80 lines for brevity._

```csv
record_id,patient_id,class_label,rel_path,wsi_filename,subtype,source_dir,vendor,objective_power,mpp_x,mpp_y,width0,height0,level_count,annotation_xml,roi_files,num_rois,xml_roi_total,xml_roi_tumor,wsi_basename,wsi_size_bytes,xml_roi_not_tumor
18cc19c8c89d7097687190708beffef143194f2c,HP19.754,ccRCC,ccRCC/HP19.754.A5.ccRCC.scn,HP19.754.A5.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.754.A5.ccRCC.xml,nan,0,8,5,HP19.754.A5.ccRCC.scn,1689627456,3
90a49842d640dfafa0064521e73d08ffd5d70d55,HP19.754,ccRCC,ccRCC/HP19.754.A8.ccRCC.scn,HP19.754.A8.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.754.A8.ccRCC.xml,nan,0,4,1,HP19.754.A8.ccRCC.scn,1173686952,3
5d3a6910bd8f790af1224c82a5fdfbe644ab8eed,HP19.10064,ccRCC,ccRCC/HP19.10064.A13.ccRCC.scn,HP19.10064.A13.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.10064.A13.ccRCC.xml,nan,0,16,8,HP19.10064.A13.ccRCC.scn,1747198590,8
35a75540461e999cdff4d5ba4bddd1cdf899c56b,HP19.4372,ccRCC,ccRCC/HP19.4372.A6.ccRCC.scn,HP19.4372.A6.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.4372.A6.ccRCC.xml,nan,0,11,9,HP19.4372.A6.ccRCC.scn,1216327640,2
617f31ba740451322c4a61999b2d44e0ff7a06ee,HP19.8394,ccRCC,ccRCC/HP19.8394.A3.ccRCC.scn,HP19.8394.A3.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.8394.A3.ccRCC.xml,nan,0,2,2,HP19.8394.A3.ccRCC.scn,982990540,0
492e3f1812f239e968e01ac2a7a6500941ac686b,HP19.5524,ccRCC,ccRCC/HP19.5524.A4.ccRCC.scn,HP19.5524.A4.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.5524.A4.ccRCC.xml,nan,0,6,2,HP19.5524.A4.ccRCC.scn,1545210746,4
cb79544f1a3b5ad38c509e64248c5cff5989231e,HP19.7949,ccRCC,ccRCC/HP19.7949.2A1.ccRCC.scn,HP19.7949.2A1.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.7949.2A1.ccRCC.xml,nan,0,8,0,HP19.7949.2A1.ccRCC.scn,1344509338,8
57c2f274d0415a8cf843616d2d50f58ce115e65e,HP19.7840,ccRCC,ccRCC/HP19.7840.A6.ccRCC.scn,HP19.7840.A6.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.7840.A6.ccRCC.xml,nan,0,3,3,HP19.7840.A6.ccRCC.scn,1009951182,0
78a2116a34d73fa7581dd1099cd32b916aa2982b,HP19.754,ccRCC,ccRCC/HP19.754.A3.ccRCC.scn,HP19.754.A3.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.754.A3.ccRCC.xml,nan,0,5,2,HP19.754.A3.ccRCC.scn,1450516040,3
f41496ad8bbbe2369eada7d309c34ee3c35459f3,HP19.9347,ccRCC,ccRCC/HP19.9347.2A.ccRCC.scn,HP19.9347.2A.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.9347.2A.ccRCC.xml,nan,0,8,6,HP19.9347.2A.ccRCC.scn,1315974208,2
1a8f1a0eb1dc6976cd3f1b4db3c9386d140f2bb1,HP19.5524,ccRCC,ccRCC/HP19.5524.A2.ccRCC.scn,HP19.5524.A2.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.5524.A2.ccRCC.xml,nan,0,4,2,HP19.5524.A2.ccRCC.scn,1146430324,2
8db9d3485dcd55ef4350ee67fc669dc315a940de,HP19.4075,ccRCC,ccRCC/HP19.4075.A1.ccRCC.scn,HP19.4075.A1.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.4075.A1.ccRCC.xml,nan,0,7,3,HP19.4075.A1.ccRCC.scn,1580373822,4
4e86c84e8dae18485fe1272e4d4e3b6fc1ee129d,HP19.754,ccRCC,ccRCC/HP19.754.A4.ccRCC.scn,HP19.754.A4.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.754.A4.ccRCC.xml,nan,0,3,1,HP19.754.A4.ccRCC.scn,1028482616,2
4f648cdb68f05627b0e5629e12bf68c6a981206e,HP19.7840,ccRCC,ccRCC/HP19.7840.A1.ccRCC.scn,HP19.7840.A1.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.7840.A1.ccRCC.xml,nan,0,1,1,HP19.7840.A1.ccRCC.scn,786989730,0
a65e7bf84a9d6b0e4da796d26827765ce8b86e00,HP19.3695,ccRCC,ccRCC/HP19.3695.2A3.ccRCC.scn,HP19.3695.2A3.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.3695.2A3.ccRCC.xml,nan,0,4,0,HP19.3695.2A3.ccRCC.scn,1334337876,4
1ef88c24eabcc8a43bbc92e21e02f593d7001884,HP19.754,ccRCC,ccRCC/HP19.754.A9.ccRCC.scn,HP19.754.A9.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.754.A9.ccRCC.xml,nan,0,4,1,HP19.754.A9.ccRCC.scn,1045740184,3
9fd9827369ab9997702f7d0255215290f3197fa1,HP19.4372,ccRCC,ccRCC/HP19.4372.A7.ccRCC.scn,HP19.4372.A7.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.4372.A7.ccRCC.xml,nan,0,8,6,HP19.4372.A7.ccRCC.scn,1132280774,2
ab99fbc26500430e7eee843ca03c5aa9469265ba,HP19.9421,ccRCC,ccRCC/HP19.9421.A2.ccRCC.scn,HP19.9421.A2.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.9421.A2.ccRCC.xml,nan,0,7,3,HP19.9421.A2.ccRCC.scn,393061440,4
1a37c76a10b1742ea4103e3668f59d8ac2b6bb77,HP19.7840,ccRCC,ccRCC/HP19.7840.A7.ccRCC.scn,HP19.7840.A7.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.7840.A7.ccRCC.xml,nan,0,5,5,HP19.7840.A7.ccRCC.scn,891074832,0
345bb1a01db69310c1abd8312dc5607587d6a70f,HP19.999,ccRCC,ccRCC/HP19.999.A.ccRCC.scn,HP19.999.A.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.999.A.ccRCC.xml,nan,0,5,3,HP19.999.A.ccRCC.scn,1413223708,2
6161cc5b59034f199f34616d88a96f12fbb60163,HP19.5524,ccRCC,ccRCC/HP19.5524.A3.ccRCC.scn,HP19.5524.A3.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.5524.A3.ccRCC.xml,nan,0,7,4,HP19.5524.A3.ccRCC.scn,1319936988,3
b884f717df1d05d67a3e10d861684531f5970921,HP19.7715,ccRCC,ccRCC/HP19.7715.2A4.ccRCC.scn,HP19.7715.2A4.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.7715.2A4.ccRCC.xml,nan,0,5,2,HP19.7715.2A4.ccRCC.scn,1205324006,3
9451ee5d0d2d7b9e865f698ad29fc6bda7799a0b,HP19.10064,ccRCC,ccRCC/HP19.10064.A14.ccRCC.scn,HP19.10064.A14.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.10064.A14.ccRCC.xml,nan,0,1,1,HP19.10064.A14.ccRCC.scn,355647246,0
117dcac0ab5317326777f25bc0af1d0968061e9b,HP19.2434,ccRCC,ccRCC/HP19.2434.A6.ccRCC.scn,HP19.2434.A6.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.2434.A6.ccRCC.xml,nan,0,8,6,HP19.2434.A6.ccRCC.scn,952202036,2
5aaebe497610a5e1fadc41fde2b7e1590325aed7,HP19.7840,ccRCC,ccRCC/HP19.7840.A9.ccRCC-1.scn,HP19.7840.A9.ccRCC-1.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.7840.A9.ccRCC-1.xml,nan,0,3,3,HP19.7840.A9.ccRCC-1.scn,797684844,0
fa989a0646ef029034d108fcdcf52cf92723063e,HP19.7421,ccRCC,ccRCC/HP19.7421.A6.ccRCC.scn,HP19.7421.A6.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.7421.A6.ccRCC.xml,nan,0,8,3,HP19.7421.A6.ccRCC.scn,757777460,5
3ca24a78f59699e06affe97255405a9715f1ce0e,HP19.10064,ccRCC,ccRCC/HP19.10064.A1.ccRCC.scn,HP19.10064.A1.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.10064.A1.ccRCC.xml,nan,0,6,4,HP19.10064.A1.ccRCC.scn,2344675570,2
126d5e08fe0715a052e406a914604c9ce4e0edda,HP19.754,ccRCC,ccRCC/HP19.754.A7.ccRCC.scn,HP19.754.A7.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.754.A7.ccRCC.xml,nan,0,9,4,HP19.754.A7.ccRCC.scn,1142213292,5
4d99b8ea8e8e8f8838c6ca94c3260900e880263f,HP19.7840,ccRCC,ccRCC/HP19.7840.A2.ccRCC.scn,HP19.7840.A2.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.7840.A2.ccRCC.xml,nan,0,1,1,HP19.7840.A2.ccRCC.scn,867502582,0
358f8a7ebf2614579b15cdad3f365eed27b417c2,HP19.7840,ccRCC,ccRCC/HP19.7840.A7.ccRCC-2.scn,HP19.7840.A7.ccRCC-2.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.7840.A7.ccRCC-2.xml,nan,0,5,5,HP19.7840.A7.ccRCC-2.scn,792711090,0
63bd9e217c3340194473cc31fa426e1ef61b48d7,HP19.4075,ccRCC,ccRCC/HP19.4075.A5.ccRCC.scn,HP19.4075.A5.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.4075.A5.ccRCC.xml,nan,0,5,3,HP19.4075.A5.ccRCC.scn,1445701748,2
7a0e397642ffa2012933f803f4b25630b7e30e8c,HP19.8394,ccRCC,ccRCC/HP19.8394.A1.ccRCC.scn,HP19.8394.A1.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.8394.A1.ccRCC.xml,nan,0,8,4,HP19.8394.A1.ccRCC.scn,987972966,4
b01ed64e6eea572b78ae11864b82bb5a6318371e,HP19.2434,ccRCC,ccRCC/HP19.2434.A3.ccRCC.scn,HP19.2434.A3.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.2434.A3.ccRCC.xml,nan,0,3,1,HP19.2434.A3.ccRCC.scn,1204778242,2
a0a2866a9975b6ec4bd022a5e6a363ff6b8b3f48,HP19.4372,ccRCC,ccRCC/HP19.4372.A4.ccRCC.scn,HP19.4372.A4.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.4372.A4.ccRCC.xml,nan,0,6,3,HP19.4372.A4.ccRCC.scn,1230482890,3
d4a2c3e1cc182973439fc0374e75e9119233bc56,HP19.7840,ccRCC,ccRCC/HP19.7840.A7.ccRCC-1.scn,HP19.7840.A7.ccRCC-1.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.7840.A7.ccRCC-1.xml,nan,0,7,7,HP19.7840.A7.ccRCC-1.scn,715480236,0
161ee228b623d32c45700aa87f61559fd5759645,HP19.7840,ccRCC,ccRCC/HP19.7840.A4.ccRCC.scn,HP19.7840.A4.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.7840.A4.ccRCC.xml,nan,0,1,1,HP19.7840.A4.ccRCC.scn,982678652,0
7a1e21a4df64ef44fb3f0217d37657f3fce58f4a,HP19.754,ccRCC,ccRCC/HP19.754.A1.ccRCC.scn,HP19.754.A1.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.754.A1.ccRCC.xml,nan,0,3,2,HP19.754.A1.ccRCC.scn,783811206,1
a6cbf9f15e5f23bf7c1b4f79d0396294003ffa3b,HP19.10064,ccRCC,ccRCC/HP19.10064.A7.ccRCC.scn,HP19.10064.A7.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.10064.A7.ccRCC.xml,nan,0,11,6,HP19.10064.A7.ccRCC.scn,2177317658,5
b27835cf965dd568d064008f3a52c55b083f28e5,HP19.2434,ccRCC,ccRCC/HP19.2434.A5.ccRCC.scn,HP19.2434.A5.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.2434.A5.ccRCC.xml,nan,0,5,4,HP19.2434.A5.ccRCC.scn,1553820284,1
a79c70b811f3d424c03703207a545407b3cd1ddf,HP19.7840,ccRCC,ccRCC/HP19.7840.A9.ccRCC.scn,HP19.7840.A9.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.7840.A9.ccRCC.xml,nan,0,3,2,HP19.7840.A9.ccRCC.scn,868679938,1
f4f3c805d03923adbf56c1249e107c19bce326a5,HP19.4075,ccRCC,ccRCC/HP19.4075.A3.ccRCC.scn,HP19.4075.A3.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.4075.A3.ccRCC.xml,nan,0,5,4,HP19.4075.A3.ccRCC.scn,1180389218,1
8331a7dd29e1eba45df0841b7daa52f2b791344c,HP19.4075,ccRCC,ccRCC/HP19.4075.A9.ccRCC.scn,HP19.4075.A9.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.4075.A9.ccRCC.xml,nan,0,6,4,HP19.4075.A9.ccRCC.scn,1131750352,2
727c85d5134abdf8828b0b2edf5f6725c73dd79f,HP19.3695,ccRCC,ccRCC/HP19.3695.2A1.ccRCC.scn,HP19.3695.2A1.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.3695.2A1.ccRCC.xml,nan,0,15,0,HP19.3695.2A1.ccRCC.scn,2161134484,15
6d08fbc5f4ea40bd50944df37413b03de3ff26c5,H19.754,ccRCC,ccRCC/H19.754.IHC.ccRCC.scn,H19.754.IHC.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,H19.754.IHC.ccRCC.xml,nan,0,7,2,H19.754.IHC.ccRCC.scn,567371250,5
468012c6ecfd79be748ae3b05fdc45c44e42f0f4,HP19.754,ccRCC,ccRCC/HP19.754.A6.ccRCC.scn,HP19.754.A6.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.754.A6.ccRCC.xml,nan,0,17,9,HP19.754.A6.ccRCC.scn,1412185688,8
db26f64ecb0ded15534923b43f07ef05de9cfea6,HP19.4372,ccRCC,ccRCC/HP19.4372.A8.ccRCC.scn,HP19.4372.A8.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.4372.A8.ccRCC.xml,nan,0,4,4,HP19.4372.A8.ccRCC.scn,1101100858,0
79efdda70808d027288db6b3c82e642060cf4552,HP19.7840,ccRCC,ccRCC/HP19.7840.A1.ccRCC-1.scn,HP19.7840.A1.ccRCC-1.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.7840.A1.ccRCC-1.xml,nan,0,1,1,HP19.7840.A1.ccRCC-1.scn,852836242,0
46add5f9a296c2f6a71cef56119335957a34f209,HP19.10064,ccRCC,ccRCC/HP19.10064.A10.ccRCC.scn,HP19.10064.A10.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.10064.A10.ccRCC.xml,nan,0,9,7,HP19.10064.A10.ccRCC.scn,2003084070,2
b0f758fd54cd2d094bb2963a850e0ecc2d154d40,HP19.2434,ccRCC,ccRCC/HP19.2434.A2.ccRCC.scn,HP19.2434.A2.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.2434.A2.ccRCC.xml,nan,0,4,4,HP19.2434.A2.ccRCC.scn,937979584,0
7ebcbd66bb6d9bca8d0aa72dc33576fa263d2ecb,HP19.7840,ccRCC,ccRCC/HP19.7840.A5.ccRCC.scn,HP19.7840.A5.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.7840.A5.ccRCC.xml,nan,0,3,3,HP19.7840.A5.ccRCC.scn,943259724,0
b13cf01f46fed4a39b64d83b803ae3350548a8e9,HP19.7840,ccRCC,ccRCC/HP19.7840.A12.ccRCC.scn,HP19.7840.A12.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.7840.A12.ccRCC.xml,nan,0,1,1,HP19.7840.A12.ccRCC.scn,798412652,0
c42eaa9075bcb7550ad1e76e013d382dafbe101d,HP19.10064,ccRCC,ccRCC/HP19.10064.A1-1.ccRCC.scn,HP19.10064.A1-1.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.10064.A1-1.ccRCC.xml,nan,0,6,4,HP19.10064.A1-1.ccRCC.scn,2426658640,2
062ede50d0626e0b961dfd1572d68b1fa1c8ff26,HP19.7949,ccRCC,ccRCC/HP19.7949.2A2.ccRCC.scn,HP19.7949.2A2.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.7949.2A2.ccRCC.xml,nan,0,6,0,HP19.7949.2A2.ccRCC.scn,1483004042,6
01fee3d5a206a78b73fa067ab49a98488a30e12e,HP19.10064,ccRCC,ccRCC/HP19.10064.A6.ccRCC.scn,HP19.10064.A6.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.10064.A6.ccRCC.xml,nan,0,13,8,HP19.10064.A6.ccRCC.scn,2610779244,5
297e1155e062f180737107cb7d918bab3c80a011,HP19.5254,ccRCC,ccRCC/HP19.5254.A.ccRCC.scn,HP19.5254.A.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.5254.A.ccRCC.xml,nan,0,7,6,HP19.5254.A.ccRCC.scn,1649902336,1
bab44c6d253b4ead174c67b1890d4379a93c6b0d,HP19.4372,ccRCC,ccRCC/HP19.4372.A3.ccRCC.scn,HP19.4372.A3.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.4372.A3.ccRCC.xml,nan,0,19,12,HP19.4372.A3.ccRCC.scn,1137569160,7
fc749af378a84e34aba0749804d6f09ef0ba68e9,HP19.10064,ccRCC,ccRCC/HP19.10064.A16.ccRCC.scn,HP19.10064.A16.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.10064.A16.ccRCC.xml,nan,0,14,8,HP19.10064.A16.ccRCC.scn,1404478636,6
70517d11fab36f0d9f6514cca6806106486cad14,HP19.2434,ccRCC,ccRCC/HP19.2434.A4.ccRCC.scn,HP19.2434.A4.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.2434.A4.ccRCC.xml,nan,0,3,1,HP19.2434.A4.ccRCC.scn,1084773056,2
a8ea2382091b4675aa41a23b109ece8d739cb317,HP19.7864,ccRCC,ccRCC/HP19.7864.1A.ccRCC.scn,HP19.7864.1A.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.7864.1A.ccRCC.xml,nan,0,8,4,HP19.7864.1A.ccRCC.scn,254027346,4
0c3f070d6c5266bf9683e397e17bb73718b72377,HP19.4075,ccRCC,ccRCC/HP19.4075.A2.ccRCC.scn,HP19.4075.A2.ccRCC.scn,ccRCC,ccRCC,leica,40,0.25,0.25,106259,306939,6,HP19.4075.A2.ccRCC.xml,nan,0,8,4,HP19.4075.A2.ccRCC.scn,1527461720,4
be29a4b1fc327e5c31f54bc1d462b75c5d4368d0,HP15.12550,ccRCC,pre/ccRCC/HP15.12550.A7.ccRCC.scn,HP15.12550.A7.ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP15.12550.A7.ccRCC.xml,nan,0,12,6,HP15.12550.A7.ccRCC.scn,767723142,6
c8510ba2f7c4bc0b59f30586739e6e42e3f61228,HP12.8355,ccRCC,pre/ccRCC/HP12.8355.A8.ccRCC.scn,HP12.8355.A8.ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP12.8355.A8.ccRCC.xml,nan,0,36,10,HP12.8355.A8.ccRCC.scn,863823010,26
47cb6bfb459fd5ece3ce81fe0a616ae6fdfecfaa,HP12.9282,ccRCC,pre/ccRCC/HP12.9282.A10.ccRCC.scn,HP12.9282.A10.ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP12.9282.A10.ccRCC.xml,nan,0,2,2,HP12.9282.A10.ccRCC.scn,982577414,0
f663267028e7cb66060a4667bbb53e56bbcde64a,HP12.9282,ccRCC,pre/ccRCC/HP12.9282.A5.ccRCC.scn,HP12.9282.A5.ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP12.9282.A5.ccRCC.xml,nan,0,14,7,HP12.9282.A5.ccRCC.scn,1164863608,7
5a5919267a7e1f93fcc24c4a6b5cc1287da80b39,HP14.1749,ccRCC,pre/ccRCC/HP14.1749.A6.ccRCC.scn,HP14.1749.A6.ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP14.1749.A6.ccRCC.xml,nan,0,16,14,HP14.1749.A6.ccRCC.scn,622227788,2
01109a1bfd659983ee423543354cdedf88859140,HP13.7465,ccRCC,pre/ccRCC/HP13.7465.A5.ccRCC.scn,HP13.7465.A5.ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP13.7465.A5.ccRCC.xml,nan,0,5,3,HP13.7465.A5.ccRCC.scn,474568420,2
5dc900d15863f55daacdf513907689eed41b5890,HP12.6691,ccRCC,pre/ccRCC/HP12.6691.T1.ccRCC.scn,HP12.6691.T1.ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP12.6691.T1.ccRCC.xml,nan,0,10,3,HP12.6691.T1.ccRCC.scn,770613564,7
feddbf872a45f991acb2a06f280bce36184b93cb,HP14.7813,ccRCC,pre/ccRCC/HP14.7813.A8.ccRCC.scn,HP14.7813.A8.ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP14.7813.A8.ccRCC.xml,nan,0,4,1,HP14.7813.A8.ccRCC.scn,1194545462,3
f9493ed81c783a0971e8f049dd5a445bef587590,HP14.11034,ccRCC,pre/ccRCC/HP14.11034.A.ccRCC.scn,HP14.11034.A.ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP14.11034.A.ccRCC.xml,nan,0,19,15,HP14.11034.A.ccRCC.scn,616673724,4
661a98be124a37b012332fd910e79f52b7d2c587,HP11.12318,ccRCC,pre/ccRCC/HP11.12318.A1.ccRCC.scn,HP11.12318.A1.ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP11.12318.A1.ccRCC.xml,nan,0,27,19,HP11.12318.A1.ccRCC.scn,710682134,8
8dd1733d29ee1dd57965856ce4c19e222a6c9ba3,HP10.2695,ccRCC,pre/ccRCC/HP10.2695.A4.ccRCC.scn,HP10.2695.A4.ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP10.2695.A4.ccRCC.xml,nan,0,25,10,HP10.2695.A4.ccRCC.scn,1013859340,15
b98520a37a57621f49096f60e14839eb60ef576d,HP12.7225,ccRCC,pre/ccRCC/HP12.7225.A.ccRCC.scn,HP12.7225.A.ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP12.7225.A.ccRCC.xml,nan,0,13,7,HP12.7225.A.ccRCC.scn,727979688,6
606877b553fa618e87a2af70b62bd32e8ac58205,HP16.819,ccRCC,pre/ccRCC/HP16.819.A2.ccRCC.scn,HP16.819.A2.ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP16.819.A2.ccRCC.xml,nan,0,5,2,HP16.819.A2.ccRCC.scn,763773246,3
a4d95afe48c52f027b431377d80f8f312c48864a,HP12.390,ccRCC,pre/ccRCC/HP12.390.A5.ccRCC.scn,HP12.390.A5.ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP12.390.A5.ccRCC.xml,nan,0,21,3,HP12.390.A5.ccRCC.scn,1106790912,18
beb064f0ee611ce72f0b9bb5c411e5d2213125d0,HP10.2986,ccRCC,pre/ccRCC/HP10.2986_A4_ccRCC.scn,HP10.2986_A4_ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP10.2986_A4_ccRCC.xml,nan,0,21,21,HP10.2986_A4_ccRCC.scn,833342700,0
281a0f53b7f22e07fd5b72d26b6f641b520f8299,HP15.12550,ccRCC,pre/ccRCC/HP15.12550.A1.ccRCC.scn,HP15.12550.A1.ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP15.12550.A1.ccRCC.xml,nan,0,7,3,HP15.12550.A1.ccRCC.scn,746920644,4
32b47efd3a77dec2c7ebca684f9e7447083d92dd,HP13.1799,ccRCC,pre/ccRCC/HP13.1799.B2.ccRCC.scn,HP13.1799.B2.ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP13.1799.B2.ccRCC.xml,nan,0,4,0,HP13.1799.B2.ccRCC.scn,1181953034,4
b1fe4945a8a5dbb0a75d0d0167c0274c53c78029,HP12.6073,ccRCC,pre/ccRCC/HP12.6073.A5-1.ccRCC.scn,HP12.6073.A5-1.ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP12.6073.A5-1.ccRCC.xml,nan,0,15,8,HP12.6073.A5-1.ccRCC.scn,889956856,7
2305e6f39e6452abf4c671c165c03ca4c445c0f9,HP15.12550,ccRCC,pre/ccRCC/HP15.12550.A6.ccRCC.scn,HP15.12550.A6.ccRCC.scn,ccRCC,pre/ccRCC,leica,40,0.25,0.25,106259,306939,6,HP15.12550.A6.ccRCC.xml,nan,0,11,8,HP15.12550.A6.ccRCC.scn,771117104,3
```

### File: `reports/02_parquet/slides.parquet`

_Parquet file (binary) – not inlined. Path: `reports/02_parquet/slides.parquet`_

