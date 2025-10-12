from __future__ import annotations
import sys, pandas as pd
from .config_ingest import OUT_MANIFEST

def main() -> int:
    try:
        df = pd.read_parquet(OUT_MANIFEST)
        assert df.shape[0] > 0, "empty manifest"
        assert df["patient_id"].notna().all(), "null patient_id"
        print(df.sample(min(5, len(df))))
        print("[OK] smoketest")
        return 0
    except Exception as e:
        print("[FAIL] smoketest:", e)
        return 3

if __name__ == "__main__":
    sys.exit(main())
