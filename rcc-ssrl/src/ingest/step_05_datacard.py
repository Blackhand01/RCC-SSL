from __future__ import annotations
from pathlib import Path
import pandas as pd, yaml
from .config_ingest import OUT_MANIFEST, OUT_DATACARD

def main() -> Path:
    df = pd.read_parquet(OUT_MANIFEST)
    summary = {
        "rows": int(df.shape[0]),
        "classes": {c:int(v) for c,v in df.groupby("class_label").size().items()},
        "by_source": {c:int(v) for c,v in df.groupby("source").size().items()},
        "qc_flags": {k:int(v) for k,v in df["notes"].value_counts(dropna=True).items()},
        "columns": list(df.columns),
    }
    OUT_DATACARD.parent.mkdir(parents=True, exist_ok=True)
    with OUT_DATACARD.open("w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    return OUT_DATACARD

if __name__ == "__main__":
    p = main()
    print(f"[OK] datacard → {p}")
