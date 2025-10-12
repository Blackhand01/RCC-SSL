"""One-shot orchestrator: esegue tutti gli step in sequenza.
Usage: python -m ingest.run_all_ingest
"""
from __future__ import annotations
import sys
from . import step_01_scan as s1
from . import step_03_build_manifest as s3
from . import step_04_validate as s4
from . import step_05_datacard as s5
from . import step_06_smoketest as s6

def main() -> int:
    print("[1/5] scan files…")
    lookup = s1.main()

    print("[2/5] build manifest…")
    df = s3.main(lookup)
    print(f"[INFO] manifest rows: {len(df)}")

    print("[3/5] validate manifest…")
    rc = s4.main()
    if rc != 0:
        return rc

    print("[4/5] write data card…")
    s5.main()

    print("[5/5] smoketest…")
    return s6.main()

if __name__ == "__main__":
    sys.exit(main())
