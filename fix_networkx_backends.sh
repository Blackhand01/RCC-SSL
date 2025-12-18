#!/usr/bin/env bash
# Fix per: RuntimeWarning: networkx backend defined more than once: nx-loopback

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERRORE] Attiva prima il venv (es: source .venvs/xai-concept_pa-llava/bin/activate)" >&2
  exit 1
fi

PYBIN="${VIRTUAL_ENV}/bin/python"

"$PYBIN" - << 'PY'
from importlib import metadata as md
from pathlib import Path
import shutil
import datetime
import sys

def show_backends():
    print("=== NetworkX backends registrati ===")
    for dist in md.distributions():
        name = dist.metadata.get("Name")
        for ep in dist.entry_points:
            if ep.group == "networkx.backends":
                print(f"{name}  ->  {ep.name} = {ep.value}")

print("---- PRIMA ----")
show_backends()

dists = [d for d in md.distributions() if d.metadata.get("Name") == "networkx"]
if not dists:
    print("[INFO] Nessuna distribuzione networkx trovata in questo env.")
    sys.exit(0)

print(f"[INFO] Trovate {len(dists)} distribuzioni networkx")
primary = dists[0]
secondary = dists[1:]

def patch_primary(dist):
    ep_files = [f for f in (dist.files or []) if str(f).endswith("entry_points.txt")]
    for rel in ep_files:
        ep_path = Path(dist.locate_file(rel))
        if not ep_path.is_file():
            continue
        backup = ep_path.with_suffix(
            ep_path.suffix + ".bak." + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        shutil.copy2(ep_path, backup)
        print(f"[PRIMARY] Patch {ep_path} (backup {backup})")

        lines = ep_path.read_text(encoding="utf-8").splitlines()
        new_lines = []
        in_group = False
        seen_loopback = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                in_group = (stripped == "[networkx.backends]")
                if in_group:
                    seen_loopback = False
                new_lines.append(line)
                continue

            if in_group and stripped.startswith("nx-loopback"):
                if seen_loopback:
                    # dup nella stessa dist-info → scartalo
                    continue
                seen_loopback = True
                new_lines.append(line)
                continue

            new_lines.append(line)

        ep_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

def patch_secondary(dist):
    ep_files = [f for f in (dist.files or []) if str(f).endswith("entry_points.txt")]
    for rel in ep_files:
        ep_path = Path(dist.locate_file(rel))
        if not ep_path.is_file():
            continue
        backup = ep_path.with_suffix(
            ep_path.suffix + ".bak." + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        shutil.copy2(ep_path, backup)
        print(f"[SECONDARY] Patch {ep_path} (backup {backup})")

        lines = ep_path.read_text(encoding="utf-8").splitlines()
        new_lines = []
        in_backends = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                # nuova sezione
                if stripped == "[networkx.backends]":
                    in_backends = True
                    # NON scriviamo il titolo della sezione → sezione rimossa
                    continue
                else:
                    in_backends = False
                    new_lines.append(line)
                    continue

            if in_backends:
                # siamo dentro [networkx.backends] → scarta tutte le righe
                continue

            new_lines.append(line)

        ep_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

# sistema la prima dist come "sorgente" di nx-loopback
patch_primary(primary)

# tutte le altre: eliminano proprio la sezione [networkx.backends]
for dist in secondary:
    patch_secondary(dist)

print("---- DOPO ----")
show_backends()
PY

echo "[OK] Patch completata. Rilancia il tuo script Python."
