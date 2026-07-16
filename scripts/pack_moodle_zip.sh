#!/usr/bin/env bash
# Build a Moodle-ready code-only ZIP (no datasets, checkpoints, or large results).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${1:-$ROOT}"
NAME="AmitaiGal_ShalevShiriki_046211_code"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

mkdir -p "$STAGING/$NAME/proj"
# Copy project tree without heavy / generated paths
rsync -a \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'LOG/' \
  --exclude 'data/' \
  --exclude 'old_versions/' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.ckpt' \
  --exclude '*.parquet' \
  --exclude 'data_urls.json' \
  --exclude 'SUBMISSION.md' \
  --exclude '.cursor/' \
  --exclude 'AmitaiGal_ShalevShiriki_046211_code.zip' \
  "$ROOT/" "$STAGING/$NAME/proj/project/"

# Tiny placeholder so graders know where data goes
mkdir -p "$STAGING/$NAME/proj/project/data/processed/combined_filtered_smart_coastal"
cat > "$STAGING/$NAME/proj/project/data/processed/README.md" <<'EOF'
Processed datasets are **not** shipped in the Moodle ZIP.

Run from `proj/project`:

```bash
cp data_urls.example.json data_urls.json   # paste Google Drive links
python scripts/download_processed_data.py
```

Or rebuild from public NOAA AIS (README §5).
EOF

ZIP_PATH="$OUT_DIR/${NAME}.zip"
rm -f "$ZIP_PATH"
python3 - <<PY
import zipfile
from pathlib import Path
staging = Path("$STAGING")
zip_path = Path("$ZIP_PATH")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in staging.rglob("*"):
        if p.is_file():
            zf.write(p, p.relative_to(staging).as_posix())
print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
PY
echo "Contents root: $NAME/proj/project/  →  export PYTHONPATH=\$PWD/$NAME"
