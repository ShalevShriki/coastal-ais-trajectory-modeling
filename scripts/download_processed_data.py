#!/usr/bin/env python3
"""Download the processed coastal training artifacts used by exp_coastal.

Moodle / graders: this repo ships **code only**. The ~2.8 GB coastal parquet and
the land-penalty grid must be fetched once from cloud URLs listed in
``data_urls.json`` (see ``data_urls.example.json``).

Usage:
  cp data_urls.example.json data_urls.json   # then paste your Drive links
  python scripts/download_processed_data.py

Requires: pip install gdown requests
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_URLS = PROJECT / "data_urls.json"
OUT_PARQUET = PROJECT / "data/processed/combined_filtered_smart_coastal/train.parquet"
OUT_LAND = PROJECT / "data/processed/land_grid_us.npz"


def _drive_file_id(url: str) -> str | None:
    """Extract a Google Drive file id from common share / uc URL forms."""
    if not url:
        return None
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "id" in qs and qs["id"]:
        return qs["id"][0]
    parts = [p for p in parsed.path.split("/") if p]
    if "d" in parts:
        i = parts.index("d")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def download(url: str, dest: Path, force: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[skip] already exists: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
        return
    if not url or "YOUR_FILE_ID" in url:
        raise SystemExit(
            f"Missing real URL for {dest.name}. Edit data_urls.json "
            "(copy from data_urls.example.json) and paste Google Drive links."
        )

    print(f"[download] {url}\n       -> {dest}")
    file_id = _drive_file_id(url)
    try:
        import gdown  # type: ignore
    except ImportError:
        gdown = None

    if file_id and gdown is not None:
        # fuzzy=True handles Drive virus-scan interstitial for large files
        ok = gdown.download(id=file_id, output=str(dest), quiet=False, fuzzy=True)
        if not ok or not dest.exists():
            raise SystemExit(f"gdown failed for file id={file_id}")
        return

    # Generic HTTPS fallback (direct links / non-Drive hosts)
    import requests

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urls", type=Path, default=DEFAULT_URLS)
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument(
        "--from-scratch-hint",
        action="store_true",
        help="Print NOAA rebuild instructions and exit",
    )
    args = parser.parse_args()

    if args.from_scratch_hint:
        print(
            "Alternative to cloud download: rebuild from public NOAA AIS "
            "(multi-day). See README §5 Data pipeline:\n"
            "  processing/INCREMENTAL_PROCESS *.py\n"
            "  scripts/combine_datasets.py\n"
            "  scripts/apply_training_filters.py\n"
            "  scripts/filter_inland_windows.py\n"
        )
        return

    if not args.urls.exists():
        example = PROJECT / "data_urls.example.json"
        raise SystemExit(
            f"Missing {args.urls}. Copy {example.name} → data_urls.json "
            "and paste Google Drive (or HTTPS) URLs for the processed files."
        )

    urls = json.loads(args.urls.read_text(encoding="utf-8"))
    download(urls.get("combined_filtered_smart_coastal_train_parquet", ""), OUT_PARQUET, args.force)
    download(urls.get("land_grid_us_npz", ""), OUT_LAND, args.force)
    print("Done. Point training at:")
    print(f"  --input {OUT_PARQUET.relative_to(PROJECT)}")
    print(f"  land grid: {OUT_LAND.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
