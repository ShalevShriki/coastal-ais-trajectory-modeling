from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests


def _parquet_row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows

_PROCESS_PATH = Path(__file__).with_name("PROCESS_noaa_long_coastal Mexcany Beach.py")
_spec = importlib.util.spec_from_file_location("process_mexican_coast", _PROCESS_PATH)
_process = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _process
_spec.loader.exec_module(_process)

add_coastal_argument_group = _process.add_coastal_argument_group
days_dir = _process.days_dir
ensure_dirs = _process.ensure_dirs
finalize_coastal_pipeline = _process.finalize_coastal_pipeline
format_duration = _process.format_duration
resolve_geo_args = _process.resolve_geo_args
state_file = _process.state_file
write_cleaned_parquet = _process.write_cleaned_parquet

NOAA_BASE_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler"
TEMP_DIR = Path("data/temp")


def resolve_dataset_label(args: argparse.Namespace) -> tuple[tuple[float, float], tuple[float, float], str]:
    lat_range, lon_range, region = resolve_geo_args(args)
    tag = getattr(args, "dataset_tag", None)
    label = f"{region}_{tag}" if tag else region
    return lat_range, lon_range, label


def iter_dates(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def day_tag(day: date) -> str:
    return day.strftime("%Y-%m-%d")


def download_url(day: date) -> str:
    return f"{NOAA_BASE_URL}/{day.year}/ais-{day_tag(day)}.csv.zst"


def archive_name(day: date) -> str:
    return f"ais-{day_tag(day)}.csv.zst"


def load_processed_days(region_label: str) -> set[str]:
    path = state_file(region_label)
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def mark_day_processed(day: date, region_label: str) -> None:
    path = state_file(region_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    processed = load_processed_days(region_label)
    processed.add(day_tag(day))
    path.write_text("\n".join(sorted(processed)) + "\n", encoding="utf-8")


def download_day(day: date) -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    zst_path = TEMP_DIR / archive_name(day)
    if zst_path.exists():
        return zst_path

    url = download_url(day)
    print(f"Downloading {url} ...", flush=True)
    start = time.perf_counter()
    with requests.get(url, stream=True, timeout=300) as response:
        if response.status_code == 404:
            raise FileNotFoundError(f"No AIS file for {day_tag(day)} at {url}")
        response.raise_for_status()
        with zst_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    print(
        f"Downloaded {zst_path.name} ({zst_path.stat().st_size / 1e9:.2f} GB) "
        f"in {format_duration(time.perf_counter() - start)}",
        flush=True,
    )
    return zst_path


def resolve_input_path(day: date, local_input: Path | None) -> Path:
    if local_input is not None:
        if not local_input.exists():
            raise FileNotFoundError(f"Local input not found: {local_input}")
        return local_input
    return download_day(day)


def delete_raw_files(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()
            print(f"Deleted raw file: {path}", flush=True)


def day_output_path(day: date, region_label: str) -> Path:
    return days_dir(region_label) / f"{day_tag(day)}.parquet"


def process_day(
    day: date,
    commercial_only: bool,
    *,
    region_label: str,
    source: str,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    local_input: Path | None = None,
    keep_raw: bool = False,
) -> int:
    tag = day_tag(day)
    output_path = day_output_path(day, region_label)
    if tag in load_processed_days(region_label) and output_path.exists():
        print(f"Skipping {tag} (already processed).", flush=True)
        return 0

    input_path = resolve_input_path(day, local_input)
    print(f"Processing {input_path.name} ...", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_kept = write_cleaned_parquet(
        [input_path],
        output_path,
        commercial_only,
        source=source,
        lat_range=lat_range,
        lon_range=lon_range,
    )

    if not keep_raw and local_input is None and input_path.exists():
        delete_raw_files(input_path)

    mark_day_processed(day, region_label)
    return rows_kept


def load_all_cleaned_days(region_label: str) -> pd.DataFrame:
    day_dir = days_dir(region_label)
    day_files = sorted(day_dir.glob("*.parquet"))
    if not day_files:
        raise FileNotFoundError(f"No day parquet files found in {day_dir}")
    frames = [pd.read_parquet(path) for path in day_files]
    return pd.concat(frames, ignore_index=True)


def finalize(args: argparse.Namespace) -> Path:
    lat_range, lon_range, dataset_label = resolve_dataset_label(args)
    print(f"Finalizing coastal long-horizon dataset [{dataset_label}]...", flush=True)

    cleaned = load_all_cleaned_days(dataset_label)
    day_count = len(list(days_dir(dataset_label).glob("*.parquet")))
    print(
        f"Cleaned rows from {day_count} day files: "
        f"{len(cleaned):,} | vessels: {cleaned['mmsi'].nunique():,}",
        flush=True,
    )

    return finalize_coastal_pipeline(
        cleaned,
        region_label=dataset_label,
        lat_range=lat_range,
        lon_range=lon_range,
        max_gap_hours=args.max_gap_hours,
        max_implausible_kmh=args.max_implausible_kmh,
        min_points_per_segment=args.min_points_per_segment,
        min_segment_hours=args.min_segment_hours,
        min_total_distance_km=args.min_total_distance_km,
        history_hours=args.history_hours,
        future_hours=args.future_hours,
        resample_minutes=args.resample_minutes,
        max_windows_per_traj=args.max_windows_per_traj,
        no_windows=args.no_windows,
        summary_args=args,
    )


def run_incremental(
    start_date: date,
    end_date: date,
    args: argparse.Namespace,
    *,
    finalize_at_end: bool,
    keep_raw: bool,
) -> None:
    ensure_dirs()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    lat_range, lon_range, dataset_label = resolve_dataset_label(args)
    commercial_only = not args.all_vessels
    pipeline_start = time.perf_counter()

    days = list(iter_dates(start_date, end_date))
    total_days = len(days)
    total_kept = 0

    print(
        f"Dataset: {dataset_label} | lat {lat_range} | lon {lon_range}",
        flush=True,
    )

    for idx, day in enumerate(days, start=1):
        print(f"\n=== Day {idx}/{total_days}: {day_tag(day)} ===", flush=True)
        rows_kept = process_day(
            day,
            commercial_only,
            region_label=dataset_label,
            source=args.source,
            lat_range=lat_range,
            lon_range=lon_range,
            keep_raw=keep_raw,
        )
        total_kept += rows_kept
        if rows_kept:
            print(f"Day kept rows: {rows_kept:,}", flush=True)

    print(f"\nIncremental cleaning done. Total kept this run: {total_kept:,}", flush=True)

    if finalize_at_end and any(days_dir(dataset_label).glob("*.parquet")):
        final_path = finalize(args)
        print(f"Final output: {final_path} | samples: {_parquet_row_count(final_path):,}", flush=True)

    print(f"Total runtime: {format_duration(time.perf_counter() - pipeline_start)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download NOAA AIS day-by-day for the Mexican coast, clean by region, "
            "and build long-horizon trajectory windows under data/processed/Mexcany Beach."
        )
    )
    parser.add_argument("--start-date", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end-date", help="End date YYYY-MM-DD.")
    parser.add_argument(
        "--input",
        help="Process one local .csv/.csv.zst/.zip file (requires --start-date).",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "noaa", "danish"),
        default="noaa",
        help="AIS data source format.",
    )
    add_coastal_argument_group(parser)
    parser.add_argument("--all-vessels", action="store_true", help="Disable commercial filter.")
    parser.add_argument(
        "--dataset-tag",
        help="Suffix for separate dataset storage, e.g. '4d' -> mexican_coast_4d.",
    )
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep downloaded raw files after processing.",
    )
    args = parser.parse_args()

    if args.max_windows_per_traj == 0:
        args.max_windows_per_traj = None

    if args.finalize_only:
        final_path = finalize(args)
        print(f"Final output: {final_path} | samples: {_parquet_row_count(final_path):,}")
        return

    if args.input:
        if not args.start_date:
            raise SystemExit("--input requires --start-date (the date of that file).")
        lat_range, lon_range, dataset_label = resolve_dataset_label(args)
        process_day(
            date.fromisoformat(args.start_date),
            commercial_only=not args.all_vessels,
            region_label=dataset_label,
            source=args.source,
            lat_range=lat_range,
            lon_range=lon_range,
            local_input=Path(args.input),
            keep_raw=args.keep_raw,
        )
        if not args.no_finalize and any(days_dir(dataset_label).glob("*.parquet")):
            final_path = finalize(args)
            print(f"Final output: {final_path} | samples: {_parquet_row_count(final_path):,}")
        return

    if not args.start_date or not args.end_date:
        raise SystemExit("Provide --start-date and --end-date, or use --input.")

    run_incremental(
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        args=args,
        finalize_at_end=not args.no_finalize,
        keep_raw=args.keep_raw,
    )


if __name__ == "__main__":
    main()
