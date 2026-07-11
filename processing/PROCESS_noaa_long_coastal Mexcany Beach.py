from __future__ import annotations

import argparse
import io
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK_SIZE = 200_000

# ---------------------------------------------------------------------------
# Long-horizon coastal AIS settings
# ---------------------------------------------------------------------------

# Default window task: past 24 hours -> predict next 12 hours.
# All three values are CLI-overridable.
DEFAULT_HISTORY_HOURS = 24.0
DEFAULT_FUTURE_HOURS = 12.0
DEFAULT_RESAMPLE_MINUTES = 10

KNOTS_TO_KMH = 1.852
EARTH_RADIUS_KM = 6371.0
MAX_AIS_SOG_KMH = 102.2 * KNOTS_TO_KMH
MAX_IMPLAUSIBLE_KMH = 50.0 * KNOTS_TO_KMH
MAX_GAP_HOURS = 6.0

# Keep only long continuous segments.
MIN_POINTS_PER_SEGMENT = 50
MIN_SEGMENT_HOURS = 12.0
MIN_TOTAL_DISTANCE_KM = 10.0 * KNOTS_TO_KMH

# Default region: Mexican Pacific and Gulf coasts.
# Override by --region or explicit --lat-min/--lat-max/--lon-min/--lon-max.
DEFAULT_REGION = "mexican_coast"
PROCESSED_ROOT = Path("data/processed/Mexcany Beach")

REGIONS = {
    "mexican_coast": (14.0, 32.5, -118.0, -86.0),
    "mexico_pacific": (14.0, 32.5, -118.0, -105.0),
    "mexico_gulf": (18.0, 26.5, -98.0, -86.0),
}

DANISH_USECOLS = [
    "# Timestamp",
    "MMSI",
    "Latitude",
    "Longitude",
    "SOG",
    "COG",
    "Heading",
    "Ship type",
]

COMMERCIAL_SHIP_TYPES = {
    "Cargo",
    "Tanker",
    "Passenger",
    "Towing",
    "Tug",
    "Port tender",
    "Dredging",
}

# NOAA / AIS numeric type groups:
# 30-39 special/fishing/towing, 50-59 special craft/tug-like,
# 60-69 passenger, 70-79 cargo, 80-89 tanker.
COMMERCIAL_VESSEL_TYPE_CODES = set(range(30, 40)) | set(range(50, 90))


@dataclass(frozen=True)
class AISFormat:
    source: str
    usecols: list[str]
    rename: dict[str, str]
    timestamp_format: str | None


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# Input handling: CSV / CSV.ZST / NOAA ZIP / directory of files
# ---------------------------------------------------------------------------

def is_zst_path(path: Path) -> bool:
    return path.suffix == ".zst" or path.name.endswith(".csv.zst")


def is_zip_path(path: Path) -> bool:
    return path.suffix.lower() == ".zip"


def expand_inputs(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        files = []
        for pattern in ("*.csv", "*.csv.zst", "*.zip"):
            files.extend(input_path.glob(pattern))
        files = sorted(files)
        if not files:
            raise FileNotFoundError(
                f"No .csv/.csv.zst/.zip files found in {input_path}\n"
                "Place NOAA AIS files there, or pass --input to a specific file."
            )
        return files

    if not input_path.exists():
        raise FileNotFoundError(
            f"AIS input not found: {input_path}\n\n"
            "How to get data:\n"
            "  1) Download day-by-day (recommended):\n"
            "     python INCREMENTAL_PROCESS.py --start-date 2025-03-01 "
            "--end-date 2025-03-10 --region mexican_coast\n"
            "  2) Point to one local file:\n"
            "     python \"PROCESS_noaa_long_coastal Mexcany Beach.py\" "
            "--input data/temp/ais-2025-03-10.csv.zst --region mexican_coast\n"
            "  3) Put files under data/raw/noaa_ais/ and run:\n"
            "     python \"PROCESS_noaa_long_coastal Mexcany Beach.py\" "
            "--input data/raw/noaa_ais --region mexican_coast"
        )

    return [input_path]


def discover_ais_inputs() -> list[Path]:
    patterns = (
        "data/raw/noaa_ais/**/*.csv.zst",
        "data/raw/noaa_ais/**/*.csv",
        "data/raw/noaa_ais/**/*.zip",
        "data/temp/*.csv.zst",
        "data/temp/*.csv",
        "data/temp/*.zip",
        "data/processed/Mexcany Beach/days/**/*.parquet",
        "*.csv.zst",
        "ais-*.csv.zst",
        "aisdk-*.csv",
    )
    found: list[Path] = []
    for pattern in patterns:
        found.extend(Path(".").glob(pattern))
    return sorted({path.resolve() for path in found})


def resolve_input_path(input_arg: str | None) -> Path:
    if input_arg:
        return Path(input_arg)

    discovered = discover_ais_inputs()
    if not discovered:
        raise FileNotFoundError(
            "No AIS input provided and no .csv/.csv.zst/.zip files were found.\n\n"
            "Examples:\n"
            "  python INCREMENTAL_PROCESS.py --start-date 2025-03-01 "
            "--end-date 2025-03-10 --region mexican_coast\n"
            "  python \"PROCESS_noaa_long_coastal Mexcany Beach.py\" "
            "--input data/temp/ais-2025-03-10.csv.zst --region mexican_coast"
        )

    raw_files = [
        path
        for path in discovered
        if path.suffix.lower() in {".csv", ".zst", ".zip"} or path.name.endswith(".csv.zst")
    ]
    if len(raw_files) == 1:
        print(f"Auto-selected input: {raw_files[0]}", flush=True)
        return raw_files[0]

    raw_dir = Path("data/raw/noaa_ais")
    if raw_dir.is_dir():
        try:
            files = expand_inputs(raw_dir)
            if files:
                print(f"Auto-selected input directory: {raw_dir} ({len(files)} files)", flush=True)
                return raw_dir
        except FileNotFoundError:
            pass

    if len(raw_files) > 1:
        listing = "\n".join(f"  - {path}" for path in raw_files[:10])
        extra = f"\n  ... and {len(raw_files) - 10} more" if len(raw_files) > 10 else ""
        raise FileNotFoundError(
            "Multiple AIS files found. Pass --input explicitly:\n"
            f"{listing}{extra}"
        )

    raise FileNotFoundError(
        "No AIS raw files found. Only processed parquet exists.\n"
        "Download or provide a .csv/.csv.zst file with --input."
    )


def open_text_stream(path: Path):
    """
    Opens CSV or CSV.ZST as text.
    ZIP is handled separately by iter_csv_chunks().
    """
    if is_zst_path(path):
        import zstandard as zstd

        handle = path.open("rb")
        reader = zstd.ZstdDecompressor().stream_reader(handle)
        return handle, io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace"), None


def peek_header(path: Path) -> list[str]:
    if is_zip_path(path):
        with zipfile.ZipFile(path) as zf:
            csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV file found inside {path}")
            with zf.open(csv_names[0]) as raw:
                first = raw.readline().decode("utf-8", errors="replace")
                return first.strip().split(",")

    raw_handle, text_handle = open_text_stream(path)
    try:
        stream = text_handle if text_handle is not None else raw_handle
        return stream.readline().strip().split(",")
    finally:
        if text_handle is not None:
            text_handle.close()
        raw_handle.close()


def iter_csv_chunks(path: Path, usecols: list[str], chunksize: int):
    if is_zip_path(path):
        with zipfile.ZipFile(path) as zf:
            csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV file found inside {path}")
            with zf.open(csv_names[0]) as raw:
                yield from pd.read_csv(raw, usecols=usecols, chunksize=chunksize, low_memory=False)
        return

    compression = "zstd" if is_zst_path(path) else None
    yield from pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunksize,
        compression=compression,
        low_memory=False,
    )


def count_input_rows(path: Path) -> int | None:
    """
    Counting ZIP rows cheaply is not worth it here, so return None for ZIP.
    Progress is still printed per chunk.
    """
    if is_zip_path(path):
        return None

    print(f"Counting rows in {path.name}...", flush=True)
    start = time.perf_counter()
    lines = 0
    raw_handle, text_handle = open_text_stream(path)
    try:
        stream = text_handle if text_handle is not None else raw_handle
        for _ in stream:
            lines += 1
    finally:
        if text_handle is not None:
            text_handle.close()
        raw_handle.close()

    rows = max(lines - 1, 0)
    print(f"Found {rows:,} rows in {format_duration(time.perf_counter() - start)}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Format detection and cleaning
# ---------------------------------------------------------------------------

def detect_format(path: Path, source: str = "auto") -> AISFormat:
    if source == "danish":
        return AISFormat(
            source="danish",
            usecols=DANISH_USECOLS,
            rename={
                "# Timestamp": "timestamp",
                "MMSI": "mmsi",
                "Latitude": "lat",
                "Longitude": "lon",
                "SOG": "sog",
                "COG": "cog",
                "Heading": "heading",
                "Ship type": "vessel_type",
            },
            timestamp_format="%d/%m/%Y %H:%M:%S",
        )

    header = peek_header(path)
    lookup = {col.lower().replace(" ", "_"): col for col in header}

    if source == "noaa" or "base_date_time" in lookup or "basedatetime" in lookup:
        timestamp_col = lookup.get("base_date_time") or lookup.get("basedatetime")
        lat_col = lookup.get("latitude") or lookup.get("lat")
        lon_col = lookup.get("longitude") or lookup.get("lon")
        vessel_type_col = lookup.get("vessel_type") or lookup.get("vesseltype")

        required = {
            "mmsi": lookup.get("mmsi"),
            "timestamp": timestamp_col,
            "lat": lat_col,
            "lon": lon_col,
            "sog": lookup.get("sog"),
            "cog": lookup.get("cog"),
            "heading": lookup.get("heading"),
            "vessel_type": vessel_type_col,
        }
        missing = [name for name, col in required.items() if col is None]
        if missing:
            raise ValueError(f"NOAA format detected but missing columns: {missing}")

        usecols = list(required.values())
        return AISFormat(
            source="noaa",
            usecols=usecols,
            rename={col: name for name, col in required.items()},
            timestamp_format=None,
        )

    if "#_timestamp" in lookup or any(col.startswith("#") for col in lookup):
        return AISFormat(
            source="danish",
            usecols=DANISH_USECOLS,
            rename={
                "# Timestamp": "timestamp",
                "MMSI": "mmsi",
                "Latitude": "lat",
                "Longitude": "lon",
                "SOG": "sog",
                "COG": "cog",
                "Heading": "heading",
                "Ship type": "vessel_type",
            },
            timestamp_format="%d/%m/%Y %H:%M:%S",
        )

    raise ValueError(
        f"Could not detect AIS format for {path.name}. Use --source noaa or --source danish."
    )


def normalize_chunk(df: pd.DataFrame, ais_format: AISFormat) -> pd.DataFrame:
    clean = df.rename(columns=ais_format.rename).copy()

    if ais_format.timestamp_format:
        clean["timestamp"] = pd.to_datetime(
            clean["timestamp"],
            format=ais_format.timestamp_format,
            errors="coerce",
        )
    else:
        clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce", utc=True)
        clean["timestamp"] = clean["timestamp"].dt.tz_convert(None)

    for col in ("mmsi", "lat", "lon", "sog", "cog", "heading"):
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    if ais_format.source == "danish":
        clean["vessel_type"] = clean["vessel_type"].astype(str).str.strip()
    else:
        clean["vessel_type"] = pd.to_numeric(clean["vessel_type"], errors="coerce")

    # AIS heading 511 usually means unavailable.
    clean.loc[clean["heading"] == 511, "heading"] = np.nan

    # Missing heading can be handled later with a missing flag.
    # AIS SOG is reported in knots; store everything in km/h.
    clean["sog"] = clean["sog"].fillna(0) * KNOTS_TO_KMH
    clean["cog"] = clean["cog"].fillna(clean["heading"]).fillna(0)

    return clean


def is_commercial(df: pd.DataFrame, ais_format: AISFormat) -> pd.Series:
    if ais_format.source == "danish":
        return df["vessel_type"].isin(COMMERCIAL_SHIP_TYPES)
    return df["vessel_type"].isin(COMMERCIAL_VESSEL_TYPE_CODES)


def filter_chunk(
    df: pd.DataFrame,
    *,
    commercial_only: bool,
    ais_format: AISFormat,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> pd.DataFrame:
    mask = (
        df["timestamp"].notna()
        & df["mmsi"].notna()
        & (df["mmsi"] > 0)
        & df["lat"].between(*lat_range)
        & df["lon"].between(*lon_range)
        & df["sog"].between(0, MAX_AIS_SOG_KMH)
        & df["cog"].between(0, 360)
    )

    if commercial_only:
        mask = mask & is_commercial(df, ais_format)

    return df.loc[mask].drop_duplicates(subset=["mmsi", "timestamp"]).copy()


def days_dir(region_label: str) -> Path:
    return PROCESSED_ROOT / "days" / region_label


def state_file(region_label: str) -> Path:
    return PROCESSED_ROOT / f"processed_days_{region_label}.txt"


def ensure_dirs() -> tuple[Path, Path]:
    interim_dir = Path("data/interim")
    interim_dir.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    return interim_dir, PROCESSED_ROOT


def write_cleaned_parquet(
    input_paths: list[Path],
    output_path: Path,
    commercial_only: bool,
    *,
    source: str,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> int:
    if output_path.exists():
        output_path.unlink()

    writer: pq.ParquetWriter | None = None
    total_kept = 0

    for file_idx, input_path in enumerate(input_paths, start=1):
        ais_format = detect_format(input_path, source=source)
        print(
            f"\nInput {file_idx}/{len(input_paths)}: {input_path.name} | "
            f"source={ais_format.source.upper()} | "
            f"lat={lat_range[0]}..{lat_range[1]} | lon={lon_range[0]}..{lon_range[1]}",
            flush=True,
        )

        total_rows = count_input_rows(input_path)
        rows_read = 0
        rows_kept = 0
        start = time.perf_counter()

        for chunk in iter_csv_chunks(input_path, ais_format.usecols, CHUNK_SIZE):
            rows_read += len(chunk)

            clean = filter_chunk(
                normalize_chunk(chunk, ais_format),
                commercial_only=commercial_only,
                ais_format=ais_format,
                lat_range=lat_range,
                lon_range=lon_range,
            )

            if clean.empty:
                continue

            table = pa.Table.from_pandas(clean, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)

            rows_kept += len(clean)
            total_kept += len(clean)

            if total_rows:
                pct = rows_read / total_rows * 100
                print(
                    f"\rCleaning {input_path.name}: {pct:5.1f}% | "
                    f"read {rows_read:,}/{total_rows:,} | kept {rows_kept:,}",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"\rCleaning {input_path.name}: read {rows_read:,} | kept {rows_kept:,}",
                    end="",
                    flush=True,
                )

        print(
            f"\nDone {input_path.name}: kept {rows_kept:,}/{rows_read:,} rows "
            f"in {format_duration(time.perf_counter() - start)}",
            flush=True,
        )

    if writer is not None:
        writer.close()

    if total_kept == 0:
        raise ValueError(
            "No valid rows remained after cleaning. "
            "Try a wider region, more days, or --all-vessels."
        )

    print(f"\nCleaned parquet saved: {output_path} | total kept rows: {total_kept:,}", flush=True)
    return total_kept


# ---------------------------------------------------------------------------
# Features, segmentation, and model-ready sequence windows
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def circular_diff_deg(current: pd.Series, previous: pd.Series) -> pd.Series:
    raw = current - previous
    return ((raw + 180) % 360) - 180


def add_motion_features_and_segments(
    cleaned_df: pd.DataFrame,
    *,
    max_gap_hours: float,
    max_speed_kmh: float,
    min_points_per_segment: int,
    min_segment_hours: float,
    min_total_distance_km: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = time.perf_counter()
    print("\nBuilding motion features and trajectory segments...", flush=True)

    df = cleaned_df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)
    grouped = df.groupby("mmsi", sort=False)

    df["prev_timestamp"] = grouped["timestamp"].shift(1)
    df["prev_lat"] = grouped["lat"].shift(1)
    df["prev_lon"] = grouped["lon"].shift(1)
    df["prev_sog"] = grouped["sog"].shift(1)
    df["prev_cog"] = grouped["cog"].shift(1)

    df["dt_sec"] = (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds()
    df["dt_hours"] = df["dt_sec"] / 3600.0

    df["dist_km"] = haversine_km(df["prev_lat"], df["prev_lon"], df["lat"], df["lon"])
    df["speed_between_kmh"] = df["dist_km"] / df["dt_hours"].replace(0, np.nan)

    df["dlat"] = df["lat"] - df["prev_lat"]
    df["dlon"] = df["lon"] - df["prev_lon"]
    df["dsog"] = df["sog"] - df["prev_sog"]
    df["dcog"] = circular_diff_deg(df["cog"], df["prev_cog"])

    cog_rad = np.deg2rad(df["cog"])
    df["cog_sin"] = np.sin(cog_rad)
    df["cog_cos"] = np.cos(cog_rad)
    df["v_north_kmh"] = df["sog"] * df["cog_cos"]
    df["v_east_kmh"] = df["sog"] * df["cog_sin"]

    df["heading_missing"] = df["heading"].isna().astype(int)
    df["heading"] = df["heading"].fillna(df["cog"])
    heading_rad = np.deg2rad(df["heading"])
    df["heading_sin"] = np.sin(heading_rad)
    df["heading_cos"] = np.cos(heading_rad)

    new_segment = (
        df["dt_sec"].isna()
        | (df["dt_sec"] <= 0)
        | (df["dt_hours"] > max_gap_hours)
        | (~np.isfinite(df["speed_between_kmh"]))
        | (df["speed_between_kmh"] > max_speed_kmh)
    )

    df["new_segment"] = new_segment.astype(int)
    df["segment_id"] = df.groupby("mmsi")["new_segment"].cumsum()
    df["traj_id"] = (
        df["mmsi"].astype("int64").astype(str)
        + "_"
        + df["segment_id"].astype("int64").astype(str)
    )

    # Remove invalid rows after segment id was assigned.
    df = df[
        df["dt_sec"].notna()
        & (df["dt_sec"] > 0)
        & np.isfinite(df["speed_between_kmh"])
        & (df["speed_between_kmh"] <= max_speed_kmh)
    ].copy()

    seg_stats = df.groupby("traj_id").agg(
        mmsi=("mmsi", "first"),
        n_points=("timestamp", "size"),
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        mean_sog=("sog", "mean"),
        median_dt_sec=("dt_sec", "median"),
        total_distance_km=("dist_km", "sum"),
    )
    seg_stats["duration_hours"] = (
        seg_stats["end_time"] - seg_stats["start_time"]
    ).dt.total_seconds() / 3600.0

    keep = (
        (seg_stats["n_points"] >= min_points_per_segment)
        & (seg_stats["duration_hours"] >= min_segment_hours)
        & (seg_stats["total_distance_km"] >= min_total_distance_km)
    )

    valid_traj = seg_stats.index[keep]
    df = df[df["traj_id"].isin(valid_traj)].copy()
    seg_stats = seg_stats.loc[valid_traj].sort_values(
        ["duration_hours", "n_points"],
        ascending=False,
    )

    print(
        f"Segments kept: {len(seg_stats):,} | rows kept: {len(df):,} | "
        f"runtime {format_duration(time.perf_counter() - start)}",
        flush=True,
    )

    if df.empty:
        raise ValueError(
            "No valid trajectory segments remained. "
            "Lower --min-segment-hours / --min-points-per-segment, "
            "increase date range, or widen region."
        )

    return df.reset_index(drop=True), seg_stats


WINDOW_FEATURE_COLS = [
    "lat",
    "lon",
    "sog",
    "cog_sin",
    "cog_cos",
    "heading_sin",
    "heading_cos",
    "heading_missing",
    "dt_sec",
    "dlat",
    "dlon",
    "dsog",
    "dcog",
    "v_north_kmh",
    "v_east_kmh",
]

TARGET_COLS = ["lat", "lon"]


def resample_one_trajectory(
    traj: pd.DataFrame,
    every_minutes: int,
    max_gap_steps: int = 6,
) -> pd.DataFrame:
    """
    Resample one trajectory to a regular time grid.

    Numeric fields are time-interpolated up to max_gap_steps consecutive missing
    steps (default 6 = 1 hour at 10-min grid). Longer gaps stay NaN and are
    dropped, preventing the model from training on fabricated positions.
    Vessel metadata is forward-filled.
    """
    traj = traj.sort_values("timestamp").drop_duplicates("timestamp").copy()
    traj = traj.set_index("timestamp")

    # Deduplicate: TARGET_COLS (lat/lon) are already in WINDOW_FEATURE_COLS.
    # Without this, pd.concat creates duplicate lat/lon columns and to_numpy()
    # silently flattens / mis-aligns, corrupting windows (lon copied from lat).
    numeric_cols = list(dict.fromkeys(
        col for col in WINDOW_FEATURE_COLS + TARGET_COLS
        if col in traj.columns
    ))
    meta_cols = [col for col in ["mmsi", "traj_id", "vessel_type"] if col in traj.columns]

    numeric = traj[numeric_cols].resample(f"{every_minutes}min").mean().interpolate(
        method="time",
        limit_direction="both",
        limit=max_gap_steps,
    )
    meta = traj[meta_cols].resample(f"{every_minutes}min").ffill().bfill()

    out = pd.concat([meta, numeric], axis=1).dropna(subset=["lat", "lon"])
    out = out.reset_index()
    return out


def build_sequence_windows(
    segmented_df: pd.DataFrame,
    *,
    history_hours: float,
    future_hours: float,
    resample_minutes: int,
    max_windows_per_traj: int | None = None,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    max_gap_steps: int = 6,
    output_path: Path | None = None,
    flush_rows: int = 20_000,
) -> pd.DataFrame | int:
    """Build fixed-length windows.

    If ``output_path`` is given, windows are streamed to parquet in batches and
    the total sample count (int) is returned. This keeps peak memory bounded for
    very large coasts. Otherwise the full DataFrame is returned (legacy behavior).
    """
    start = time.perf_counter()
    print("\nBuilding fixed-length sequence windows...", flush=True)

    history_steps = int(round(history_hours * 60 / resample_minutes))
    future_steps = int(round(future_hours * 60 / resample_minutes))
    total_steps = history_steps + future_steps

    if history_steps <= 0 or future_steps <= 0:
        raise ValueError("history_steps and future_steps must be positive.")

    streaming = output_path is not None
    rows: list[dict] = []
    rng = np.random.default_rng(42)

    writer: pq.ParquetWriter | None = None
    schema_columns: list[str] | None = None
    total_written = 0

    def flush_batch(force: bool = False) -> None:
        nonlocal writer, schema_columns, total_written, rows
        if not rows:
            return
        if not force and len(rows) < flush_rows:
            return
        batch_df = pd.DataFrame(rows)
        if schema_columns is None:
            schema_columns = list(batch_df.columns)
        else:
            batch_df = batch_df.reindex(columns=schema_columns)
        table = pa.Table.from_pandas(batch_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
        else:
            table = table.cast(writer.schema)
        writer.write_table(table)
        total_written += len(rows)
        rows = []

    grouped = segmented_df.groupby("traj_id", sort=False)
    total_traj = segmented_df["traj_id"].nunique()

    for idx, (traj_id, traj) in enumerate(grouped, start=1):
        regular = resample_one_trajectory(traj, resample_minutes, max_gap_steps=max_gap_steps)

        if len(regular) < total_steps:
            continue

        possible_starts = np.arange(0, len(regular) - total_steps + 1)

        if max_windows_per_traj and len(possible_starts) > max_windows_per_traj:
            possible_starts = rng.choice(possible_starts, size=max_windows_per_traj, replace=False)
            possible_starts = np.sort(possible_starts)

        feature_values = regular[WINDOW_FEATURE_COLS].to_numpy(dtype=np.float32)
        target_values = regular[TARGET_COLS].to_numpy(dtype=np.float32)

        for start_i in possible_starts:
            hist = feature_values[start_i : start_i + history_steps]
            fut = target_values[start_i + history_steps : start_i + total_steps]

            # Skip windows where any predicted position exits the geographic frame.
            if lat_range is not None and lon_range is not None:
                if (
                    fut[:, 0].min() < lat_range[0] or fut[:, 0].max() > lat_range[1]
                    or fut[:, 1].min() < lon_range[0] or fut[:, 1].max() > lon_range[1]
                ):
                    continue

            row = {
                "traj_id": traj_id,
                "mmsi": int(regular["mmsi"].iloc[start_i]),
                "start_time": regular["timestamp"].iloc[start_i],
                "split_time": regular["timestamp"].iloc[start_i + history_steps - 1],
                "target_end_time": regular["timestamp"].iloc[start_i + total_steps - 1],
                "history_steps": history_steps,
                "future_steps": future_steps,
                "resample_minutes": resample_minutes,
            }

            for t in range(history_steps):
                for f_idx, col in enumerate(WINDOW_FEATURE_COLS):
                    row[f"x_t{t:03d}_{col}"] = float(hist[t, f_idx])

            for t in range(future_steps):
                row[f"y_t{t:03d}_lat"] = float(fut[t, 0])
                row[f"y_t{t:03d}_lon"] = float(fut[t, 1])

            rows.append(row)

        if streaming:
            flush_batch()

        if idx == 1 or idx == total_traj or idx % 50 == 0:
            seen = total_written + len(rows)
            print(
                f"\rWindows: trajectories {idx:,}/{total_traj:,} | rows {seen:,}",
                end="",
                flush=True,
            )

    print()

    if streaming:
        flush_batch(force=True)
        if writer is not None:
            writer.close()
        if total_written == 0:
            raise ValueError(
                "No sequence windows could be created. "
                "Use smaller --history-hours/--future-hours, smaller --resample-minutes, "
                "or keep longer segments."
            )
        print(
            f"Window build done (streamed): {total_written:,} samples | "
            f"history_steps={history_steps}, future_steps={future_steps}, "
            f"runtime {format_duration(time.perf_counter() - start)}",
            flush=True,
        )
        return total_written

    if not rows:
        raise ValueError(
            "No sequence windows could be created. "
            "Use smaller --history-hours/--future-hours, smaller --resample-minutes, "
            "or keep longer segments."
        )

    windows = pd.DataFrame(rows)
    print(
        f"Window build done: {len(windows):,} samples | "
        f"history_steps={history_steps}, future_steps={future_steps}, "
        f"runtime {format_duration(time.perf_counter() - start)}",
        flush=True,
    )
    return windows


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def resolve_geo_args(args: argparse.Namespace) -> tuple[tuple[float, float], tuple[float, float], str]:
    if args.lat_min is not None:
        needed = [args.lat_min, args.lat_max, args.lon_min, args.lon_max]
        if any(v is None for v in needed):
            raise ValueError(
                "For custom bounds, provide all of --lat-min --lat-max --lon-min --lon-max."
            )
        return (args.lat_min, args.lat_max), (args.lon_min, args.lon_max), "custom"

    if args.region not in REGIONS:
        raise ValueError(f"Unknown region {args.region}. Options: {sorted(REGIONS)}")

    lat_min, lat_max, lon_min, lon_max = REGIONS[args.region]
    return (lat_min, lat_max), (lon_min, lon_max), args.region


def save_summary(
    output_dir: Path,
    args: argparse.Namespace,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    segmented_df: pd.DataFrame,
    seg_stats: pd.DataFrame,
    window_count: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "processing_summary.txt"

    lines = [
        "AIS coastal long-horizon processing summary",
        "============================================",
        "",
        f"input: {getattr(args, 'input', 'incremental days')}",
        f"source: {args.source}",
        f"region: {args.region}",
        f"lat_range: {lat_range}",
        f"lon_range: {lon_range}",
        f"commercial_only: {not args.all_vessels}",
        "",
        f"segmented rows: {len(segmented_df):,}",
        f"unique vessels: {segmented_df['mmsi'].nunique():,}",
        f"valid trajectories: {segmented_df['traj_id'].nunique():,}",
        "",
        f"max_gap_hours: {args.max_gap_hours}",
        f"max_implausible_kmh: {args.max_implausible_kmh}",
        f"min_points_per_segment: {args.min_points_per_segment}",
        f"min_segment_hours: {args.min_segment_hours}",
        f"min_total_distance_km: {args.min_total_distance_km}",
        "",
    ]

    if window_count is not None:
        lines += [
            f"window samples: {window_count:,}",
            f"history_hours: {args.history_hours}",
            f"future_hours: {args.future_hours}",
            f"resample_minutes: {args.resample_minutes}",
            "",
        ]

    if len(seg_stats):
        lines += [
            "Trajectory duration quantiles [hours]:",
            str(seg_stats["duration_hours"].quantile([0.5, 0.75, 0.9, 0.95, 0.99])),
            "",
            "Top trajectories:",
            str(seg_stats.head(20)),
            "",
        ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved summary: {summary_path}", flush=True)


def finalize_coastal_pipeline(
    cleaned: pd.DataFrame,
    *,
    region_label: str,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    max_gap_hours: float,
    max_implausible_kmh: float,
    min_points_per_segment: int,
    min_segment_hours: float,
    min_total_distance_km: float,
    history_hours: float,
    future_hours: float,
    resample_minutes: int,
    max_windows_per_traj: int | None,
    no_windows: bool,
    summary_args: argparse.Namespace | None = None,
) -> Path:
    _, processed_dir = ensure_dirs()
    output_dir = processed_dir / f"ais_{region_label}_long_horizon"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Segmentation + feature engineering...", flush=True)
    segmented_df, seg_stats = add_motion_features_and_segments(
        cleaned,
        max_gap_hours=max_gap_hours,
        max_speed_kmh=max_implausible_kmh,
        min_points_per_segment=min_points_per_segment,
        min_segment_hours=min_segment_hours,
        min_total_distance_km=min_total_distance_km,
    )

    segments_path = output_dir / "coastal_segments.parquet"
    stats_path = output_dir / "trajectory_stats.csv"
    segmented_df.to_parquet(segments_path, index=False)
    seg_stats.to_csv(stats_path)
    print(f"Saved segments: {segments_path}", flush=True)
    print(f"Saved stats:    {stats_path}", flush=True)

    window_count: int | None = None
    if not no_windows:
        print("Building long-horizon model windows (streaming)...", flush=True)
        windows_path = output_dir / "model_ready_windows.parquet"
        window_count = build_sequence_windows(
            segmented_df,
            history_hours=history_hours,
            future_hours=future_hours,
            resample_minutes=resample_minutes,
            max_windows_per_traj=max_windows_per_traj,
            output_path=windows_path,
        )
        print(f"Saved model windows: {windows_path}", flush=True)
        final_path = windows_path
    else:
        final_path = segments_path

    if summary_args is not None:
        save_summary(
            output_dir=output_dir,
            args=summary_args,
            lat_range=lat_range,
            lon_range=lon_range,
            segmented_df=segmented_df,
            seg_stats=seg_stats,
            window_count=window_count,
        )

    return final_path


def add_coastal_argument_group(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--region",
        choices=sorted(REGIONS.keys()),
        default=DEFAULT_REGION,
        help="Mexican coastal region filter (default: full Pacific + Gulf coasts).",
    )
    parser.add_argument("--lat-min", type=float, default=None)
    parser.add_argument("--lat-max", type=float, default=None)
    parser.add_argument("--lon-min", type=float, default=None)
    parser.add_argument("--lon-max", type=float, default=None)
    parser.add_argument("--max-gap-hours", type=float, default=MAX_GAP_HOURS)
    parser.add_argument("--max-implausible-kmh", type=float, default=MAX_IMPLAUSIBLE_KMH)
    parser.add_argument("--min-points-per-segment", type=int, default=MIN_POINTS_PER_SEGMENT)
    parser.add_argument("--min-segment-hours", type=float, default=MIN_SEGMENT_HOURS)
    parser.add_argument("--min-total-distance-km", type=float, default=MIN_TOTAL_DISTANCE_KM)
    parser.add_argument("--history-hours", type=float, default=DEFAULT_HISTORY_HOURS)
    parser.add_argument("--future-hours", type=float, default=DEFAULT_FUTURE_HOURS)
    parser.add_argument("--resample-minutes", type=int, default=DEFAULT_RESAMPLE_MINUTES)
    parser.add_argument(
        "--max-windows-per-traj",
        type=int,
        default=200,
        help="Cap windows per trajectory. Use 0 for no cap.",
    )
    parser.add_argument("--no-windows", action="store_true", help="Only save segmented trajectories.")


def run_pipeline(args: argparse.Namespace) -> Path:
    pipeline_start = time.perf_counter()
    interim_dir, _ = ensure_dirs()

    lat_range, lon_range, region_label = resolve_geo_args(args)
    input_paths = expand_inputs(resolve_input_path(args.input))

    interim_path = interim_dir / f"cleaned_{region_label}.parquet"

    print("Step 1/3: Cleaning raw AIS input...", flush=True)
    write_cleaned_parquet(
        input_paths,
        interim_path,
        commercial_only=not args.all_vessels,
        source=args.source,
        lat_range=lat_range,
        lon_range=lon_range,
    )

    print("\nReading cleaned parquet...", flush=True)
    cleaned = pd.read_parquet(interim_path)

    print("Step 2-3/3: Segmentation and model windows...", flush=True)
    final_path = finalize_coastal_pipeline(
        cleaned,
        region_label=region_label,
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

    print(f"Total runtime: {format_duration(time.perf_counter() - pipeline_start)}", flush=True)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Process NOAA MarineCadastre AIS data for long-horizon coastal "
            "trajectory prediction on the Mexican coast."
        )
    )

    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Path to AIS .csv/.csv.zst/.zip file, or directory containing such files. "
            "If omitted, auto-discovers files under data/raw/noaa_ais/ or data/temp/."
        ),
    )
    parser.add_argument(
        "--source",
        choices=("auto", "noaa", "danish"),
        default="auto",
        help="AIS data source format.",
    )
    add_coastal_argument_group(parser)
    parser.add_argument("--all-vessels", action="store_true", help="Disable commercial-vessel filter.")

    args = parser.parse_args()
    if args.max_windows_per_traj == 0:
        args.max_windows_per_traj = None

    final_path = run_pipeline(args)

    print()
    print(f"Saved final output: {final_path}")
    if final_path.exists():
        df = pd.read_parquet(final_path)
        print(f"Rows/samples: {len(df):,}")


if __name__ == "__main__":
    main()
