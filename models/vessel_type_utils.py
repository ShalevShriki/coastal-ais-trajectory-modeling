"""AIS vessel type lookup and coarse class encoding for gate conditioning."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Coarse AIS ship-type buckets (NOAA numeric codes).
VESSEL_CLASS_NAMES: dict[int, str] = {
    0: "unknown",
    1: "passenger",  # 60-69 ferries, cruise, etc.
    2: "cargo",      # 70-79
    3: "tanker",     # 80-89
    4: "service",    # 50-59 tug, pilot, law enforcement
    5: "fishing",    # 30-39
    6: "misc",       # 20-29
    7: "hsc",        # 40-49 high-speed craft
    8: "other",
}
NUM_VESSEL_CLASSES = len(VESSEL_CLASS_NAMES)


def ais_code_to_class(vessel_type: float | int | None) -> int:
    if vessel_type is None or (isinstance(vessel_type, float) and np.isnan(vessel_type)):
        return 0
    code = int(vessel_type)
    if 60 <= code <= 69:
        return 1
    if 70 <= code <= 79:
        return 2
    if 80 <= code <= 89:
        return 3
    if 50 <= code <= 59:
        return 4
    if 30 <= code <= 39:
        return 5
    if 20 <= code <= 29:
        return 6
    if 40 <= code <= 49:
        return 7
    return 8


def default_lookup_path(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[1]
    return root / "data/processed/vessel_type_mmsi_lookup.parquet"


def build_mmsi_vessel_type_lookup(
    processed_root: Path,
    *,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Scan coastal_segments.parquet files and build one type per MMSI (mode)."""
    frames: list[pd.DataFrame] = []
    for segments_path in sorted(processed_root.rglob("coastal_segments.parquet")):
        df = pd.read_parquet(segments_path, columns=["mmsi", "vessel_type"])
        frames.append(df.dropna(subset=["mmsi"]))

    if not frames:
        raise FileNotFoundError(f"No coastal_segments.parquet under {processed_root}")

    combined = pd.concat(frames, ignore_index=True)
    combined["mmsi"] = combined["mmsi"].astype(np.int64)
    combined["vessel_type"] = pd.to_numeric(combined["vessel_type"], errors="coerce")

    def _mode_type(series: pd.Series) -> float:
        counts = series.dropna().astype(int).value_counts()
        if counts.empty:
            return np.nan
        return float(counts.index[0])

    lookup = (
        combined.groupby("mmsi", as_index=False)["vessel_type"]
        .agg(_mode_type)
        .rename(columns={"vessel_type": "vessel_type_code"})
    )
    lookup["vessel_class"] = lookup["vessel_type_code"].map(ais_code_to_class).astype(np.int64)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lookup.to_parquet(output_path, index=False)

    return lookup


def load_vessel_type_lookup(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Vessel type lookup not found: {path}")
    return pd.read_parquet(path)


def vessel_class_indices_for_mmsi(
    mmsi: np.ndarray,
    lookup: pd.DataFrame,
) -> np.ndarray:
    """Map window MMSI array to coarse vessel-class indices (0 = unknown)."""
    lut = lookup.set_index("mmsi")["vessel_class"]
    mmsi_series = pd.Series(mmsi.astype(np.int64))
    mapped = mmsi_series.map(lut).fillna(0).astype(np.int64).to_numpy()
    return mapped


def resolve_vessel_type_lookup(
    *,
    project_root: Path | None = None,
    lookup_path: Path | None = None,
    rebuild: bool = False,
) -> pd.DataFrame:
    root = project_root or Path(__file__).resolve().parents[1]
    path = lookup_path or default_lookup_path(root)
    if rebuild or not path.is_file():
        print(f"Building vessel type lookup -> {path}")
        return build_mmsi_vessel_type_lookup(root / "data/processed", output_path=path)
    return load_vessel_type_lookup(path)
