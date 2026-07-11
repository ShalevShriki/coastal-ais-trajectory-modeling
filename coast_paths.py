from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATASET_PREFIX = "ais_"
DATASET_SUFFIX = "_long_horizon"


@dataclass(frozen=True)
class CoastConfig:
    name: str
    default_region: str
    processed_root: Path
    results_root: Path
    visualizations_root: Path
    regions: dict[str, tuple[float, float, float, float]]


COAST_CONFIGS: dict[str, CoastConfig] = {
    "USA Combined": CoastConfig(
        name="USA Combined",
        default_region="usa_combined",
        processed_root=Path("data/processed/combined"),
        results_root=Path("data/results/USA Combined"),
        visualizations_root=Path("data/visualizations/USA Combined"),
        regions={
            "usa_combined": (14.0, 50.0, -130.0, -60.0),
        },
    ),
    "West Coast": CoastConfig(
        name="West Coast",
        default_region="west_coast",
        processed_root=Path("data/processed/West Coast"),
        results_root=Path("data/results/West Coast"),
        visualizations_root=Path("data/visualizations/West Coast"),
        regions={
            "west_coast": (32.0, 49.5, -126.0, -117.0),
            "california": (32.0, 38.5, -124.5, -117.0),
            "pnw": (45.0, 49.5, -126.0, -122.0),
        },
    ),
    "Mexcany Beach": CoastConfig(
        name="Mexcany Beach",
        default_region="mexican_coast",
        processed_root=Path("data/processed/Mexcany Beach"),
        results_root=Path("data/results/Mexcany Beach"),
        visualizations_root=Path("data/visualizations/Mexcany Beach"),
        regions={
            "mexican_coast": (14.0, 32.5, -118.0, -86.0),
            "mexico_pacific": (14.0, 32.5, -118.0, -105.0),
            "mexico_gulf": (18.0, 26.5, -98.0, -86.0),
        },
    ),
    "Eastern coast": CoastConfig(
        name="Eastern coast",
        default_region="east_coast",
        processed_root=Path("data/processed/Eastern coast"),
        results_root=Path("data/results/Eastern coast"),
        visualizations_root=Path("data/visualizations/Eastern coast"),
        regions={
            "gulf": (24.0, 31.0, -98.0, -80.0),
            "east_coast": (35.0, 43.5, -76.5, -68.0),
            "west_coast": (32.0, 49.5, -126.0, -117.0),
            "mexican_coast": (14.0, 32.5, -118.0, -86.0),
            "mexico_pacific": (14.0, 32.5, -118.0, -105.0),
            "mexico_gulf": (18.0, 26.5, -98.0, -86.0),
            "california": (32.0, 38.5, -124.5, -117.0),
            "pnw": (45.0, 49.5, -126.0, -122.0),
            "danish": (54.5, 58.0, 7.5, 13.0),
        },
    ),
}

REGIONS: dict[str, tuple[float, float, float, float]] = {
    region: bounds
    for coast in COAST_CONFIGS.values()
    for region, bounds in coast.regions.items()
}


def parse_dataset_folder(parent_name: str) -> tuple[str | None, str | None]:
    if not (
        parent_name.startswith(DATASET_PREFIX)
        and parent_name.endswith(DATASET_SUFFIX)
    ):
        return None, None

    middle = parent_name[len(DATASET_PREFIX) : -len(DATASET_SUFFIX)]
    for region in sorted(REGIONS.keys(), key=len, reverse=True):
        if middle == region:
            return region, None
        region_prefix = f"{region}_"
        if middle.startswith(region_prefix):
            return region, middle[len(region_prefix) :]
    return middle, None


def region_from_dataset_path(path: Path) -> str | None:
    region, _ = parse_dataset_folder(path.parent.name)
    return region


def days_suffix_from_dataset_path(path: Path) -> str | None:
    _, days_suffix = parse_dataset_folder(path.parent.name)
    return days_suffix


def days_label_from_timestamps(df: pd.DataFrame) -> str:
    for column in ("timestamp", "history_end", "future_start"):
        if column not in df.columns:
            continue
        ts = pd.to_datetime(df[column], errors="coerce").dropna()
        if not ts.empty:
            unique_days = int(ts.dt.normalize().nunique())
            return f"{unique_days} dey"
    return "unknown"


def days_label_for_dataset(path: Path, df: pd.DataFrame | None = None) -> str:
    suffix = days_suffix_from_dataset_path(path)
    if suffix:
        day_match = re.fullmatch(r"(\d+)d", suffix)
        if day_match:
            return f"{day_match.group(1)} dey"
        month_match = re.fullmatch(r"(\d+)m", suffix)
        if month_match and df is not None:
            return days_label_from_timestamps(df)

    if df is not None:
        return days_label_from_timestamps(df)

    segments_path = path.parent / "coastal_segments.parquet"
    if segments_path.exists():
        segments = pd.read_parquet(segments_path, columns=["timestamp"])
        return days_label_from_timestamps(segments)

    return "unknown"


def coast_from_processed_path(path: Path) -> CoastConfig | None:
    resolved = path.resolve()
    for coast in COAST_CONFIGS.values():
        try:
            resolved.relative_to(coast.processed_root.resolve())
            return coast
        except ValueError:
            continue
    return None


def default_windows_path(coast: CoastConfig, region: str) -> Path:
    return (
        coast.processed_root
        / f"ais_{region}_long_horizon"
        / "model_ready_windows.parquet"
    )


def discover_windows_path(processed_root: Path) -> Path | None:
    if not processed_root.exists():
        return None
    candidates = sorted(
        processed_root.rglob("model_ready_windows.parquet"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_windows_path(
    coast_name: str | None,
    region: str,
    input_path: Path | None,
) -> tuple[Path, CoastConfig, str]:
    if input_path is not None:
        coast = coast_from_processed_path(input_path)
        if coast is None:
            # Path not under any registered coast (e.g. data/processed/combined/).
            # Fall back to the explicitly supplied coast name, or Eastern coast.
            fallback = coast_name or "Eastern coast"
            if fallback not in COAST_CONFIGS:
                raise FileNotFoundError(
                    f"--input must be under one of: "
                    f"{', '.join(str(c.processed_root) for c in COAST_CONFIGS.values())}"
                    f" or supply --coast to override."
                )
            coast = COAST_CONFIGS[fallback]
        inferred_region = region_from_dataset_path(input_path) or region
        return input_path, coast, inferred_region

    if coast_name is None:
        coast_name = "Eastern coast"

    coast = COAST_CONFIGS[coast_name]
    default_path = default_windows_path(coast, region)
    if default_path.exists():
        return default_path, coast, region

    discovered = discover_windows_path(coast.processed_root)
    if discovered is not None:
        inferred_region = region_from_dataset_path(discovered) or region
        print(f"Note: using discovered windows file for {coast.name}: {discovered}")
        return discovered, coast, inferred_region

    raise FileNotFoundError(
        f"Missing model_ready_windows.parquet for {coast.name}.\n"
        f"Expected: {default_path}\n"
        f'Run "PROCESS_noaa_long_coastal {coast.name}.py" or INCREMENTAL_PROCESS first.'
    )


def results_output_dir(
    coast: CoastConfig,
    dataset_path: Path,
    model_name: str,
    df: pd.DataFrame | None = None,
    run_tag: str | None = None,
) -> Path:
    days_label = days_label_for_dataset(dataset_path, df)
    base = coast.results_root / days_label
    if run_tag:
        base = base / run_tag
    return base / model_name


def models_output_dir(
    coast: CoastConfig,
    dataset_path: Path,
    df: pd.DataFrame | None = None,
    run_tag: str | None = None,
) -> Path:
    days_label = days_label_for_dataset(dataset_path, df)
    base = Path("data/models") / coast.name / days_label
    if run_tag:
        base = base / run_tag
    return base
