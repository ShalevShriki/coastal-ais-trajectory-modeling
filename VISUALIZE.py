from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import contextily as ctx
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.collections import LineCollection
from pyproj import Transformer

from proj.project.coast_frame import bounds_for_coast_region, segment_frame_stats, vessel_in_frame_hours

KNOTS_TO_KMH = 1.852

try:
    import folium
    from folium import plugins
except ImportError as exc:
    raise ImportError(
        "folium is not installed for this Python. Run:\n"
        "  python -m pip install -r requirements.txt"
    ) from exc

GEO_CRS = "EPSG:4326"
MAP_CRS = "EPSG:3857"
TO_MERCATOR = Transformer.from_crs(GEO_CRS, MAP_CRS, always_xy=True)

REGION_LABELS = {
    "gulf": "Gulf of Mexico",
    "east_coast": "US East Coast",
    "west_coast": "US West Coast",
    "mexican_coast": "Mexican coast",
    "mexico_pacific": "Mexico Pacific coast",
    "mexico_gulf": "Mexico Gulf coast",
    "california": "US West Coast (California)",
    "pnw": "Pacific Northwest",
    "danish": "Danish waters",
}

# Port markers per region: (name, longitude, latitude)
REGION_PORTS: dict[str, list[tuple[str, float, float]]] = {
    "danish": [
        ("Copenhagen", 12.5683, 55.6761),
        ("Aarhus", 10.2167, 56.1500),
        ("Aalborg", 9.9217, 57.0488),
        ("Fredericia", 9.7522, 55.5658),
        ("Esbjerg", 8.4500, 55.4667),
        ("Helsingør", 12.6136, 56.0361),
        ("Kalundborg", 11.0886, 55.6794),
        ("Frederikshavn", 10.5361, 57.4417),
        ("Skagen", 10.5950, 57.7208),
        ("Hanstholm", 8.6167, 57.1167),
        ("Korsør", 11.1386, 55.3294),
        ("Nyborg", 10.7897, 55.3125),
        ("Gedser", 11.9258, 54.5758),
        ("Rønne", 14.7014, 55.1011),
    ],
    "east_coast": [
        ("New York", -74.0060, 40.7128),
        ("Boston", -71.0589, 42.3601),
        ("Philadelphia", -75.1652, 39.9526),
        ("Baltimore", -76.6122, 39.2904),
        ("Norfolk", -76.2859, 36.8508),
        ("Providence", -71.4128, 41.8240),
        ("Portland ME", -70.2568, 43.6591),
        ("New Haven", -72.9279, 41.3083),
    ],
    "gulf": [
        ("Houston", -95.0195, 29.7290),
        ("New Orleans", -90.0640, 29.9360),
        ("Tampa", -82.4510, 27.9230),
        ("Mobile", -88.0399, 30.6954),
        ("Galveston", -94.7930, 29.3110),
        ("Corpus Christi", -97.3964, 27.8006),
        ("Pascagoula", -88.5561, 30.3658),
    ],
    "california": [
        ("Los Angeles", -118.2720, 33.7405),
        ("Long Beach", -118.1937, 33.7540),
        ("Oakland", -122.3000, 37.7950),
        ("San Francisco", -122.4194, 37.7749),
        ("San Diego", -117.1730, 32.7100),
        ("Stockton", -121.2900, 37.9570),
    ],
    "west_coast": [
        ("San Diego", -117.1730, 32.7100),
        ("Los Angeles", -118.2720, 33.7405),
        ("Long Beach", -118.1937, 33.7540),
        ("San Francisco", -122.4194, 37.7749),
        ("Oakland", -122.3000, 37.7950),
        ("Seattle", -122.3380, 47.6062),
        ("Tacoma", -122.4443, 47.2529),
        ("Portland", -122.6750, 45.5231),
    ],
    "mexican_coast": [
        ("Ensenada", -116.6320, 31.8660),
        ("La Paz", -110.3090, 24.1420),
        ("Mazatlán", -106.4160, 23.1950),
        ("Manzanillo", -104.3350, 19.0520),
        ("Lázaro Cárdenas", -102.1950, 17.9580),
        ("Acapulco", -99.9100, 16.8530),
        ("Veracruz", -96.1342, 19.2000),
        ("Tampico", -97.8686, 22.2553),
    ],
    "mexico_pacific": [
        ("Ensenada", -116.6320, 31.8660),
        ("La Paz", -110.3090, 24.1420),
        ("Mazatlán", -106.4160, 23.1950),
        ("Puerto Vallarta", -105.2450, 20.6530),
        ("Manzanillo", -104.3350, 19.0520),
        ("Lázaro Cárdenas", -102.1950, 17.9580),
        ("Acapulco", -99.9100, 16.8530),
    ],
    "mexico_gulf": [
        ("Tampico", -97.8686, 22.2553),
        ("Altamira", -97.9090, 22.4010),
        ("Veracruz", -96.1342, 19.2000),
        ("Coatzacoalcos", -94.4150, 18.1490),
        ("Progreso", -89.6640, 21.2830),
    ],
    "pnw": [
        ("Seattle", -122.3380, 47.6062),
        ("Tacoma", -122.4443, 47.2529),
        ("Portland", -122.6750, 45.5231),
        ("Vancouver", -123.1139, 49.2827),
        ("Everett", -122.2021, 47.9790),
    ],
}

NOAA_VESSEL_TYPE_LABELS: dict[int, str] = {
    30: "Fishing",
    31: "Towing",
    32: "Towing (large)",
    33: "Dredging",
    34: "Diving",
    35: "Military",
    36: "Sailing",
    37: "Pleasure craft",
    50: "Pilot",
    51: "SAR",
    52: "Tug",
    53: "Port tender",
    54: "Pollution control",
    55: "Law enforcement",
    58: "Medical",
    59: "Special craft",
    60: "Passenger",
    70: "Cargo",
    80: "Tanker",
}

TRACK_COLORS = plt.cm.tab10.colors
FOLIUM_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
    "#66c2a5",
    "#fc8d62",
]


@dataclass(frozen=True)
class CoastConfig:
    name: str
    default_region: str
    processed_root: Path
    output_dir: Path
    regions: dict[str, tuple[float, float, float, float]]


COAST_CONFIGS: dict[str, CoastConfig] = {
    "West Coast": CoastConfig(
        name="West Coast",
        default_region="west_coast",
        processed_root=Path("data/processed/West Coast"),
        output_dir=Path("data/visualizations/West Coast"),
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
        output_dir=Path("data/visualizations/Mexcany Beach"),
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
        output_dir=Path("data/visualizations/Eastern coast"),
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


@dataclass(frozen=True)
class VizConfig:
    region: str
    bounds: dict[str, float]
    ports: list[tuple[str, float, float]]
    area_label: str


def default_segments_path(coast: CoastConfig, region: str) -> Path:
    return coast.processed_root / f"ais_{region}_long_horizon" / "coastal_segments.parquet"


def parse_segments_folder(parent_name: str) -> tuple[str | None, str | None]:
    prefix, suffix = "ais_", "_long_horizon"
    if not (parent_name.startswith(prefix) and parent_name.endswith(suffix)):
        return None, None

    middle = parent_name[len(prefix) : -len(suffix)]
    for region in sorted(REGIONS.keys(), key=len, reverse=True):
        if middle == region:
            return region, None
        region_prefix = f"{region}_"
        if middle.startswith(region_prefix):
            return region, middle[len(region_prefix) :]
    return middle, None


def region_from_segments_path(path: Path) -> str | None:
    region, _ = parse_segments_folder(path.parent.name)
    return region


def days_suffix_from_segments_path(path: Path) -> str | None:
    _, days_suffix = parse_segments_folder(path.parent.name)
    return days_suffix


def days_label_from_timestamps(df: pd.DataFrame) -> str:
    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if ts.empty:
        return "unknown"
    unique_days = int(ts.dt.normalize().nunique())
    return f"{unique_days} dey"


def days_label_for_dataset(segments_path: Path, df: pd.DataFrame) -> str:
    suffix = days_suffix_from_segments_path(segments_path)
    if suffix:
        day_match = re.fullmatch(r"(\d+)d", suffix)
        if day_match:
            return f"{day_match.group(1)} dey"
        month_match = re.fullmatch(r"(\d+)m", suffix)
        if month_match:
            return days_label_from_timestamps(df)

    return days_label_from_timestamps(df)


def visualization_output_dir(
    coast: CoastConfig, segments_path: Path, df: pd.DataFrame
) -> Path:
    return coast.output_dir / days_label_for_dataset(segments_path, df)


def discover_segments_path(processed_root: Path) -> Path | None:
    if not processed_root.exists():
        return None
    candidates = sorted(
        processed_root.rglob("coastal_segments.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_input_path(
    coast: CoastConfig, region: str, input_path: Path | None
) -> tuple[Path, str]:
    if input_path is not None:
        inferred = region_from_segments_path(input_path)
        return input_path, inferred or region

    default_path = default_segments_path(coast, region)
    if default_path.exists():
        return default_path, region

    discovered = discover_segments_path(coast.processed_root)
    if discovered is not None:
        inferred = region_from_segments_path(discovered)
        print(
            f"Note: using discovered segments file for {coast.name}: {discovered}"
        )
        return discovered, inferred or region

    raise FileNotFoundError(
        f"Missing coastal_segments.parquet for {coast.name}.\n"
        f"Expected: {default_path}\n"
        f"Run the matching PROCESS script first, e.g.:\n"
        f'  python "PROCESS_noaa_long_coastal {coast.name}.py" --region {region}'
    )


def route_group_col(df: pd.DataFrame) -> str:
    return "traj_id" if "traj_id" in df.columns else "mmsi"


def format_vessel_type(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Unknown"
    if isinstance(value, str):
        text = value.strip()
        return text if text else "Unknown"
    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return str(value)
    if code in NOAA_VESSEL_TYPE_LABELS:
        return NOAA_VESSEL_TYPE_LABELS[code]
    if 60 <= code <= 69:
        return "Passenger"
    if 70 <= code <= 79:
        return "Cargo"
    if 80 <= code <= 89:
        return "Tanker"
    if 30 <= code <= 39:
        return "Special / fishing"
    if 50 <= code <= 59:
        return "Special craft"
    return f"Type {code}"


def vessel_type_series(df: pd.DataFrame) -> pd.Series:
    if "vessel_type" in df.columns:
        return df["vessel_type"].map(format_vessel_type)
    if "ship_type" in df.columns:
        return df["ship_type"].astype(str)
    return pd.Series(["Unknown"] * len(df), index=df.index)


def build_viz_config(region: str, df: pd.DataFrame) -> VizConfig:
    if region in REGIONS:
        lat_min, lat_max, lon_min, lon_max = REGIONS[region]
        bounds = {"south": lat_min, "north": lat_max, "west": lon_min, "east": lon_max}
    else:
        pad_lat = (df["lat"].max() - df["lat"].min()) * 0.02 or 0.1
        pad_lon = (df["lon"].max() - df["lon"].min()) * 0.02 or 0.1
        bounds = {
            "south": float(df["lat"].min() - pad_lat),
            "north": float(df["lat"].max() + pad_lat),
            "west": float(df["lon"].min() - pad_lon),
            "east": float(df["lon"].max() + pad_lon),
        }

    ports = REGION_PORTS.get(region, [])
    area_label = REGION_LABELS.get(region, region.replace("_", " ").title())
    return VizConfig(region=region, bounds=bounds, ports=ports, area_label=area_label)


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f8f9fa",
            "axes.edgecolor": "#cccccc",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "grid.alpha": 0.35,
            "legend.framealpha": 0.92,
            "font.size": 10,
        }
    )


def load_data(path: Path, sample_size: int | None, coast_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run PROCESS_noaa_long_coastal {coast_name}.py first to create "
            "coastal_segments.parquet."
        )
    df = pd.read_parquet(path)
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    return df


def top_routes_from_file(path: Path, n_routes: int) -> pd.Index:
    group_col = "traj_id" if "traj_id" in pq.read_schema(path).names else "mmsi"
    route_df = pd.read_parquet(path, columns=[group_col])
    return route_df.groupby(group_col).size().sort_values(ascending=False).head(n_routes).index


def load_full_routes(path: Path, route_ids: pd.Index) -> pd.DataFrame:
    group_col = "traj_id" if "traj_id" in pq.read_schema(path).names else "mmsi"
    if group_col == "traj_id":
        ids = [str(value) for value in route_ids]
    else:
        ids = [int(value) for value in route_ids]
    df = pd.read_parquet(path, filters=[(group_col, "in", ids)])
    return df.sort_values([group_col, "timestamp"]).reset_index(drop=True)


def top_routes(df: pd.DataFrame, n_routes: int) -> pd.Index:
    group_col = route_group_col(df)
    return df.groupby(group_col).size().sort_values(ascending=False).head(n_routes).index


def trajectory_stats_path(segments_path: Path) -> Path:
    return segments_path.parent / "trajectory_stats.csv"


def longest_routes_from_file(
    segments_path: Path, n_routes: int = 5
) -> tuple[pd.Index, dict[object, str]]:
    stats_path = trajectory_stats_path(segments_path)
    if stats_path.exists():
        stats = pd.read_csv(stats_path, index_col=0)
        if "total_distance_km" not in stats.columns and "total_distance_nm" in stats.columns:
            stats["total_distance_km"] = stats["total_distance_nm"] * KNOTS_TO_KMH
        top = stats.sort_values("total_distance_km", ascending=False).head(n_routes)
        metrics = {
            route_id: f"{row['total_distance_km']:.1f} km, {row['duration_hours']:.1f} h"
            for route_id, row in top.iterrows()
        }
        return top.index, metrics

    return top_routes_from_file(segments_path, n_routes), {}


def route_label(track: pd.DataFrame, route_id: object) -> str:
    group_col = route_group_col(track)
    label = f"{group_col} {route_id}"
    types = vessel_type_series(track).dropna()
    if not types.empty:
        label += f" ({types.iloc[0]})"
    if "mmsi" in track.columns and group_col == "traj_id":
        label += f" | MMSI {track['mmsi'].iloc[0]}"
    return label


def to_mercator(lon: pd.Series | float, lat: pd.Series | float) -> tuple:
    return TO_MERCATOR.transform(lon, lat)


def bounds_mercator(bounds: dict[str, float], padding: float = 0.03) -> tuple[float, float, float, float]:
    xmin, ymin = to_mercator(bounds["west"], bounds["south"])
    xmax, ymax = to_mercator(bounds["east"], bounds["north"])
    dx = (xmax - xmin) * padding
    dy = (ymax - ymin) * padding
    return xmin - dx, ymin - dy, xmax + dx, ymax + dy


BASEMAP_SOURCES = [
    ctx.providers.CartoDB.Positron,
    ctx.providers.OpenStreetMap.Mapnik,
]


def _add_basemap(ax: plt.Axes) -> None:
    last_exc: Exception | None = None
    for source in BASEMAP_SOURCES:
        for _ in range(2):
            try:
                ctx.add_basemap(
                    ax,
                    crs=MAP_CRS,
                    source=source,
                    attribution_size=6,
                )
                return
            except Exception as exc:
                last_exc = exc
    print(f"Warning: could not load map tiles ({last_exc}). Plotting without basemap.")
    ax.set_facecolor("#d8e8f0")


def _plot_ports(ax: plt.Axes, ports: list[tuple[str, float, float]], *, zorder: int = 6) -> None:
    for name, lon, lat in ports:
        x, y = to_mercator(lon, lat)
        ax.scatter(
            x,
            y,
            marker="s",
            s=55,
            c="#1d3557",
            edgecolors="white",
            linewidths=1.0,
            zorder=zorder,
        )
        ax.annotate(
            name,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7.5,
            color="#1d3557",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "#cccccc",
            },
            zorder=zorder + 1,
        )


def _create_basemap_ax(
    viz: VizConfig,
    *,
    figsize: tuple[float, float] = (11, 9),
    title: str = "",
    subtitle: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    xmin, ymin, xmax, ymax = bounds_mercator(viz.bounds)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    _add_basemap(ax)
    if viz.ports:
        _plot_ports(ax, viz.ports)
    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full_title, fontsize=12, fontweight="bold", pad=10)
    ax.set_axis_off()
    return fig, ax


def _route_segments(df: pd.DataFrame) -> list[list[tuple[float, float]]]:
    group_col = route_group_col(df)
    segments: list[list[tuple[float, float]]] = []
    for _, track in df.groupby(group_col, sort=False):
        track = track.sort_values("timestamp")
        if len(track) < 2:
            continue
        x, y = to_mercator(track["lon"], track["lat"])
        segments.append(list(zip(x, y)))
    return segments


def plot_position_routes(
    df: pd.DataFrame, output_path: Path, total_rows: int, viz: VizConfig
) -> None:
    n_routes = df[route_group_col(df)].nunique()
    sample_note = (
        f"{n_routes:,} routes | {len(df):,} of {total_rows:,} AIS points"
        if len(df) < total_rows
        else f"{n_routes:,} routes | {len(df):,} AIS points"
    )
    fig, ax = _create_basemap_ax(
        viz,
        title="Vessel routes",
        subtitle=f"{viz.area_label} | {sample_note}",
    )

    segments = _route_segments(df)
    if segments:
        routes = LineCollection(
            segments,
            colors="#1d4e89",
            linewidths=0.45,
            alpha=0.42,
            zorder=4,
        )
        ax.add_collection(routes)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def load_vessel_tracking_hours(
    segments_path: Path,
    coast_name: str,
    region: str,
    *,
    strict_frame: bool = False,
) -> tuple[pd.Series, dict[str, float | int | str]]:
    """Longest in-frame AIS segment (hours) per vessel within the coast bounding box."""
    bounds = bounds_for_coast_region(coast_name, region)
    stats = segment_frame_stats(segments_path, bounds, strict_frame=strict_frame)
    vessel = vessel_in_frame_hours(stats)
    hours = vessel.set_index("mmsi")["longest_in_frame_hours"]
    meta = {
        "frame_region": region,
        "lat_min": bounds.lat_min,
        "lat_max": bounds.lat_max,
        "lon_min": bounds.lon_min,
        "lon_max": bounds.lon_max,
        "out_of_frame_points": int(stats["n_out_of_frame"].sum()),
        "vessels_fully_in_frame": int(vessel["fully_in_frame"].sum()),
        "vessel_count": len(vessel),
    }
    return hours, meta


def dataset_span_hours(segments_path: Path) -> float | None:
    stats_path = trajectory_stats_path(segments_path)
    if stats_path.exists():
        stats = pd.read_csv(stats_path, usecols=["start_time", "end_time"])
        start = pd.to_datetime(stats["start_time"]).min()
        end = pd.to_datetime(stats["end_time"]).max()
        if pd.notna(start) and pd.notna(end):
            return (end - start).total_seconds() / 3600.0

    df = pd.read_parquet(segments_path, columns=["timestamp"])
    ts = pd.to_datetime(df["timestamp"])
    if ts.empty:
        return None
    return (ts.max() - ts.min()).total_seconds() / 3600.0


def plot_vessel_tracking_hours_distribution(
    hours: pd.Series,
    output_path: Path,
    area_label: str,
    *,
    dataset_span_h: float | None = None,
    frame_meta: dict[str, float | int | str] | None = None,
) -> None:
    hours = hours.clip(lower=0).dropna()
    if hours.empty:
        return
    if dataset_span_h is not None:
        hours = hours.clip(upper=dataset_span_h)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    median_h = float(hours.median())
    mean_h = float(hours.mean())
    x_max = float(hours.quantile(0.995))
    if dataset_span_h is not None:
        x_max = max(x_max, dataset_span_h * 1.02)

    ax.hist(
        hours,
        bins=40,
        range=(0, x_max),
        color="#457b9d",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
    )
    ax.axvline(
        median_h,
        color="#e76f51",
        linewidth=2,
        linestyle="--",
        label=f"Median: {median_h:.1f} h",
    )
    ax.axvline(
        mean_h,
        color="#264653",
        linewidth=2,
        linestyle=":",
        label=f"Mean: {mean_h:.1f} h",
    )
    if dataset_span_h is not None:
        ax.axvline(
            dataset_span_h,
            color="#999999",
            linewidth=1.5,
            linestyle="-.",
            label=f"Data window: {dataset_span_h:.0f} h",
        )

    ax.set_title(
        f"Longest in-frame AIS segment per vessel\n{area_label}",
        pad=12,
    )
    ax.set_xlabel("In-frame tracking time (hours)")
    ax.set_ylabel("Number of vessels")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", linestyle="--")

    stats_text = (
        f"Vessels: {len(hours):,}\n"
        f"p90: {hours.quantile(0.9):.1f} h\n"
        f"max: {hours.max():.1f} h\n"
        "Hours inside coast\nbounding box only"
    )
    if frame_meta:
        fully = frame_meta.get("vessels_fully_in_frame", 0)
        total = frame_meta.get("vessel_count", len(hours))
        oof_pts = frame_meta.get("out_of_frame_points", 0)
        stats_text += (
            f"\n100% in-frame vessels:\n{fully:,}/{total:,}"
            f"\nOut-of-frame points:\n{oof_pts:,}"
        )
    ax.text(
        0.98,
        0.72,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#cccccc"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_sog_distribution(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sog = df["sog"].clip(lower=0, upper=30 * KNOTS_TO_KMH)
    median_sog = float(sog.median())
    mean_sog = float(sog.mean())

    ax.hist(
        sog,
        bins=40,
        color="#2a9d8f",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
    )
    ax.axvline(median_sog, color="#e76f51", linewidth=2, linestyle="--", label=f"Median: {median_sog:.1f} km/h")
    ax.axvline(mean_sog, color="#264653", linewidth=2, linestyle=":", label=f"Mean: {mean_sog:.1f} km/h")

    ax.set_title("Speed over ground (SOG) distribution", pad=12)
    ax.set_xlabel("SOG (km/h)")
    ax.set_ylabel("Number of AIS reports")
    ax.set_xlim(0, 30 * KNOTS_TO_KMH)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle="--")

    stats_text = (
        f"Reports: {len(sog):,}\n"
        f"Stationary (SOG < 0.9 km/h): {(sog < 0.5 * KNOTS_TO_KMH).mean() * 100:.1f}%\n"
        f"Cruising (SOG ≥ 9.3 km/h): {(sog >= 5 * KNOTS_TO_KMH).mean() * 100:.1f}%"
    )
    ax.text(
        0.98,
        0.72,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#cccccc"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_ship_types(df: pd.DataFrame, output_path: Path) -> None:
    if "vessel_type" not in df.columns and "ship_type" not in df.columns:
        return

    counts = vessel_type_series(df).value_counts().head(10)
    if counts.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color="#457b9d", edgecolor="white")
    ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=9)
    ax.set_title("Top vessel types in sample", pad=12)
    ax.set_xlabel("Number of AIS reports")
    ax.grid(True, axis="x", linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _draw_routes_on_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    route_ids: pd.Index,
    *,
    route_metrics: dict[object, str] | None = None,
) -> None:
    group_col = route_group_col(df)
    for idx, route_id in enumerate(route_ids):
        track = df[df[group_col] == route_id].sort_values("timestamp")
        if len(track) < 2:
            continue

        color = TRACK_COLORS[idx % len(TRACK_COLORS)]
        label = route_label(track, route_id)
        if route_metrics and route_id in route_metrics:
            label += f" | {route_metrics[route_id]}"

        x, y = to_mercator(track["lon"], track["lat"])
        (line,) = ax.plot(
            x,
            y,
            linewidth=2.5,
            alpha=0.92,
            color=color,
            label=label,
            zorder=4,
        )
        line.set_path_effects(
            [pe.Stroke(linewidth=4.5, foreground="white", alpha=0.8), pe.Normal()]
        )
        ax.scatter(
            x[0],
            y[0],
            s=75,
            marker="o",
            color=color,
            edgecolors="white",
            linewidths=1.4,
            zorder=5,
        )
        ax.scatter(
            x[-1],
            y[-1],
            s=95,
            marker="X",
            color=color,
            edgecolors="white",
            linewidths=1.4,
            zorder=5,
        )


def plot_sample_trajectories(
    df: pd.DataFrame, output_path: Path, n_routes: int, viz: VizConfig
) -> None:
    routes = top_routes(df, n_routes)
    group_col = route_group_col(df)
    total_points = sum(len(df[df[group_col] == route_id]) for route_id in routes)
    fig, ax = _create_basemap_ax(
        viz,
        title=f"Sample trajectories ({n_routes} busiest routes)",
        subtitle=f"Full routes | {total_points:,} AIS points | Circle = start | X = end",
    )

    _draw_routes_on_ax(ax, df, routes)

    ax.legend(
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        title="Route",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_longest_trajectories(
    df: pd.DataFrame,
    output_path: Path,
    route_ids: pd.Index,
    route_metrics: dict[object, str],
    viz: VizConfig,
) -> None:
    n_routes = len(route_ids)
    fig, ax = _create_basemap_ax(
        viz,
        title=f"Longest trajectories (top {n_routes} by distance)",
        subtitle="Ranked by total_distance_km | Circle = start | X = end",
    )

    _draw_routes_on_ax(ax, df, route_ids, route_metrics=route_metrics)

    ax.legend(
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        title="Route",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_interactive_map(
    df: pd.DataFrame, output_path: Path, n_routes: int, viz: VizConfig
) -> None:
    center_lat = float(df["lat"].median())
    center_lon = float(df["lon"].median())
    ship_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles="CartoDB positron",
        control_scale=True,
    )

    heat_data = df[["lat", "lon"]].dropna().values.tolist()
    if heat_data:
        plugins.HeatMap(
            heat_data,
            radius=12,
            blur=18,
            max_zoom=10,
            gradient={0.2: "#ffffb2", 0.5: "#fd8d3c", 0.8: "#e31a1c", 1.0: "#800026"},
        ).add_to(ship_map)

    if viz.ports:
        ports_layer = folium.FeatureGroup(name="Ports", show=True)
        for name, lon, lat in viz.ports:
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color="#1d3557",
                fill=True,
                fill_color="#1d3557",
                fill_opacity=0.95,
                popup=name,
                tooltip=name,
            ).add_to(ports_layer)
            folium.map.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    icon_size=(120, 16),
                    icon_anchor=(0, 12),
                    html=(
                        f'<div style="font-size:10px;font-weight:700;color:#1d3557;'
                        f'background:rgba(255,255,255,0.9);padding:1px 4px;border-radius:3px;'
                        f'border:1px solid #ccc;white-space:nowrap;">{name}</div>'
                    ),
                ),
            ).add_to(ports_layer)
        ports_layer.add_to(ship_map)

    tracks_layer = folium.FeatureGroup(name="Vessel tracks", show=True)
    group_col = route_group_col(df)

    for idx, route_id in enumerate(top_routes(df, n_routes)):
        track = df[df[group_col] == route_id].sort_values("timestamp")
        points = list(zip(track["lat"], track["lon"]))
        if len(points) < 2:
            continue

        color = FOLIUM_COLORS[idx % len(FOLIUM_COLORS)]
        label = route_label(track, route_id)
        popup_html = (
            f"<b>{label}</b><br>"
            f"Points: {len(track):,}<br>"
            f"From: {track['timestamp'].iloc[0]}<br>"
            f"To: {track['timestamp'].iloc[-1]}"
        )

        folium.PolyLine(
            points,
            color=color,
            weight=4,
            opacity=0.9,
            tooltip=label,
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(tracks_layer)

        folium.CircleMarker(
            location=points[0],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.95,
            popup=f"Start — {label}",
        ).add_to(tracks_layer)
        folium.Marker(
            location=points[-1],
            icon=folium.Icon(color="green" if idx % 2 else "blue", icon="flag"),
            popup=f"End — {label}",
        ).add_to(tracks_layer)

    tracks_layer.add_to(ship_map)

    ship_map.fit_bounds(
        [
            [viz.bounds["south"], viz.bounds["west"]],
            [viz.bounds["north"], viz.bounds["east"]],
        ],
        padding=(20, 20),
    )

    legend_html = """
    <div style="
        position: fixed; bottom: 24px; left: 24px; z-index: 9999;
        background: white; border: 1px solid #ccc; border-radius: 6px;
        padding: 10px 12px; font-size: 13px; line-height: 1.5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    ">
        <b>Map legend</b><br>
        Heat layer: traffic density<br>
        Lines: vessel routes<br>
        Dot: route start | Flag: route end
    </div>
    """
    ship_map.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(ship_map)
    ship_map.save(output_path)


def print_summary(df: pd.DataFrame) -> None:
    group_col = route_group_col(df)
    print(f"Rows loaded: {len(df):,}")
    print(f"Unique routes ({group_col}): {df[group_col].nunique():,}")
    print(f"Unique vessels (mmsi): {df['mmsi'].nunique():,}")
    print(f"Time range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    types = vessel_type_series(df)
    if not types.empty:
        print("\nVessel types:")
        print(types.value_counts().head(10))


def visualize_coast(
    coast: CoastConfig,
    *,
    region: str,
    input_path: Path | None,
    sample_size: int | None,
    n_trajectories: int,
) -> list[Path]:
    input_path, region = resolve_input_path(coast, region, input_path)
    total_rows = len(pd.read_parquet(input_path, columns=["mmsi"]))
    full_extent = pd.read_parquet(input_path, columns=["lat", "lon"])
    df = load_data(input_path, sample_size=sample_size, coast_name=coast.name)
    viz = build_viz_config(region, full_extent)

    top_ids = top_routes_from_file(input_path, n_trajectories)
    tracks_df = load_full_routes(input_path, top_ids)

    longest_ids, longest_metrics = longest_routes_from_file(input_path, n_routes=5)
    longest_df = load_full_routes(input_path, longest_ids)

    output_dir = visualization_output_dir(coast, input_path, df)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {coast.name} ===")
    print_summary(df)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Full trajectories: {len(tracks_df):,} points for {len(top_ids)} routes")
    print(f"Longest trajectories: {len(longest_df):,} points for {len(longest_ids)} routes")

    routes_path = output_dir / "positions_scatter.png"
    sog_path = output_dir / "sog_distribution.png"
    tracking_path = output_dir / "tracking_hours_per_vessel.png"
    types_path = output_dir / "ship_types.png"
    tracks_path = output_dir / "sample_trajectories.png"
    longest_path = output_dir / "longest_trajectories.png"
    map_path = output_dir / "interactive_map.html"

    plot_position_routes(df, routes_path, total_rows=total_rows, viz=viz)
    plot_sog_distribution(df, sog_path)
    vessel_hours, frame_meta = load_vessel_tracking_hours(
        input_path, coast.name, region
    )
    span_hours = dataset_span_hours(input_path)
    plot_vessel_tracking_hours_distribution(
        vessel_hours,
        tracking_path,
        viz.area_label,
        dataset_span_h=span_hours,
        frame_meta=frame_meta,
    )
    plot_ship_types(df, types_path)
    plot_sample_trajectories(tracks_df, tracks_path, n_routes=n_trajectories, viz=viz)
    plot_longest_trajectories(longest_df, longest_path, longest_ids, longest_metrics, viz=viz)
    build_interactive_map(tracks_df, map_path, n_routes=n_trajectories, viz=viz)

    saved = [routes_path, sog_path, tracks_path, longest_path, map_path]
    if tracking_path.exists():
        saved.insert(2, tracking_path)
    if types_path.exists():
        idx = 3 if tracking_path.exists() else 2
        saved.insert(idx, types_path)

    print("\nSaved visualizations:")
    for path in saved:
        print(f"- {path}")
    return saved


def coast_from_input_path(path: Path) -> CoastConfig | None:
    for coast in COAST_CONFIGS.values():
        try:
            path.resolve().relative_to(coast.processed_root.resolve())
            return coast
        except ValueError:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize processed coastal AIS segments for each coast."
    )
    parser.add_argument(
        "--coast",
        choices=sorted(COAST_CONFIGS.keys()),
        default=None,
        help="Coastal area to visualize (default: all coasts).",
    )
    parser.add_argument(
        "--all-coasts",
        action="store_true",
        help="Generate visualizations for West Coast, Mexcany Beach, and Eastern coast.",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Predefined coastal region filter (default: coast-specific default).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Path to coastal_segments.parquet "
            "(default: data/processed/<coast>/ais_<region>_long_horizon/)."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100_000,
        help="Random sample size for route overview plots (0 = use all rows).",
    )
    parser.add_argument(
        "--trajectories",
        type=int,
        default=8,
        help="Number of full trajectories to draw on map/detail plots.",
    )
    args = parser.parse_args()

    _setup_style()

    input_override = Path(args.input) if args.input else None

    if args.coast is not None:
        coasts = [COAST_CONFIGS[args.coast]]
    elif input_override is not None:
        matched = coast_from_input_path(input_override)
        if matched is None:
            raise SystemExit(
                f"--input must be under one of: "
                f"{', '.join(str(c.processed_root) for c in COAST_CONFIGS.values())}"
            )
        coasts = [matched]
    else:
        coasts = [COAST_CONFIGS[name] for name in sorted(COAST_CONFIGS.keys())]

    sample_size = args.sample if args.sample > 0 else None
    failures: list[str] = []

    for coast in coasts:
        region = args.region or coast.default_region
        if region not in coast.regions:
            available = ", ".join(sorted(coast.regions))
            failures.append(
                f"{coast.name}: region '{region}' is not configured. Options: {available}"
            )
            continue
        try:
            visualize_coast(
                coast,
                region=region,
                input_path=input_override,
                sample_size=sample_size,
                n_trajectories=args.trajectories,
            )
        except FileNotFoundError as exc:
            failures.append(str(exc))

    if failures:
        print("\nSkipped coasts:")
        for message in failures:
            print(f"- {message}")
        if len(failures) == len(coasts):
            raise SystemExit(1)

    print("\nOpen interactive_map.html in each coast folder for the live map.")


if __name__ == "__main__":
    main()
