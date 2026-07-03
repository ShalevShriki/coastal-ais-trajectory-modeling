from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from proj.project.coast_frame import (
    bounds_for_segments_path,
    dataset_label_from_path,
    discover_segments_path,
    points_in_frame,
    segment_frame_stats,
    vessel_in_frame_hours,
)
from proj.project.coast_paths import COAST_CONFIGS, region_from_dataset_path

POINT_EXPORT_COLS = (
    "coast",
    "region",
    "dataset_label",
    "mmsi",
    "traj_id",
    "timestamp",
    "lat",
    "lon",
    "sog",
    "vessel_type",
    "in_frame",
)


def export_coast(
    coast_name: str,
    *,
    dataset_tag: str | None,
    output_dir: Path,
    strict_frame: bool,
    include_points: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    segments_path = discover_segments_path(coast_name, dataset_tag)
    region = region_from_dataset_path(segments_path) or COAST_CONFIGS[coast_name].default_region
    bounds = bounds_for_segments_path(segments_path, coast_name)
    label = dataset_label_from_path(segments_path)

    print(f"\n=== {coast_name} ({label}) ===", flush=True)
    print(f"Segments: {segments_path}", flush=True)
    print(
        f"Frame ({region}): lat [{bounds.lat_min}, {bounds.lat_max}], "
        f"lon [{bounds.lon_min}, {bounds.lon_max}]",
        flush=True,
    )

    stats = segment_frame_stats(segments_path, bounds, strict_frame=strict_frame)
    stats.insert(0, "coast", coast_name)
    stats.insert(1, "region", region)
    stats.insert(2, "dataset_label", label)

    vessel = vessel_in_frame_hours(stats)
    vessel.insert(0, "coast", coast_name)
    vessel.insert(1, "region", region)
    vessel.insert(2, "dataset_label", label)

    out_of_frame_pts = int(stats["n_out_of_frame"].sum())
    fully_in = int(vessel["fully_in_frame"].sum())
    print(
        f"Trajectories: {len(stats):,} | vessels: {len(vessel):,} | "
        f"out-of-frame points: {out_of_frame_pts:,} | "
        f"vessels 100% in-frame: {fully_in:,}/{len(vessel):,}",
        flush=True,
    )

    if include_points:
        _export_points(segments_path, coast_name, region, label, bounds, output_dir, strict_frame)

    return stats, vessel


def _export_points(
    segments_path: Path,
    coast_name: str,
    region: str,
    label: str,
    bounds,
    output_dir: Path,
    strict_frame: bool,
) -> None:
    schema = pq.read_schema(segments_path)
    read_cols = [
        c
        for c in ("mmsi", "traj_id", "timestamp", "lat", "lon", "sog", "vessel_type")
        if c in schema.names
    ]
    export_cols = [c for c in POINT_EXPORT_COLS if c in read_cols or c in {"coast", "region", "dataset_label", "in_frame"}]
    out_path = output_dir / f"points_{coast_name.replace(' ', '_')}.parquet"
    writer: pq.ParquetWriter | None = None
    total = 0

    parquet_file = pq.ParquetFile(segments_path)
    for batch in parquet_file.iter_batches(batch_size=250_000, columns=read_cols):
        chunk = batch.to_pandas()
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
        chunk["in_frame"] = points_in_frame(chunk, bounds)
        if strict_frame:
            chunk = chunk[chunk["in_frame"]].copy()
        chunk.insert(0, "coast", coast_name)
        chunk.insert(1, "region", region)
        chunk.insert(2, "dataset_label", label)

        table = pa.Table.from_pandas(chunk[[c for c in export_cols if c in chunk.columns]], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)
        total += len(chunk)

    if writer is not None:
        writer.close()
        print(f"Saved {total:,} points -> {out_path}", flush=True)


def write_summary(
    trajectories: pd.DataFrame,
    vessels: pd.DataFrame,
    output_dir: Path,
    *,
    strict_frame: bool,
) -> None:
    lines = [
        "Combined coast export summary",
        "=============================",
        f"trajectories: {len(trajectories):,}",
        f"vessels (coast,mmsi rows): {len(vessels):,}",
        f"unique MMSI (global): {vessels['mmsi'].nunique():,}",
        f"strict_frame filter: {strict_frame}",
        "",
        "Rows per coast:",
        str(trajectories.groupby("coast").size()),
        "",
        "Out-of-frame points per coast:",
        str(trajectories.groupby("coast")["n_out_of_frame"].sum()),
        "",
        "Vessels not fully in-frame (any out-of-frame point):",
        str(
            vessels[~vessels["fully_in_frame"]]
            .groupby("coast")["mmsi"]
            .nunique()
            .rename("vessels_with_out_of_frame")
        ),
        "",
        "In-frame hours quantiles (longest segment per vessel):",
        str(vessels["longest_in_frame_hours"].quantile([0.5, 0.9, 0.95, 0.99])),
    ]
    (output_dir / "export_summary.txt").write_text("\n".join(map(str, lines)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export combined AIS segment stats for all coasts with frame checks."
    )
    parser.add_argument(
        "--coast",
        choices=sorted(COAST_CONFIGS.keys()),
        action="append",
        help="Coast to export (repeatable). Default: all coasts.",
    )
    parser.add_argument(
        "--dataset-tag",
        default="4d",
        help="Dataset folder suffix, e.g. 4d -> ais_west_coast_4d_long_horizon.",
    )
    parser.add_argument(
        "--out",
        default="data/export/all_coasts",
        help="Output directory for combined CSV/parquet exports.",
    )
    parser.add_argument(
        "--strict-frame",
        action="store_true",
        help="Drop AIS reports outside the coast bounding box before computing stats.",
    )
    parser.add_argument(
        "--include-points",
        action="store_true",
        help="Also export filtered point-level parquet per coast (large files).",
    )
    args = parser.parse_args()

    coasts = args.coast or sorted(COAST_CONFIGS.keys())
    tag_suffix = f"_{args.dataset_tag}" if args.dataset_tag else ""
    output_dir = Path(args.out + tag_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)

    traj_parts: list[pd.DataFrame] = []
    vessel_parts: list[pd.DataFrame] = []

    for coast_name in coasts:
        stats, vessel = export_coast(
            coast_name,
            dataset_tag=args.dataset_tag,
            output_dir=output_dir,
            strict_frame=args.strict_frame,
            include_points=args.include_points,
        )
        traj_parts.append(stats)
        vessel_parts.append(vessel)

    trajectories = pd.concat(traj_parts, ignore_index=True)
    vessels = pd.concat(vessel_parts, ignore_index=True)

    trajectories.to_csv(output_dir / "trajectories_all_coasts.csv", index=False)
    vessels.to_csv(output_dir / "vessels_all_coasts.csv", index=False)
    write_summary(trajectories, vessels, output_dir, strict_frame=args.strict_frame)

    print(f"\nSaved combined export to: {output_dir}")
    print(f"  - trajectories_all_coasts.csv ({len(trajectories):,} rows)")
    print(f"  - vessels_all_coasts.csv ({len(vessels):,} rows)")
    print(f"  - export_summary.txt")


if __name__ == "__main__":
    main()
