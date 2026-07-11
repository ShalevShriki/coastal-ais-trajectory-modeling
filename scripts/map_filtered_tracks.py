#!/usr/bin/env python3
"""Sample random AIS tracks onto Folium maps to sanity-check lon and confined-radius filtering.

Modes:
  --source segments     : coastal_segments (known-good lon)
  --source filtered     : combined_filtered windows (after history-only stationary filter)
  --source windows-raw  : model_ready_windows for a coast (before combine/filter)
  --contrast            : write side-by-side comparison summary JSON

Also tags tracks kept vs would-be-dropped by StationaryFilterConfig when possible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from proj.project.window_data import (  # noqa: E402
    StationaryFilterConfig,
    compute_history_motion_metrics,
    stationary_window_mask,
)

COAST_SEGMENTS = {
    "east": Path(
        "data/processed/Eastern coast/ais_east_coast_feb_long_horizon/coastal_segments.parquet"
    ),
    "mexican": Path(
        "data/processed/Mexcany Beach/ais_mexican_coast_feb_long_horizon/coastal_segments.parquet"
    ),
    "west": Path(
        "data/processed/West Coast/ais_west_coast_feb_long_horizon/coastal_segments.parquet"
    ),
}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def window_track_cols(history_steps: int = 144, future_steps: int = 72) -> list[str]:
    cols = []
    for t in range(history_steps):
        cols.extend([f"x_t{t:03d}_lat", f"x_t{t:03d}_lon"])
    for t in range(future_steps):
        cols.extend([f"y_t{t:03d}_lat", f"y_t{t:03d}_lon"])
    return cols


def extract_window_track(row: pd.Series, history_steps: int = 144, future_steps: int = 72):
    hist = np.array(
        [[row[f"x_t{t:03d}_lat"], row[f"x_t{t:03d}_lon"]] for t in range(history_steps)],
        dtype=float,
    )
    fut = np.array(
        [[row[f"y_t{t:03d}_lat"], row[f"y_t{t:03d}_lon"]] for t in range(future_steps)],
        dtype=float,
    )
    return hist, fut


def lat_lon_sanity(lats: np.ndarray, lons: np.ndarray) -> dict:
    eq = float(np.mean(np.round(lats, 4) == np.round(lons, 4))) if len(lats) else 0.0
    return {
        "n_points": int(len(lats)),
        "lat_lon_equal_frac": eq,
        "lat_range": [float(np.min(lats)), float(np.max(lats))] if len(lats) else None,
        "lon_range": [float(np.min(lons)), float(np.max(lons))] if len(lons) else None,
        "looks_like_real_us_mex_coast": bool(
            len(lats)
            and 10 <= float(np.median(lats)) <= 55
            and -130 <= float(np.median(lons)) <= -65
            and eq < 0.05
        ),
    }


def make_map(tracks: list[dict], out: Path, title: str, center=None) -> None:
    import folium

    if not tracks:
        raise SystemExit("No tracks to plot")

    if center is None:
        pts = []
        for tr in tracks:
            pts.extend(tr["hist"])
            pts.extend(tr["fut"])
        arr = np.asarray(pts, dtype=float)
        center = [float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))]

    m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")
    legend = f"""
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;
                padding:8px 12px;border:1px solid #888;font-size:13px;max-width:320px">
      <b>{title}</b><br>
      Blue = history (24h) · Orange = future (12h) · Black = split/anchor<br>
      Kept (green border) vs would-drop (red border) when filter score available
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    for i, tr in enumerate(tracks):
        kept = tr.get("kept")
        border = "#2ca02c" if kept is True else ("#d62728" if kept is False else "#555")
        hist = [[float(a), float(b)] for a, b in tr["hist"]]
        fut = [[float(a), float(b)] for a, b in tr["fut"]]
        anchor = hist[-1]
        popup = (
            f"#{i} mmsi={tr.get('mmsi')} traj={tr.get('traj_id')}<br>"
            f"radius={tr.get('radius_km', float('nan')):.2f} km<br>"
            f"disp={tr.get('disp_km', float('nan')):.2f} km<br>"
            f"kept={kept}<br>{tr.get('note', '')}"
        )
        folium.PolyLine(hist, color="#1f77b4", weight=3, opacity=0.85, popup=popup).add_to(m)
        folium.PolyLine([anchor] + fut, color="#ff7f0e", weight=3, opacity=0.85).add_to(m)
        folium.CircleMarker(
            anchor,
            radius=5,
            color=border,
            fill=True,
            fill_color="#111",
            fill_opacity=0.9,
            weight=2,
            popup=popup,
        ).add_to(m)

    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(f"Wrote {out}")


def sample_segments(n: int, seed: int, out_dir: Path) -> dict:
    rng = np.random.default_rng(seed)
    tracks = []
    sanity_all_lats, sanity_all_lons = [], []
    for coast, path in COAST_SEGMENTS.items():
        if not path.exists():
            print(f"Missing {path}")
            continue
        # Read only needed columns; sample traj ids cheaply
        df = pd.read_parquet(path, columns=["traj_id", "mmsi", "lat", "lon", "timestamp"])
        traj_ids = df["traj_id"].dropna().unique()
        pick = rng.choice(traj_ids, size=min(n, len(traj_ids)), replace=False)
        for tid in pick:
            sub = df[df["traj_id"] == tid].sort_values("timestamp")
            if len(sub) < 50:
                continue
            # Take a middle ~36h slice if long enough (approx by index)
            step = max(1, len(sub) // 216)  # rough downsample for map
            pts = sub.iloc[::step][["lat", "lon"]].to_numpy(dtype=float)
            if len(pts) < 20:
                continue
            mid = len(pts) * 2 // 3
            hist = pts[:mid]
            fut = pts[mid:]
            # Proxies for radius/displacement on this slice history
            anchor = hist[-1]
            rad = float(
                np.max(haversine_km(anchor[0], anchor[1], hist[:, 0], hist[:, 1]))
            )
            disp = float(haversine_km(hist[0, 0], hist[0, 1], anchor[0], anchor[1]))
            cfg = StationaryFilterConfig()
            kept = not (rad <= cfg.max_confined_radius_km and disp < cfg.min_displacement_km)
            tracks.append(
                {
                    "hist": hist.tolist(),
                    "fut": fut.tolist() if len(fut) else hist[-1:].tolist(),
                    "mmsi": int(sub["mmsi"].iloc[0]),
                    "traj_id": str(tid),
                    "radius_km": rad,
                    "disp_km": disp,
                    "kept": kept,
                    "note": f"coast={coast} segments (good lon)",
                }
            )
            sanity_all_lats.extend(pts[:, 0].tolist())
            sanity_all_lons.extend(pts[:, 1].tolist())

    out = out_dir / "map_good_segments_random.html"
    make_map(tracks[: max(n * 3, n)], out, "Feb coastal_segments (good lon)")
    report = {
        "source": "coastal_segments",
        "n_tracks": len(tracks),
        "sanity": lat_lon_sanity(np.array(sanity_all_lats), np.array(sanity_all_lons)),
        "map": str(out),
    }
    return report


def sample_filtered_windows(
    path: Path,
    n: int,
    seed: int,
    out_dir: Path,
    tag: str,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    pf_cols = window_track_cols() + ["traj_id", "mmsi"]
    # Reservoir-ish: read first ~80k rows then sample (memory-safe enough)
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    take = min(80_000, pf.metadata.num_rows)
    table = pf.read_row_groups(list(range(min(4, pf.metadata.num_row_groups))), columns=[c for c in pf_cols if c in pf.schema.names])
    df = table.to_pandas()
    if len(df) > take:
        df = df.sample(n=take, random_state=seed)

    metrics = compute_history_motion_metrics(df, history_steps=144)
    cfg = StationaryFilterConfig(history_only=True)
    drop_mask = stationary_window_mask(metrics, cfg)
    kept_mask = ~drop_mask

    rng = np.random.default_rng(seed)
    idx_kept = np.where(kept_mask.to_numpy() if hasattr(kept_mask, "to_numpy") else kept_mask)[0]
    idx_drop = np.where(drop_mask.to_numpy() if hasattr(drop_mask, "to_numpy") else drop_mask)[0]

    n_kept = min(n // 2 + n % 2, len(idx_kept))
    n_drop = min(n // 2, len(idx_drop))
    pick = []
    if n_kept:
        pick.extend(rng.choice(idx_kept, size=n_kept, replace=False).tolist())
    if n_drop:
        pick.extend(rng.choice(idx_drop, size=n_drop, replace=False).tolist())
    if not pick:
        pick = rng.choice(len(df), size=min(n, len(df)), replace=False).tolist()

    tracks = []
    all_lat, all_lon = [], []
    for i in pick:
        row = df.iloc[int(i)]
        hist, fut = extract_window_track(row)
        all_lat.extend(hist[:, 0].tolist())
        all_lat.extend(fut[:, 0].tolist())
        all_lon.extend(hist[:, 1].tolist())
        all_lon.extend(fut[:, 1].tolist())
        tracks.append(
            {
                "hist": hist.tolist(),
                "fut": fut.tolist(),
                "mmsi": int(row["mmsi"]) if "mmsi" in row and pd.notna(row["mmsi"]) else None,
                "traj_id": str(row["traj_id"]) if "traj_id" in row else None,
                "radius_km": float(metrics.iloc[int(i)]["history_max_radius_km"]),
                "disp_km": float(metrics.iloc[int(i)]["history_displacement_km"]),
                "kept": bool(kept_mask.iloc[int(i)] if hasattr(kept_mask, "iloc") else kept_mask[int(i)]),
                "note": tag,
            }
        )

    out = out_dir / f"map_{tag}_random.html"
    make_map(tracks, out, f"{tag} windows")
    report = {
        "source": str(path),
        "tag": tag,
        "n_tracks": len(tracks),
        "rows_scanned": len(df),
        "kept_fraction_in_scan": float(np.mean(kept_mask)),
        "drop_fraction_in_scan": float(np.mean(drop_mask)),
        "median_radius_kept_km": float(metrics.loc[kept_mask, "history_max_radius_km"].median())
        if kept_mask.any()
        else None,
        "median_radius_dropped_km": float(metrics.loc[drop_mask, "history_max_radius_km"].median())
        if drop_mask.any()
        else None,
        "sanity": lat_lon_sanity(np.asarray(all_lat), np.asarray(all_lon)),
        "map": str(out),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["segments", "filtered", "both", "fixed-filtered"],
        default="both",
    )
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--filtered",
        type=Path,
        default=Path("data/processed/combined_filtered/test.parquet"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/results/USA Combined/unknown/track_maps"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    if args.mode in ("segments", "both"):
        print("=== Sampling good coastal_segments ===")
        reports["segments"] = sample_segments(args.n, args.seed, args.out_dir)

    if args.mode in ("filtered", "both", "fixed-filtered"):
        tag = "fixed_filtered" if args.mode == "fixed-filtered" else "corrupt_or_current_filtered"
        if args.mode == "fixed-filtered":
            tag = "fixed_filtered"
        print(f"=== Sampling filtered windows: {args.filtered} ===")
        reports[tag] = sample_filtered_windows(
            args.filtered, args.n, args.seed, args.out_dir, tag=tag
        )

    report_path = args.out_dir / "track_map_report.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
