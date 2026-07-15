#!/usr/bin/env python3
"""
Remove strongly inland (river/canal/marsh) windows from a coastal AIS dataset.

Keeps open-water and coastal-fringe / near-port traffic.
Drops windows where a majority of history points have no open water within N km.

Writes a new parquet (does not modify the source) plus a JSON report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))

from proj.project.models.land_mask_utils import inland_window_mask_from_history


def infer_history_steps(names: list[str]) -> int:
    hist = 0
    for c in names:
        if c.startswith("x_t") and c.endswith("_lat"):
            try:
                hist = max(hist, int(c[3:6]) + 1)
            except ValueError:
                pass
    return hist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT / "data/processed/combined_filtered_smart/train.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "data/processed/combined_filtered_smart_coastal/train.parquet",
    )
    parser.add_argument("--open-water-km", type=float, default=10.0)
    parser.add_argument("--inland-fraction", type=float, default=0.5)
    parser.add_argument("--subsample-every", type=int, default=6, help="History step stride (6=1h)")
    parser.add_argument("--batch-rows", type=int, default=40_000)
    args = parser.parse_args()

    t0 = time.time()
    pf = pq.ParquetFile(args.input)
    names = list(pf.schema.names)
    history_steps = infer_history_steps(names)
    lat_cols = [f"x_t{t:03d}_lat" for t in range(history_steps)]
    lon_cols = [f"x_t{t:03d}_lon" for t in range(history_steps)]
    for c in lat_cols + lon_cols:
        if c not in names:
            raise SystemExit(f"Missing column {c}")

    print(
        f"Filtering inland windows | src={args.input} | rows={pf.metadata.num_rows:,} | "
        f"history={history_steps} | open_water_km={args.open_water_km} | "
        f"inland_fraction>{args.inland_fraction}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    kept = 0
    dropped = 0
    total = 0

    for batch in pf.iter_batches(batch_size=args.batch_rows, columns=None):
        df = batch.to_pandas()
        lat = df[lat_cols].to_numpy(dtype=np.float64)
        lon = df[lon_cols].to_numpy(dtype=np.float64)
        inland = inland_window_mask_from_history(
            lat,
            lon,
            open_water_km=args.open_water_km,
            inland_fraction=args.inland_fraction,
            subsample_every=args.subsample_every,
        )
        keep = ~inland
        n_keep = int(keep.sum())
        n_drop = int((~keep).sum())
        kept += n_keep
        dropped += n_drop
        total += len(df)
        if n_keep == 0:
            print(f"  batch total={total:,} kept={kept:,} dropped={dropped:,}")
            continue
        out_df = df.loc[keep].reset_index(drop=True)
        table = pa.Table.from_pandas(out_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(args.output, table.schema, compression="zstd")
        writer.write_table(table)
        print(
            f"  batch total={total:,} kept={kept:,} ({100*kept/total:.1f}%) "
            f"dropped={dropped:,} ({100*dropped/total:.1f}%)"
        )

    if writer is not None:
        writer.close()

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "rows_in": total,
        "rows_out": kept,
        "rows_dropped": dropped,
        "keep_fraction": kept / total if total else 0.0,
        "drop_fraction": dropped / total if total else 0.0,
        "open_water_km": args.open_water_km,
        "inland_fraction_threshold": args.inland_fraction,
        "subsample_every": args.subsample_every,
        "history_steps": history_steps,
        "rule": (
            "Drop window if fraction of history points with no open water within "
            f"{args.open_water_km} km exceeds {args.inland_fraction}. "
            "Keeps open-water and coastal-fringe/port traffic; removes inland rivers/canals/marsh."
        ),
        "runtime_sec": round(time.time() - t0, 1),
    }
    report_path = args.output.parent / "inland_filter_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDone. Kept {kept:,}/{total:,} ({100*kept/max(total,1):.1f}%)")
    print(f"Wrote: {args.output}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
