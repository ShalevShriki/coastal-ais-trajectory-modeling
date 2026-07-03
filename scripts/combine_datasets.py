#!/usr/bin/env python3
"""
Combine model_ready_windows.parquet files from multiple coastal regions into one
unified dataset with train/val/test splits.

Uses incremental pyarrow writing — never loads more than one source file at a time,
so memory usage is bounded by the largest single file, not the total dataset.

Usage:
  cd <project root>

  # Combine canonical datasets (East, Mexican, West Coast proper data)
  python scripts/combine_datasets.py

  # Also include datasets stored in the wrong coast folder (East 14d / 1m)
  python scripts/combine_datasets.py --include-misplaced

  # Custom output dir / skip split
  python scripts/combine_datasets.py --out data/processed/combined_v2 --no-split
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

COAST_DIR_TO_REGION_PREFIX: dict[str, str] = {
    "Eastern coast": "east_coast",
    "Mexcany Beach": "mexican_coast",
    "West Coast":    "west_coast",
}


def _region_from_folder(folder: str) -> str | None:
    m = re.match(r"^ais_(.+?)(?:_\d+[dm])?_long_horizon$", folder)
    return m.group(1) if m else None


def _is_canonical(wf: Path, include_misplaced: bool) -> tuple[bool, str]:
    coast_dir = wf.parent.parent.name
    region = _region_from_folder(wf.parent.name)
    if region is None:
        return False, f"can't infer region from '{wf.parent.name}'"
    if "_1d_" in wf.parent.name:
        return False, "1d dataset (too short for 24h+12h windows)"
    expected = COAST_DIR_TO_REGION_PREFIX.get(coast_dir)
    if expected and region != expected:
        if not include_misplaced:
            return False, f"misplaced: {region} data in '{coast_dir}' (use --include-misplaced)"
        return True, f"misplaced but included: {region} in '{coast_dir}'"
    return True, "ok"


def collect_traj_ids(wf: Path) -> list:
    """Read only the traj_id column — cheap first pass for split assignment."""
    return pd.read_parquet(wf, columns=["traj_id"])["traj_id"].unique().tolist()


def combine(
    data_root: Path,
    out_dir: Path,
    include_misplaced: bool,
    val_frac: float,
    test_frac: float,
    no_split: bool,
    seed: int,
) -> None:
    t0 = time.perf_counter()

    window_files = sorted(data_root.rglob("model_ready_windows.parquet"))
    if not window_files:
        print(f"No model_ready_windows.parquet found under {data_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(window_files)} window file(s)\n")

    included: list[tuple[Path, str]] = []   # (path, region)
    for wf in window_files:
        ok, reason = _is_canonical(wf, include_misplaced)
        tag = "READ" if ok else "SKIP"
        region = _region_from_folder(wf.parent.name) or "?"
        label = f"{wf.parent.parent.name}/{wf.parent.name}"
        print(f"  {tag}  {label}  — {reason}")
        if ok:
            included.append((wf, region))

    if not included:
        print("\nNo datasets passed the filter.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Pass 1: collect all trajectory IDs to assign splits (cheap — one col)
    # ------------------------------------------------------------------
    if not no_split:
        print("\nPass 1: collecting trajectory IDs for split assignment…", flush=True)
        all_traj_ids: list = []
        for wf, _ in included:
            ids = collect_traj_ids(wf)
            all_traj_ids.extend(ids)
            print(f"  {wf.parent.name}: {len(ids):,} trajectories", flush=True)

        rng = np.random.default_rng(seed)
        unique_ids = np.array(list(set(all_traj_ids)))
        rng.shuffle(unique_ids)
        n = len(unique_ids)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))

        test_ids = set(unique_ids[:n_test].tolist())
        val_ids  = set(unique_ids[n_test : n_test + n_val].tolist())
        train_ids = set(unique_ids[n_test + n_val:].tolist())

        print(
            f"\nSplit: {len(train_ids):,} train / {len(val_ids):,} val / "
            f"{len(test_ids):,} test trajectories (seed={seed})\n",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Pass 2: stream-write parquet one source file at a time
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    if no_split:
        writers: dict[str, pq.ParquetWriter] = {"all": None}
        split_id_sets = {"all": None}
        out_paths = {"all": out_dir / "all_windows.parquet"}
    else:
        writers = {"train": None, "val": None, "test": None}
        split_id_sets = {"train": train_ids, "val": val_ids, "test": test_ids}
        out_paths = {s: out_dir / f"{s}.parquet" for s in writers}

    total_counts: dict[str, int] = {s: 0 for s in writers}
    region_counts: dict[str, int] = {}

    ref_schema: pa.Schema | None = None

    print("Pass 2: writing output parquet (one source file at a time)…", flush=True)
    for wf, region in included:
        print(f"  {wf.parent.name} [{region}]…", end=" ", flush=True)
        df = pd.read_parquet(wf)
        df["region"] = region
        df["source_file"] = str(wf.relative_to(data_root))
        # Normalize traj_id to string — type varies across older vs newer segment files
        df["traj_id"] = df["traj_id"].astype(str)

        # Warn if params mismatch
        if "history_steps" in df.columns:
            h  = int(df["history_steps"].iloc[0])
            f_ = int(df["future_steps"].iloc[0])
            rm = int(df["resample_minutes"].iloc[0])
            params_ok = (h == 144 and f_ == 72)
            status = f"hist={h*rm//60}h fut={f_*rm//60}h"
            if not params_ok:
                status += " *** PARAM MISMATCH — rebuild with scripts/rebuild_windows.py --all ***"
        else:
            status = "?"

        print(f"{len(df):,} windows ({status})", flush=True)
        region_counts[region] = region_counts.get(region, 0) + len(df)

        for split_name, id_set in split_id_sets.items():
            if id_set is None:
                chunk = df
            else:
                chunk = df[df["traj_id"].isin(id_set)]
            if len(chunk) == 0:
                continue
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            # Establish reference schema from first chunk; cast all subsequent tables to it
            if ref_schema is None:
                ref_schema = table.schema
            else:
                table = table.cast(ref_schema)
            if writers[split_name] is None:
                writers[split_name] = pq.ParquetWriter(out_paths[split_name], ref_schema)
            writers[split_name].write_table(table)
            total_counts[split_name] += len(chunk)

        del df

    for w in writers.values():
        if w is not None:
            w.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = sum(total_counts.values())
    print(f"\nTotal windows written: {total:,}")
    print("\nPer-region:")
    for r, cnt in sorted(region_counts.items()):
        print(f"  {r:<20s}  {cnt:>10,}  ({cnt/total*100:.1f}%)")
    print("\nOutput files:")
    for split_name, path in out_paths.items():
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            print(f"  {split_name:<6s}  {total_counts[split_name]:>10,} windows  "
                  f"({size_mb:.0f} MB)  →  {path}")

    print(f"\nDone in {time.perf_counter() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine coastal AIS window datasets — memory-efficient incremental writing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--out", default="data/processed/combined")
    parser.add_argument("--include-misplaced", action="store_true",
                        help="Include datasets in wrong coast folder (e.g. West Coast/ais_east_coast_14d).")
    parser.add_argument("--no-split", action="store_true",
                        help="Save one all_windows.parquet instead of train/val/test.")
    parser.add_argument("--val-frac",  type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    combine(
        data_root=Path(args.data_root),
        out_dir=Path(args.out),
        include_misplaced=args.include_misplaced,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        no_split=args.no_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
