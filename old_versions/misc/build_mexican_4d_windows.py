"""Build model windows for Mexcany Beach 4d in low-memory trajectory batches."""
from __future__ import annotations

import gc
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
PROCESS_PATH = ROOT / "PROCESS_noaa_long_coastal Mexcany Beach.py"
DATA_ROOT = ROOT / "data/processed/Mexcany Beach/ais_mexican_coast_4d_long_horizon"
SEGMENTS_PATH = DATA_ROOT / "coastal_segments.parquet"
WINDOWS_PATH = DATA_ROOT / "model_ready_windows.parquet"
BATCH_SIZE = 15


def load_process_module():
    spec = importlib.util.spec_from_file_location("process_mexican_coast", PROCESS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    proc = load_process_module()
    if WINDOWS_PATH.exists():
        WINDOWS_PATH.unlink()

    traj_ids = pd.read_parquet(SEGMENTS_PATH, columns=["traj_id"])["traj_id"].unique()
    temp_paths: list[Path] = []
    total = 0
    n_batches = (len(traj_ids) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, start in enumerate(range(0, len(traj_ids), BATCH_SIZE), start=1):
        batch_ids = traj_ids[start : start + BATCH_SIZE].tolist()
        batch_df = pd.read_parquet(SEGMENTS_PATH, filters=[("traj_id", "in", batch_ids)])
        temp_path = DATA_ROOT / f"_tmp_windows_{batch_idx:04d}.parquet"
        proc.build_sequence_windows(
            batch_df,
            history_hours=6.0,
            future_hours=5.0,
            resample_minutes=10,
            max_windows_per_traj=200,
            output_path=temp_path,
            batch_size=500,
        )
        del batch_df
        gc.collect()

        if not temp_path.exists():
            continue

        batch_count = pq.read_metadata(temp_path).num_rows
        total += batch_count
        temp_paths.append(temp_path)
        print(f"Batch {batch_idx}/{n_batches}: +{batch_count:,} | total {total:,}", flush=True)

    if not temp_paths:
        raise RuntimeError("No windows were created.")

    writer: pq.ParquetWriter | None = None
    for temp_path in temp_paths:
        table = pq.read_table(temp_path)
        if writer is None:
            writer = pq.ParquetWriter(WINDOWS_PATH, table.schema, compression="snappy")
        writer.write_table(table)
        temp_path.unlink()
        del table
        gc.collect()

    if writer is not None:
        writer.close()

    print(f"Done: {total:,} windows -> {WINDOWS_PATH}", flush=True)


if __name__ == "__main__":
    main()
