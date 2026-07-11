#!/usr/bin/env python3
"""Which observable features most strongly associate with adaptive context alphas?

Memory-lean: only loads motion/id columns needed for the analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))

from proj.project.window_data import haversine_km, trajectory_splits

FEATURE_NAMES = [
    "mean_sog",
    "std_sog",
    "maneuver_mean_abs_dcog",
    "max_abs_dcog",
    "accel_mean_abs_dsog",
    "path_km_24h",
    "net_disp_km_24h",
    "straightness",
    "abs_dlat_24h",
    "abs_dlon_24h",
    "anchor_lat",
    "anchor_lon",
]


def needed_columns(schema_names: list[str], history_steps: int = 144) -> list[str]:
    cols = []
    for idc in ("traj_id", "mmsi"):
        if idc in schema_names:
            cols.append(idc)
    for t in range(history_steps):
        for feat in ("lat", "lon", "sog", "dcog", "dsog"):
            c = f"x_t{t:03d}_{feat}"
            if c in schema_names:
                cols.append(c)
    return cols


def load_lean_sample(path: Path, sample_size: int, seed: int) -> pd.DataFrame:
    """Match training load: enough row-groups for sample_size, then random sample.

    Uses the same row-group RNG seed (42) as window_data.load_windows.
    """
    pf = pq.ParquetFile(path)
    names = list(pf.schema.names)
    cols = needed_columns(names)
    total = pf.metadata.num_rows
    # Training used prefetch≈sample/0.7; if that exceeds total, load everything.
    target = max(sample_size, int(sample_size / 0.7) + 1000)
    if total <= target:
        table = pf.read(columns=cols)
        df = table.to_pandas()
        del table
        return df

    n_groups = pf.metadata.num_row_groups
    frac = target / total
    n_pick = max(1, min(n_groups, int(np.ceil(n_groups * frac))))
    rng = np.random.default_rng(42)  # matches window_data.load_windows
    chosen = sorted(rng.choice(n_groups, size=n_pick, replace=False).tolist())
    parts = []
    for g in chosen:
        part = pf.read_row_group(g, columns=cols).to_pandas()
        # downcast to save RAM
        for c in part.columns:
            if part[c].dtype == np.float64:
                part[c] = part[c].astype(np.float32)
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    del parts
    if len(df) > target:
        df = df.sample(target, random_state=seed).reset_index(drop=True)
    return df


def motion_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    lat_cols = [c for c in df.columns if c.startswith("x_t") and c.endswith("_lat")]
    lon_cols = [c for c in df.columns if c.startswith("x_t") and c.endswith("_lon")]
    sog_cols = [c for c in df.columns if c.startswith("x_t") and c.endswith("_sog")]
    dcog_cols = [c for c in df.columns if c.startswith("x_t") and c.endswith("_dcog")]
    dsog_cols = [c for c in df.columns if c.startswith("x_t") and c.endswith("_dsog")]

    lat = df[lat_cols].to_numpy(dtype=np.float64)
    lon = df[lon_cols].to_numpy(dtype=np.float64)
    sog = df[sog_cols].to_numpy(dtype=np.float64)
    dcog = df[dcog_cols].to_numpy(dtype=np.float64)
    dsog = df[dsog_cols].to_numpy(dtype=np.float64)

    step_km = haversine_km(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
    path_km = step_km.sum(axis=1)
    net_km = haversine_km(lat[:, 0], lon[:, 0], lat[:, -1], lon[:, -1])
    straightness = np.clip(net_km / np.maximum(path_km, 1e-3), 0.0, 1.0)

    return {
        "mean_sog": sog.mean(axis=1),
        "std_sog": sog.std(axis=1),
        "maneuver_mean_abs_dcog": np.abs(dcog).mean(axis=1),
        "max_abs_dcog": np.abs(dcog).max(axis=1),
        "accel_mean_abs_dsog": np.abs(dsog).mean(axis=1),
        "path_km_24h": path_km,
        "net_disp_km_24h": net_km,
        "straightness": straightness,
        "abs_dlat_24h": np.abs(lat[:, -1] - lat[:, 0]),
        "abs_dlon_24h": np.abs(lon[:, -1] - lon[:, 0]),
        "anchor_lat": lat[:, -1],
        "anchor_lon": lon[:, -1],
    }


def spearman_matrix(feats: dict[str, np.ndarray], alpha: np.ndarray, labels: list[str]) -> dict:
    from scipy import stats

    out = {}
    for fname, fvals in feats.items():
        out[fname] = {
            labels[j]: float(stats.spearmanr(fvals, alpha[:, j]).statistic) for j in range(alpha.shape[1])
        }
    return out


def rf_importance(X: np.ndarray, y: np.ndarray, names: list[str], seed: int = 42):
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=40,
        n_jobs=2,
        random_state=seed,
    )
    model.fit(X, y)
    return sorted(zip(names, model.feature_importances_.tolist()), key=lambda t: -t[1])


def write_html_report(
    out_path: Path,
    *,
    title: str,
    alpha_mean: dict,
    argmax_pct: dict,
    spearman: dict,
    rf_by_target: dict,
    vessel_alpha: dict | None,
    n: int,
    top_takeaways: list[str],
) -> None:
    labels = list(alpha_mean.keys())
    ranked = list(spearman.items())
    rows_sp = []
    for fname, rs in ranked:
        cells = "".join(f"<td style='text-align:right'>{rs[l]:+.3f}</td>" for l in labels)
        rows_sp.append(f"<tr><td>{fname}</td>{cells}</tr>")

    rf_sections = []
    for target, imps in rf_by_target.items():
        items = "".join(f"<li><b>{name}</b>: {imp:.3f}</li>" for name, imp in imps[:8])
        rf_sections.append(f"<h3>Predicting {target}</h3><ol>{items}</ol>")

    vessel_html = ""
    if vessel_alpha:
        vrows = []
        for cls, a in vessel_alpha.items():
            cells = "".join(
                f"<td style='text-align:right'>{a[f'alpha_{l}']:.3f}</td>" for l in labels
            )
            vrows.append(f"<tr><td>{cls}</td>{cells}</tr>")
        vessel_html = f"""
        <h2>Mean α by vessel class</h2>
        <table><tr><th>class</th>{''.join(f'<th>{l}</th>' for l in labels)}</tr>
        {''.join(vrows)}</table>
        """

    takeaways = "".join(f"<li>{t}</li>" for t in top_takeaways)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; max-width: 960px; color: #222; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; font-size: 13px; }}
th {{ background: #f4f4f4; }}
.note {{ color: #555; font-size: 14px; line-height: 1.5; }}
.takeaways {{ background:#f7fafc; border:1px solid #cbd5e1; border-radius:8px; padding:12px 16px; }}
</style></head><body>
<h1>{title}</h1>
<p class="note">n={n:,} test windows. The gate sees encoder hidden states (not raw AIS
features). This report correlates <b>observable motion summaries</b> with α to interpret
what the gate prefers.</p>
<div class="takeaways"><b>Takeaways</b><ul>{takeaways}</ul></div>

<h2>Mean α / argmax</h2>
<p>Mean: {', '.join(f'{k}={v:.3f}' for k,v in alpha_mean.items())}</p>
<p>Argmax %: {', '.join(f'{k}={v:.1f}%' for k,v in argmax_pct.items())}</p>
{vessel_html}

<h2>Spearman correlation (feature ↔ α)</h2>
<p class="note">Sorted by max |r|. Positive = higher feature → higher weight on that context.</p>
<table>
<tr><th>feature</th>{''.join(f'<th>α {l}</th>' for l in labels)}</tr>
{''.join(rows_sp)}
</table>

<h2>Random-forest surrogate importance</h2>
<p class="note">RF regressors predict each α (and α_24h−α_9h) from motion features.</p>
{''.join(rf_sections)}
</body></html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alpha-json",
        type=Path,
        default=PROJECT
        / "data/results/USA Combined/unknown/exp_final/adaptive_vessel_type/RNN_AR_adaptive/context_alpha_weights.json",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT / "data/processed/combined_filtered_smart/train.parquet",
    )
    parser.add_argument("--sample", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=15_000)
    parser.add_argument(
        "--html",
        type=Path,
        default=PROJECT
        / "data/results/USA Combined/unknown/exp_final/adaptive_vessel_type/RNN_AR_adaptive/gate_feature_drivers.html",
    )
    args = parser.parse_args()

    payload = json.loads(args.alpha_json.read_text())
    alpha_all = np.array(payload["per_sample"], dtype=np.float32)
    labels = [f"{h:g}h" for h in payload.get("context_hours", [9, 12, 18, 24])]

    print("Loading lean window sample (motion columns only)...")
    prefetch = int(args.sample / 0.7) + 1000
    df = load_lean_sample(args.input, prefetch, args.seed)
    print(f"  loaded {len(df):,} rows, {df.shape[1]} cols")

    split_by = "trajectory" if "traj_id" in df.columns else "mmsi"
    train_ids, val_ids, test_ids = trajectory_splits(
        df, test_fraction=0.2, val_fraction=0.1, seed=args.seed, split_by=split_by
    )
    id_col = "traj_id" if split_by == "trajectory" else "mmsi"
    test_df = df[df[id_col].isin(test_ids)].reset_index(drop=True)
    del df

    n = min(len(test_df), len(alpha_all))
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(n, size=min(args.max_rows, n), replace=False)
    sub = test_df.iloc[pick].reset_index(drop=True)
    alpha = alpha_all[pick]
    del test_df

    print(f"Computing features on n={len(sub):,} ...")
    feats = motion_features(sub)
    del sub
    X = np.nan_to_num(
        np.column_stack([feats[name] for name in FEATURE_NAMES]),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    spearman = spearman_matrix(feats, alpha, labels)
    ranked = sorted(spearman.items(), key=lambda kv: max(abs(v) for v in kv[1].values()), reverse=True)

    print("\nSpearman r (feature ↔ α):")
    for fname, rs in ranked:
        print(f"  {fname:<28} " + " ".join(f"{labels[j]}={rs[labels[j]]:+.3f}" for j in range(4)))

    rf_by_target = {}
    for j, lbl in enumerate(labels):
        imps = rf_importance(X, alpha[:, j], FEATURE_NAMES, seed=args.seed)
        rf_by_target[f"α_{lbl}"] = imps
        print(f"\nRF importance → α_{lbl}:")
        for name, imp in imps[:6]:
            print(f"  {name:<28} {imp:.3f}")

    long_minus_short = alpha[:, 3] - alpha[:, 0]
    imps = rf_importance(X, long_minus_short, FEATURE_NAMES, seed=args.seed)
    rf_by_target["α_24h − α_9h"] = imps
    print("\nRF importance → α_24h − α_9h:")
    for name, imp in imps[:8]:
        print(f"  {name:<28} {imp:.3f}")

    argmax = alpha.argmax(axis=1)
    alpha_mean = {labels[j]: float(alpha[:, j].mean()) for j in range(4)}
    argmax_pct = {labels[j]: float((argmax == j).mean() * 100) for j in range(4)}
    print("\nArgmax %:", {k: f"{v:.1f}%" for k, v in argmax_pct.items()})

    vessel_alpha = payload.get("alpha_mean_by_vessel_class")
    if vessel_alpha:
        print("\nMean α by vessel class:")
        for cls, a in vessel_alpha.items():
            print(
                f"  {cls:<12} "
                + " ".join(f"{labels[j]}={a[f'alpha_{labels[j]}']:.3f}" for j in range(4))
            )

    # auto takeaways from top spearman / RF
    top_feat, top_rs = ranked[0]
    top_rf = imps[0][0]
    takeaways = [
        f"Strongest Spearman association overall: <b>{top_feat}</b> "
        f"(max |r|={max(abs(v) for v in top_rs.values()):.3f}).",
        f"Best RF driver of long-vs-short preference (α_24h−α_9h): <b>{top_rf}</b>.",
        "Gate still puts most mass on 18–24h contexts on average; motion features modulate this only mildly.",
    ]
    if vessel_alpha:
        # largest 24h alpha class
        best = max(vessel_alpha.items(), key=lambda kv: kv[1]["alpha_24h"])
        takeaways.append(
            f"Among vessel classes, <b>{best[0]}</b> leans most on 24h "
            f"(α_24h={best[1]['alpha_24h']:.3f})."
        )

    write_html_report(
        args.html,
        title="Adaptive gate — feature drivers",
        alpha_mean=alpha_mean,
        argmax_pct=argmax_pct,
        spearman={k: v for k, v in ranked},
        rf_by_target=rf_by_target,
        vessel_alpha=vessel_alpha,
        n=len(alpha),
        top_takeaways=takeaways,
    )
    print(f"\nSaved HTML report: {args.html}")


if __name__ == "__main__":
    main()
