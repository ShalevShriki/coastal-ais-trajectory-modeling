"""
Build a clean folium map with 2 trajectory examples (low / high error).

Markers
-------
  Green circle   — start of 24h history
  Gray line      — observed history (24h)
  Yellow star    — end of history / start of prediction ("NOW")
  Blue line      — ground-truth future (12h)
  Red circle     — true final position @ 12h
  Orange dashed  — LSTM flat prediction
  Dark-green     — LSTM-AR prediction
  Purple dashed  — Transformer prediction
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUBROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SUBROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from proj.project.window_data import (
    StationaryFilterConfig,
    haversine_km,
    stationary_window_mask,
    trajectory_splits,
)

RESULTS = SUBROOT / "data" / "results" / "USA Combined" / "unknown"
DEFAULT_OUT = RESULTS / "visualizations" / "map_clean_trajectories.html"

COLORS = {
    "history": "#9E9E9E",
    "gt": "#1565C0",
    "rnn": "#FF9800",
    "rnn_ar": "#1B5E20",
    "transformer": "#7B1FA2",
}


def load_traj_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: np.asarray(data[k], dtype=np.float64) for k in ("y_true", "y_pred", "anchor")}


def map_indices_into_test(seed: int, n_test: int, n_map: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(n_test, size=min(n_map, n_test), replace=False)


def load_histories_for_map_rows(
    input_path: Path,
    sample_size: int,
    seed: int,
    map_rows: list[int],
    anchors: dict[int, np.ndarray],
    history_steps: int = 144,
) -> dict[int, np.ndarray]:
    """Load 24h history lat/lon for specific map-json rows without loading all features."""
    import pyarrow.parquet as pq

    hist_cols: list[str] = []
    for t in range(history_steps):
        hist_cols.extend([f"x_t{t:03d}_lat", f"x_t{t:03d}_lon"])
    id_cols = ["traj_id"] if "traj_id" in pq.ParquetFile(input_path).schema.names else ["mmsi"]
    read_cols = id_cols + hist_cols

    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    if total_rows <= sample_size:
        df = pd.read_parquet(input_path, columns=read_cols)
    else:
        num_groups = pf.metadata.num_row_groups
        frac = sample_size / total_rows
        n_groups = max(1, min(num_groups, int(np.ceil(num_groups * frac))))
        rng = np.random.default_rng(42)
        chosen = sorted(rng.choice(num_groups, size=n_groups, replace=False).tolist())
        frames = [pf.read_row_group(g, columns=read_cols).to_pandas() for g in chosen]
        df = pd.concat(frames, ignore_index=True)
        del frames
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    split_col = id_cols[0]
    _, _, test_ids = trajectory_splits(df, test_fraction=0.2, val_fraction=0.1, seed=seed)
    test_mask = df[split_col].isin(test_ids).to_numpy()
    test_positions = np.flatnonzero(test_mask)
    map_idx = map_indices_into_test(seed, len(test_positions), n_map=200)

    out: dict[int, np.ndarray] = {}
    last_lat = f"x_t{history_steps - 1:03d}_lat"
    last_lon = f"x_t{history_steps - 1:03d}_lon"

    for row in map_rows:
        anchor = anchors[row]
        sample_i = int(test_positions[int(map_idx[row])])
        row_df = df.iloc[[sample_i]]
        if not (
            np.isclose(row_df[last_lat].iloc[0], anchor[0], atol=1e-4)
            and np.isclose(row_df[last_lon].iloc[0], anchor[1], atol=1e-4)
        ):
            # fallback: nearest anchor match in test set
            test_df = df.iloc[test_positions]
            dlat = test_df[last_lat].to_numpy() - anchor[0]
            dlon = test_df[last_lon].to_numpy() - anchor[1]
            sample_i = int(test_positions[int(np.argmin(dlat * dlat + dlon * dlon))])
            row_df = df.iloc[[sample_i]]

        history = np.empty((history_steps, 2), dtype=np.float64)
        for t in range(history_steps):
            history[t, 0] = row_df[f"x_t{t:03d}_lat"].iloc[0]
            history[t, 1] = row_df[f"x_t{t:03d}_lon"].iloc[0]
        out[row] = history

    return out


def pick_examples(err_km: np.ndarray, stationary: np.ndarray, seed: int) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    moving = ~stationary
    low_pool = np.where(moving & (err_km < 0.5))[0]
    high_pool = np.where(err_km > 50)[0]
    if len(low_pool) == 0:
        low_pool = np.argsort(err_km)[:5]
    if len(high_pool) == 0:
        high_pool = np.argsort(err_km)[-5:]
    return int(rng.choice(low_pool)), int(rng.choice(high_pool))


def add_example(
    m,
    layer,
    *,
    title: str,
    history: np.ndarray,
    anchor: np.ndarray,
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    horizon_step: int,
    horizon_hours: float,
) -> None:
    import folium

    hist_pts = [[float(p[0]), float(p[1])] for p in history]
    anchor_pt = [float(anchor[0]), float(anchor[1])]
    start_pt = hist_pts[0]

    gt_pts = [[float(y_true[t, 0]), float(y_true[t, 1])] for t in range(horizon_step + 1)]
    gt_end = gt_pts[-1]
    err_rnn = float(
        haversine_km(y_true[horizon_step, 0], y_true[horizon_step, 1],
                     preds["rnn"][horizon_step, 0], preds["rnn"][horizon_step, 1])
    )

    # 24h history
    folium.PolyLine(
        hist_pts, color=COLORS["history"], weight=3, opacity=0.85,
        popup="היסטוריה — 24 שעות (נתונים שנצפו)",
    ).add_to(layer)

    # future ground truth
    folium.PolyLine(
        [anchor_pt] + gt_pts, color=COLORS["gt"], weight=4, opacity=0.95,
        popup="עתיד אמיתי — 12 שעות (GT)",
    ).add_to(layer)

    for tag, label, dash in [
        ("rnn", "LSTM flat", "8 6"),
        ("rnn_ar", "LSTM-AR", None),
        ("transformer", "Transformer", "6 4"),
    ]:
        pred = preds[tag]
        pred_pts = [[float(pred[t, 0]), float(pred[t, 1])] for t in range(horizon_step + 1)]
        folium.PolyLine(
            [anchor_pt] + pred_pts,
            color=COLORS[tag],
            weight=2.5,
            opacity=0.8,
            dash_array=dash,
            popup=f"חיזוי {label}",
        ).add_to(layer)

    # markers
    folium.CircleMarker(
        start_pt, radius=10, color="#2E7D32", fill=True, fill_color="#4CAF50",
        fill_opacity=1.0, weight=2,
        popup="<b>START</b><br>תחילת היסטוריה (לפני 24 שעות)",
    ).add_to(layer)

    folium.Marker(
        anchor_pt,
        icon=folium.Icon(color="orange", icon="star", prefix="fa"),
        popup="<b>NOW</b><br>סוף 24h היסטוריה — כאן מתחיל החיזוי",
    ).add_to(layer)

    folium.CircleMarker(
        gt_end, radius=11, color="#B71C1C", fill=True, fill_color="#F44336",
        fill_opacity=1.0, weight=2,
        popup=f"<b>TARGET</b><br>יעד אמיתי @ {horizon_hours:.0f}h",
    ).add_to(layer)

    folium.Marker(
        location=[(anchor_pt[0] + start_pt[0]) / 2, (anchor_pt[1] + start_pt[1]) / 2],
        icon=folium.DivIcon(
            icon_size=(260, 48),
            icon_anchor=(0, 0),
            html=(
                f'<div style="font-size:12px;font-weight:700;background:rgba(255,255,255,0.95);'
                f"padding:4px 8px;border-radius:5px;border:2px solid #333;'>"
                f"{title}<br>שגיאת LSTM @ {horizon_hours:.0f}h: {err_rnn:.2f} km</div>"
            ),
        ),
    ).add_to(layer)


def build_map(
    *,
    input_path: Path,
    out_path: Path,
    sample_size: int,
    seed: int,
    horizon_hours: float,
) -> Path:
    try:
        import folium
        from folium import Element
    except ImportError as exc:
        raise SystemExit("folium is required: pip install folium") from exc

    traj_rnn = load_traj_json(RESULTS / "RNN" / "lstm_sample_trajectories.json")
    traj_ar = load_traj_json(RESULTS / "RNN_AR_LSTM" / "lstm_ar_sample_trajectories.json")
    traj_xf = load_traj_json(RESULTS / "Transformer" / "transformer_sample_trajectories.json")

    metrics = json.loads((RESULTS / "RNN" / "lstm_metrics.json").read_text(encoding="utf-8"))
    future_steps = metrics["future_steps"]
    horizon_step = metrics["horizon_step_index"]

    err_km = haversine_km(
        traj_rnn["y_true"][:, horizon_step, 0],
        traj_rnn["y_true"][:, horizon_step, 1],
        traj_rnn["y_pred"][:, horizon_step, 0],
        traj_rnn["y_pred"][:, horizon_step, 1],
    )

    # Mark confined windows so "low error" examples are actually moving vessels.
    motion_cfg = StationaryFilterConfig()
    fut = traj_rnn["y_true"]
    max_radius = np.maximum(
        haversine_km(traj_rnn["anchor"][:, 0], traj_rnn["anchor"][:, 1], fut[:, :, 0], fut[:, :, 1]).max(axis=1),
        0.0,
    )
    future_disp = haversine_km(
        traj_rnn["anchor"][:, 0], traj_rnn["anchor"][:, 1],
        fut[:, horizon_step, 0], fut[:, horizon_step, 1],
    )
    motion_df = pd.DataFrame({"max_radius_km": max_radius, "future_displacement_km": future_disp, "mean_sog_kn": 0.0})
    stationary = stationary_window_mask(motion_df, motion_cfg)

    low_i, high_i = pick_examples(err_km, stationary, seed)
    histories = load_histories_for_map_rows(
        input_path,
        sample_size,
        seed,
        [low_i, high_i],
        anchors={low_i: traj_rnn["anchor"][low_i], high_i: traj_rnn["anchor"][high_i]},
        history_steps=metrics["history_steps"],
    )

    examples = [
        ("שגיאה נמוכה", low_i, float(err_km[low_i])),
        ("שגיאה גבוהה", high_i, float(err_km[high_i])),
    ]

    center_lat = float(np.mean([traj_rnn["anchor"][i, 0] for _, i, _ in examples]))
    center_lon = float(np.mean([traj_rnn["anchor"][i, 1] for _, i, _ in examples]))

    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="CartoDB positron", control_scale=True)

    legend = """
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;
    border:2px solid #444;border-radius:8px;padding:12px 14px;font-size:13px;line-height:1.7;
    box-shadow:0 2px 10px rgba(0,0,0,.2);max-width:320px;">
    <b>מקרא — קרא מימין לשמאל בזמן</b><br>
    <span style="color:#4CAF50">&#9679;</span> <b>START</b> — תחילת היסטוריה (לפני 24h)<br>
  <span style="color:#9E9E9E">&#9644;</span> אפור — מסלול שנצפה (24h)<br>
    <span style="color:orange">&#9733;</span> <b>NOW</b> — סוף היסטוריה, תחילת חיזוי<br>
    <span style="color:#1565C0">&#9644;</span> כחול — עתיד אמיתי (12h)<br>
    <span style="color:#F44336">&#9679;</span> <b>TARGET</b> — יעד אמיתי @ 12h<br>
    <span style="color:#FF9800">- -</span> כתום — LSTM &nbsp;
    <span style="color:#1B5E20">&#9644;</span> ירוק כהה — AR &nbsp;
    <span style="color:#7B1FA2">- -</span> סגול — Transformer<br>
    <i>הפעל/כבה שכבות בפינה הימנית העליונה</i>
    </div>
    """
    m.get_root().html.add_child(Element(legend))

    for title, row_i, _ in examples:
        layer = folium.FeatureGroup(name=f"{title} ({err_km[row_i]:.1f} km)", show=True)
        history = histories[row_i]
        preds = {
            "rnn": traj_rnn["y_pred"][row_i],
            "rnn_ar": traj_ar["y_pred"][row_i],
            "transformer": traj_xf["y_pred"][row_i],
        }
        add_example(
            m,
            layer,
            title=title,
            history=history,
            anchor=traj_rnn["anchor"][row_i],
            y_true=traj_rnn["y_true"][row_i],
            preds=preds,
            horizon_step=horizon_step,
            horizon_hours=horizon_hours,
        )
        layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))

    meta = {
        "low_error_row": low_i,
        "high_error_row": high_i,
        "low_error_km": float(err_km[low_i]),
        "high_error_km": float(err_km[high_i]),
        "seed": seed,
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean trajectory comparison map.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/combined/train.parquet"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon-hours", type=float, default=12.0)
    args = parser.parse_args()

    out = build_map(
        input_path=args.input,
        out_path=args.output,
        sample_size=args.sample,
        seed=args.seed,
        horizon_hours=args.horizon_hours,
    )
    print(f"Saved: {out}")
    print(f"Meta:  {out.with_suffix('.json')}")
    print(f"Open:  http://127.0.0.1:8765/{out.name}")


if __name__ == "__main__":
    main()
