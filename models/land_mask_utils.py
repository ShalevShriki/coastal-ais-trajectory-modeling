"""Coastal vs inland classification and soft land-grid penalty for training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Contiguous US + nearby coasts used by this project
DEFAULT_LAT_MIN = 18.0
DEFAULT_LAT_MAX = 50.0
DEFAULT_LON_MIN = -126.0
DEFAULT_LON_MAX = -66.0
DEFAULT_RES_DEG = 0.05  # ~5.5 km


def _globe():
    from global_land_mask import globe

    return globe


def is_land_latlon(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Vectorized land mask (True = land)."""
    globe = _globe()
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    out = np.empty(lat.shape, dtype=bool)
    flat_lat = lat.ravel()
    flat_lon = lon.ravel()
    flat_out = out.ravel()
    chunk = 200_000
    for i in range(0, flat_lat.size, chunk):
        j = min(i + chunk, flat_lat.size)
        flat_out[i:j] = globe.is_land(flat_lat[i:j], flat_lon[i:j])
    return out


def has_open_water_within(
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    km: float = 10.0,
) -> np.ndarray:
    """True if the point itself is water OR any of 8 neighbors at ~km is water."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    land0 = is_land_latlon(lat, lon)
    water = ~land0
    if km <= 0:
        return water

    dlat = km / 111.0
    dlon = km / (111.0 * np.cos(np.deg2rad(np.clip(lat, -60.0, 60.0))))
    offsets = [
        (dlat, 0.0),
        (-dlat, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (dlat, 1.0),
        (dlat, -1.0),
        (-dlat, 1.0),
        (-dlat, -1.0),
    ]
    for dla, sign_lo in offsets:
        lo2 = lon + sign_lo * dlon
        la2 = lat + dla
        water |= ~is_land_latlon(la2, lo2)
    return water


def inland_point_mask(lat: np.ndarray, lon: np.ndarray, *, open_water_km: float = 10.0) -> np.ndarray:
    """Strongly inland: land and no open water within open_water_km."""
    return ~has_open_water_within(lat, lon, km=open_water_km)


def inland_window_mask_from_history(
    lat_hist: np.ndarray,
    lon_hist: np.ndarray,
    *,
    open_water_km: float = 10.0,
    inland_fraction: float = 0.5,
    subsample_every: int = 6,
) -> np.ndarray:
    """
    Mark windows as inland when a majority of (subsampled) history points are inland.

    lat_hist/lon_hist: (N, T)
    """
    if lat_hist.ndim != 2:
        raise ValueError("expected (N, T) history arrays")
    idx = np.arange(0, lat_hist.shape[1], max(1, subsample_every))
    inland = inland_point_mask(lat_hist[:, idx], lon_hist[:, idx], open_water_km=open_water_km)
    frac = inland.mean(axis=1)
    return frac > inland_fraction


def build_or_load_land_grid(
    cache_path: Path,
    *,
    lat_min: float = DEFAULT_LAT_MIN,
    lat_max: float = DEFAULT_LAT_MAX,
    lon_min: float = DEFAULT_LON_MIN,
    lon_max: float = DEFAULT_LON_MAX,
    res_deg: float = DEFAULT_RES_DEG,
) -> dict:
    """Return dict with land grid (1=land, 0=water) and geo metadata; cache as .npz."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = np.load(cache_path)
        return {
            "land": data["land"].astype(np.float32),
            "lat_min": float(data["lat_min"]),
            "lat_max": float(data["lat_max"]),
            "lon_min": float(data["lon_min"]),
            "lon_max": float(data["lon_max"]),
            "res_deg": float(data["res_deg"]),
        }

    lats = np.arange(lat_min, lat_max + 1e-9, res_deg, dtype=np.float64)
    lons = np.arange(lon_min, lon_max + 1e-9, res_deg, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    land = is_land_latlon(lat_grid, lon_grid).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        land=land,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        res_deg=res_deg,
    )
    return {
        "land": land,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "res_deg": res_deg,
    }


class SoftLandPenalty(torch.nn.Module):
    """Bilinear-sample a coarse land raster at predicted lat/lon (differentiable)."""

    def __init__(self, grid: dict, weight: float = 0.1):
        super().__init__()
        self.weight = float(weight)
        land = torch.as_tensor(grid["land"], dtype=torch.float32)  # (H, W)
        # grid_sample expects (N, C, H, W)
        self.register_buffer("land", land.unsqueeze(0).unsqueeze(0))
        self.lat_min = float(grid["lat_min"])
        self.lat_max = float(grid["lat_max"])
        self.lon_min = float(grid["lon_min"])
        self.lon_max = float(grid["lon_max"])

    def forward(self, pred_abs_latlon: torch.Tensor) -> torch.Tensor:
        """
        pred_abs_latlon: (B, T, 2) with [...,0]=lat, [...,1]=lon in degrees.
        Returns scalar penalty (mean land occupancy of predicted points).
        """
        if self.weight <= 0:
            return pred_abs_latlon.new_zeros(())

        lat = pred_abs_latlon[..., 0]
        lon = pred_abs_latlon[..., 1]
        # Normalize to [-1, 1] for grid_sample (x=lon, y=lat). align_corners=True.
        x = 2.0 * (lon - self.lon_min) / (self.lon_max - self.lon_min) - 1.0
        y = 2.0 * (lat - self.lat_min) / (self.lat_max - self.lat_min) - 1.0
        # Outside bbox → treat as 0 (no penalty) by clamping sample coords? Better: zero via mask
        inside = (x >= -1) & (x <= 1) & (y >= -1) & (y <= 1)
        grid = torch.stack([x, y], dim=-1)  # (B, T, 2)
        b, t, _ = grid.shape
        sample = F.grid_sample(
            self.land.expand(b, -1, -1, -1),
            grid.view(b, 1, t, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (B, 1, 1, T)
        land_prob = sample.view(b, t)
        land_prob = land_prob * inside.float()
        return self.weight * land_prob.mean()
