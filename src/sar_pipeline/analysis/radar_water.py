"""Per-pixel radar confirmation of the water at the optical transplant date.

Why
---
Rice is transplanted into water, always; what varies is how much and for how long. The optical
water test (LSWI) sees deep, prolonged flooding well and shallow, short flooding badly: on the first
field-plot delivery it found water in 94-100 % of delta plots and in 0-13 % of the capital-region
plots, where the radar found it in 91 %. So water is confirmed here from Sentinel-1, per pixel, the
way ``sar_water_check`` confirmed it per field plot:

* line every pixel's radar series up on **that pixel's** optical trough date;
* **dry level** = median backscatter 40 to 15 days before the trough (after the previous harvest,
  bare and dry, the brightest the field gets);
* **flood level** = the minimum 10 days before to 15 days after the trough;
* **dip** = dry - flood, in dB, per polarisation, the best over the season's tracks.

A dip of ``DIP_MIN_DB`` or more in either polarisation is ``radar_wet``. Where the season run holds
no dates in the dry window (a trough at the very start of the run), the pixel is ``not checkable``
rather than dry: the map must not confuse "no water" with "could not look".

The radar stacks come from the season run of the Sentinel-1 pipeline (stages 1-5), read as 5x5
linear-power means in dB by ``sar_curve.read_track``; the radar grid must be the optical grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import pixel_report as pr
from . import sar_curve

DRY_WINDOW = (-40, -15)
FLOOD_WINDOW = (-10, 15)
DIP_MIN_DB = 3.0


def dips_for_track(dates, cube, trough, dry=DRY_WINDOW, flood=FLOOD_WINDOW):
    """Dry level, flood level and dip per pixel for one track and one polarisation.

    ``dates`` (n_dates), ``cube`` (n_dates, n_pixels) in dB, ``trough`` (n_pixels) datetime64 with
    NaT where the pixel has no trough. Returns ``(dry_level, flood_level, dip)``; NaN where a window
    holds no date.
    """
    day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
    t = np.asarray(trough).astype("datetime64[D]")
    gap = (day[:, None] - t[None, :]) / np.timedelta64(1, "D")          # NaT -> nan
    with np.errstate(invalid="ignore"):
        in_dry = (gap >= dry[0]) & (gap <= dry[1])
        in_flood = (gap >= flood[0]) & (gap <= flood[1])
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN windows are the "not checkable" case
        dry_level = np.nanmedian(np.where(in_dry, cube, np.nan), axis=0)
        flood_level = np.nanmin(np.where(in_flood, cube, np.nan), axis=0)
    return dry_level, flood_level, dry_level - flood_level


def pixel_dips(aoi_id: int, trough, grid: dict, season_key: str = "monsoon2026", window: int = 5) -> pd.DataFrame:
    """Best dip over tracks per pixel, for VV and VH, plus whether the check was possible at all."""
    loc = pr.locate(aoi_id, 0, season_key=season_key)
    for key in ("crs", "x0", "y0", "width", "height"):
        if loc["grid"][key] != grid[key]:
            raise ValueError(f"radar grid differs from the optical grid in {key}")
    n_pix = int(grid["width"]) * int(grid["height"])
    best = {pol: np.full(n_pix, np.nan) for pol in ("VV", "VH")}
    checkable = np.zeros(n_pix, dtype=bool)
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        dates, cubes = sar_curve.read_track(loc, track, window)
        for pol in ("VV", "VH"):
            _, _, dip = dips_for_track(dates, cubes[pol].reshape(len(dates), -1), trough)
            best[pol] = np.fmax(best[pol], dip)
            checkable |= np.isfinite(dip)
    out = pd.DataFrame({"VV_dip": best["VV"], "VH_dip": best["VH"], "radar_checkable": checkable})
    out["radar_wet"] = checkable & ((out["VV_dip"] >= DIP_MIN_DB) | (out["VH_dip"] >= DIP_MIN_DB))
    return out


def season_run_exists(aoi_id: int, season_key: str = "monsoon2026") -> bool:
    """True when the AOI has a config and a stacked run for the season."""
    try:
        loc = pr.locate(aoi_id, 0, season_key=season_key)
    except Exception:  # noqa: BLE001 - no config for that season is the normal "no" here
        return False
    from pathlib import Path

    return any((Path(loc["run"]) / "stack" / f"track_{t['track_id']}" / "stack_VH.vrt").exists()
               for t in loc["cfg"]["s1"]["tracks"])
