"""How the rice calendar and its signature shift across the AOIs, north to south and east to west.

Why
---
The AOIs span several hundred kilometres. The field plots showed the same crop transplanted two
months apart between regions, and its water showing differently (deep and long in one region,
shallow and short in another). A rule that is checked in one region and applied everywhere has to
carry that knowledge along, and the people reading the maps need to see the gradient: transplant
date, canopy timing, water signature and the previous crop's harvest, laid out by latitude and
longitude. This module builds that table from what the other modules already measured — nothing is
re-derived — and draws it.

Coordinates and region names are client data: the table is written under ``processed/`` and never
into the repository's docs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def aoi_centroids(polygons: dict) -> pd.DataFrame:
    """``aoi``, ``lon``, ``lat`` of every AOI polygon (lon/lat GeoJSON, as ``season_screen.aoi_polygons``)."""
    from shapely.geometry import shape

    rows = []
    for key, geom in polygons.items():
        c = shape(geom).centroid
        rows.append({"aoi": key, "lon": round(c.x, 4), "lat": round(c.y, 4)})
    return pd.DataFrame(rows)


def plot_calendar(plot_events: pd.DataFrame, plot_signatures: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per AOI, the medians of what the field plots showed: trough, onset, last NDVI, water shares."""
    def med_date(s):
        s = pd.to_datetime(s.dropna())
        return None if s.empty else s.median().strftime("%d %b")

    g = plot_events.groupby("aoi")
    out = pd.DataFrame({
        "plots": g.size(),
        "trough": g["trough_date"].agg(med_date),
        "trough_ndvi": g["trough_ndvi"].median().round(2),
        "greenup_onset": g["onset_date"].agg(med_date),
        "trough_to_onset_days": g["trough_to_onset_days"].median(),
        "last_ndvi": g["last_ndvi"].median().round(2),
        "optical_open_water_pct": (100 * g["wet_at_trough"].mean()).round(0),
    })
    if plot_signatures is not None and len(plot_signatures):
        s = plot_signatures.groupby("aoi")
        out["transplant_from_cycle"] = s["transplant_date"].agg(med_date)
        out["peak_ndvi"] = s["peak_ndvi"].median().round(2)
        out["radar_VH_dip_db"] = s["VH_dip"].median().round(1)
        out["radar_wet_pct"] = (100 * s["radar_wet"].mean()).round(0)
    return out.reset_index()


def previous_harvest(curves: pd.DataFrame, before="2026-05-01", min_pixels: int = 3) -> pd.DataFrame:
    """Per AOI, when the crop before the season was cut: the median date the plot's NDVI last fell
    below half of its pre-season peak, read off the plot curves."""
    rows = []
    for aoi, g in curves[curves["n_pixels"] >= min_pixels].groupby("aoi"):
        dates = []
        for _, p in g.groupby("plot_id"):
            p = p.sort_values("window")
            pre = p[p["window"] < before]
            if pre.empty or pre["ndvi"].max() < 0.5:
                continue
            peak_i = pre["ndvi"].idxmax()
            after = pre.loc[peak_i:]
            half = pre.loc[peak_i, "ndvi"] / 2
            fell = after[after["ndvi"] <= half]
            if not fell.empty:
                dates.append(fell["window"].iloc[0])
        rows.append({"aoi": aoi, "plots_with_previous_crop": len(dates),
                     "previous_harvest": pd.Series(dates).median().strftime("%d %b") if dates else None,
                     "previous_harvest_month": pd.Series(dates).median().month if dates else None})
    return pd.DataFrame(rows)


def shift_table(centroids: pd.DataFrame, calendar: pd.DataFrame, harvest: pd.DataFrame | None = None,
                rule: pd.DataFrame | None = None, region_of: dict | None = None) -> pd.DataFrame:
    """Everything per AOI, sorted north to south, with a rough days-per-degree-latitude figure in ``attrs``."""
    t = centroids.merge(calendar, on="aoi", how="inner")
    if harvest is not None:
        t = t.merge(harvest, on="aoi", how="left")
    if rule is not None:
        keep = [c for c in ("aoi", "rice_acres", "rice_unconfirmed_acres", "young_acres", "harvested_acres",
                            "radar_wet_pct_of_rice", "trough_median", "climb_median") if c in rule]
        t = t.merge(rule[keep], on="aoi", how="left")
    if region_of:
        t.insert(1, "region", t["aoi"].map(region_of))
    t = t.sort_values("lat", ascending=False).reset_index(drop=True)
    doy = pd.to_datetime(t["trough"] + " 2026", format="%d %b %Y", errors="coerce").dt.dayofyear
    ok = doy.notna() & t["lat"].notna()
    if ok.sum() >= 3:
        slope = np.polyfit(t.loc[ok, "lat"], doy[ok], 1)[0]
        t.attrs["trough_days_per_degree_latitude"] = round(float(slope), 1)
    return t


def figure(t: pd.DataFrame, out_path=None):
    """Two panels: the AOIs on a lon/lat plane coloured by trough date, and trough date against latitude."""
    import matplotlib.pyplot as plt

    from .curves import INK, INK_MUTED, SURFACE

    doy = pd.to_datetime(t["trough"] + " 2026", format="%d %b %Y", errors="coerce").dt.dayofyear
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6), facecolor=SURFACE)
    for a in (a1, a2):
        a.set_facecolor(SURFACE)
        a.grid(True, alpha=.2)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    sc = a1.scatter(t["lon"], t["lat"], c=doy, cmap="viridis", s=90, edgecolor=INK, linewidth=.5)
    for _, r in t.iterrows():
        a1.annotate(r["aoi"], (r["lon"], r["lat"]), fontsize=7, xytext=(4, 3), textcoords="offset points", color=INK)
    cb = fig.colorbar(sc, ax=a1, fraction=.04, pad=.02)
    cb.set_label("trough (transplant) date, day of year")
    a1.set_xlabel("longitude")
    a1.set_ylabel("latitude")
    a1.set_title("where each AOI is, coloured by when its rice went in", loc="left", fontsize=10)
    a2.scatter(doy, t["lat"], s=70, color="#2a78d6", edgecolor=INK, linewidth=.5)
    for _, r in t.iterrows():
        a2.annotate(r["aoi"], (pd.to_datetime(r["trough"] + " 2026", format="%d %b %Y").dayofyear, r["lat"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points", color=INK_MUTED)
    ticks = pd.to_datetime(["2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01", "2026-09-01"])
    a2.set_xticks(ticks.dayofyear)
    a2.set_xticklabels([d.strftime("%b") for d in ticks])
    a2.set_ylabel("latitude")
    slope = t.attrs.get("trough_days_per_degree_latitude")
    a2.set_title(f"transplant date against latitude" + (f"  (~{slope:+.0f} days per degree north)" if slope is not None else ""),
                 loc="left", fontsize=10)
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110, facecolor=SURFACE)
    return fig
