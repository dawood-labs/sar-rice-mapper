"""Systematic review of the delivered map, AOI by AOI, before anything is fixed.

Why
---
The user's visual review of two AOIs found eight kinds of error in a few hours (docs outside the
repository: ``review_issues_2026.md``). Before any rule is changed, the errors must be found and
understood across many AOIs, so each fix addresses a cause and its size is known. This module does
the finding; people (or reviewers) decide what is wrong.

For one AOI:

1. ``panel`` — one picture: the latest clear Sentinel-2 image (5-3-2 and true colour), the radar
   rice composite (``sar_composites``: red = paddy, yellow = dry-land crop, white = trees/houses),
   and the field-labelled class map, side by side. Where they disagree, look closer.
2. ``field_evidence`` — for every field (>= 4 pixels) the evidence the rule does not combine:
   the last clear optical NDVI (Cloud Score+ >= 60, last 30 days), the dry-season NDVI (March-April,
   clear), the field-mean radar at the end of the season against its canopy peak (a harvest is a
   drop), the monsoon flood seen anywhere (darkest VH, local drop), and whether the ground was ever
   dark at all (villages and trees are not).
3. ``suspects`` — fields whose label contradicts that evidence, in named categories that match the
   review issues (harvested but still standing, water and canopy but not rice, built-up or trees
   in a crop class, young vs young-rice on the water threshold, rice with weak evidence, standing
   judged on an old observation). Each category is a question, not a verdict.
4. ``render`` — curve and chip sheets (``qgis_review.inspect_field``) for the largest suspects.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SRC = "processed/_batch/s2_2026"
OUT = f"{SRC}/review"
MAP_DATE = pd.Timestamp("2026-09-21")
CLEAR_CS = 60

CATEGORIES = {
    "harvested_but_standing": "class 4, but a clear late-season NDVI >= 0.5 or no radar harvest drop",
    "water_and_canopy_not_rice": "class 0/3, but a monsoon flood in the radar and a canopy now",
    "builtup_or_trees_in_crop_class": "class 1/2/3/4/6, but green in the dry season or radar never dark",
    "young_on_water_threshold": "class 2, a young crop with a shallow flood (2.5-4 dB) - or class 6 just above it",
    "rice_weak_evidence": "class 1, but the field-mean rule disagrees or no flood seen at any date",
    "standing_extrapolated": "class 1/6, but the last clear observation is more than 30 days old",
}


# ---------------------------------------------------------------- evidence per field
def _nanmed(a, axis=0):
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def field_evidence(aoi_id: int, drop_passes=()) -> pd.DataFrame:
    """One row per field with >= 4 pixels inside the AOI: label, field_id and the evidence columns.

    `drop_passes` (dates like "2026-06-11") removes those radar passes before the water evidence is
    computed. Why: a pass can be partly broken (one polarisation dark, the other normal) yet stay
    under the bad-pass cut; comparing the evidence with and without it shows which fields rely on it.
    """
    import pyogrio

    from .analysis import ndvi_5day as nd
    from .analysis import pixel_report as pr
    from .analysis import radar_water as rw
    from .analysis import sar_curve
    from .analysis.field_level import field_means, to_db, to_power
    from .analysis.field_rice import NODATA, label_aoi

    fields, idx, classes = label_aoi(aoi_id)
    gpkg = pyogrio.read_dataframe(Path(SRC) / "fields" / f"aoi{aoi_id}_fields_monsoon2026.gpkg", read_geometry=False)
    owner = np.where(classes.ravel() != NODATA, idx.ravel(), -1)
    n_f = len(fields)
    npx = np.bincount(owner[owner >= 0], minlength=n_f)
    keep = npx >= 4
    # optical, strict clear mask
    dates, ndvi, _, _, ok, _, _ = nd.read_dates(aoi_id, cs_min=CLEAR_CS)
    dates = pd.DatetimeIndex(dates)
    v = np.where(ok, ndvi, np.nan).reshape(len(dates), -1)
    fm = field_means(v, owner, n_f)[:, keep]                    # NaN where no clear pixel
    seen = field_means(ok.reshape(len(dates), -1).astype(float), owner, n_f)[:, keep] >= 0.5
    fm = np.where(seen, fm, np.nan)
    late = np.asarray(dates >= MAP_DATE - pd.Timedelta(days=30))
    dry = np.asarray((dates >= "2026-03-01") & (dates < "2026-05-01"))
    last_i = np.array([np.flatnonzero(c)[-1] if c.any() else -1 for c in seen.T])
    out = gpkg.loc[keep, ["field_id", "area_acres", "pixels", "label", "rice_share", "unconfirmed_share",
                          "label_confidence"]].reset_index(drop=True)
    # where the field is on the AOI grid (same rows/cols as the review panel's images)
    width = classes.shape[1]
    rr, cc = np.divmod(np.arange(owner.size), width)
    has = owner >= 0
    out["row"] = (np.bincount(owner[has], weights=rr[has], minlength=n_f) / np.maximum(npx, 1))[keep].round(0)
    out["col"] = (np.bincount(owner[has], weights=cc[has], minlength=n_f) / np.maximum(npx, 1))[keep].round(0)
    out["last_clear_date"] = np.where(last_i >= 0, dates.to_numpy()[np.clip(last_i, 0, None)], np.datetime64("NaT"))
    out["last_clear_ndvi"] = np.where(last_i >= 0, fm[np.clip(last_i, 0, None), np.arange(fm.shape[1])], np.nan)
    out["late_clear_ndvi_max"] = np.nanmax(np.where(late[:, None], fm, -np.inf), axis=0)
    out.loc[~np.isfinite(out["late_clear_ndvi_max"]), "late_clear_ndvi_max"] = np.nan
    out["dry_season_ndvi"] = _nanmed(np.where(dry[:, None], fm, np.nan))
    # radar, field means in power over single pixels
    loc = pr.locate(aoi_id, 0, season_key="monsoon2026")
    vh_all, drops_all, days_all = [], [], []
    vh_end, vh_peak = [], []
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        rd, cubes = sar_curve.read_track(loc, track, 1)
        rdi = pd.DatetimeIndex(rd)
        if len(drop_passes):
            k = ~rdi.normalize().isin(pd.DatetimeIndex(drop_passes))
            rd, rdi, cubes = rd[k], rdi[k], {b: c[k] for b, c in cubes.items()}
        vh = to_db(field_means(to_power(cubes["VH"].reshape(len(rd), -1)), owner, n_f))[:, keep]
        vh_all.append(vh)
        days_all.append(rdi)
        drops_all.append(rw.local_drops(rd, vh))
        end = np.asarray(rdi >= MAP_DATE - pd.Timedelta(days=20))
        canopy = np.asarray((rdi >= "2026-07-15") & (rdi < MAP_DATE - pd.Timedelta(days=20)))
        vh_end.append(_nanmed(np.where(end[:, None], vh, np.nan)))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vh_peak.append(np.nanmax(np.where(canopy[:, None], vh, np.nan), axis=0))
    VH = np.concatenate(vh_all)
    D = np.concatenate(drops_all)
    days = pd.DatetimeIndex(np.concatenate([d.to_numpy() for d in days_all]))
    monsoon = np.asarray(days >= "2026-05-15")[:, None]
    with np.errstate(invalid="ignore"):
        flood_pass = monsoon & (D >= 4) & (VH <= -19)
        dark_pass = monsoon & (VH <= -20)
    out["flood_passes"] = flood_pass.sum(axis=0)
    out["dark_monsoon_passes"] = dark_pass.sum(axis=0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["best_monsoon_drop"] = np.nanmax(np.where(monsoon, D, np.nan), axis=0)
        out["darkest_monsoon_vh"] = np.nanmin(np.where(monsoon, VH, np.nan), axis=0)
        srt = np.sort(np.where(np.isfinite(VH), VH, np.inf), axis=0)
    out["vh_second_darkest"] = np.where(np.isfinite(srt[1]), srt[1], np.nan)
    out["vh_median"] = _nanmed(VH)
    out["vh_end"] = np.nanmean(np.stack(vh_end), axis=0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["vh_canopy_peak"] = np.nanmax(np.stack(vh_peak), axis=0)
    out["vh_drop_end"] = out["vh_canopy_peak"] - out["vh_end"]
    # the rule re-run on field means (analysis/field_level), same order
    fl_path = Path(SRC) / "report" / "field_level" / f"aoi{aoi_id}.parquet"
    if fl_path.exists():
        fl = pd.read_parquet(fl_path)
        if len(fl) == len(out) and (fl["label"].to_numpy() == out["label"].to_numpy()).all():
            for c in ("field_rule_label", "trough_date", "climb_date", "last_ndvi", "flood_drop", "flood_vh"):
                out[c] = fl[c].to_numpy()
    out["days_since_clear"] = (MAP_DATE - pd.to_datetime(out["last_clear_date"])).dt.days
    out["aoi"] = f"aoi{aoi_id}"
    nd.forget()
    return out


def pass_dependence(aoi_id: int, date: str) -> pd.DataFrame:
    """Fields whose flood evidence disappears when the radar pass on `date` is removed.

    Why: a review claim like "this rice is only water on one suspicious pass" must be measured, not
    eyeballed. Returns every field with its label, acres and flood passes with / without the pass,
    plus `relies_on_pass` (had a flood pass, has none without it).
    """
    a = field_evidence(aoi_id)
    b = field_evidence(aoi_id, drop_passes=[date])
    out = a[["field_id", "area_acres", "pixels", "label", "flood_passes", "dark_monsoon_passes"]].copy()
    out["flood_passes_without"] = b["flood_passes"].to_numpy()
    out["dark_passes_without"] = b["dark_monsoon_passes"].to_numpy()
    out["relies_on_pass"] = (out["flood_passes"] >= 1) & (out["flood_passes_without"] == 0)
    out["dropped_pass"] = date
    return out


def nan_spread(aoi_id: int, window: int = 5, pol: str = "VH") -> pd.DataFrame:
    """Per radar pass: pixels with no data, and pixels lost after the 5x5 box mean of ``read_track``.

    Why: ``scipy.ndimage.uniform_filter`` keeps a running sum along each row and column, so one NaN
    pixel turns every later pixel of that line into NaN. A pass with a small no-data strip (issue 14)
    can then vanish over most of the AOI for the rule, the field curves and the radar composite.
    ``expected_lost_pct`` is what a NaN-aware box mean would lose (the no-data pixels grown by the
    window), ``lost_pct`` is what ``read_track`` loses now.
    """
    import rasterio
    from scipy.ndimage import binary_dilation, uniform_filter

    from .analysis import pixel_report as pr

    loc = pr.locate(aoi_id, 0, season_key="monsoon2026")
    rows = []
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        with rasterio.open(Path(loc["run"]) / "stack" / f"track_{track}" / f"stack_{pol}.vrt") as ds:
            cube = ds.read().astype("float32")
            nodata, names = ds.nodata, ds.descriptions
        for i, name in enumerate(names):
            a = cube[i]
            bad = ~np.isfinite(a) | ((a == nodata) if nodata is not None else False)
            if bad.any():
                power = np.where(bad, np.nan, 10 ** (a / 10))
                lost = np.isnan(uniform_filter(power, size=window, mode="nearest"))
                grown = binary_dilation(bad, np.ones((window, window), bool))
            else:
                lost = grown = bad
            rows.append({"aoi": f"aoi{aoi_id}", "track": track, "date": name.rsplit("_", 1)[1],
                         "nodata_pct": round(100 * bad.mean(), 2),
                         "expected_lost_pct": round(100 * grown.mean(), 2),
                         "lost_pct": round(100 * lost.mean(), 2)})
    return pd.DataFrame(rows)


def suspects(ev: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``ev`` that fall into a review category (a field can fall into several)."""
    lab = ev["label"]
    canopy_now = (ev["late_clear_ndvi_max"] >= 0.5) | (ev.get("last_ndvi", pd.Series(np.nan, index=ev.index)) >= 0.5)
    flood = (ev["flood_passes"] >= 1) | (ev["dark_monsoon_passes"] >= 2)
    never_dark = (ev["vh_median"] > -15.5) & (ev["vh_second_darkest"] > -17.5)
    rules = {
        "harvested_but_standing": (lab == 4) & ((ev["late_clear_ndvi_max"] >= 0.5) | (ev["vh_drop_end"] < 1.5)),
        "water_and_canopy_not_rice": lab.isin([0, 3]) & flood & canopy_now,
        "builtup_or_trees_in_crop_class": lab.isin([1, 2, 3, 4, 6]) & ((ev["dry_season_ndvi"] >= 0.6) | never_dark),
        "young_on_water_threshold": ((lab == 2) & (ev["best_monsoon_drop"].between(2.5, 4)))
                                    | ((lab == 6) & (ev["best_monsoon_drop"].between(4, 5))),
        "rice_weak_evidence": (lab == 1) & ((ev.get("field_rule_label", lab) != 1) | ~flood),
        "standing_extrapolated": lab.isin([1, 6]) & (ev["days_since_clear"] > 30),
    }
    parts = []
    for name, m in rules.items():
        s = ev[m.fillna(False)].copy()
        s.insert(0, "category", name)
        parts.append(s)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------- pictures
def _stretch(a, lo_p=2, hi_p=98):
    good = np.isfinite(a) & (a > 0)
    if not good.any():
        return np.zeros_like(a)
    lo, hi = np.percentile(a[good], [lo_p, hi_p])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def latest_clear(aoi_id: int, min_clear: float = 0.8, since: str = "2026-08-01"):
    """(path, date, clear share) of the latest Sentinel-2 date since ``since`` whose AOI is at least
    ``min_clear`` Cloud Score+ clear; else the clearest date since then."""
    import rasterio

    from .analysis import ndvi_5day as nd
    from .analysis import pixel_report as pr
    from .optical_export import band_index

    loc = pr.locate(aoi_id, 0)
    inside = nd.inside_aoi(aoi_id)
    rows = []
    for p in sorted(pr.sync_s2(loc, f"data/{nd.FOLDER}", nd.FOLDER).glob("*.tif")):
        d = pd.Timestamp(p.stem.rsplit("_S2_", 1)[1])
        if d < pd.Timestamp(since):
            continue
        with rasterio.open(p) as ds:
            cs = ds.read(band_index(ds, "clear")).ravel()[inside]
            b4 = ds.read(band_index(ds, "B4")).ravel()[inside]
        valid = b4 > 0
        rows.append((p, d, float(((cs >= CLEAR_CS) & valid).mean()), float(valid.mean())))
    nd.forget()
    good = [r for r in rows if r[2] >= min_clear]
    pick = max(good, key=lambda r: r[1]) if good else max(rows, key=lambda r: r[2])
    return pick[0], pick[1], pick[2]


def panel(aoi_id: int, out_dir=OUT) -> Path:
    """S2 latest clear (5-3-2 and 4-3-2), radar rice RGB and field-labelled class map, one PNG."""
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.colors import ListedColormap

    from .analysis.field_rice import field_label_raster, label_aoi
    from .optical_export import band_index
    from .qgis_review import COLOURS
    from .sar_composites import OUT as SAR_OUT

    path, date, share = latest_clear(aoi_id)
    with rasterio.open(path) as ds:
        b = {n: ds.read(band_index(ds, n)).astype("float32") for n in ("B2", "B3", "B4", "B5")}
    s532 = np.dstack([_stretch(b["B5"]), _stretch(b["B3"]), _stretch(b["B2"])])
    s432 = np.dstack([_stretch(b["B4"]), _stretch(b["B3"]), _stretch(b["B2"])])
    with rasterio.open(Path(SAR_OUT) / "per_aoi" / f"aoi{aoi_id}_sar_rice_rgb.tif") as ds:
        rgb = ds.read().astype("float32")
    rgb = np.where(rgb == -32768, np.nan, rgb / 100)
    sar = np.nan_to_num(np.clip((rgb - (-26)) / 14, 0, 1)).transpose(1, 2, 0)
    fields, idx, classes = label_aoi(aoi_id)
    fmap = field_label_raster(idx, fields["label"].to_numpy(), classes).astype("float32") if len(fields) else classes
    fmap[fmap == 255] = np.nan
    fig, axes = plt.subplots(1, 4, figsize=(24, 7.5))
    for a, img, title in zip(axes[:3], (s432, s532, sar),
                             (f"S2 {date.date()} true colour (clear {100 * share:.0f} %)", "S2 5-3-2",
                              "radar: R VH Aug-Sep, G VH min Jun-Jul, B VH Apr (red = paddy)")):
        a.imshow(img, interpolation="nearest")
        a.set_title(title, fontsize=9)
    axes[3].imshow(fmap, cmap=ListedColormap([COLOURS[k][1] for k in range(7)]), vmin=-0.5, vmax=6.5,
                   interpolation="nearest")
    axes[3].set_title("field labels: dark green rice, light green young rice, pale young, orange 3, "
                      "brown harvested, purple never bare, grey not rice", fontsize=8)
    for a in axes:
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle(f"aoi{aoi_id}", x=0.01, ha="left")
    fig.tight_layout()
    out = Path(out_dir) / f"aoi{aoi_id}"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"aoi{aoi_id}_panel.png"
    fig.savefig(p, dpi=70)
    plt.close(fig)
    return p


def render(sus: pd.DataFrame, per_category: int = 3, out_dir=OUT) -> list[str]:
    """Curve and chip sheets for the largest suspects of each category."""
    import matplotlib.pyplot as plt

    from . import qgis_review as q

    done = []
    for cat, g in sus.groupby("category"):
        for fid in g.sort_values("area_acres", ascending=False)["field_id"].head(per_category):
            aoi = fid.split("_")[0]
            try:
                r = q.inspect_field(fid, out_dir=str(Path(out_dir) / aoi / "fields"), keep_cache=True)
                plt.close("all")
                done.append(f"{cat}: {fid}")
            except Exception as exc:          # one bad field must not stop the review of an AOI
                done.append(f"{cat}: {fid} FAILED {type(exc).__name__}: {exc}")
    from .analysis import ndvi_5day as nd

    nd.forget()
    return done


def review_aoi(aoi_id: int, per_category: int = 3, out_dir=OUT) -> dict:
    out = Path(out_dir) / f"aoi{aoi_id}"
    out.mkdir(parents=True, exist_ok=True)
    ev = field_evidence(aoi_id)
    ev.to_parquet(out / f"aoi{aoi_id}_field_evidence.parquet")
    sus = suspects(ev)
    sus.to_csv(out / f"aoi{aoi_id}_suspects.csv", index=False)
    summary = (sus.groupby("category").agg(fields=("field_id", "size"), acres=("area_acres", "sum")).round(1)
               if len(sus) else pd.DataFrame())
    summary.to_csv(out / f"aoi{aoi_id}_suspects_summary.csv")
    panel(aoi_id, out_dir)
    dq = delineation_qa(aoi_id)
    dq.drop(columns="geometry").to_csv(out / f"aoi{aoi_id}_delineation_flags.csv", index=False)
    pd.DataFrame([delineation_summary(dq)]).to_csv(out / f"aoi{aoi_id}_delineation_summary.csv", index=False)
    delineation_examples(aoi_id, dq, out_dir=out_dir)
    rendered = render(sus, per_category, out_dir) if len(sus) else []
    (out / "rendered.txt").write_text("\n".join(rendered))
    return {"aoi": f"aoi{aoi_id}", "fields": len(ev), "suspects": len(sus), "rendered": len(rendered)}


# ---------------------------------------------------------------- delineation boundaries
DELIN_TYPES = {
    "monster_over_smaller": "a large polygon at least 30 % covered by smaller polygons traced on top of it",
    "tail_or_spike": "a polygon losing >= 15 % of its area to a 10 m opening (thin tails, spikes, necks)",
    "tiny_same_class_inside": "a polygon under 0.05 acre (2 pixels) whose neighbours all carry its label",
    "sub_pixel": "a polygon smaller than one 10 m pixel (0.025 acre)",
    "jagged_outline": "an outline with more than 2 vertices per metre of perimeter (pixel staircase)",
}


def delineation_qa(aoi_id: int) -> pd.DataFrame:
    """Per polygon of the AOI's delineation: geometry flags that make the final boundaries look wrong."""
    import geopandas as gpd

    import shapely

    g = gpd.read_file(Path(SRC) / "fields" / f"aoi{aoi_id}_fields_monsoon2026.gpkg").to_crs("EPSG:32646")
    g["geometry"] = g.geometry.make_valid()
    g["a"] = g.geometry.area
    # vertex density: traced outlines follow the high-resolution pixels as staircases; very dense
    # outlines are slow to process and look jagged (vertices per metre of perimeter)
    g["vertices"] = shapely.get_num_coordinates(g.geometry.to_numpy())
    g["vertices_per_m"] = g["vertices"] / g.geometry.length.clip(lower=1)
    # the geometry checks below run on outlines simplified by 0.5 m (far below the 10 m pixel)
    simple = g.geometry.simplify(0.5, preserve_topology=True)
    # overlaps: area of each polygon covered by SMALLER polygons
    gs = g[["a", "label"]].assign(geometry=simple)
    gs = gpd.GeoDataFrame(gs, geometry="geometry", crs=g.crs)
    pairs = gpd.sjoin(gs[["geometry", "a"]], gs[["geometry", "a"]], predicate="intersects", how="inner")
    pairs = pairs[pairs.index != pairs["index_right"]]
    pairs = pairs[pairs["a_right"] < pairs["a_left"]]
    geom = gs.geometry
    inter = shapely.area(shapely.intersection(geom.loc[pairs.index].to_numpy(),
                                              geom.loc[pairs["index_right"]].to_numpy()))
    covered = pd.Series(inter, index=pairs.index).groupby(level=0).sum()
    g["covered_by_smaller"] = (covered.reindex(g.index).fillna(0) / g["a"]).clip(0, 1)
    # tails: area lost to an opening of 5 m (removes parts narrower than 10 m)
    opened = geom.buffer(-5, resolution=2, join_style="mitre").buffer(5, resolution=2, join_style="mitre")
    g["opening_loss"] = (1 - opened.area / geom.area.clip(lower=1e-6)).clip(0, 1)
    # tiny polygons whose touching neighbours (within 1 m) all share their label
    touch = gpd.sjoin(gs[["geometry", "label"]], gs[["geometry", "label"]], predicate="dwithin",
                      distance=1.0, how="inner")
    touch = touch[touch.index != touch["index_right"]]
    same = touch.groupby(level=0).apply(lambda t: bool((t["label_right"] == t["label_left"]).all()))
    g["neighbours_same_label"] = same.reindex(g.index).fillna(False)
    acre = 4046.8564224
    big = g["a"] >= max(np.percentile(g["a"], 99), 2 * acre)
    flags = {
        "monster_over_smaller": big & (g["covered_by_smaller"] >= 0.3),
        "tail_or_spike": (g["opening_loss"] >= 0.15) & (g["a"] >= 0.05 * acre),
        "tiny_same_class_inside": (g["a"] < 0.05 * acre) & g["neighbours_same_label"],
        "sub_pixel": g["a"] < 0.025 * acre,
        "jagged_outline": g["vertices_per_m"] > 2,
    }
    for k, m in flags.items():
        g[k] = m.to_numpy()
    g["overlap_area"] = g["covered_by_smaller"] * g["a"]
    return g


def delineation_summary(g) -> dict:
    acre = 4046.8564224
    out = {"polygons": len(g), "acres": round(float(g["a"].sum() / acre), 1),
           "overlapping_acres": round(float(g["overlap_area"].sum() / acre), 1),
           "vertices_median": float(g["vertices"].median()), "vertices_max": int(g["vertices"].max()),
           "vertices_per_m_median": round(float(g["vertices_per_m"].median()), 2)}
    for k in DELIN_TYPES:
        out[f"{k}_n"] = int(g[k].sum())
        out[f"{k}_acres"] = round(float(g.loc[g[k], "a"].sum() / acre), 1)
    return out


def delineation_examples(aoi_id: int, g, n: int = 2, half_m: float = 150, out_dir=OUT) -> list[Path]:
    """PNG crops of the latest clear true-colour image with all polygon outlines, the flagged one in magenta."""
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.windows import from_bounds

    from .optical_export import band_index

    path, date, _ = latest_clear(aoi_id)
    made = []
    with rasterio.open(path) as ds:
        for k in DELIN_TYPES:
            pick = g[g[k]].sort_values("a", ascending=False).head(n)
            for _, row in pick.iterrows():
                c = row.geometry.centroid
                minx, miny, maxx, maxy = row.geometry.bounds
                r = max(half_m, (maxx - minx) / 2 + 30, (maxy - miny) / 2 + 30)
                bx = (c.x - r, c.y - r, c.x + r, c.y + r)
                win = from_bounds(*bx, transform=ds.transform)
                img = np.dstack([_stretch(ds.read(band_index(ds, bnd), window=win, boundless=True, fill_value=0)
                                          .astype("float32")) for bnd in ("B4", "B3", "B2")])
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img, extent=(bx[0], bx[2], bx[1], bx[3]))
                near = g[g.intersects(__import__("shapely").geometry.box(*bx))]
                near.boundary.plot(ax=ax, color="yellow", linewidth=0.6)
                g.loc[[row.name]].boundary.plot(ax=ax, color="magenta", linewidth=1.8)
                ax.set_xlim(bx[0], bx[2]), ax.set_ylim(bx[1], bx[3])
                ax.set_xticks([]), ax.set_yticks([])
                ax.set_title(f"{k}: {row['field_id']} ({row['a'] / 4046.86:.2f} ac, label {row['label']})\n"
                             f"S2 {date.date()} true colour; yellow = all polygons", fontsize=8)
                out = Path(out_dir) / f"aoi{aoi_id}" / "delineation"
                out.mkdir(parents=True, exist_ok=True)
                p = out / f"{k}_{row['field_id']}.png"
                fig.savefig(p, dpi=90, bbox_inches="tight")
                plt.close(fig)
                made.append(p)
    return made


# ---------------------------------------------------------------- cloud masks
def mask_disagreement(aoi_id: int, start: str = "2026-05-15", end: str = "2026-09-24") -> pd.DataFrame:
    """Per Sentinel-2 date: share of the AOI flagged by QA60 (opaque, cirrus) and share Cloud Score+ clear.

    Why: the series uses QA60 (bits 10 and 11) as its only cloud mask. A reviewer found a monsoon date
    that Cloud Score+ called 100 % clear while QA60 flagged every pixel as cirrus, i.e. the one clear
    view of the transplanting water was thrown away. This table shows how often that happens. One file
    is read at a time (light on memory).
    """
    import rasterio

    from .analysis import ndvi_5day as nd
    from .analysis import pixel_report as pr
    from .optical_export import band_index

    loc = pr.locate(aoi_id, 0)
    inside = nd.inside_aoi(aoi_id)
    nd.forget()
    rows = []
    for p in sorted(pr.sync_s2(loc, f"data/{nd.FOLDER}", nd.FOLDER).glob("*.tif")):
        d = pd.Timestamp(p.stem.rsplit("_S2_", 1)[1])
        if not (pd.Timestamp(start) <= d < pd.Timestamp(end)):
            continue
        with rasterio.open(p) as ds:
            qa = ds.read(band_index(ds, "QA60")).ravel()[inside].astype("int64")
            cs = ds.read(band_index(ds, "clear")).ravel()[inside]
            b4 = ds.read(band_index(ds, "B4")).ravel()[inside]
        v = b4 > 0
        n = max(int(v.sum()), 1)
        opaque = ((qa >> 10) & 1).astype(bool) & v
        cirrus = ((qa >> 11) & 1).astype(bool) & v
        clear_cs = (cs >= CLEAR_CS) & v
        rows.append({"aoi": f"aoi{aoi_id}", "date": d.date(), "data_pct": round(100 * float(v.mean()), 1),
                     "qa60_opaque_pct": round(100 * opaque.sum() / n, 1),
                     "qa60_cirrus_pct": round(100 * cirrus.sum() / n, 1),
                     "cs_clear_pct": round(100 * clear_cs.sum() / n, 1),
                     "cs_clear_but_qa60_flagged_pct": round(100 * (clear_cs & (opaque | cirrus)).sum() / n, 1),
                     "qa60_clear_but_cs_cloudy_pct": round(100 * (~clear_cs & ~(opaque | cirrus) & v).sum() / n, 1)})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.review", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", nargs="+", type=int, default=[])
    p.add_argument("--per-category", type=int, default=3)
    p.add_argument("--render", nargs="+", default=[], metavar="FIELD_ID",
                   help="only draw curve + chip sheets for these fields (same AOI loaded once)")
    p.add_argument("--out", default=None, help="folder for --render (default: review/<aoi>/fields_extra)")
    p.add_argument("--masks", action="store_true", help="only the QA60 vs Cloud Score+ table per date for --ids")
    p.add_argument("--without-pass", default=None, metavar="DATE",
                   help="only the per-field flood evidence with / without the radar pass on DATE for --ids")
    p.add_argument("--nan-spread", action="store_true",
                   help="only the per-pass no-data / lost-after-5x5 table for --ids (all AOIs if none)")
    args = p.parse_args(argv)
    if args.render:
        import matplotlib.pyplot as plt

        from . import qgis_review as q
        from .analysis import ndvi_5day as nd

        for fid in args.render:
            aoi = fid.split("_")[0]
            out = args.out or str(Path(OUT) / aoi / "fields_extra")
            try:
                info = q.inspect_field(fid, out_dir=out, keep_cache=True)
                info.pop("figures", None)
                plt.close("all")
                print(f"{fid}: {info}")
            except Exception as exc:
                print(f"{fid}: FAILED {type(exc).__name__}: {exc}")
        nd.forget()
        return 0
    import resource
    import time

    if args.nan_spread:
        from .prep import batch

        ids = args.ids or sorted(int(str(a).replace("aoi", "")) for a in batch.aoi_index()["aoi"])
        t = pd.concat([nan_spread(a) for a in ids], ignore_index=True)
        out = Path(SRC) / "report" / "final_audit" / "nan_spread.csv"
        t.to_csv(out, index=False)
        hit = t[t["lost_pct"] > t["expected_lost_pct"] + 1]
        print(f"{len(t)} passes; {len(hit)} lose more than expected, in {hit['aoi'].nunique()} AOIs -> {out}")
        return 0
    if args.without_pass:
        for a in args.ids:
            t = pass_dependence(a, args.without_pass)
            out = Path(OUT) / f"aoi{a}"
            out.mkdir(parents=True, exist_ok=True)
            t.to_csv(out / f"aoi{a}_without_{args.without_pass}.csv", index=False)
            s = t[t["relies_on_pass"]].groupby("label")["area_acres"].agg(["size", "sum"]).round(1)
            print(f"aoi{a}: fields whose only flood pass is {args.without_pass} (polygon acres, ~9 % high):")
            print(s.to_string())
        return 0
    if args.masks:
        for a in args.ids:
            t = mask_disagreement(a)
            out = Path(OUT) / f"aoi{a}"
            out.mkdir(parents=True, exist_ok=True)
            t.to_csv(out / f"aoi{a}_mask_disagreement.csv", index=False)
            print(f"aoi{a}: {len(t)} dates, {int((t['cs_clear_but_qa60_flagged_pct'] >= 50).sum())} dates where "
                  f">= 50 % of the AOI is Cloud Score+ clear but QA60-flagged")
        return 0
    for a in args.ids:
        t0 = time.time()
        r = review_aoi(a, args.per_category)
        # peak memory of this process (kB on Linux): the batch runs several AOIs side by side on a
        # machine with limited RAM, so every AOI reports what it needed
        r["seconds"] = round(time.time() - t0)
        r["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2)
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


