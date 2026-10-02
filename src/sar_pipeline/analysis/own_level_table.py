"""Table for choosing the radar canopy test by each pixel's own levels (fix round 3, stage 2.2, step 2).

Why
---
The rule decides "young rice" / "full canopy" from the radar with fixed dB cut-offs (4, 5, 8 dB, VH -15 dB).
Fields in the same state fall on both sides of those lines (issues 24, 26, aoi72_000108). The replacement reads
each pixel against itself (``radar_water.own_levels``): ``recovery`` = how far VH has come back from the pixel's
own flood level towards its own pre-flood level (0 = still water, 1 = back), and ``rise_z`` = the rise in units
of the pixel's own pass-to-pass noise. For a double crop the pre-flood level is the first crop's canopy, so a
second measure, ``recovery_aoi``, uses the AOI's confirmed-rice canopy level on the map date as the target.

Before any class changes, this module writes those numbers for the pixels we can judge: surveyed plots, the
negatives (evergreen, cut, bare, water), the reviewers' noted fields and a sample of delivered rice / young rice,
so the user can choose the one unitless noise factor ``k`` from evidence. It changes nothing in the map.

Memory: one AOI at a time (the rule's evidence of a large AOI takes several GB); run under ``ulimit -v``.

Use::

    python -m sar_pipeline.analysis.own_level_table --ids 13 28 39 63 72 116 160
    python -m sar_pipeline.analysis.own_level_table --summary     # table by set, from the written files
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = "processed/_batch/s2_2026/report/own_level"
SAMPLE_PER_CLASS = 3000
#: Fields named in the review log (issues 24, 26, 27, 31, 33): the cases the new test must explain.
NOTED_FIELDS = ("aoi13_000095", "aoi13_000549", "aoi13_000269", "aoi13_000438", "aoi13_000470", "aoi13_000277",
                "aoi39_001390", "aoi39_001425", "aoi72_000077", "aoi72_000108", "aoi28_000129", "aoi63_000764",
                "aoi63_001574", "aoi63_000393", "aoi63_000460", "aoi63_001332", "aoi63_001229", "aoi63_001233",
                "aoi63_000909", "aoi63_000897", "aoi63_000184", "aoi63_000413")
COLUMNS = ("flood_ok", "flood_date", "flood_vh", "vh_dry_own", "vh_end_own", "vh_noise_own", "recovery", "rise_z",
           "radar_canopy_rise", "vh_end", "peak_after", "standing", "last_clear_ndvi", "last_clear_age")


def aoi_rice_level(events: pd.DataFrame, classes: np.ndarray) -> float:
    """VH canopy level of the AOI's confirmed rice on the map date: median ``vh_end_own`` of class-1 pixels
    whose optical series shows the full canopy standing (the double-crop target)."""
    from . import monsoon_rule as mr

    full = (classes == 1) & (events["peak_after"].to_numpy() >= mr.CANOPY_MIN) & events["standing"].to_numpy(bool)
    v = events["vh_end_own"].to_numpy()[full]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) >= 50 else np.nan


def aoi_rows(aoi_id: int, plots=None, seed: int = 0) -> pd.DataFrame:
    """One row per judged pixel of one AOI with the own-level measures and the delivered class."""
    import rasterio

    from ..qgis_review import SRC, field_by_id
    from . import monsoon_rule as mr
    from . import ndvi_5day as nd
    from . import validation as va

    _, events, radar = mr.aoi_events(aoi_id)
    if not radar:
        nd.forget()
        return pd.DataFrame()
    with rasterio.open(Path(SRC) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026_final.tif") as ds:
        classes = ds.read(1).ravel()
    level = aoi_rice_level(events, classes)
    p = plots[plots["aoi"] == f"aoi{aoi_id}"] if plots is not None else None
    refs = va.reference_sets(aoi_id, p)
    parts = [refs[["pixel", "set"]].assign(field_id="")]
    for fid in NOTED_FIELDS:
        if fid.startswith(f"aoi{aoi_id}_"):
            _, _, pids = field_by_id(fid)
            parts.append(pd.DataFrame({"pixel": pids, "set": "noted_field", "field_id": fid}))
    rng = np.random.default_rng(seed)
    for c, name in ((1, "delivered_rice"), (6, "delivered_young_rice")):
        pix = np.flatnonzero(classes == c)
        if len(pix):
            pick = rng.choice(pix, min(SAMPLE_PER_CLASS, len(pix)), replace=False)
            parts.append(pd.DataFrame({"pixel": pick, "set": name, "field_id": ""}))
    t = pd.concat(parts, ignore_index=True)
    ev = events.iloc[t["pixel"].to_numpy()][[c for c in COLUMNS if c in events]].reset_index(drop=True)
    t = pd.concat([t.reset_index(drop=True), ev], axis=1)
    t["class"] = classes[t["pixel"].to_numpy()]
    t["aoi"] = f"aoi{aoi_id}"
    t["aoi_rice_vh"] = level
    with np.errstate(invalid="ignore", divide="ignore"):
        t["recovery_aoi"] = (t["vh_end_own"] - t["flood_vh"]) / (level - t["flood_vh"])
    nd.forget()
    return t


def per_track(aoi_id: int, pixels, flood_dates, window: int = 5) -> pd.DataFrame:
    """Every Sentinel-1 track read on its own for the given pixels: does THIS track see the water, and the canopy after it?

    Why: the rule measures the canopy only on the track where the flood was found; an AOI has 2-3 tracks. If the other
    tracks see the same water and the same rise, combining them makes a small rise certain (more passes, less noise).
    Per track and pixel: ``flood`` = darkest VH within 15 days of the pixel's flood date, ``dry`` = the median of 45 to 12
    days before that pass, ``end`` / ``noise`` as in ``radar_water.own_levels``, and ``drop``, ``recovery``, ``rise_z``.
    """
    import warnings

    from . import pixel_report as pr
    from . import radar_water as rw
    from . import sar_curve

    pixels = np.asarray(pixels)
    fd = pd.to_datetime(pd.Series(flood_dates)).to_numpy().astype("datetime64[D]")
    loc = pr.locate(aoi_id, 0, season_key="monsoon2026")
    rows = []
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        dates, cubes = sar_curve.read_track(loc, track, window)
        vh = cubes["VH"].reshape(len(dates), -1)[:, pixels]
        del cubes
        day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
        rel = (day[:, None] - fd[None, :]) / np.timedelta64(1, "D")
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            near = np.where(np.abs(rel) <= 15, vh, np.nan)
            flood = np.nanmin(near, axis=0)
            i_min = np.nanargmin(np.where(np.isfinite(near), near, np.inf), axis=0)
            lag = (day[i_min][None, :] - day[:, None]) / np.timedelta64(1, "D")
            dry = np.nanmedian(np.where((lag >= rw.REF_WINDOW[1]) & (lag <= rw.REF_WINDOW[0]), vh, np.nan), axis=0)
        own = rw.own_levels([(day, None, {"VH": vh})], np.zeros(len(pixels), dtype=int), day.max())
        rows.append(pd.DataFrame({"pixel": pixels, "track": track, "flood": flood, "dry": dry, "end": own["end"],
                                  "noise": own["noise"]}))
    t = pd.concat(rows, ignore_index=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        t["drop"] = t["dry"] - t["flood"]
        t["recovery"] = (t["end"] - t["flood"]) / t["drop"]
        t["rise_z"] = (t["end"] - t["flood"]) / (t["noise"] * np.sqrt(1.0 / rw.END_PASSES + 1.0))
    return t


def combine_tracks(pt: pd.DataFrame) -> pd.DataFrame:
    """Per pixel over its tracks: ``rise_z_all`` = sum of the tracks' rise_z / sqrt(number of tracks) (independent
    looks at the same field add up: three tracks that each show a modest rise make a certain one), mean
    ``recovery_all``, and ``tracks_rising`` = how many tracks show a positive rise."""
    g = pt.dropna(subset=["rise_z"]).groupby("pixel")
    return pd.DataFrame({"rise_z_all": g["rise_z"].sum() / np.sqrt(g["rise_z"].count()),
                         "recovery_all": g["recovery"].mean(), "tracks": g["rise_z"].count(),
                         "tracks_rising": g["rise_z"].apply(lambda z: int((z > 0).sum()))}).reset_index()


def summary(folder: str = OUT, ks=(2.0, 3.0)) -> pd.DataFrame:
    """Per set: flood share, median recovery / rise_z, and for each k the share a pixel-relative test would call
    'radar canopy' (flood ok and rise_z >= k) and 'back at its own level' (also recovery >= 1)."""
    t = pd.concat([pd.read_csv(f) for f in sorted(Path(folder).glob("aoi*.csv"))], ignore_index=True)
    t["group"] = np.where(t["set"] == "noted_field", t["field_id"], t["set"])
    rows = []
    for g, d in t.groupby("group"):
        f = d[d["flood_ok"].astype(bool)]
        row = {"group": g, "pixels": len(d), "flood_ok": round(len(f) / max(len(d), 1), 2),
               "recovery_med": round(float(f["recovery"].median()), 2) if len(f) else np.nan,
               "recovery_aoi_med": round(float(f["recovery_aoi"].median()), 2) if len(f) else np.nan,
               "rise_z_med": round(float(f["rise_z"].median()), 1) if len(f) else np.nan,
               "old_rise_db_med": round(float(f["radar_canopy_rise"].median()), 1) if len(f) else np.nan,
               "class_mode": int(d["class"].mode().iloc[0])}
        for k in ks:
            canopy = d["flood_ok"].astype(bool) & (d["rise_z"] >= k)
            row[f"canopy_k{k:g}"] = round(float(canopy.mean()), 2)
            row[f"full_k{k:g}"] = round(float((canopy & (d[["recovery", "recovery_aoi"]].max(axis=1) >= 1)).mean()), 2)
        rows.append(row)
    return pd.DataFrame(rows)


NEGATIVE_SETS = ("evergreen", "bare_or_built", "water", "cut_before_map_date")
EVIDENCE = ("flood_ok_v2", "flood_vh_v2", "flood_ok", "flood_by_peers", "flood_vh", "flood_z", "flood_pol", "flood_tracks",
            "support", "rise_z_all", "bare_near_flood", "vh_premonsoon", "vh_end_track", "trough_ndvi", "rise",
            "peak_after", "last_ndvi", "last_clear_ndvi", "never_bare", "peer_gap", "peer_level", "peer_spread", "pixel_noise",
            "flood_date", "flood_by_pattern", "open_water_year", "last_clear_age", "ndvi_at_flood_fit", "ndvi_noise_own",
            "ndvi_max_own", "rise_from_flood", "peak_after_flood")


def negative_evidence(aoi_id: int, sandbox: str, baseline: str = "processed/_batch/s2_2026/baseline_v4") -> pd.DataFrame:
    """The negatives (evergreen, bare, water, cut) that a sandbox rule delivers as rice and the baseline did not, with the
    evidence that made them rice. Why: a new water test that raises false rice must be fixed at its cause, not tuned;
    this shows which condition let each pixel through (the radar flood, the peers' water level, the optical cycle) and how
    near it lies to delivered rice (the radar reads a 5 x 5 box, so trees beside paddies share their flood)."""
    import rasterio
    from scipy.ndimage import uniform_filter

    from . import monsoon_rule as mr
    from . import ndvi_5day as nd
    from . import validation as va

    refs = va.reference_sets(aoi_id)
    refs = refs[refs["set"].isin(NEGATIVE_SETS)]
    with rasterio.open(Path(baseline) / f"aoi{aoi_id}_monsoon2026.tif") as ds:
        old = ds.read(1)
    with rasterio.open(Path(sandbox) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026.tif") as ds:
        new = ds.read(1)
    rice_new = np.isin(new, (1, 6))
    near = uniform_filter(rice_new.astype("float32"), size=5).ravel()
    pix = refs["pixel"].to_numpy()
    moved = rice_new.ravel()[pix] & ~np.isin(old.ravel()[pix], (1, 6))
    refs = refs[moved].reset_index(drop=True)
    if refs.empty:
        return refs
    _, events, _ = mr.aoi_events(aoi_id, out_root=sandbox, water="v3")
    ev = events.iloc[refs["pixel"].to_numpy()][[c for c in EVIDENCE if c in events]].reset_index(drop=True)
    nd.forget()
    t = pd.concat([refs, ev], axis=1)
    t["class_old"] = old.ravel()[t["pixel"]]
    t["class_new"] = new.ravel()[t["pixel"]]
    t["rice_share_5x5"] = near[t["pixel"]]
    t["aoi"] = f"aoi{aoi_id}"
    return t


def negative_fields(aoi_id: int, sandbox: str = "processed/_batch/s2_2026_water3", top: int = 3) -> pd.DataFrame:
    """Per negative set, the sandbox fields holding most of the negatives that became rice (from ``negative_evidence``'s
    file), so the user can open them in QGIS and give the verdict."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    t = pd.read_csv(Path(OUT) / f"aoi{aoi_id}_negatives_water3.csv")
    f = gpd.read_file(Path(sandbox) / "fields" / f"aoi{aoi_id}_fields_monsoon2026.gpkg")
    with rasterio.open(Path(sandbox) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026_final.tif") as ds:
        idx = rasterize([(g, i + 1) for i, g in enumerate(f.to_crs(ds.crs).geometry)], out_shape=ds.shape,
                        transform=ds.transform, fill=0, dtype="int32").ravel()
    t["field"] = idx[t["pixel"].to_numpy()]
    t = t[t["field"] > 0]
    rows = []
    for s_, g in t.groupby("set"):
        for fi, n in g["field"].value_counts().head(top).items():
            r = f.iloc[fi - 1]
            rows.append({"set": s_, "field_id": r["field_id"], "label": int(r["label"]), "negative_pixels": int(n),
                         "acres": round(float(r["area_acres"]), 2) if "area_acres" in f else None})
    return pd.DataFrame(rows)


#: The user's verdicts in QGIS (round 3, 30 Sep): what a field is on the latest clear image.
VERDICTS = {"aoi116_006946": "water", "aoi116_008939": "water", "aoi116_006865": "water", "aoi160_004004": "water",
            "aoi160_003847": "shallow water", "aoi160_004502": "water", "aoi39_000837": "bare soil", "aoi39_000829": "wet soil",
            "aoi13_000598": "young rice", "aoi13_000528": "young rice", "aoi13_000769": "rice", "aoi13_000095": "young rice",
            "aoi13_000421": "rice", "aoi13_000594": "rice", "aoi13_000161": "rice"}
VERDICT_COLUMNS = ("flood_ok", "flood_date", "trough_ndvi", "last_clear_ndvi", "last_clear_age", "ndvi_noise_own",
                   "ndvi_max_own", "rise_z_all", "vh_end_track", "flood_vh", "behind_rice_z")


def verdict_table(verdicts=None, sandbox: str = "processed/_batch/s2_2026_water3") -> pd.DataFrame:
    """The rule's evidence (field medians) beside the user's verdict, one AOI loaded at a time. Why: a rule change that
    separates "water / bare soil" from "young rice" must be read off fields the user has judged, not guessed."""
    from ..qgis_review import field_by_id
    from . import monsoon_rule as mr
    from . import ndvi_5day as nd

    verdicts = VERDICTS if verdicts is None else verdicts
    rows = []
    for aoi in sorted({f.split("_")[0] for f in verdicts}):
        _, events, _ = mr.aoi_events(int(aoi[3:]), out_root=sandbox, water="v3")
        for fid in [f for f in verdicts if f.startswith(aoi + "_")]:
            _, _, pids = field_by_id(fid)
            e = events.iloc[pids]
            r = {"field_id": fid, "verdict": verdicts[fid]}
            for c in VERDICT_COLUMNS:
                if c not in e:
                    continue
                v = e[c]
                if v.dtype == bool:
                    r[c] = round(float(v.mean()), 2)
                elif str(v.dtype).startswith("datetime"):
                    r[c] = v.mode().iloc[0] if v.notna().any() else None
                else:
                    r[c] = round(float(np.nanmedian(v.astype(float))), 3)
            with np.errstate(invalid="ignore", divide="ignore"):
                r["last_above_trough_in_noise"] = round(float(np.nanmedian(
                    (e["last_clear_ndvi"] - e["trough_ndvi"]) / e["ndvi_noise_own"])), 2)
            rows.append(r)
        nd.forget()
        del events
    return pd.DataFrame(rows)


def compare_reference_pixels(aoi_id: int, sandbox_a: str, sandbox_b: str, sets=("rice_plot_interior", "rice_plot_edge",
                             "evergreen"), retrough_a: bool = False, retrough_b: bool = True) -> pd.DataFrame:
    """Reference pixels (plots, evergreen) whose delivered status differs between two sandboxes' raw rule maps, with the
    trough date / value and flood date of each under both. Why: a rule change that moves the scores a little must be
    explained pixel by pixel before it is kept (round 3, the season-long trough)."""
    import rasterio

    from . import compare_runs as cr
    from . import monsoon_rule as mr
    from . import ndvi_5day as nd
    from . import validation as va

    plots = cr.load_plots()
    refs = va.reference_sets(aoi_id, plots[plots["aoi"] == f"aoi{aoi_id}"])
    refs = refs[refs["set"].isin(sets)].reset_index(drop=True)
    nd.forget()
    maps = {}
    for tag, root in (("a", sandbox_a), ("b", sandbox_b)):
        with rasterio.open(Path(root) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026.tif") as ds:
            maps[tag] = ds.read(1).ravel()
    pix = refs["pixel"].to_numpy()
    da, db = np.isin(maps["a"][pix], (1, 6)), np.isin(maps["b"][pix], (1, 6))
    moved = refs[da != db].reset_index(drop=True)
    if moved.empty:
        return moved
    keep = mr.RETROUGH_V3
    for tag, root, retro in (("a", sandbox_a, retrough_a), ("b", sandbox_b, retrough_b)):
        mr.RETROUGH_V3 = retro
        _, ev, _ = mr.aoi_events(aoi_id, out_root=root, water="v3")
        e = ev.iloc[moved["pixel"].to_numpy()]
        for c in ("trough_date", "trough_ndvi", "flood_date", "flood_ok", "peak_after", "rise_z_all", "last_ndvi"):
            moved[f"{c}_{tag}"] = e[c].to_numpy()
        moved[f"class_{tag}"] = maps[tag][moved["pixel"].to_numpy()]
        nd.forget()
        del ev
    mr.RETROUGH_V3 = keep
    return moved


def plot_share_final(aoi_id: int, root: str) -> pd.DataFrame:
    """Share of surveyed plot pixels (interior / edge) delivered as rice (1 or 6) on the FINAL map (after relabels and
    the sieve) of a sandbox. Why: ``mask_experiment score`` reads the raw rule map, so a change in the relabel step
    (round 3: per-pixel class-3 relabel) must be scored on the final map."""
    import rasterio

    from . import compare_runs as cr
    from . import ndvi_5day as nd
    from . import validation as va

    plots = cr.load_plots()
    refs = va.reference_sets(aoi_id, plots[plots["aoi"] == f"aoi{aoi_id}"])
    nd.forget()
    refs = refs[refs["set"].str.startswith("rice_plot")]
    with rasterio.open(Path(root) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026_final.tif") as ds:
        c = ds.read(1).ravel()[refs["pixel"].to_numpy()]
    refs = refs.assign(delivered=np.isin(c, (1, 6)))
    return refs.groupby("set")["delivered"].agg(["size", "mean"]).reset_index().assign(aoi=f"aoi{aoi_id}")


def plot_moves(aoi_id: int, root: str, before_final: str, sets=("rice_plot",), map_name: str = "_final") -> pd.DataFrame:
    """Reference pixels whose delivered status (rice 1 / 6 or not) differs between an earlier map (``before_final``, a
    backup file) and the ``<aoi>_monsoon2026<map_name>.tif`` of ``root`` now, with the evidence the recent rules read
    (latest clear view against the AOI's rice at sowing, radar rises, clear views above the trough). ``sets``: reference
    set name prefixes (plots, or negatives such as ``cut_before_map_date``). Why (round 3): a rule change that moves
    plot recall or negatives must be explained pixel by pixel."""
    import rasterio

    from . import compare_runs as cr
    from . import monsoon_rule as mr
    from . import ndvi_5day as nd
    from . import validation as va

    plots = cr.load_plots()
    refs = va.reference_sets(aoi_id, plots[plots["aoi"] == f"aoi{aoi_id}"])
    refs = refs[refs["set"].str.startswith(tuple(sets))].drop_duplicates("pixel").reset_index(drop=True)
    nd.forget()
    with rasterio.open(before_final) as ds:
        old = ds.read(1).ravel()
    with rasterio.open(Path(root) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026{map_name}.tif") as ds:
        new = ds.read(1).ravel()
    pix = refs["pixel"].to_numpy()
    moved = refs[np.isin(old[pix], (1, 6)) != np.isin(new[pix], (1, 6))].reset_index(drop=True)
    if moved.empty:
        return moved
    d, ev, _ = mr.aoi_events(aoi_id, out_root=root, water="v3")
    e = ev.iloc[moved["pixel"].to_numpy()].reset_index(drop=True)
    last_day = pd.DatetimeIndex(d["windows"])[-1] - pd.to_timedelta(e["last_clear_age"].fillna(0) * mr.STEP_DAYS, "D")
    out = moved.assign(old=old[moved["pixel"]], new=new[moved["pixel"]], flood_date=e["flood_date"].to_numpy(),
                       trough_ndvi=e["trough_ndvi"].round(2).to_numpy(), last_clear_ndvi=e["last_clear_ndvi"].round(2).to_numpy(),
                       last_clear_day=last_day.to_numpy(),
                       level_z=np.round(mr.level_vs_rice_sowing(ev, ev["last_clear_ndvi"])[moved["pixel"]], 2),
                       rise_since_z=e["rise_since_last_clear_z"].round(2).to_numpy() if "rise_since_last_clear_z" in e else np.nan,
                       last_ndvi=e["last_ndvi"].round(2).to_numpy(),
                       peak_after=e["peak_after"].round(2).to_numpy(),
                       views_above=e["views_above_trough"].to_numpy() if "views_above_trough" in e else np.nan,
                       vh_rise_since_trough=e["rise_since_trough_z"].round(2).to_numpy() if "rise_since_trough_z" in e
                       else np.nan, rise_z_all=e["rise_z_all"].round(2).to_numpy(), flood_ok=e["flood_ok"].to_numpy())
    return out


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis.own_level_table", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", nargs="*", type=int, default=[])
    p.add_argument("--summary", action="store_true")
    p.add_argument("--negatives", nargs="*", type=int, metavar="AOI",
                   help="false-rice evidence of the negatives in the water3 sandbox (one AOI at a time)")
    p.add_argument("--fields", action="store_true", help="with --negatives: list the fields holding most of them")
    p.add_argument("--verdicts", action="store_true", help="the rule's evidence beside the user's field verdicts (VERDICTS)")
    p.add_argument("--per-track", type=int, metavar="AOI",
                   help="read every track on its own for the plot pixels and noted fields of this AOI (needs its table file)")
    p.add_argument("--plot-share", nargs="+", type=int, metavar="AOI",
                   help="share of plot pixels delivered as rice on the final map of --root (plot_share_final)")
    p.add_argument("--root", default="processed/_batch/s2_2026", help="with --plot-share: the sandbox")
    p.add_argument("--plot-moves", type=int, metavar="AOI", help="plot pixels no longer delivered (with --root, --before)")
    p.add_argument("--before", help="with --plot-moves: the earlier map (a backup .tif)")
    p.add_argument("--sets", nargs="+", default=["rice_plot"], help="with --plot-moves: reference set prefixes")
    p.add_argument("--map", default="_final", help="with --plot-moves: '_final' or '' (the raw rule map)")
    args = p.parse_args(argv)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    if args.plot_moves:
        t = plot_moves(args.plot_moves, args.root, args.before, args.sets, args.map)
        print(len(t), "pixels moved; old -> new:", t.groupby(["old", "new"]).size().to_dict() if len(t) else {})
        print(t.drop(columns=[c for c in ("pixel",) if c in t]).describe(include="all").T.to_string() if len(t) else "")
        print(t.head(30).to_string(index=False))
        return 0
    if args.plot_share:
        t = pd.concat([plot_share_final(a, args.root) for a in args.plot_share])
        print(t.assign(mean=(100 * t["mean"]).round(1)).to_string(index=False))
        return 0
    if args.ids:
        from . import compare_runs as cr

        plots = cr.load_plots()
        for a in args.ids:
            t = aoi_rows(a, plots)
            t.to_csv(Path(OUT) / f"aoi{a}.csv", index=False)
            print(f"aoi{a}: {len(t)} rows, AOI rice VH level {t['aoi_rice_vh'].iloc[0] if len(t) else None}", flush=True)
    if args.verdicts:
        t = verdict_table()
        t.to_csv(Path(OUT) / "verdicts_water3.csv", index=False)
        print(t.to_string(index=False))
        return 0
    if args.negatives and args.fields:
        for a in args.negatives:
            print(negative_fields(a).to_string(index=False))
        return 0
    if args.negatives:
        for a in args.negatives:
            t = negative_evidence(a, "processed/_batch/s2_2026_water3")
            t.to_csv(Path(OUT) / f"aoi{a}_negatives_water3.csv", index=False)
            print(f"aoi{a}: {len(t)} negative pixels became rice", flush=True)
    if args.per_track:
        t = pd.read_csv(Path(OUT) / f"aoi{args.per_track}.csv")
        t = t[t["set"].isin(["rice_plot_interior", "noted_field"]) & t["flood_ok"].astype(bool)]
        pt = per_track(args.per_track, t["pixel"].to_numpy(), t["flood_date"])
        pt = pt.merge(t[["pixel", "set", "field_id"]].drop_duplicates("pixel"), on="pixel")
        pt.to_csv(Path(OUT) / f"aoi{args.per_track}_per_track.csv", index=False)
        pt["group"] = np.where(pt["set"] == "noted_field", pt["field_id"], pt["set"])
        print(pt.groupby(["group", "track"])[["drop", "flood", "recovery", "rise_z"]].median().round(2).to_string())
        c = combine_tracks(pt).merge(pt[["pixel", "group"]].drop_duplicates("pixel"), on="pixel")
        print("\nall tracks together:")
        print(c.groupby("group").agg(pixels=("pixel", "size"), rise_z_all=("rise_z_all", "median"),
                                     recovery_all=("recovery_all", "median"),
                                     pass_k2=("rise_z_all", lambda z: (z >= 2).mean()),
                                     pass_k3=("rise_z_all", lambda z: (z >= 3).mean())).round(2).to_string())
    if args.summary:
        s = summary()
        s.to_csv(Path(OUT) / "summary.csv", index=False)
        print(s.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
