"""Optical mask experiment: which cloud mask gives the rule the best map? (fix plan, stage 1.4)

Why
---
The delivered map used QA60 (opaque + cirrus bits) as its only cloud mask. The review (issue 11)
showed the cirrus bit throwing away whole clear scenes, including the only clear view of a
transplanting flood, while hazy scenes passed and turned standing rice into "harvested" at the
series end. Every exported Sentinel-2 file also carries the Cloud Score+ ``clear`` band, so the
mask can be changed here without a new export. This module builds the 5-day series under several
masks into separate folders, runs the rule on each, and scores every variant on the same
references: the surveyed plots and negatives (``validation.reference_sets``, always from the
baseline series so the references do not move with the mask) and the reviewers' field verdicts
(``review_verdicts``). The variant with the best scores becomes the mask of the re-run.

Variants (``VARIANTS``): ``qa60`` is the delivered mask rebuilt with the current code (the control:
any difference between it and the baseline comes from the radar fixes, not the mask); ``cs60``
is a plain Cloud Score+ mask (QA60 opaque only plus ``clear`` >= 60), kept for the record because it
removes the flooded-field observations; ``hyb50`` / ``hyb60`` / ``hyb70`` are the hybrid: QA60
opaque only, Cloud Score+ at that threshold for bright observations, dark observations always kept
(``ndvi_5day.clear_mask``).

Use::

    python -m sar_pipeline.analysis.mask_experiment build --ids ... [--variants cs60 cs70] [--jobs 3]
    python -m sar_pipeline.analysis.mask_experiment rule  --ids ... [--variants ...] [--jobs 3]
    python -m sar_pipeline.analysis.mask_experiment score --ids ... [--variants ...]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = "processed/_batch/s2_2026"
VARIANTS = {
    "qa60": {"qa60_mode": "both", "cs_min": None, "keep_dark": False, "drop_haze": False},
    # plain Cloud Score+: kept for the record only; it removes flooded fields (see ndvi_5day.DARK_NIR_MAX)
    "cs60": {"qa60_mode": "opaque", "cs_min": 60, "keep_dark": False, "drop_haze": False},
    # hybrid: QA60 opaque cloud, Cloud Score+ for bright observations (cloud, haze), dark ones kept
    # ... and a blue-bright canopy removed as haze (ndvi_5day.HAZE_B2_MIN)
    "hyb20": {"qa60_mode": "opaque", "cs_min": 20, "keep_dark": True, "drop_haze": True},
    "hyb30": {"qa60_mode": "opaque", "cs_min": 30, "keep_dark": True, "drop_haze": True},
    "hyb40": {"qa60_mode": "opaque", "cs_min": 40, "keep_dark": True, "drop_haze": True},
    "hyb50": {"qa60_mode": "opaque", "cs_min": 50, "keep_dark": True, "drop_haze": True},
    "hyb60": {"qa60_mode": "opaque", "cs_min": 60, "keep_dark": True, "drop_haze": True},
    "hyb70": {"qa60_mode": "opaque", "cs_min": 70, "keep_dark": True, "drop_haze": True},
    # QA60 opaque cloud + the blue-band haze test only, no Cloud Score+: does the haze test do the work?
    "hazeonly": {"qa60_mode": "opaque", "cs_min": None, "keep_dark": False, "drop_haze": True},
}
OUT = f"{BASE}/report/mask_experiment"


#: Set by ``--into-standard`` (stage 4): the chosen mask's series go to the standard folder, where
#: the batch, the field labels and the delivery read them.
_INTO_STANDARD = False


def root(variant: str) -> str:
    return BASE if _INTO_STANDARD else f"{BASE}_{variant}"


def build_one(args) -> dict:
    aoi_id, variant = args
    from . import ndvi_5day as nd

    marker = Path(root(variant)) / f"aoi{aoi_id}" / f"aoi{aoi_id}_ndvi5d.tif"
    if marker.exists() and not _INTO_STANDARD:
        return {"aoi": f"aoi{aoi_id}", "variant": variant, "skipped": True}
    s = nd.build(aoi_id, out_root=root(variant), log=lambda *_: None, **VARIANTS[variant])
    nd.forget()
    return {"aoi": f"aoi{aoi_id}", "variant": variant, **{k: v for k, v in s.items() if not k.startswith("path_")}}


def rule_one(args) -> dict:
    aoi_id, variant = args
    from . import monsoon_rule as mr
    from . import ndvi_5day as nd

    r = mr.run_aoi(aoi_id, out_root=root(variant))
    nd.forget()
    return {"variant": variant, **{k: v for k, v in r.items() if k != "path"}}


def run_many(func, ids, variants, jobs: int = 3) -> pd.DataFrame:
    """``func`` over every (aoi, variant) pair, ``jobs`` processes; failures are recorded, not raised."""
    from concurrent.futures import ProcessPoolExecutor

    tasks = [(a, v, _INTO_STANDARD) for v in variants for a in ids] if _INTO_STANDARD else [(a, v) for v in variants for a in ids]
    rows = []
    if jobs <= 1:
        rows = [_safe_call((func, t)) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            rows = list(ex.map(_safe_call, [(func, t) for t in tasks]))
    return pd.DataFrame(rows)


def _safe_call(pair):
    func, t = pair
    global _INTO_STANDARD
    if len(t) == 3:                       # (aoi, variant, into_standard) from a worker process
        _INTO_STANDARD = t[2]
        t = t[:2]
    try:
        return func(t)
    except Exception as exc:  # noqa: BLE001
        return {"aoi": f"aoi{t[0]}", "variant": t[1], "error": f"{type(exc).__name__}: {exc}"}


def score(ids, variants, plots=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per variant: delivered-rice share of the reference sets (baseline references, variant map),
    and agreement with the reviewers' verdicts (field majority label from the variant map)."""
    from . import compare_runs as cr
    from . import field_rice
    from . import monsoon_rule as mr
    from . import ndvi_5day as nd
    from . import review_verdicts as rv
    from . import validation as va

    import rasterio

    plots = cr.load_plots() if plots is None else plots
    verdicts = pd.read_csv(rv.OUT) if Path(rv.OUT).exists() else None
    ref_rows, verdict_rows = [], []
    for aoi_id in ids:
        p = plots[plots["aoi"] == f"aoi{aoi_id}"]
        refs_base = va.reference_sets(aoi_id, p, out_root=BASE)
        nd.forget()
        for v in variants:
            path = Path(root(v)) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026.tif"
            if not path.exists():
                continue
            with rasterio.open(path) as ds:
                classes = ds.read(1).ravel()
            # the negatives are defined from a series: under the baseline mask a hazy end makes
            # false "cut" negatives, so each variant is also judged on negatives from its own series
            refs_own = va.reference_sets(aoi_id, p, out_root=root(v))
            nd.forget()
            for refs, origin in ((refs_base, "baseline"), (refs_own, "own")):
                c = classes[refs["pixel"].to_numpy()]
                for (s, region), g in refs.assign(cls=c).groupby(["set", "region"]):
                    if origin == "own" and s.startswith("rice_plot"):
                        continue                              # the plots do not move with the mask
                    ref_rows.append({"variant": v, "refs": origin, "aoi": f"aoi{aoi_id}", "set": s, "region": region,
                                     "pixels": len(g),
                                     "delivered_pct": round(100 * float(np.isin(g["cls"], (1, 6)).mean()), 2),
                                     "harvested_pct": round(100 * float((g["cls"] == 4).mean()), 2),
                                     "young_pct": round(100 * float((g["cls"] == 2).mean()), 2),
                                     "class3_pct": round(100 * float((g["cls"] == 3).mean()), 2)})
            if verdicts is not None and (verdicts["aoi"] == f"aoi{aoi_id}").any():
                fields, _, _ = field_rice.label_aoi(aoi_id, map_suffix="", src_root=root(v))
                labels = fields[["field_id", "label"]]
                for _, r in rv.score(labels, verdicts[verdicts["aoi"] == f"aoi{aoi_id}"]).iterrows():
                    verdict_rows.append({"variant": v, "aoi": f"aoi{aoi_id}", **r.to_dict()})
    return pd.DataFrame(ref_rows), pd.DataFrame(verdict_rows)


def diagnose(aoi_id: int, variant: str, control: str = "qa60") -> pd.DataFrame:
    """Where two masks disagree: per date, the share of the AOI each mask keeps, and for the
    pixels whose class changed, the rule's events under both masks.

    Why: a mask that removes the wrong observations (e.g. a stricter mask that takes a dark flooded
    field for a cloud shadow) breaks the rule silently; this shows which dates it removes and which
    condition (trough, bare run, rise, standing) then fails.
    """
    import rasterio

    from . import monsoon_rule as mr
    from . import ndvi_5day as nd

    dates, ndvi, _, _, ok_c, _, _ = nd.read_dates(aoi_id, **VARIANTS[control])
    _, _, _, _, ok_v, _, _ = nd.read_dates(aoi_id, **VARIANTS[variant])
    inside = nd.inside_aoi(aoi_id)
    n = inside.sum()
    per_date = pd.DataFrame({"date": dates,
                             f"{control}_clear_pct": [round(100 * float(o.ravel()[inside].mean()), 1) for o in ok_c],
                             f"{variant}_clear_pct": [round(100 * float(o.ravel()[inside].mean()), 1) for o in ok_v]})
    with np.errstate(invalid="ignore"):
        per_date["median_ndvi_kept_by_control_only"] = [
            round(float(np.nanmedian(v.ravel()[inside & c.ravel() & ~w.ravel()])), 2)
            if (inside & c.ravel() & ~w.ravel()).sum() > 50 else np.nan for v, c, w in zip(ndvi, ok_c, ok_v)]
    maps = {}
    for v in (control, variant):
        r = root(v) if v != control or Path(root(v)).exists() else BASE
        with rasterio.open(Path(r) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026.tif") as ds:
            maps[v] = ds.read(1).ravel()
    changed = (maps[control] != maps[variant]) & (maps[control] != 255)
    print(f"aoi{aoi_id}: {round(mr.acres(int(changed.sum())), 1)} ac changed class between {control} and {variant}")
    print(pd.crosstab(maps[control][changed], maps[variant][changed]).rename(index=mr.CLASSES, columns=mr.CLASSES))
    ev = {}
    for v in (control, variant):
        r = root(v) if v != control or Path(root(v)).exists() else BASE
        _, e, _ = mr.aoi_events(aoi_id, out_root=r)
        cols = [c for c in ("trough_ndvi", "low_windows", "rise", "peak_after", "standing", "last_ndvi", "radar_wet",
                            "flood_ok", "flood_drop", "flood_vh", "ndvi_at_flood", "support", "flood_other_drop",
                            "rise_from_flood", "peak_after_flood", "standing_after_flood") if c in e]
        ev[v] = e.loc[changed, cols]
        nd.forget()
    print("events of the changed pixels, medians (bool columns: share True):")
    print(pd.DataFrame({v: ev[v].astype(float).median() for v in ev}).round(2).to_string())
    lost = ev[variant][(maps[control][changed] == 1) & (maps[variant][changed] != 1)]
    if len(lost):
        print(f"pixels rice under {control} only ({round(mr.acres(len(lost)), 1)} ac), {variant} events, medians / shares:")
        print(lost.astype(float).median().round(2).to_string())
        print("  flood_ok share", round(float(lost["flood_ok"].mean()), 2) if "flood_ok" in lost else None,
              "| of those, rise_from_flood>=0.3:", round(float((lost["rise_from_flood"] >= 0.3)[lost["flood_ok"]].mean()), 2) if "flood_ok" in lost and lost["flood_ok"].any() else None,
              "| standing_after_flood:", round(float(lost["standing_after_flood"][lost["flood_ok"]].mean()), 2) if "flood_ok" in lost and lost["flood_ok"].any() else None)
    return per_date


def bright_profile(aoi_id: int, dates, variant: str, control: str = "qa60") -> pd.DataFrame:
    """For each date, the reflectances and Cloud Score+ of the pixels the control mask keeps and
    the variant mask removes: quartiles of B2 (blue), B4, B8, NDVI and ``clear``.

    Why: to set the "dark observation" rule (``ndvi_5day.DARK_NIR_MAX``) from what the removed
    flooded fields actually look like, rather than from a guess.
    """
    import rasterio

    from ..optical_export import band_index
    from . import ndvi_5day as nd
    from . import pixel_report as pr

    loc = pr.locate(aoi_id, 0)
    inside = nd.inside_aoi(aoi_id).reshape(loc["grid"]["height"], loc["grid"]["width"])
    rows = []
    for path in sorted(pr.sync_s2(loc, f"data/{nd.FOLDER}", nd.FOLDER).glob("*.tif")):
        d = path.stem.rsplit("_S2_", 1)[1]
        if d not in set(dates):
            continue
        with rasterio.open(path) as ds:
            b = {n: ds.read(band_index(ds, n)).astype("float32") for n in ("B2", "B4", "B8", "QA60", "clear")}
        data = (b["B4"] > 0) & (b["B8"] > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = (b["B8"] - b["B4"]) / (b["B8"] + b["B4"])
        keep = {}
        for name in (control, variant):
            v = VARIANTS[name]
            bits = (1 << 10) | (1 << 11) if v["qa60_mode"] == "both" else (1 << 10)
            keep[name] = nd.clear_mask(data, b["QA60"], b["clear"], b["B8"], ndvi, v["cs_min"], bits, v["keep_dark"],
                                       v["drop_haze"], b["B2"], b["B4"])
        sel = inside & keep[control] & ~keep[variant]
        if sel.sum() < 20:
            continue
        for q in (0.1, 0.5, 0.9):
            rows.append({"date": d, "pixels": int(sel.sum()), "quantile": q,
                         **{k: round(float(np.quantile(b[k][sel], q)), 0) for k in ("B2", "B4", "B8", "clear")},
                         "ndvi": round(float(np.quantile(ndvi[sel], q)), 2)})
    return pd.DataFrame(rows)


def write_sidecars(variants=None) -> int:
    """Write the ``<aoi>_series_mask.json`` sidecar (what ``ndvi_5day.load`` reads) into every
    variant folder built before the sidecar existed. Returns the number written."""
    import json

    n = 0
    for v in (variants or VARIANTS):
        for d in Path(root(v)).glob("aoi*"):
            if (d / f"{d.name}_ndvi5d.tif").exists() and not (d / f"{d.name}_series_mask.json").exists():
                (d / f"{d.name}_series_mask.json").write_text(json.dumps(VARIANTS[v]))
                n += 1
    return n


def verdict_detail(ids, variant: str) -> pd.DataFrame:
    """Every reviewed field of ``ids`` with its old label, its label under ``variant`` (field
    majority of the variant's rule map on the current delineation) and whether it agrees with the
    reviewer. Why: the summary counts say how many right fields broke; this says which, so the cause
    can be found."""
    from . import field_rice
    from . import review_verdicts as rv

    v = pd.read_csv(rv.OUT)
    rows = []
    for aoi_id in ids:
        sub = v[v["aoi"] == f"aoi{aoi_id}"]
        if sub.empty or not (Path(root(variant)) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026.tif").exists():
            continue
        fields, _, _ = field_rice.label_aoi(aoi_id, map_suffix="", src_root=root(variant))
        m = sub.merge(fields[["field_id", "label", "rice_share", "pixels"]].rename(
            columns={"label": "new_label", "rice_share": "new_rice_share", "pixels": "new_pixels"}), on="field_id", how="left")
        rows.append(m)
    t = pd.concat(rows, ignore_index=True)
    t["new_rice"] = t["new_label"].isin(rv.DELIVERED)
    t["old_rice"] = t["label"].isin(rv.DELIVERED)
    t["agree_now"] = np.where(t["expected_rice"].isna(), np.nan, (t["new_rice"] == (t["expected_rice"] == 1)))
    t["agreed_before"] = np.where(t["expected_rice"].isna(), np.nan, (t["old_rice"] == (t["expected_rice"] == 1)))
    return t


def field_obs(aoi_id: int, field_id: str, variant: str, since: str = "2026-08-01") -> pd.DataFrame:
    """Per Sentinel-2 date since ``since``: the field's mean raw NDVI, blue, NIR and Cloud Score+, and
    the share of its pixels the variant's mask keeps. Why: to see with one table why a field's end
    of season looks the way it does under a mask (haze, shadow, cloud, or a real fall)."""
    import rasterio

    from ..optical_export import band_index
    from . import field_rice
    from . import ndvi_5day as nd
    from . import pixel_report as pr

    fields, idx, _ = field_rice.label_aoi(aoi_id, map_suffix="", src_root=root(variant))
    i = int(np.flatnonzero(fields["field_id"].to_numpy() == field_id)[0])
    sel = np.asarray(idx) == i
    loc = pr.locate(aoi_id, 0)
    v = VARIANTS[variant]
    bits = (1 << 10) | (1 << 11) if v["qa60_mode"] == "both" else (1 << 10)
    rows = []
    for path in sorted(pr.sync_s2(loc, f"data/{nd.FOLDER}", nd.FOLDER).glob("*.tif")):
        d = path.stem.rsplit("_S2_", 1)[1]
        if d < since:
            continue
        with rasterio.open(path) as ds:
            b = {n: ds.read(band_index(ds, n)).astype("float32")[sel] for n in ("B2", "B4", "B8", "QA60", "clear")}
        data = (b["B4"] > 0) & (b["B8"] > 0)
        if not data.any():
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = (b["B8"] - b["B4"]) / (b["B8"] + b["B4"])
        keep = nd.clear_mask(data, b["QA60"], b["clear"], b["B8"], ndvi, v["cs_min"], bits, v["keep_dark"],
                             v["drop_haze"], b["B2"], b["B4"])
        rows.append({"date": d, "ndvi": round(float(np.nanmedian(ndvi[data])), 2), "B2": round(float(np.median(b["B2"][data]))),
                     "B8": round(float(np.median(b["B8"][data]))), "cs": round(float(np.median(b["clear"][data]))),
                     "qa60_opaque_pct": round(100 * float(((b["QA60"][data].astype(int) & (1 << 10)) != 0).mean())),
                     "kept_pct": round(100 * float(keep[data].mean()))})
    return pd.DataFrame(rows)


def grey_rule_effect(aoi_id: int, variant: str = "hyb40", since: str = "2026-05-01") -> pd.DataFrame:
    """Per date: share of the AOI kept with and without the grey-haze rule (blue >= 0.9 x red), and
    what the observations removed by that rule alone look like (median B2/B4, NDVI, Cloud Score+).
    Why: the rule targets haze; pale dry soils are also grey-ish, and a rule that removes the bare
    observations of a dry-zone AOI would cost the troughs there."""
    import rasterio

    from ..optical_export import band_index
    from . import ndvi_5day as nd
    from . import pixel_report as pr

    loc = pr.locate(aoi_id, 0)
    inside = nd.inside_aoi(aoi_id).reshape(loc["grid"]["height"], loc["grid"]["width"])
    v = VARIANTS[variant]
    bits = (1 << 10) | (1 << 11) if v["qa60_mode"] == "both" else (1 << 10)
    rows = []
    for path in sorted(pr.sync_s2(loc, f"data/{nd.FOLDER}", nd.FOLDER).glob("*.tif")):
        d = path.stem.rsplit("_S2_", 1)[1]
        if d < since:
            continue
        with rasterio.open(path) as ds:
            b = {n: ds.read(band_index(ds, n)).astype("float32") for n in ("B2", "B4", "B8", "QA60", "clear")}
        data = (b["B4"] > 0) & (b["B8"] > 0) & inside
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = (b["B8"] - b["B4"]) / (b["B8"] + b["B4"])
            ratio = b["B2"] / b["B4"]
        without = nd.clear_mask(data, b["QA60"], b["clear"], b["B8"], ndvi, v["cs_min"], bits, v["keep_dark"], v["drop_haze"], b["B2"])
        with_ = nd.clear_mask(data, b["QA60"], b["clear"], b["B8"], ndvi, v["cs_min"], bits, v["keep_dark"], v["drop_haze"], b["B2"], b["B4"])
        gone = without & ~with_
        n = data.sum()
        if n == 0 or without.sum() == 0:
            continue
        rows.append({"date": d, "kept_without_pct": round(100 * without.sum() / n, 1), "kept_with_pct": round(100 * with_.sum() / n, 1),
                     "removed_by_grey_pct": round(100 * gone.sum() / n, 1),
                     "gone_B2_B4": round(float(np.median(ratio[gone])), 2) if gone.any() else np.nan,
                     "gone_ndvi": round(float(np.median(ndvi[gone])), 2) if gone.any() else np.nan,
                     "gone_cs": round(float(np.median(b["clear"][gone]))) if gone.any() else np.nan,
                     "gone_B2": round(float(np.median(b["B2"][gone]))) if gone.any() else np.nan})
    return pd.DataFrame(rows)


EVENT_COLS = ("trough_ndvi", "low_windows", "rise", "peak_after", "standing", "last_ndvi", "radar_wet", "radar_wet_v1",
              "flood_ok", "flood_drop", "flood_vh", "ndvi_at_flood", "support", "bare_near_flood", "peak_after_flood",
              "standing_after_flood", "radar_canopy_rise", "vh_end", "never_bare")


def pixel_events_table(aoi_id: int, variant: str, pixels, classes=None) -> pd.DataFrame:
    """The rule's events (medians; shares for booleans) for a set of pixels under ``variant``, split by
    the class they got. Why: to read why a group of known pixels (plot interiors, one field) ended
    up in a class."""
    import rasterio

    from . import monsoon_rule as mr
    from . import ndvi_5day as nd

    r = root(variant)
    if classes is None:
        with rasterio.open(Path(r) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026.tif") as ds:
            classes = ds.read(1).ravel()
    _, e, _ = mr.aoi_events(aoi_id, out_root=r)
    nd.forget()
    pixels = np.asarray(pixels)
    sub = e.iloc[pixels][[c for c in EVENT_COLS if c in e]].astype(float)
    sub["cls"] = classes[pixels]
    out = sub.groupby("cls").median().T
    out.columns = [f"{mr.CLASSES.get(int(c), c)} (n={int((sub['cls'] == c).sum())})" for c in out.columns]
    return out.round(2)


def summarise(refs: pd.DataFrame, verdicts: pd.DataFrame) -> str:
    lines = []
    if len(refs):
        if "refs" not in refs:
            refs = refs.assign(refs="baseline")
        w = refs.assign(wt=refs["pixels"] * refs["delivered_pct"])
        t = (w.groupby(["refs", "variant", "set", "region"]).agg(pixels=("pixels", "sum"), wt=("wt", "sum")))
        t["delivered_pct"] = (t["wt"] / t["pixels"]).round(1)
        lines += ["Delivered-rice share (%) of the reference sets, per mask variant (refs: negatives from the "
                  "baseline series or the variant's own):",
                  t.reset_index().pivot_table(index=["set", "region", "refs"], columns="variant",
                                              values="delivered_pct").to_string(), ""]
    if len(verdicts):
        g = verdicts.groupby(["variant", "verdict"])[["decidable", "agree_now", "disagree_now", "agreed_before"]].sum()
        lines += ["Reviewed fields per variant (agree_now should rise for 'wrong', stay for 'right'):", g.to_string()]
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis.mask_experiment", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["build", "rule", "score", "diagnose", "sidecars", "verdicts"])
    p.add_argument("--ids", nargs="*", type=int, default=[])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--into-standard", action="store_true",
                   help="build: write the series into the standard folder (the chosen mask, stage 4)")
    args = p.parse_args(argv)
    if args.into_standard:
        if args.step != "build" or len(args.variants) != 1:
            p.error("--into-standard takes exactly one variant with the build step")
        global _INTO_STANDARD
        _INTO_STANDARD = True
    Path(OUT).mkdir(parents=True, exist_ok=True)
    if args.step == "verdicts":
        t = verdict_detail(args.ids, args.variants[0])
        t.to_csv(Path(OUT) / f"verdict_detail_{args.variants[0]}.csv", index=False)
        broke = t[(t["verdict"] == "right") & (t["agreed_before"] == 1) & (t["agree_now"] == 0)]
        print(f"{len(t)} reviewed fields; right fields broken: {len(broke)}")
        print(broke[["field_id", "label", "new_label", "new_rice_share", "expected_rice", "true_class_text", "issue"]].to_string(index=False))
        still = t[(t["verdict"] == "wrong") & (t["agree_now"] == 0)]
        print(f"wrong fields still wrong: {len(still)}")
        print(still[["field_id", "label", "new_label", "new_rice_share", "expected_rice", "true_class_text", "issue"]].to_string(index=False))
        return 0
    if args.step == "sidecars":
        print(f"{write_sidecars(args.variants)} sidecars written")
        return 0
    if args.step == "diagnose":
        for a in args.ids:
            t = diagnose(a, args.variants[0])
            print(t[(t["date"] >= "2026-05-01")].to_string(index=False))
        return 0
    if args.step in ("build", "rule"):
        t = run_many(build_one if args.step == "build" else rule_one, args.ids, args.variants, args.jobs)
        t.to_csv(Path(OUT) / f"{args.step}_log.csv", mode="a", header=not (Path(OUT) / f"{args.step}_log.csv").exists(), index=False)
        bad = t[t.get("error", pd.Series(dtype=str)).notna()] if "error" in t else t.iloc[0:0]
        print(f"{args.step}: {len(t)} AOI-variants, {len(bad)} failed")
        if len(bad):
            print(bad[["aoi", "variant", "error"]].to_string(index=False))
    else:
        refs, verdicts = score(args.ids, args.variants)
        refs.to_csv(Path(OUT) / "reference_scores.csv", index=False)
        verdicts.to_csv(Path(OUT) / "verdict_scores.csv", index=False)
        print(summarise(refs, verdicts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
