"""Break an AOI into every combination of the factors that shape a rice-like season: a check map, not a product.

Why
---
The delivered classes hide the reasons behind them. The user (30 Sep) listed what makes one field's season differ from
another's: when it was sown, whether there was any sign of water at sowing, whether it was cut by the latest view, when
it was cut, and how green it got at its peak; and, for the ground without a rice-like season, what it is instead (water,
trees, built-up, other vegetation, empty land). Each pixel gets one short code made of these factors, e.g.
``S1 W1 cut H1 P2`` = sown before June, water seen, cut between 20 Aug and 16 Sep, peak below the AOI's rice. The code is
put next to the final class, so a combination that says "not rice" while the map says rice (or the other way) stands out
and can be checked in QGIS field by field. Nothing here feeds the map.

Factors (each pixel against its own values and against this AOI's radar-confirmed rice; ``WATER_K`` = k):

* **S sowing**: the radar flood date where a sign of water was found (the transplanting), else the optical trough (the
  field's last bare spell). S1 before ``SOW_PERIODS[0]``, S2 up to ``SOW_PERIODS[1]`` (the user's 20 July), S3 after.
* **W water at sowing**: W1 a sign of water (``monsoon_rule.water_sign``) at the pixel's own sowing
  (``monsoon_rule.at_sowing``: the fitted NDVI on the flood date near its own low point, or below half-way from that
  low point to its peak); W4 a sign of water, but the "flood" fell while the canopy stood more than half-way up (a radar
  dip under a crop is not transplanting water); W3 no water, and the radar
  brightened at sowing instead (``brightened_at_sowing``: a crop sown into dry soil); W2 none of these.
  The sowing date of a W4 pixel is its optical trough, not the flood.
* **state on the latest view**: ``cut`` the fitted NDVI fell from its peak to below half-way back to its own bare
  level (a ripening crop that yellows stays above half-way); ``young`` still rising on the last window and its peak
  so far below the AOI's rice peaks; ``full`` otherwise; ``flooded`` water at sowing but no crop since, neither
  optical (rise under k x own noise) nor radar (``rise_z_all`` under k); ``radar-only`` water, then a radar rise, but
  the optical never rose above its own low point (flooded fields whose water roughens, or a crop hidden by cloud).
* **H cut date** (cut only): H0 before ``HARVEST_PERIODS[0]`` (20 Aug), H1 up to ``HARVEST_PERIODS[1]`` (16 Sep), H2 later.
* **P peak** (full and cut only): P1 at the AOI's rice peak, P2 more than k x sqrt(rice spread^2 + own noise^2) below.

Ground with no rice-like season (no low point as low as the AOI's paddies at sowing followed by a crop):
``N water`` open water most of the year; ``N trees`` never as low as the paddies at sowing and green all season (at the
rice peak level); ``N built`` never green, radar as bright as the rice canopy all season; ``N empty`` never green, radar
darker; ``N veg`` everything else green at some time (grass, shrubs, other crops); ``no data``.

Outputs (in ``<root>/aoi<N>/``): ``aoi<N>_factors.tif`` (ids) + ``.qml`` (QGIS labels), ``aoi<N>_factors.csv`` (id, code,
plain words, acres, acres per final class, example fields) and ``aoi<N>_factors_fields.gpkg`` (each field's majority code
and its majority class on the same final map).

Use::

    python -m sar_pipeline.analysis.factor_breakdown --aoi 160 --root processed/_batch/s2_2026_hyb40m1late
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import monsoon_rule as mr
from .optical_phenology import acres

#: Sowing periods (user, 30 Sep): May, June to 20 July (the normal window, ``monsoon_rule.YOUNG_SOWN_AFTER``), later.
SOW_PERIODS = ("2026-06-01", mr.YOUNG_SOWN_AFTER)
#: Cut periods (user, 30 Sep): before 20 August, 20 August to 16 September, later.
HARVEST_PERIODS = ("2026-08-20", "2026-09-16")
NO_DATA_ID = 0

WORDS = {
    "S1": "sown in May", "S2": "sown June-20 Jul", "S3": "sown after 20 Jul",
    "W1": "water seen at sowing", "W2": "no water seen", "W3": "radar brightened (dry sowing)",
    "W4": "water only under a standing canopy", "radar-only": "radar rose, optical still bare / water",
    "full": "standing full canopy", "young": "young, still rising", "cut": "cut", "flooded": "flooded, no crop yet",
    "H0": "cut before 20 Aug", "H1": "cut 20 Aug-16 Sep", "H2": "cut after 16 Sep",
    "P1": "peak like the AOI's rice", "P2": "peak below the AOI's rice",
    "N water": "open water all year", "N trees": "trees / orchard (green all season)",
    "N built": "built-up / road (never green, radar bright)", "N empty": "empty land (never green)",
    "N veg": "other vegetation, no rice-like season", "no data": "no data",
}


def _band(values, noise, k):
    """(median, lower edge) of a reference set: the edge is k x sqrt(spread^2 + each pixel's own noise^2) below."""
    vals = values[np.isfinite(values)]
    if len(vals) < 1:
        return np.nan, np.full(len(noise), np.nan)
    med = np.median(vals)
    spread = 1.4826 * np.median(np.abs(vals - med))
    return med, med - k * np.sqrt(spread ** 2 + np.nan_to_num(noise) ** 2)


def _cut_date(ndvi_fit, windows, trough_idx, halfway):
    """First window after the pixel's peak (after its trough) where the fitted NDVI is below ``halfway``; NaT if none."""
    fit = np.asarray(ndvi_fit)
    n_win, n = fit.shape
    step = np.arange(n_win)[:, None]
    after_trough = np.where(step >= trough_idx[None, :], fit, -np.inf)
    peak_idx = after_trough.argmax(axis=0)
    with np.errstate(invalid="ignore"):
        below = (step > peak_idx[None, :]) & (fit < halfway[None, :])
    first = below.argmax(axis=0)
    w = pd.DatetimeIndex(windows).to_numpy().astype("datetime64[D]")
    return np.where(below.any(axis=0), w[first], np.datetime64("NaT"))


def factors(events: pd.DataFrame, ndvi_fit, windows, season=mr.SEASON) -> pd.DataFrame:
    """Per pixel: the factor columns and the combined ``code`` (see the module docstring).

    ``events``: ``monsoon_rule.aoi_events(..., water="v3")``; ``ndvi_fit``: the fitted (windows, pixels) series."""
    from .radar_water import WATER_K as k

    n = len(events)
    num = lambda c, fill=np.nan: np.asarray(events[c], dtype=float) if c in events else np.full(n, fill)  # noqa: E731
    fit = np.asarray(ndvi_fit)
    w = pd.DatetimeIndex(windows)
    inside = (w >= pd.Timestamp(season[0])) & (w < pd.Timestamp(season[1]) + pd.Timedelta(days=5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        season_max = np.nanmax(fit[inside], axis=0)
        season_med = np.nanmedian(fit[inside], axis=0)
    noise = np.nan_to_num(num("ndvi_noise_own"))
    trough, peak, last = num("trough_ndvi"), num("peak_after"), num("last_ndvi")
    wet = mr.water_sign(events)
    against = np.asarray(events["brightened_at_sowing"], dtype=bool) & ~wet if "brightened_at_sowing" in events \
        else np.zeros(n, dtype=bool)
    rise_r = num("rise_z_all")

    # this AOI's radar-confirmed rice: a peer-confirmed flood (not a pattern flood) and the radar crop rise after it
    ref = np.asarray(events["flood_ok"], dtype=bool) & (rise_r >= k)
    if "flood_by_pattern" in events:
        ref &= ~np.asarray(events["flood_by_pattern"], dtype=bool)
    with np.errstate(invalid="ignore"):
        grew_opt = (peak - trough) >= k * noise
        grew_opt &= (peak - trough) > 0
        grew_rad = wet & (rise_r >= k)
        grew = grew_opt | grew_rad
        ref &= grew_opt
        # water at sowing: the flood fell at the pixel's own low point, or at least before the crop was half-way up
        # (a field that stays flooded reaches its lowest NDVI weeks after the water came); W4 = a dip under a canopy
        sowing_water = wet & (mr.at_sowing(events) | ~(num("ndvi_at_flood_fit") >= (peak + trough) / 2))
        peak_med, peak_low = _band(peak[ref], noise, k)
        trough_med, _ = _band(trough[ref], noise, k)
        trough_ref = trough[ref & np.isfinite(trough)]
        t_spread = 1.4826 * np.median(np.abs(trough_ref - np.median(trough_ref))) if len(trough_ref) else np.nan
        paddy_low = ~(num("trough_peer_z") >= k)                   # unknown is not held against a field
        never_green = season_max < trough_med + k * np.sqrt(t_spread ** 2 + noise ** 2)
        vh_end = num("vh_end")
        vh_rice = np.nanmedian(vh_end[ref]) if ref.any() else np.nan
        vh_spread = 1.4826 * np.nanmedian(np.abs(vh_end[ref] - vh_rice)) if ref.any() else np.nan

        valid = np.asarray(events["valid"], dtype=bool)
        open_water = np.asarray(events["open_water_year"], dtype=bool) if "open_water_year" in events \
            else np.zeros(n, dtype=bool)
        crop = valid & ~open_water & paddy_low & grew
        flooded = valid & ~open_water & ~crop & wet & ~grew

        halfway = (peak + trough) / 2
        cut = crop & (last < halfway)
        radar_only = crop & ~grew_opt
        young = crop & ~cut & ~radar_only & (last >= peak - k * noise) & (peak < peak_low)
        full = crop & ~cut & ~young & ~radar_only

    # sowing: the flood (transplanting water) where water was seen, else the last bare spell
    flood_day = pd.to_datetime(events["flood_date"]).to_numpy().astype("datetime64[D]") if "flood_date" in events \
        else np.full(n, np.datetime64("NaT"), dtype="datetime64[D]")
    trough_day = pd.to_datetime(events["trough_date"]).to_numpy().astype("datetime64[D]")
    sown = np.where(sowing_water & ~np.isnat(flood_day), flood_day, trough_day)
    s = np.where(sown < np.datetime64(SOW_PERIODS[0]), "S1", np.where(sown < np.datetime64(SOW_PERIODS[1]), "S2", "S3"))
    water = np.where(sowing_water, "W1", np.where(wet, "W4", np.where(against, "W3", "W2")))
    state = np.select([cut, young, full, flooded, radar_only], ["cut", "young", "full", "flooded", "radar-only"], "")
    trough_idx = np.searchsorted(w.to_numpy().astype("datetime64[D]"), np.where(np.isnat(trough_day), w[0].to_numpy()
                                                                                 .astype("datetime64[D]"), trough_day))
    cut_day = _cut_date(fit, w, np.clip(trough_idx, 0, len(w) - 1), halfway)
    h = np.where(cut_day < np.datetime64(HARVEST_PERIODS[0]), "H0",
                 np.where(cut_day <= np.datetime64(HARVEST_PERIODS[1]), "H1", "H2"))
    with np.errstate(invalid="ignore"):
        p = np.where(peak < peak_low, "P2", "P1")

    code = np.full(n, "", dtype=object)
    for i in np.flatnonzero(crop | flooded):
        parts = [s[i], water[i], state[i]]
        if state[i] == "cut":
            parts.append(h[i])
        if state[i] in ("cut", "full"):
            parts.append(p[i])
        code[i] = " ".join(parts)
    with np.errstate(invalid="ignore"):
        rest = valid & ~crop & ~flooded
        trees = rest & ~open_water & ~paddy_low & (season_med >= peak_low)
        built = rest & ~open_water & ~trees & never_green & (vh_end >= vh_rice - k * vh_spread)
        empty = rest & ~open_water & ~trees & never_green & ~built
    code[rest & open_water] = "N water"
    code[trees] = "N trees"
    code[built] = "N built"
    code[empty] = "N empty"
    code[rest & (code == "")] = "N veg"
    code[~valid] = "no data"
    return pd.DataFrame({"code": code, "sow": np.where(crop | flooded, s, ""), "water": water,
                         "state": state, "cut_date": cut_day, "peak_ndvi": peak,
                         "rice_peak_median": peak_med})


def words(code: str) -> str:
    """Plain words for a code: ``S1 W1 cut H1 P2`` -> "sown in May; water seen; cut; cut 20 Aug-16 Sep; ..."."""
    if code in WORDS:
        return WORDS[code]
    return "; ".join(WORDS.get(part, part) for part in code.split())


def ids_for(codes) -> dict[str, int]:
    """A stable id per code: crop codes sorted by text from 1, the non-crop codes after them, no data = 0."""
    codes = sorted(set(codes) - {"no data"})
    crop = [c for c in codes if not c.startswith("N ")]
    other = [c for c in codes if c.startswith("N ")]
    return {"no data": NO_DATA_ID, **{c: i + 1 for i, c in enumerate(crop + other)}}


def table(code, final, field_code=None, field_ids=None, examples: int = 3) -> pd.DataFrame:
    """Acres per code, split by the final map's class, with a few example fields whose majority is that code."""
    frame = pd.DataFrame({"code": np.asarray(code), "final": np.asarray(final)})
    frame = frame[frame["final"] != 255]
    cross = pd.crosstab(frame["code"], frame["final"])
    out = pd.DataFrame({"acres": cross.sum(axis=1).map(lambda v: round(acres(v), 1))})
    for c in cross.columns:
        out[f"final_{int(c)}_{mr.CLASSES.get(int(c), c)}"] = cross[c].map(lambda v: round(acres(v), 1))
    out.insert(0, "words", [words(c) for c in out.index])
    if field_code is not None:
        fc = pd.Series(np.asarray(field_ids), index=np.asarray(field_code))
        out["example_fields"] = [" ".join(fc.loc[[c]].head(examples)) if c in fc.index else "" for c in out.index]
    return out.sort_values("acres", ascending=False)


def field_majority(field_index, code_ids, n_fields: int) -> tuple[np.ndarray, np.ndarray]:
    """Per field: the most common code id among its pixels and its share (-1 / NaN for a field without pixels)."""
    f = np.asarray(field_index).ravel()
    c = np.asarray(code_ids).ravel()
    ok = f >= 0
    n_ids = int(c.max()) + 1 if len(c) else 1
    counts = np.bincount(f[ok] * n_ids + c[ok], minlength=n_fields * n_ids).reshape(n_fields, n_ids)
    total = counts.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, counts.argmax(axis=1), -1), np.where(total > 0, counts.max(axis=1) / total, np.nan)


def run(aoi_id: int, root: str = "processed/_batch/s2_2026") -> pd.DataFrame:
    """Write the factor map, its QGIS style, the acre table and the field file for one AOI; return the table."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    from . import field_rice
    from . import ndvi_5day as nd
    from .maps import PALETTE

    d, events, _ = mr.aoi_events(aoi_id, out_root=root, water="v3")
    shape = d["ndvi5d"].shape[1:]
    fit = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)
    f = factors(events, fit, d["windows"], mr.season_to_series(mr.SEASON, d["windows"]))
    del events, fit
    nd.forget()
    out_dir = Path(root) / f"aoi{aoi_id}"
    with rasterio.open(out_dir / f"aoi{aoi_id}_monsoon2026_final.tif") as ds:
        final = ds.read(1)
        profile = ds.profile
    code = f["code"].to_numpy()
    code[final.ravel() == 255] = "no data"
    ids = ids_for(code)
    id_map = np.vectorize(ids.get, otypes=[np.uint16])(code)

    fpath = Path(root) / "fields" / f"aoi{aoi_id}_fields_monsoon2026.gpkg"
    if not fpath.exists():
        fpath = Path(field_rice.SRC) / "fields" / fpath.name
    fields = gpd.read_file(fpath).to_crs(profile["crs"])
    order = np.argsort(-fields.geometry.area.to_numpy())
    fidx = rasterize(((g, int(i)) for i, g in zip(order, fields.geometry.to_numpy()[order])), out_shape=shape,
                     transform=profile["transform"], fill=-1, dtype="int32")
    maj, share = field_majority(fidx, id_map, len(fields))
    back = {v: c for c, v in ids.items()}
    fields["factor_code"] = [back.get(int(m), "") for m in maj]
    fields["factor_words"] = [words(c) if c else "" for c in fields["factor_code"]]
    fields["factor_share"] = np.round(share, 2)
    # the field's class on THIS final map (the delivered file's label may belong to an older map)
    fin, fin_share = field_majority(fidx, np.where(final.ravel() == 255, mr.N_CLASSES, final.ravel()), len(fields))
    fields["final_class"] = np.where(fin == mr.N_CLASSES, 255, fin)
    fields["final_name"] = fields["final_class"].map(lambda c: mr.CLASSES.get(int(c), ""))
    fields["final_share"] = np.round(fin_share, 2)
    keep = [c for c in ("field_id", "final_class", "final_name", "final_share", "factor_code", "factor_words",
                        "factor_share", "area_acres", "is_field") if c in fields] + ["geometry"]
    fields = fields[keep]
    fields.to_file(out_dir / f"aoi{aoi_id}_factors_fields.gpkg", driver="GPKG")

    big = fields[fields.get("is_field", True) != False].sort_values("area_acres", ascending=False)  # noqa: E712
    t = table(code, final.ravel(), big["factor_code"], big["field_id"])
    t.insert(0, "id", [ids[c] for c in t.index])
    t.index.name = "code"
    t.to_csv(out_dir / f"aoi{aoi_id}_factors.csv")

    prof = dict(profile, dtype="uint16", nodata=NO_DATA_ID)
    with rasterio.open(out_dir / f"aoi{aoi_id}_factors.tif", "w", **prof) as ds:
        ds.write(id_map.reshape(shape), 1)
    entries = "\n".join(f'        <paletteEntry value="{i}" color="{PALETTE[(i - 1) % len(PALETTE)]}" alpha="255" '
                        f'label="{c} ({t.loc[c, "acres"] if c in t.index else 0} ac)"/>'
                        for c, i in sorted(ids.items(), key=lambda kv: kv[1]) if i != NO_DATA_ID)
    (out_dir / f"aoi{aoi_id}_factors.qml").write_text(f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" nodataColor="">
      <colorPalette>
{entries}
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
""")
    return t


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--aoi", type=int, required=True)
    p.add_argument("--root", default="processed/_batch/s2_2026", help="series folder (a sandbox)")
    args = p.parse_args(argv)
    t = run(args.aoi, args.root)
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 60):
        print(t.drop(columns=["words"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
