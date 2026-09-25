"""Run the whole rule again on each field's mean curves, and compare with the field's pixel-majority label.

Why
---
The field labels of ``field_rice`` are a vote of pixel decisions; each pixel decided on its own
noisy curves (5 x 5 radar means, fitted NDVI). A field is one crop with one transplanting and one
flood, so its mean curves carry the same signal with much less noise: the radar is averaged in
linear power over every pixel of the field (speckle falls with the square root of the pixel count),
the NDVI over every fitted pixel. Re-deciding each field on its own means is an independent check of
every class at once:

* where both agree, the label is robust to how the evidence was aggregated;
* a field the pixel vote calls "rice-like, water unconfirmed" but whose mean radar shows the flood
  is a field whose water was hidden in pixel noise;
* a field the vote calls rice but whose mean curves fail the rule deserves a look.

The same rule and thresholds are used (``monsoon_rule.pixel_events`` / ``classify``, the v1 dip and
the v2 flood of ``radar_water``); only the unit changes. Fields under ``MIN_PX`` pixels are skipped:
their mean is not better than a pixel.

It also records, per field, when it was last seen clear by Sentinel-2 (at least half of its pixels
clear on that date): a field last seen in August is "standing on the map date" only by the fitted
curve's extrapolation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import monsoon_rule as mr
from . import ndvi_5day as nd
from . import pixel_report as pr
from . import radar_water as rw
from . import sar_curve
from .field_rice import NODATA, label_aoi

SRC = "processed/_batch/s2_2026"
MIN_PX = 4
#: "Green before the flood" (issue 23): field-mean NDVI at or above this within the 60 days before
#: the flood date means a crop stood there just before the wetting.
GREEN_BEFORE_NDVI = 0.50
GREEN_BEFORE_DAYS = 60
#: The radar's version (clouds hide the optical one for months in the double-crop AOIs): field-mean
#: VH >= -15.5 and VV >= -8.5 dB over the passes 60 to 15 days before the flood = a canopy stood there.
BRIGHT_BEFORE_DAYS = (60, 15)
BRIGHT_BEFORE_VH_MIN = -15.5
BRIGHT_BEFORE_VV_MIN = -8.5


def field_means(values, index, n_fields: int, weights_mask=None) -> np.ndarray:
    """(rows, n_fields) mean over each field's pixels of a (rows, pixels) array; NaN-safe."""
    idx = np.asarray(index)
    ok_pix = idx >= 0 if weights_mask is None else (idx >= 0) & weights_mask
    f = idx[ok_pix]
    out = np.full((values.shape[0], n_fields), np.nan)
    for r in range(values.shape[0]):
        v = values[r, ok_pix]
        good = np.isfinite(v)
        s = np.bincount(f[good], weights=v[good], minlength=n_fields)
        c = np.bincount(f[good], minlength=n_fields)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[r] = np.where(c > 0, s / c, np.nan)
    return out


def to_db(power):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(power)


def to_power(db):
    return 10 ** (np.asarray(db) / 10.0)


def field_audit(aoi_id: int, out_dir=f"{SRC}/report/field_level") -> pd.DataFrame:
    fields, idx, classes = label_aoi(aoi_id)
    if fields.empty:
        return pd.DataFrame()
    inside = (classes != NODATA).ravel()
    idx_flat = np.where(inside, idx.ravel(), -1)
    n_f = len(fields)
    npx = np.bincount(idx_flat[idx_flat >= 0], minlength=n_f)
    keep = npx >= MIN_PX
    d = nd.load(aoi_id)
    W = d["ndvi5d"].shape[0]
    ndvi_f = field_means(d["ndvi5d"].reshape(W, -1), idx_flat, n_f)[:, keep]
    lswi_f = field_means(d["lswi5d"].reshape(W, -1), idx_flat, n_f)[:, keep]
    windows = pd.DatetimeIndex(d["windows"])
    # when was each field last seen clear (>= half its pixels clear on that date)?
    dates = pd.DatetimeIndex(d["dates"])
    ok_share = field_means(d["ok"].reshape(len(dates), -1).astype(float), idx_flat, n_f)[:, keep]
    seen = ok_share >= 0.5
    last_seen = np.array([dates[np.flatnonzero(col)[-1]] if col.any() else pd.NaT for col in seen.T])
    monsoon_seen = seen[(dates >= "2026-05-15")].sum(axis=0)
    # radar: single-pixel series, averaged over the field in linear power
    loc = pr.locate(aoi_id, 0, season_key="monsoon2026")
    series = []
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        rd, cubes = sar_curve.read_track(loc, track, 1)
        flat = {p: to_db(field_means(to_power(cubes[p].reshape(len(rd), -1)), idx_flat, n_f))[:, keep]
                for p in ("VV", "VH")}
        series.append((rd, flat))
    ev = mr.pixel_events(ndvi_f, lswi_f, windows)
    trough = np.where(ev["valid"].to_numpy(), ev["trough_date"].to_numpy().astype("datetime64[D]"), np.datetime64("NaT"))
    climb = np.where(ev["valid"].to_numpy(), pd.to_datetime(ev["climb_date"]).to_numpy().astype("datetime64[D]"),
                     np.datetime64("NaT"))
    vv_dip = np.full(keep.sum(), np.nan)
    vh_dip = np.full(keep.sum(), np.nan)
    checkable = np.zeros(keep.sum(), dtype=bool)
    for rd, flat in series:
        _, _, a = rw.dips_for_track(rd, flat["VV"], trough)
        _, _, b = rw.dips_for_track(rd, flat["VH"], trough)
        vv_dip, vh_dip = np.fmax(vv_dip, a), np.fmax(vh_dip, b)
        checkable |= np.isfinite(a)
    v1 = checkable & ((vv_dip >= rw.DIP_MIN_DB) | (vh_dip >= rw.DIP_MIN_DB))
    raw_f = field_means(d["ndvi5d_raw"].reshape(W, -1), idx_flat, n_f)[:, keep] if "ndvi5d_raw" in d else None
    v2 = rw.water_evidence(aoi_id, trough, climb, ndvi_f, windows, series=series, ndvi_raw=raw_f)
    wet = v1 | v2["flood_ok"].to_numpy()
    # the same decision path as the pixel rule (radar-defined trough, report classes)
    ev2 = pd.concat([ev, v2], axis=1)
    ev2 = pd.concat([ev2, mr.radar_trough_events(ev2, ndvi_f, windows)], axis=1)
    field_class = mr.classify(ev2, radar_wet=wet, never_bare=v2["never_bare"].to_numpy(),
                              radar_trough=mr.RADAR_TROUGH_DEFAULT, map_date=windows[-1])
    # was the field green shortly before its flood? (issue 23: a mid-August wetting after a
    # June-July crop may be a second transplanting or rain under a standing crop; the data cannot
    # tell, so the field is delivered with low confidence and listed for a field check)
    flood = pd.to_datetime(v2["flood_date"]).to_numpy().astype("datetime64[D]")
    win = windows.to_numpy().astype("datetime64[D]")
    rel = (win[:, None] - flood[None, :]) / np.timedelta64(1, "D")
    with np.errstate(invalid="ignore"):
        before = (rel >= -GREEN_BEFORE_DAYS) & (rel <= -5)
        peak_before = np.where(before.any(axis=0), np.nanmax(np.where(before, ndvi_f, -np.inf), axis=0), np.nan)
    # the radar's view of the same question (issue 23 under cloud): a canopy stood on the field in
    # the weeks before the flood when the field-mean VH and VV sat at canopy levels then
    bright = np.zeros(keep.sum(), dtype=bool)
    for rd, flat in series:
        day = pd.DatetimeIndex(rd).to_numpy().astype("datetime64[D]")
        lag = (flood[None, :] - day[:, None]) / np.timedelta64(1, "D")
        with np.errstate(invalid="ignore"):
            inwin = (lag >= BRIGHT_BEFORE_DAYS[1]) & (lag <= BRIGHT_BEFORE_DAYS[0])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vh_b = np.nanmedian(np.where(inwin, flat["VH"], np.nan), axis=0)
            vv_b = np.nanmedian(np.where(inwin, flat["VV"], np.nan), axis=0)
        bright |= (vh_b >= BRIGHT_BEFORE_VH_MIN) & (vv_b >= BRIGHT_BEFORE_VV_MIN)
    cols = ["field_id", "uid", "area_acres", "pixels", "label", "rice_share", "unconfirmed_share"]
    out = fields.loc[keep, [c for c in cols if c in fields]].reset_index(drop=True)
    out["field_rule_label"] = field_class
    out["trough_date"] = ev["trough_date"].to_numpy()
    out["climb_date"] = ev["climb_date"].to_numpy()
    out["last_ndvi"] = ev["last_ndvi"].to_numpy()
    out["VV_dip"], out["VH_dip"] = vv_dip, vh_dip
    out["v1_wet"], out["v2_flood"] = v1, v2["flood_ok"].to_numpy()
    out["flood_drop"], out["flood_vh"] = v2["flood_drop"].to_numpy(), v2["flood_vh"].to_numpy()
    out["flood_date"] = v2["flood_date"].to_numpy()
    out["peak_before_flood"] = peak_before
    out["green_before_flood"] = np.isfinite(peak_before) & (peak_before >= GREEN_BEFORE_NDVI)
    out["radar_bright_before_flood"] = bright
    out["never_bare"] = v2["never_bare"].to_numpy()
    out["last_seen_clear"] = last_seen
    out["days_since_clear"] = np.array([(windows[-1] - pd.Timestamp(t)).days if pd.notna(t) else np.nan for t in last_seen])
    out["monsoon_clear_dates"] = monsoon_seen
    out["aoi"] = f"aoi{aoi_id}"
    nd.forget()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out.to_parquet(Path(out_dir) / f"aoi{aoi_id}.parquet")
    return out


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis.field_level")
    p.add_argument("--ids", nargs="+", type=int, required=True)
    args = p.parse_args(argv)
    for a in args.ids:
        t = field_audit(a)
        if len(t):
            agree = (t["label"] == t["field_rule_label"]).mean()
            print(f"aoi{a}: {len(t)} fields, agreement {100 * agree:.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def reference_scores(aoi_ids, plots, out_dir=f"{SRC}/report/field_level") -> pd.DataFrame:
    """For the plots and negative sets: share of their pixels whose FIELD is called rice by the pixel
    vote (``label``) and by the field-mean rule (``field_rule_label``)."""
    from . import validation as va

    rows = []
    for aoi_id in aoi_ids:
        path = Path(out_dir) / f"aoi{aoi_id}.parquet"
        if not path.exists():
            continue
        fl = pd.read_parquet(path).set_index("uid")
        fields, idx, _ = label_aoi(aoi_id)
        refs = va.reference_sets(aoi_id, plots[plots["aoi"] == f"aoi{aoi_id}"])
        fi = np.asarray(idx).ravel()[refs["pixel"].to_numpy()]
        uid = np.where(fi >= 0, fields["uid"].to_numpy()[np.clip(fi, 0, None)], None)
        refs = refs.assign(uid=uid)
        refs = refs[refs["uid"].isin(fl.index)]
        refs = refs.join(fl[["label", "field_rule_label"]], on="uid")
        for (s, region), g in refs.groupby(["set", "region"]):
            rows.append({"aoi": f"aoi{aoi_id}", "set": s, "region": region, "pixels": len(g),
                         "vote_rice_pct": round(100 * float((g["label"] == 1).mean()), 1),
                         "mean_rule_rice_pct": round(100 * float((g["field_rule_label"] == 1).mean()), 1),
                         "vote_unconf_pct": round(100 * float((g["label"] == 3).mean()), 1),
                         "mean_rule_unconf_pct": round(100 * float((g["field_rule_label"] == 3).mean()), 1)})
        nd.forget()
    return pd.DataFrame(rows)
