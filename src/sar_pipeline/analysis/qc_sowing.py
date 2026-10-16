"""Sowing date and crop age at mid-September for the rice fields of the QC'd layers (``qc_post``).

Why (manager via the user, 15 Oct 2026): for the AOIs that had no clear Sentinel-2 view in September, each rice field
gets its sowing date and its crop age on 15 September as attributes. The other AOIs follow later.

How (user, 15 Oct 2026: "pixel pehle, phir median ... thek h"):

1. per pixel, the rule's own sowing date (``curve_rules.own_range_features`` column ``sowing_day``) under the rules
   of the layer the QC was made on (the AOI's own rules for v1, the adopted ``weak_rules`` rules for v2 / v3);
2. per field (each QC polygon, all its parts together), the median of its pixels' dates, so a few odd pixels or a
   wet corner do not move it; ``sowing_source`` is where most of its pixels' dates came from;
3. ``crop_age_15sep`` = 15 September 2026 minus the sowing date, in days; a field sown after that day gets no age and
   the note "sown after 15 Sep".

Before the dates go into every layer, a blind review checks them (user, 15 Oct 2026): fields of every sowing source
are drawn as blind sheets (``field_review.render``'s sheets, no rule date shown) and reviewers write the water start,
the radar rise and the planting they see (``docs/sowing_review_brief.md``); :func:`score` compares.

Run::

    python -m sar_pipeline.analysis.qc_sowing fields --aois 1198      # sowing per field, one AOI
    python -m sar_pipeline.analysis.qc_sowing sample --per-source 40  # blind-review sample over the test AOIs
    python -m sar_pipeline.analysis.qc_sowing render
    python -m sar_pipeline.analysis.qc_sowing score
    python -m sar_pipeline.analysis.qc_sowing detach                  # 16 Oct: sowing columns removed again (wrong)

Status (user, 16 Oct 2026): the dates were found almost all wrong and were removed from the delivery (``detach``); the
module stays for the record and for a later, better attempt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .qc_post import OUT, out_path

#: The AOIs with no clear (>= 80 %) Sentinel-2 view in September 2026 (delivery MANIFEST, 12 Oct 2026).
NO_CLEAR_SEPTEMBER = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 127, 128, 129, 132, 134, 135, 136, 137,
                      138, 139, 140, 141, 142, 143, 144, 1158, 1159, 1160, 1162, 1164, 1174, 1175, 1176, 1177, 1214,
                      1227, 1228)
AGE_DAY = pd.Timestamp("2026-09-15")
#: ``sowing_from`` codes of ``curve_rules.own_range_features``.
SOURCES = {0: "NDVI: end of the empty spell", 1: "radar: end of the water spell", 2: "radar: start of the rise",
           3: "radar: start of the water spell", 4: "NDVI: lowest view before the crop was seen rising"}
#: Sowing from the radar for every field (``curve_rules.SOWING_FROM_RADAR``), dated this many days before the radar
#: date (end of the water spell / start of the rise). Why (blind review of 160 fields, user 15 Oct 2026): the start
#: of the water spell came ~3 weeks before the planting the reviewers read (30 % within 14 days); radar first put it
#: ~8 days late; radar first minus 7 days ("radar rise se 7 din pehle"): 76 % within 14 days, 50 % within 7, median
#: 3 days. The 7 days were chosen on those same fields, so the true rate is likely a little lower.
RADAR_FIRST = True
RADAR_LEAD_DAYS = 7
RADAR_CODES = (1, 2)
REVIEW = Path(OUT) / "sowing_review"


def _source(code) -> str:
    if pd.isna(code):
        return ""
    name = SOURCES.get(int(code), "")
    return f"{name} - {RADAR_LEAD_DAYS} days" if RADAR_FIRST and int(code) in RADAR_CODES else name


def rules_for_layer(aoi: int, base_layer: str) -> dict:
    """The rules the QC'd layer was made with: v3 / v2 from the adopted ``weak_rules`` records, else the AOI's own."""
    from . import curve_rules as cr
    from . import weak_rules as wr

    for ver in ("v3", "v2"):
        if base_layer.startswith(f"fields_{ver}"):
            p = wr.VERSIONS[ver][0] / f"aoi{aoi}.json"
            if not p.exists():                        # moved to the backup with the rest of the data (13 Oct 2026)
                from .. import storage_plan as sp

                sp.restore(under=f"sar-rice-mapper/{p}")
            rec = json.loads(p.read_text())
            return dict(rec["rules"])
    return {k: v for k, v in cr.AOI_OVERRIDES.get(aoi, {}).items() if k != "SIEVE_ACRES"}


def pixel_index(aoi: int, layer) -> tuple[np.ndarray, np.ndarray]:
    """(pixel ids, row index in ``layer``) of every grid pixel whose centre lies in a polygon of ``layer``."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    from . import pixel_report as pr

    g = pr.locate(aoi, 0, season_key="monsoon2026")["grid"]
    lay = layer.to_crs(g.get("crs", "EPSG:32646"))
    ids = rasterize(((geom, i + 1) for i, geom in enumerate(lay.geometry)), out_shape=(int(g["height"]), int(g["width"])),
                    transform=from_origin(g["x0"], g["y0"], g["res"], g["res"]), fill=0, dtype="int32").ravel()
    px = np.flatnonzero(ids)
    return px, ids[px] - 1


def field_sowing(aoi: int, layer=None, base_layer: str | None = None) -> pd.DataFrame:
    """Per field of the QC'd layer: ``sowing_date``, ``crop_age_15sep``, ``sowing_source``, ``sowing_pixels_pct``.
    Fetches the AOI's inputs from the backup first (``storage_plan.ensure_aoi``)."""
    import geopandas as gpd

    from .. import storage_plan as sp
    from . import curve_rules as cr

    sp.ensure_aoi(aoi, parts=("series", "radar"))
    if layer is None:
        layer = gpd.read_file(out_path(aoi))
    if base_layer is None:
        base_layer = json.loads(out_path(aoi).with_suffix(".json").read_text())["base_layer"]
    if len(layer) == 0:
        return pd.DataFrame(columns=["field_id", "sowing_date", "crop_age_15sep", "sowing_source", "sowing_pixels_pct",
                                     "sowing_note"])
    px, row = pixel_index(aoi, layer)
    key = -994
    rules = rules_for_layer(aoi, base_layer)
    if RADAR_FIRST:
        rules = {**rules, "SOWING_FROM_RADAR": True}
    cr.AOI_OVERRIDES[key] = rules
    try:
        with cr.rules_for(key):
            f = cr.own_range_features(aoi, px)
    finally:
        cr.AOI_OVERRIDES.pop(key, None)
    t = pd.DataFrame({"field_id": layer["field_id"].to_numpy()[row],
                      "day": pd.to_numeric(f["sowing_day"], errors="coerce").to_numpy(),
                      "src": pd.to_numeric(f.get("sowing_from", pd.Series(np.nan, index=f.index)),
                                           errors="coerce").to_numpy()})
    t.loc[t["day"] < 0, "day"] = np.nan
    if RADAR_FIRST:                                   # radar dates: the planting is a week before the radar's date
        t.loc[t["src"].isin(RADAR_CODES), "day"] -= RADAR_LEAD_DAYS
    n_all = t.groupby("field_id").size()
    g = t.dropna(subset=["day"]).groupby("field_id")
    out = pd.DataFrame({"day": g["day"].median(), "n": g.size()}).reindex(n_all.index)
    out["src"] = t.dropna(subset=["day", "src"]).groupby("field_id")["src"].agg(lambda s: s.mode().iloc[0])
    date = pd.to_datetime(out["day"].round(), unit="D")
    age = (AGE_DAY - date).dt.days
    res = pd.DataFrame({"field_id": out.index, "sowing_date": date.dt.strftime("%Y-%m-%d").to_numpy(),
                        "crop_age_15sep": age.where(age >= 0).astype("Int64").to_numpy(),
                        "sowing_source": out["src"].map(lambda s: _source(s)).to_numpy(),
                        "sowing_pixels_pct": (100 * out["n"].fillna(0) / n_all).round(1).to_numpy()})
    res["sowing_note"] = np.where(age.to_numpy() < 0, "sown after 15 Sep",
                                  np.where(res["sowing_date"].isna(), "no sowing date", ""))
    fields_without_pixels = sorted(set(layer["field_id"]) - set(res["field_id"]))
    if fields_without_pixels:                     # a polygon too small to hold a pixel centre
        res = pd.concat([res, pd.DataFrame({"field_id": fields_without_pixels, "sowing_note": "no pixel inside"})],
                        ignore_index=True)
    return res


SOWING_DIR = Path(OUT) / "sowing"


def _fields_one(aoi: int) -> dict:
    try:
        t = field_sowing(aoi)
        SOWING_DIR.mkdir(parents=True, exist_ok=True)
        t.to_csv(SOWING_DIR / f"aoi{aoi}_sowing.csv", index=False)
        return {"aoi": aoi, "fields": len(t), "with_date": int(t["sowing_date"].notna().sum()),
                "after_15sep": int((t["sowing_note"] == "sown after 15 Sep").sum())}
    except Exception as e:                             # one AOI's failure does not stop the others
        return {"aoi": aoi, "error": f"{type(e).__name__}: {e}"}


def fields_all(aois=NO_CLEAR_SEPTEMBER, jobs: int = 0) -> pd.DataFrame:
    """:func:`field_sowing` for every AOI, one process per core (each AOI needs about 1 GB), each AOI's table in
    ``<OUT>/sowing/aoi<N>_sowing.csv``."""
    from concurrent.futures import ProcessPoolExecutor

    from .. import resources

    r = resources.detect_resources()
    n = jobs or max(1, min(r.cpus, int(r.memory_available_bytes / 2e9)))
    with ProcessPoolExecutor(n) as ex:
        return pd.DataFrame(list(ex.map(_fields_one, aois)))


TRUTH = SOWING_DIR / "review_sample_with_rule_dates.csv"     # outside the review folder: reviewers never see it


def sample(per_source: int = 40, per_aoi: int = 6, min_ac: float = 0.25, seed: int = 0) -> pd.DataFrame:
    """Fields for the blind review: per sowing source up to ``per_source`` fields, at most ``per_aoi`` from one AOI so
    many AOIs (both sets, every rules version) take part; only fields of at least ``min_ac`` whose pixels mostly got
    a date. The rule's dates go to :data:`TRUTH`; the review folder gets only ``fields.txt`` / ``groups.txt``."""
    import geopandas as gpd

    rows = []
    for f in sorted(SOWING_DIR.glob("aoi*_sowing.csv")):
        aoi = int(f.stem.split("_")[0][3:])
        t = pd.read_csv(f)
        lay = gpd.read_file(out_path(aoi))
        ac = lay.groupby("field_id")["area_ac"].sum()
        base = json.loads(out_path(aoi).with_suffix(".json").read_text())["base_layer"]
        t = t.assign(aoi=aoi, acres=t["field_id"].map(ac), base_layer=base)
        rows.append(t[(t["acres"] >= min_ac) & (t["sowing_pixels_pct"] >= 50) & t["sowing_date"].notna()])
    allf = pd.concat(rows, ignore_index=True)
    rng = np.random.default_rng(seed)
    picks = []
    for src, g in allf.groupby("sowing_source"):
        g = g.sample(frac=1, random_state=int(rng.integers(1 << 31)))
        g = g.groupby("aoi").head(per_aoi)
        picks.append(g.head(per_source))
    s = pd.concat(picks, ignore_index=True)
    TRUTH.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(TRUTH, index=False)
    REVIEW.mkdir(parents=True, exist_ok=True)
    ids = s.sample(frac=1, random_state=seed)["field_id"].tolist()          # mixed order: no source runs in a group
    (REVIEW / "fields.txt").write_text("\n".join(ids) + "\n")
    (REVIEW / "groups.txt").write_text("\n".join(" ".join(ids[i:i + 20]) for i in range(0, len(ids), 20)) + "\n")
    return s


def _render_aoi(aoi: int) -> int:
    """Blind sheets of the sampled fields of one AOI (``field_review.render``'s drawing: no rule class, no sowing)."""
    import geopandas as gpd
    import matplotlib
    import matplotlib.pyplot as plt
    import shapely

    from .. import qgis_review as q
    from .. import storage_plan as sp
    from . import field_review as fr
    from . import ndvi_5day as nd
    from . import pixel_2026 as p26
    from . import pixel_report as pr
    from . import sar_curve
    from . import water_investigation as wi

    matplotlib.use("Agg")
    sp.ensure_aoi(aoi)                                     # the chips need the AOI's per-date Sentinel-2 files too
    ids = pd.read_csv(TRUTH).query("aoi == @aoi")["field_id"].tolist()
    lay = gpd.read_file(out_path(aoi))
    px, row = pixel_index(aoi, lay)
    series = [(dd, {p: c.reshape(len(dd), -1) for p, c in cubes.items()})
              for dd, cubes in sar_curve.cache_tracks(pr.locate(aoi, 0, season_key="monsoon2026"))]
    g = nd.load(aoi, out_root=q.series_root(aoi))["loc"]["grid"]
    out = REVIEW / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    for fid in ids:
        if (out / f"{fid}.txt").exists():
            continue
        sel = (lay["field_id"] == fid).to_numpy()
        fpids = px[sel[row]]
        geom = shapely.union_all(lay.geometry[sel].to_numpy())
        geom = geom if geom.geom_type == "Polygon" else max(geom.geoms, key=lambda x: x.area)
        rr, cc = np.divmod(fpids, int(g["width"]))
        centre = int(np.round(np.median(rr)) * int(g["width"]) + np.round(np.median(cc)))
        r0, c0 = divmod(centre, int(g["width"]))
        xs, ys = geom.exterior.xy
        ox = (np.asarray(xs) - g["x0"]) / g["res"] - 0.5 - (c0 - p26.HALF)
        oy = (g["y0"] - np.asarray(ys)) / g["res"] - 0.5 - (r0 - p26.HALF)
        wi.group_sheet(aoi, centre, out_dir=out, file_stem=fid, label=f"field {fid}", pids=fpids, outline=(ox, oy),
                       series_root=q.series_root(aoi), sowing=np.nan, sowing_label="")
        plt.close("all")
        (out / f"{fid}.txt").write_text(f"field {fid}: {len(fpids)} pixels, {float(lay.area[sel].sum()) / 4046.86:.2f}"
                                        f" acres\n" + fr._numbers(aoi, fpids, series=series))
        done += 1
    return done


def render(jobs: int = 0) -> int:
    """:func:`_render_aoi` for every AOI in the sample, one process per core."""
    from concurrent.futures import ProcessPoolExecutor

    from .. import resources

    aois = sorted(pd.read_csv(TRUTH)["aoi"].unique().tolist())
    with ProcessPoolExecutor(jobs or resources.detect_resources().cpus) as ex:
        return sum(ex.map(_render_aoi, aois))


def score() -> pd.DataFrame:
    """Rule vs reviewers, per sowing source: days from the rule's date to the reviewer's planting, water start and rise
    start (median, and share within 14 days). Written to ``<OUT>/sowing/review_score.csv``."""
    t = pd.read_csv(TRUTH)
    # the dates as the fields' tables hold them NOW (after a method change), not as they were when sampled
    cur = pd.concat([pd.read_csv(SOWING_DIR / f"aoi{a}_sowing.csv") for a in sorted(t["aoi"].unique())])
    t = t.drop(columns=["sowing_date"]).merge(cur[["field_id", "sowing_date"]], on="field_id", how="left")
    v = []
    for f in sorted((REVIEW / "verdicts").glob("*.json")):
        v.append(json.loads(f.read_text()))
    v = pd.DataFrame(v)
    m = t.merge(v, on="field_id", how="inner")
    rule = pd.to_datetime(m["sowing_date"])
    for k in ("planting", "water_start", "rise_start"):
        m[f"{k}_minus_rule"] = (pd.to_datetime(m[k].replace("", None), errors="coerce") - rule).dt.days
    rows = []
    for src, g in [("all", m)] + list(m.groupby("sowing_source")):
        r = {"sowing_source": src, "fields": len(g)}
        for k in ("planting", "water_start", "rise_start"):
            d = g[f"{k}_minus_rule"].dropna()
            r[f"{k}_n"] = len(d)
            r[f"{k}_median_days"] = float(d.median()) if len(d) else None
            r[f"{k}_within14_pct"] = round(100 * float((d.abs() <= 14).mean()), 1) if len(d) else None
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(SOWING_DIR / "review_score.csv", index=False)
    m.to_csv(SOWING_DIR / "review_fields.csv", index=False)
    return out


def annotated_sheet(aoi: int, field_id: str, lines: dict[str, str], out_dir: Path | None = None) -> Path:
    """The field's curve sheet (as the reviewers saw it) with vertical lines at the given dates ({label: YYYY-MM-DD}),
    for showing the user how the rule's and the reviewers' dates sit on the curves. Not used by the review itself."""
    import geopandas as gpd
    import matplotlib
    import matplotlib.pyplot as plt
    import shapely

    from .. import qgis_review as q
    from . import ndvi_5day as nd
    from . import pixel_2026 as p26
    from . import water_investigation as wi

    matplotlib.use("Agg")
    out_dir = Path(out_dir or SOWING_DIR / "examples")
    out_dir.mkdir(parents=True, exist_ok=True)
    lay = gpd.read_file(out_path(aoi))
    px, row = pixel_index(aoi, lay)
    sel = (lay["field_id"] == field_id).to_numpy()
    fpids = px[sel[row]]
    g = nd.load(aoi, out_root=q.series_root(aoi))["loc"]["grid"]
    geom = shapely.union_all(lay.geometry[sel].to_numpy())
    geom = geom if geom.geom_type == "Polygon" else max(geom.geoms, key=lambda x: x.area)
    rr, cc = np.divmod(fpids, int(g["width"]))
    centre = int(np.round(np.median(rr)) * int(g["width"]) + np.round(np.median(cc)))
    r0, c0 = divmod(centre, int(g["width"]))
    xs, ys = geom.exterior.xy
    ox = (np.asarray(xs) - g["x0"]) / g["res"] - 0.5 - (c0 - p26.HALF)
    oy = (g["y0"] - np.asarray(ys)) / g["res"] - 0.5 - (r0 - p26.HALF)
    fig, _, _ = wi.group_sheet(aoi, centre, out_dir=out_dir / "_tmp", file_stem=field_id, label=f"field {field_id}",
                               pids=fpids, outline=(ox, oy), series_root=q.series_root(aoi), sowing=np.nan,
                               sowing_label="")
    colours = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    for k, (lab, d) in enumerate(lines.items()):
        if not d:
            continue
        for ax in fig.axes:
            ax.axvline(pd.Timestamp(d), color=colours[k % len(colours)], lw=2, ls="--")
        fig.axes[0].text(pd.Timestamp(d), 0.98 - 0.07 * k, f" {lab}: {d}", color=colours[k % len(colours)],
                         transform=fig.axes[0].get_xaxis_transform(), fontsize=10, va="top", weight="bold")
    path = out_dir / f"{field_id}_annotated.png"
    fig.savefig(path, dpi=90, facecolor="white")
    plt.close("all")
    return path


ATTRS = ("sowing_date", "crop_age_15sep", "sowing_source", "sowing_pixels_pct", "sowing_note")


def attach(aoi: int) -> dict:
    """Adds the sowing columns (:data:`ATTRS`) to the AOI's QC layer, by ``field_id`` (every part of a field gets its
    field's values). Only attribute columns change; each geometry is written back exactly as it was read."""
    import geopandas as gpd

    from .qc_post import qml

    path = out_path(aoi)
    lay = gpd.read_file(path)
    t = pd.read_csv(SOWING_DIR / f"aoi{aoi}_sowing.csv", dtype={"field_id": str})
    lay = lay.drop(columns=[c for c in ATTRS if c in lay.columns])
    lay["field_id"] = lay["field_id"].astype(str) if len(lay) else lay["field_id"].astype(object)
    lay = lay.merge(t[["field_id", *ATTRS]], on="field_id", how="left")      # an empty layer gets the empty columns
    lay["crop_age_15sep"] = pd.to_numeric(lay["crop_age_15sep"], errors="coerce").astype("Int64")
    cols = [c for c in lay.columns if c != "geometry"] + ["geometry"]
    tmp = path.with_name(f".{path.name}")
    gpd.GeoDataFrame(lay[cols], geometry="geometry", crs=lay.crs).to_file(tmp, driver="GPKG", layer=path.stem)
    tmp.replace(path)
    qml(path)
    rep = json.loads(path.with_suffix(".json").read_text())
    rep["sowing"] = {"fields": int(lay["field_id"].nunique()), "with_date": int(lay["sowing_date"].notna().sum()),
                     "method": f"rule sowing, radar first minus {RADAR_LEAD_DAYS} days; field = median of its pixels",
                     "age_day": str(AGE_DAY.date())}
    path.with_suffix(".json").write_text(json.dumps(rep, indent=1, default=str))
    return {"aoi": aoi, **rep["sowing"]}


BACKUP = Path(OUT) / "_backup_with_sowing_2026-10-16"


def detach(aoi: int) -> dict:
    """Removes the sowing columns (:data:`ATTRS`) and the report's ``sowing`` entry from the AOI's QC layer again.

    Why (user, 16 Oct 2026): the sowing dates and crop ages turned out almost all wrong, so they leave the delivery;
    the polygons stay exactly as they are. The columns are dropped inside the GeoPackage (SQLite ``ALTER TABLE ...
    DROP COLUMN``), so every geometry blob stays byte for byte the same; a copy of the layer and its report as they
    were goes to :data:`BACKUP` first."""
    import shutil
    import sqlite3

    path = out_path(aoi)
    keep = BACKUP / path.parent.relative_to(OUT)
    keep.mkdir(parents=True, exist_ok=True)
    for f in (path, path.with_suffix(".json")):
        if not (keep / f.name).exists():
            shutil.copy2(f, keep / f.name)
    con = sqlite3.connect(path)
    try:
        table = con.execute("SELECT table_name FROM gpkg_contents WHERE data_type = 'features'").fetchone()[0]
        cols = {r[1] for r in con.execute(f'PRAGMA table_info("{table}")')}
        dropped = [c for c in ATTRS if c in cols]
        for c in dropped:
            con.execute(f'ALTER TABLE "{table}" DROP COLUMN "{c}"')
        if con.execute("SELECT 1 FROM sqlite_master WHERE name = 'gpkg_data_columns'").fetchone():
            con.execute(f"DELETE FROM gpkg_data_columns WHERE table_name = ? AND column_name IN "
                        f"({','.join('?' * len(ATTRS))})", (table, *ATTRS))
        con.commit()
    finally:
        con.close()
    rep = json.loads(path.with_suffix(".json").read_text())
    had = rep.pop("sowing", None) is not None
    path.with_suffix(".json").write_text(json.dumps(rep, indent=1, default=str))
    return {"aoi": aoi, "columns_dropped": len(dropped), "report_entry_dropped": had}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["fields", "fields-all", "sample", "render", "score", "attach", "detach"])
    p.add_argument("--per-source", type=int, default=40)
    p.add_argument("--aois", type=int, nargs="*", default=[])
    a = p.parse_args(argv)
    if a.step == "sample":
        s_ = sample(a.per_source)
        print(s_.groupby(["sowing_source", "base_layer"]).size().to_string())
        print(f"{len(s_)} fields from {s_['aoi'].nunique()} AOIs")
        return 0
    if a.step == "render":
        print(render(), "sheets drawn")
        return 0
    if a.step == "attach":
        for x in a.aois or NO_CLEAR_SEPTEMBER:
            print(attach(x))
        return 0
    if a.step == "detach":
        for x in a.aois or NO_CLEAR_SEPTEMBER:
            print(detach(x))
        return 0
    if a.step == "score":
        print(score().to_string(index=False))
        return 0
    if a.step == "fields-all":
        print(fields_all(a.aois or NO_CLEAR_SEPTEMBER).to_string(index=False))
        return 0
    if a.step == "fields":
        for x in a.aois:
            t = field_sowing(x)
            print(f"aoi{x}: {len(t)} fields; " + t["sowing_source"].value_counts().to_string().replace("\n", "; "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
