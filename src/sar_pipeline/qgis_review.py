"""Everything needed to check the maps by eye in QGIS, as few files as possible.

Why
---
The maps exist per AOI and per version (the first water test, the current rule, the sieved finals),
which makes downloading and comparing them in QGIS tedious. This module builds:

* ``maps_all_versions_mosaic.tif`` — one GeoTIFF over all AOIs, **one band per version**, same 10 m
  grid (all AOIs share EPSG:32646), 255 = no data. Bands (also in the band descriptions):
  1 ``rule_v1`` (first water test, per pixel, unsieved), 2 ``rule_current`` (water test v2 with
  artefact passes dropped and class 6 young rice, per pixel, unsieved), 3 ``final_previous``
  (v2 + the user's relabel + 4-pixel sieve, before class 6), 4 ``final_current`` (current rule +
  relabel + sieve; the map the field labels come from);
* ``classes.qml`` — a QGIS style with the class colours (load it on any band);
* ``fields_all_aois.gpkg`` — every delineated field with its label, in one file;
* ``review_aois.csv`` — the AOIs picked for review (mostly rice, least rice, mixed; from all
  directions), with their class shares and the cloud-storage folder of their Sentinel-2 images.

The GDAL command-line tools do the mosaicking (``gdalbuildvrt``, ``gdal_translate``, ``ogr2ogr``).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

SRC = "processed/_batch/s2_2026"
OUT = f"{SRC}/qgis_review"
#: Map versions shown per pixel by ``inspect_field`` / ``inspect_pixel``. ``first_map`` is the frozen copy of
#: the map delivered on 24 Sep 2026 (``baseline_v3/``, fix plan stage 0); ``final_current`` is the map as it
#: is now; ``rule_current`` the same before relabels and sieve; ``rule_v1`` the first water test.
VERSIONS = (("rule_v1", "_v1"), ("rule_current", ""), ("first_map", "baseline_v3"), ("final_current", "_final"))
COLOURS = {0: ("not rice", "#e6e6e6"), 1: ("rice, standing", "#008c3c"), 2: ("young", "#aadc78"),
           3: ("rice-like, water not confirmed", "#f5a028"), 4: ("harvested", "#96643c"),
           5: ("never bare (trees/houses)", "#8c5abe"), 6: ("young rice, standing", "#6ec83c"),
           7: ("flooded, not yet green", "#3c78c8"), 8: ("cut crop, water not confirmed", "#c8aa78"),
           9: ("rice-like, no sign of water", "#009696")}


def version_path(aoi: str, suffix: str, src_root=SRC) -> Path:
    """The file of one map version for one AOI. A suffix naming a folder (``baseline_v3``) points at
    the frozen copy of a delivered map in that folder instead of a suffix on the AOI's own file."""
    if suffix.startswith("baseline"):
        return Path(src_root) / suffix / f"{aoi}_monsoon2026_final.tif"
    return Path(src_root) / aoi / f"{aoi}_monsoon2026{suffix}.tif"


def version_files(suffix: str, src_root=SRC) -> list[str]:
    """Each AOI's map of one version (see :func:`version_path`)."""
    return sorted(str(version_path(d.name, suffix, src_root)) for d in Path(src_root).glob("aoi*")
                  if version_path(d.name, suffix, src_root).exists())


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def mosaic(out_dir=OUT, src_root=SRC) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vrts = []
    for name, suffix in VERSIONS:
        files = version_files(suffix, src_root)
        lst = out / f"{name}_files.txt"
        lst.write_text("\n".join(files))
        vrt = out / f"{name}.vrt"
        run(["gdalbuildvrt", "-q", "-srcnodata", "255", "-vrtnodata", "255", "-input_file_list", str(lst), str(vrt)])
        vrts.append(str(vrt))
    stack = out / "maps_all_versions.vrt"
    run(["gdalbuildvrt", "-q", "-separate", "-srcnodata", "255", "-vrtnodata", "255", str(stack), *vrts])
    tif = out / "maps_all_versions_mosaic.tif"
    run(["gdal_translate", "-q", "-of", "GTiff", "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES",
         "-co", "BIGTIFF=IF_SAFER", str(stack), str(tif)])
    import rasterio

    with rasterio.open(tif, "r+") as ds:
        for i, (name, _) in enumerate(VERSIONS, start=1):
            ds.set_band_description(i, name)
    run(["gdaladdo", "-q", "-r", "nearest", str(tif), "2", "4", "8", "16"])
    return tif


def qml(out_dir=OUT) -> Path:
    items = "\n".join(f'        <paletteEntry value="{k}" label="{k} {n}" color="{c}" alpha="255"/>'
                      for k, (n, c) in COLOURS.items())
    text = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" nodataColor="">
      <colorPalette>
{items}
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
"""
    p = Path(out_dir) / "classes.qml"
    p.write_text(text)
    return p


def merge_fields(out_dir=OUT, src_root=SRC) -> Path:
    out = Path(out_dir) / "fields_all_aois.gpkg"
    out.unlink(missing_ok=True)
    for i, f in enumerate(sorted(Path(src_root, "fields").glob("aoi*_fields_monsoon2026.gpkg"))):
        cmd = ["ogr2ogr", "-f", "GPKG", str(out), str(f), "-nln", "fields"]
        if i:
            cmd.insert(1, "-append")
        run(cmd)
    return out


def review_aois(ids, out_dir=OUT, bucket_folder: str | None = None, shares_csv=f"{SRC}/report/aoi_final_shares.csv") -> pd.DataFrame:
    """The AOIs picked for review with their shares and where their Sentinel-2 images are."""
    t = pd.read_csv(shares_csv)
    t = t[t["aoi"].isin([f"aoi{i}" for i in ids])].copy()
    if bucket_folder:
        t["s2_images_gcs"] = t["aoi"].map(lambda a: f"gs://{bucket_folder}/s2_dates_masks/{a}/")
    t["s2_images_local"] = t["aoi"].map(lambda a: f"data/s2_dates_masks/{a}/")
    keep = [c for c in ("aoi", "region", "lat", "lon", "acres", "rice_pct", "unconf_pct", "notrice_pct",
                        "s2_images_gcs", "s2_images_local") if c in t]
    t = t[keep].round(2)
    t.to_csv(Path(out_dir) / "review_aois.csv", index=False)
    return t


# --- looking at one place: point -> AOI, pixel, field; curves and chips --------------------------------
def locate(lon: float, lat: float, src_root="processed", aoi_dir="data/aoi"):
    """(aoi_id, pixel id, row, col) of the 10 m pixel under a lon/lat point inside an AOI polygon, or None.

    Copy the coordinates from QGIS (right-click > Copy Coordinate, in EPSG:4326)."""
    import json

    import geopandas as gpd
    import pyproj
    from shapely.geometry import Point

    x, y = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32646", always_xy=True).transform(lon, lat)
    for gpath in sorted(Path(src_root).glob("aoi*/monsoon2026/grid/grid_def.json")):
        g = json.loads(gpath.read_text())
        col = int((x - g["x0"]) // g["res"])
        row = int((g["y0"] - y) // g["res"])
        if not (0 <= col < int(g["width"]) and 0 <= row < int(g["height"])):
            continue
        aoi_id = int(gpath.parts[-4][3:])
        poly = gpd.read_file(Path(aoi_dir) / f"aoi_{aoi_id:03d}.gpkg").to_crs(4326).geometry.union_all()
        if poly.contains(Point(lon, lat)):
            return aoi_id, row * int(g["width"]) + col, row, col
    return None


def field_at(aoi_id: int, lon: float, lat: float):
    """The delineated field under the point (GeoDataFrame row in EPSG:32646) and its pixel ids."""
    import geopandas as gpd
    import numpy as np
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from shapely.geometry import Point

    from .analysis import field_rice as fr
    from .analysis import pixel_report as pr

    path = Path(SRC) / "fields" / f"aoi{aoi_id}_fields_monsoon2026.gpkg"
    fields = gpd.read_file(path) if path.exists() else fr.load_fields(aoi_id)
    pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(fields.crs).iloc[0]
    hit = fields[fields.contains(pt)]
    if hit.empty:
        return None, None
    f = hit.iloc[[int(np.argmin(hit.to_crs("EPSG:32646").geometry.area.to_numpy()))]].to_crs("EPSG:32646")  # finest tracing
    g = pr.locate(aoi_id, 0, season_key="monsoon2026")["grid"]
    mask = rasterize([(f.geometry.iloc[0], 1)], out_shape=(int(g["height"]), int(g["width"])),
                     transform=from_origin(g["x0"], g["y0"], g["res"], g["res"]), fill=0, dtype="uint8",
                     all_touched=False)
    pids = np.flatnonzero(mask.ravel())
    return f, pids


#: The series the sheets read. None (default): each AOI's own pinned series (``ndvi_5day.analysis_series_root``, the
#: same one the rule reads, so notebook 08 shows what the map was made from; 2 Oct: newer images live in a new series
#: for some AOIs only). Set a path to force one series for every AOI. The fresh-start step outputs (``vegetation_types``
#: step 1, ``sowing_fresh`` step 2) are read from ``FRESH``.
SERIES_ROOT = None
FRESH = "processed/_batch/s2_2026/rice_fresh"


def series_root(aoi_id: int) -> str:
    """The series notebook 08 reads for this AOI: ``SERIES_ROOT`` when set, else the AOI's pinned series."""
    from .analysis import ndvi_5day as nd

    return SERIES_ROOT or nd.analysis_series_root(aoi_id)


def inputs_used(aoi_id: int) -> str:
    """One line naming the inputs behind the curves and the rule: series folder, its newest date, radar run."""
    from .analysis import ndvi_5day as nd
    from .analysis import pixel_report as pr

    root = series_root(aoi_id)
    dates = nd.series_dates(root, f"aoi{aoi_id}")
    run = Path(pr.locate(aoi_id, 0, season_key="monsoon2026")["run"]).name
    return f"series {Path(root).name}" + (f" (S2 to {dates[-1]})" if dates else "") + f", radar run {run}"


def fresh_at(aoi_id: int, pids, fresh_root: str = FRESH) -> dict:
    """The fresh-start results for these pixels (empty when the AOI has none yet): step-1 cover shares, step-2 sowing
    period shares and the median sowing date (``sowing_date``, a Timestamp or NaT)."""
    import numpy as np
    import pandas as pd
    import rasterio

    from .analysis import sowing_fresh as sf
    from .analysis import vegetation_types as vt

    stem = Path(fresh_root) / f"aoi{aoi_id}" / f"aoi{aoi_id}_step"
    out = {}
    pids = np.asarray(pids, dtype=int)
    if Path(f"{stem}1_cover.tif").exists():
        with rasterio.open(f"{stem}1_cover.tif") as ds:
            c = ds.read(1).ravel()[pids]
        out["step1_cover"] = {vt.CLASSES[int(k)]: f"{100 * n / len(c):.0f} %" for k, n in zip(*np.unique(c, return_counts=True))}
    if Path(f"{stem}2_sowing.tif").exists():
        with rasterio.open(f"{stem}2_period.tif") as ds:
            p = ds.read(1).ravel()[pids]
        with rasterio.open(f"{stem}2_sowing.tif") as ds:
            doy = ds.read(1).ravel()[pids].astype(float)
        out["step2_sowing_period"] = {sf.PERIODS.get(int(k), "not crop ground"): f"{100 * n / len(p):.0f} %"
                                      for k, n in zip(*np.unique(p, return_counts=True))}
        doy = doy[doy > 0]
        out["sowing_date"] = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=float(np.median(doy)) - 1)).normalize() \
            if len(doy) else pd.NaT
    return out


def classes_at(aoi_id: int, pid: int) -> dict:
    """The class of the pixel in every map version (``VERSIONS``)."""
    import rasterio

    out = {}
    for name, suffix in VERSIONS:
        p = version_path(f"aoi{aoi_id}", suffix)
        if p.exists():
            with rasterio.open(p) as ds:
                v = int(ds.read(1).ravel()[pid])
            out[name] = f"{v} {COLOURS.get(v, ('no data', ''))[0]}"
    return out


def rule_at(aoi_id: int, pids) -> dict:
    """The CURRENT relative rule (``analysis.curve_rules``) computed live for these pixels, so the notebook always shows
    what the rules say now, not an older map (user, 1 Oct). Returns the class of every pixel (shares for a block or a
    field), the median of each measured feature, and the user's own label when the pixel has one
    (``analysis.curve_labels``)."""
    import numpy as np

    from .analysis import curve_labels as cl
    from .analysis import curve_rules as cr

    pids = np.asarray(pids, dtype=int)
    with cr.rules_for(aoi_id):                 # the AOI's own rules (curve_rules.AOI_OVERRIDES), as on its map
        f = cr.own_range_features(aoi_id, pids, series_root(aoi_id))
        cls = cr.classify_relative(f)
    shares = (cls.value_counts(normalize=True) * 100).round(0)
    feats = {c: round(float(f[c].median()), 2) for c in f.columns
             if c not in ("pixel", "sowing_day") and f[c].notna().any()}
    import pandas as pd

    sd = f["sowing_day"].where(f["sowing_day"] >= 0) if "sowing_day" in f else pd.Series(dtype=float)
    out = {"rule_class": shares.index[0], "rule_shares": {k: f"{v:.0f} %" for k, v in shares.items()},
           "rule_features": feats,
           "inputs": inputs_used(aoi_id),
           "rule_set": ("aoi160 rules" + (" + this AOI's own: " + ", ".join(f"{k}={v}" for k, v in
                                                                         cr.AOI_OVERRIDES[aoi_id].items())
                                          if cr.AOI_OVERRIDES.get(aoi_id) else "")),
           "rule_sowing": (pd.Timestamp("1970-01-01") + pd.Timedelta(days=float(sd.median()))).normalize()
           if sd.notna().any() else pd.NaT,
           "rule_sowing_from": SOWING_SOURCES.get(int(f["sowing_from"].mode().iloc[0]), "")
           if "sowing_from" in f and f["sowing_from"].notna().any() else ""}
    lab = cl.load()
    lab = lab[(lab["aoi"] == aoi_id) & lab["pixel"].isin(pids)]
    if len(lab):
        out["your_labels"] = {int(r.pixel): " / ".join(x for x in (r.label, r.state, r.establishment) if x)
                              for r in lab.itertuples()}
    return out


#: Short names of the rule features shown on the curve plot, in reading order.
RULE_FEATURE_NAMES = {
    "vh_pos_end": "VH now in own range (0 low, 1 high)", "vv_pos_end": "VV now in own range",
    "vh_slope_end": "VH change, last 3 passes", "vv_slope_end": "VV change, last 3 passes",
    "vh_accel_end": "VH acceleration", "ndvi_low_peak": "NDVI low / peak (tree if high)",
    "ndvi_left": "NDVI share of own rise left", "ndvi_slope_end": "NDVI change, last view",
    "ndvi_rise_days": "days 10 -> 90 % green-up (fit)", "ndvi_rise_seen": "days 10 -> 90 % (clear views)",
    "days_since_peak": "days since NDVI peak", "crop_age": "crop age, days since last at low", "days_at_top": "days held at top", "days_off_top": "days since last at top",
    "vh_rise_recent": "VH rise since recent low", "vv_rise_recent": "VV rise since recent low",
    "vv_step_end": "VV change, last pass", "water_spell": "water while crop small (1 = yes)",
    "sowing_from": "sowing date from (0 NDVI, 1 radar water end, 2 radar rise, 3 radar water start)"}


#: Where the rule's sowing date came from (``curve_rules.own_range_features`` column ``sowing_from``), for the plot.
SOWING_SOURCES = {0: "NDVI: end of the empty spell", 1: "radar: end of the water spell (transplanting)",
                  2: "radar: VH leaves its low (no clear view)", 3: "radar: start of the water spell"}


def _sowing_label(rule: dict) -> str:
    """The text on the sowing line: where the date came from (user, 2 Oct: when the optical cannot see the sowing the
    radar gives the date, and the plot must say so)."""
    src = rule.get("rule_sowing_from")
    return f"sowing ({src})" if src else "sowing (last empty spell)"


def _sowing(rule: dict, fresh: dict):
    """The sowing line on the plot: the current rule's own date (water spell start for a transplanted crop, end of the
    empty spell otherwise), else step 2's (user, 2 Oct: the plot showed step 2's June trough for aoi28 pixel 4227)."""
    import pandas as pd

    d = rule.get("rule_sowing")
    return d if d is not None and not pd.isna(d) else fresh.get("sowing_date")


def annotate_rule(fig, rule: dict) -> None:
    """Writes the rule's class (and the user's label) in the curve figure's title and the features in a box."""
    if fig is None or not rule:
        return
    head = f"rule now: {rule['rule_class']}"
    if len(rule.get("rule_shares", {})) > 1:
        head += "  (" + ", ".join(f"{k} {v}" for k, v in rule["rule_shares"].items()) + ")"
    if rule.get("your_labels"):
        head += "   |   your label: " + "; ".join(rule["your_labels"].values())
    if rule.get("rule_set"):
        head += f"\nrules used: {rule['rule_set']}"
    if rule.get("inputs"):
        head += f"\ninputs: {rule['inputs']}"
    # Wrapped title and the feature box INSIDE the figure (user, 5 Oct: "the curves plot is too small"): a long
    # "rules used" line (an AOI with many own switches) or a box drawn outside the axes widened the saved figure, and
    # the notebook shrank the whole figure, curves included, to its cell width.
    import textwrap

    head = "\n".join(textwrap.fill(h, 170, subsequent_indent="    ") for h in head.split("\n"))
    n_head = head.count("\n") + 1
    fig.set_size_inches(20, 10 + 0.25 * n_head)
    fig.subplots_adjust(left=0.05, right=0.80, bottom=0.06, top=1 - (0.35 + 0.25 * n_head) / (10 + 0.25 * n_head),
                        hspace=0.12)
    fig.suptitle(head, fontsize=11, fontweight="bold", x=0.01, ha="left", y=0.995)
    lines = [f"{RULE_FEATURE_NAMES[k]}: {rule['rule_features'][k]}" for k in RULE_FEATURE_NAMES
             if k in rule.get("rule_features", {})]
    ax = fig.axes[0]
    ax.text(1.01, 1.0, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            family="monospace", bbox=dict(boxstyle="round", fc="white", ec="#999999", alpha=0.9))


def inspect(lon: float, lat: float, block: int = 3, out_dir=f"{OUT}/inspect") -> dict:
    """Curves (NDVI with clear/hazy observations, VV/VH per track) and every clear 5-3-2 chip for the
    pixel block at the point and, if the point is in a delineated field, for the whole field (outline
    drawn on the chips). PNGs and a text summary go to ``out_dir``; the summary is also returned."""
    import numpy as np

    from .analysis import ndvi_5day as nd
    from .analysis import pixel_2026 as p26
    from .analysis import water_investigation as wi

    hit = locate(lon, lat)
    if hit is None:
        return {"error": "the point is not inside any AOI"}
    aoi_id, pid, row, col = hit
    stem = f"aoi{aoi_id}_{lat:.5f}_{lon:.5f}"
    info = {"aoi": f"aoi{aoi_id}", "pixel_id": pid, "row": row, "col": col, "lon": lon, "lat": lat,
            "classes_at_pixel": classes_at(aoi_id, pid)}
    figs = {}
    figs["pixels_curve"], figs["pixels_chips"], _ = wi.group_sheet(
        aoi_id, pid, block=block, out_dir=out_dir, file_stem=f"{stem}_pixels{block}x{block}",
        label=f"({block}x{block} pixels at the point)")
    f, fpids = field_at(aoi_id, lon, lat)
    if f is not None and len(fpids):
        d = nd.load(aoi_id)
        g = d["loc"]["grid"]
        rr, cc = np.divmod(fpids, int(g["width"]))
        centre = int(np.round(np.median(rr)) * int(g["width"]) + np.round(np.median(cc)))
        r0, c0 = divmod(centre, int(g["width"]))
        xs, ys = f.geometry.iloc[0].exterior.xy if f.geometry.iloc[0].geom_type == "Polygon" else \
            max(f.geometry.iloc[0].geoms, key=lambda q: q.area).exterior.xy
        ox = (np.asarray(xs) - g["x0"]) / g["res"] - 0.5 - (c0 - p26.HALF)
        oy = (g["y0"] - np.asarray(ys)) / g["res"] - 0.5 - (r0 - p26.HALF)
        row_ = f.iloc[0]
        info["field"] = {k: (row_[k].item() if hasattr(row_[k], "item") else row_[k])
                         for k in ("uid", "area_acres", "pixels", "label", "class_name", "rice_share",
                                   "unconfirmed_share", "label_confidence") if k in f.columns}
        figs["field_curve"], figs["field_chips"], _ = wi.group_sheet(
            aoi_id, centre, block=block, out_dir=out_dir, file_stem=f"{stem}_field",
            label=f"(whole field, {len(fpids)} px)", pids=fpids, outline=(ox, oy))
    else:
        info["field"] = None
    nd.forget()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}_info.txt").write_text("\n".join(f"{k}: {v}" for k, v in info.items()))
    info["figures"] = figs        # matplotlib figures, for display in a notebook
    return info


def field_by_id(field_id: str):
    """(aoi_id, field row in EPSG:32646, pixel ids) for a ``field_id`` such as ``aoi116_000123``."""
    import geopandas as gpd
    import numpy as np
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    from .analysis import pixel_report as pr

    aoi_id = int(field_id.split("_")[0][3:])
    path = Path(SRC) / "fields" / f"aoi{aoi_id}_fields_monsoon2026.gpkg"
    if field_id.split("_", 1)[1].startswith("p"):
        # a field of the delivered (locked) field layer, id aoi<N>_p<polygon> (analysis.rice_map_delivery)
        path = Path(FRESH) / "locked" / f"aoi{aoi_id}" / f"aoi{aoi_id}_rel_fields_sliver015.gpkg"
        f = gpd.read_file(path, where=f"polygon_id = '{field_id.split('_', 1)[1]}'")
        if not f.empty:
            f = f.assign(field_id=field_id)
    else:
        f = gpd.read_file(path, where=f"field_id = '{field_id}'")
    if f.empty:
        raise ValueError(f"{field_id} not found in {path}")
    f = f.to_crs("EPSG:32646")
    g = pr.locate(aoi_id, 0, season_key="monsoon2026")["grid"]
    mask = rasterize([(f.geometry.iloc[0], 1)], out_shape=(int(g["height"]), int(g["width"])),
                     transform=from_origin(g["x0"], g["y0"], g["res"], g["res"]), fill=0, dtype="uint8")
    return aoi_id, f, np.flatnonzero(mask.ravel())


def spread_sample(fields, labels, per_label: int = 10, min_acres: float = 0.5) -> list[str]:
    """``field_id`` list: for each label, ``per_label`` fields spread over the AOI in all directions.

    Why: checking a few random fields misses whole corners of an AOI. The AOI extent is cut into a 3 x 3
    grid (north-west ... south-east, plus the centre); in each cell the field closest to the cell centre is
    taken (only fields of at least ``min_acres`` that are real fields, so slivers are not picked), and the
    largest remaining field fills up to ``per_label``. The result is deterministic: the same file gives the
    same list. ``fields`` is a GeoDataFrame in a metric CRS with ``field_id``, ``label``, ``area_acres``.
    """
    import numpy as np

    f = fields[fields["area_acres"] >= min_acres]
    if "is_field" in f:
        f = f[f["is_field"].astype(bool)]
    x0, y0, x1, y1 = fields.total_bounds
    cx, cy = f.geometry.centroid.x.to_numpy(), f.geometry.centroid.y.to_numpy()
    f = f.assign(_cx=cx, _cy=cy)
    centres = [(x0 + (i + 0.5) * (x1 - x0) / 3, y0 + (j + 0.5) * (y1 - y0) / 3) for j in (2, 1, 0) for i in (0, 1, 2)]
    out: list[str] = []
    for lab in labels:
        g = f[f["label"] == lab]
        chosen: list[str] = []
        for (px, py) in centres:
            rest = g[~g["field_id"].isin(chosen)]
            if len(chosen) >= per_label or rest.empty:
                continue
            d = np.hypot(rest["_cx"] - px, rest["_cy"] - py)
            chosen.append(rest.loc[d.idxmin(), "field_id"])
        rest = g[~g["field_id"].isin(chosen)].sort_values("area_acres", ascending=False)
        chosen += rest["field_id"].tolist()[: max(per_label - len(chosen), 0)]
        out += chosen
    return out


def inspect_pixel(aoi_id: int, pid: int, block: int = 3, out_dir=f"{OUT}/inspect", keep_cache: bool = True) -> dict:
    """Curves and chips for a pixel block, by AOI and pixel id (the pixel id is the value of the AOI's
    ``grid/pixel_index.tif`` in QGIS)."""
    from .analysis import ndvi_5day as nd
    from .analysis import water_investigation as wi

    info = {"aoi": f"aoi{aoi_id}", "pixel_id": pid, "classes_at_pixel": classes_at(aoi_id, pid)}
    g = nd.load(aoi_id, out_root=series_root(aoi_id))["loc"]["grid"]
    block_pids = wi.block_pids(int(pid), int(g["width"]), int(g["height"]), block)
    fresh = fresh_at(aoi_id, block_pids)
    info.update({k: v for k, v in fresh.items() if k != "sowing_date"})
    rule = rule_at(aoi_id, block_pids)
    info.update(rule)
    figs = {}
    figs["curve"], figs["chips"], ev = wi.group_sheet(aoi_id, int(pid), block=block, out_dir=out_dir,
                                                      file_stem=f"aoi{aoi_id}_pid{pid}_{block}x{block}",
                                                      label=f"({block}x{block} pixels)", series_root=series_root(aoi_id),
                                                      sowing=_sowing(rule, fresh),
                                                      sowing_label=_sowing_label(rule))
    annotate_rule(figs["curve"], rule)
    if "sowing_date" in fresh:
        info["sowing_date"] = str(ev.get("sowing_date", ""))
    else:                                   # no fresh-start output for this AOI yet: the old rule's events
        info["old_rule_events"] = {k: str(ev[k]) for k in ("trough_date", "climb_date", "trough_ndvi", "peak_after",
                                                           "last_ndvi")}
    if not keep_cache:
        nd.forget()
    info["figures"] = figs
    return info


def inspect_field(field_id: str, out_dir=f"{OUT}/inspect", keep_cache: bool = True) -> dict:
    """Curves (averaged over all the field's pixels) and chips with the field outline, by ``field_id``.
    ``keep_cache`` (default): the AOI's series stays in memory, so the next field of the same AOI plots in seconds
    (``ndvi_5day.CACHE_AOIS`` bounds it to one AOI)."""
    import numpy as np

    from .analysis import ndvi_5day as nd
    from .analysis import pixel_2026 as p26
    from .analysis import water_investigation as wi

    aoi_id, f, fpids = field_by_id(field_id)
    row = f.iloc[0]
    info = {"field_id": field_id, "aoi": f"aoi{aoi_id}", "pixels": int(len(fpids)),
            **{k: (row[k].item() if hasattr(row[k], "item") else row[k])
               for k in ("area_acres", "label", "class_name", "rice_share", "unconfirmed_share",
                         "label_confidence", "field_rule_label") if k in f.columns}}
    if not len(fpids):
        info["note"] = "the field is smaller than one pixel: no curve of its own"
        return info
    d = nd.load(aoi_id, out_root=series_root(aoi_id))
    g = d["loc"]["grid"]
    rr, cc = np.divmod(fpids, int(g["width"]))
    centre = int(np.round(np.median(rr)) * int(g["width"]) + np.round(np.median(cc)))
    r0, c0 = divmod(centre, int(g["width"]))
    geom = row.geometry if row.geometry.geom_type == "Polygon" else max(row.geometry.geoms, key=lambda q: q.area)
    xs, ys = geom.exterior.xy
    ox = (np.asarray(xs) - g["x0"]) / g["res"] - 0.5 - (c0 - p26.HALF)
    oy = (g["y0"] - np.asarray(ys)) / g["res"] - 0.5 - (r0 - p26.HALF)
    info["classes_at_centre_pixel"] = classes_at(aoi_id, centre)
    fresh = fresh_at(aoi_id, fpids)
    info.update({k: v for k, v in fresh.items() if k != "sowing_date"})
    rule = rule_at(aoi_id, fpids)
    info.update(rule)
    figs = {}
    figs["curve"], figs["chips"], ev = wi.group_sheet(aoi_id, centre, out_dir=out_dir, file_stem=field_id,
                                                      label=f"field {field_id} ({len(fpids)} px)",
                                                      pids=fpids, outline=(ox, oy), series_root=series_root(aoi_id),
                                                      sowing=_sowing(rule, fresh),
                                                      sowing_label=_sowing_label(rule))
    annotate_rule(figs["curve"], rule)
    if "sowing_date" in fresh:
        info["sowing_date"] = str(ev.get("sowing_date", ""))
    else:                                   # no fresh-start output for this AOI yet: the old rule's events
        info["old_rule_events"] = {k: str(ev[k]) for k in ("trough_date", "climb_date", "trough_ndvi", "peak_after",
                                                           "last_ndvi")}
    if not keep_cache:
        nd.forget()
    info["figures"] = figs
    return info


def main(argv=None) -> int:
    import argparse
    import os

    os.environ["PATH"] = "/opt/gis/bin:" + os.environ.get("PATH", "")      # GDAL tools on this machine
    p = argparse.ArgumentParser(prog="python -m sar_pipeline.qgis_review", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    b = sub.add_parser("build", help="mosaic of all versions, style, merged fields, review AOI list")
    b.add_argument("--review-ids", nargs="*", type=int, default=[])
    f = sub.add_parser("field", help="curves and chips of one field, by field_id")
    f.add_argument("field_id")
    f.add_argument("--repeat", type=int, default=1, help="run it this many times and print the seconds of each "
                   "(the second run shows the speed of the next field in a notebook session)")
    x = sub.add_parser("pixel", help="curves and chips of a pixel block, by AOI and pixel id")
    x.add_argument("--aoi", type=int, required=True)
    x.add_argument("--pid", type=int, required=True)
    x.add_argument("--block", type=int, default=3)
    sm = sub.add_parser("sample", help="field_ids spread over an AOI in all directions, per label")
    sm.add_argument("--aoi", type=int, required=True)
    sm.add_argument("--labels", nargs="+", type=int, default=[1, 6])
    sm.add_argument("--per-label", type=int, default=10)
    i = sub.add_parser("inspect", help="curves and chips at a lon/lat point (pixel block and its field)")
    i.add_argument("--lon", type=float, required=True)
    i.add_argument("--lat", type=float, required=True)
    i.add_argument("--block", type=int, default=3, help="pixel block size (1, 3 or 5)")
    args = p.parse_args(argv)
    if args.step == "build":
        import yaml

        mosaic(), qml(), merge_fields()
        cfg = yaml.safe_load(next(Path("config").glob("aoi*_monsoon2026.yaml")).read_text())
        print(review_aois(args.review_ids, bucket_folder=f"{cfg['gcs']['bucket']}/{cfg['gcs']['base_folder']}"))
    elif args.step == "sample":
        import geopandas as gpd

        fl = gpd.read_file(Path(SRC) / "fields" / f"aoi{args.aoi}_fields_monsoon2026.gpkg").to_crs("EPSG:32646")
        print("\n".join(spread_sample(fl, args.labels, args.per_label)))
    elif args.step in ("field", "pixel"):
        import time

        for i in range(getattr(args, "repeat", 1)):
            t0 = time.perf_counter()
            r = inspect_field(args.field_id) if args.step == "field" else inspect_pixel(args.aoi, args.pid, args.block)
            if getattr(args, "repeat", 1) > 1:
                print(f"run {i + 1}: {time.perf_counter() - t0:.1f} s")
        for k, v in r.items():
            if k != "figures":
                print(f"{k}: {v}")
        print(f"pictures in {OUT}/inspect/")
    else:
        for k, v in inspect(args.lon, args.lat, args.block).items():
            if k != "figures":
                print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
