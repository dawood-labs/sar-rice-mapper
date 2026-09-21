"""QGIS layers: spatial-CV results on the fields, and a class map of a run's AOI.

The class map is EXPLORATORY: the model is trained on all fields and applied to every AOI pixel, including
boundaries between different crops and places without ground truth, which the CV never tested.

Two rasters are written because GeoTIFF has one nodata value per file:
  rf_<set>_classes.tif      class code (codes from the ground truth), nodata 0
  rf_<set>_<target>_prob.tif  target-class probability in percent, nodata 255 (0 % stays a valid value)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

from .. import config as config_mod
from .. import manifest, run_meta
from . import gcp_qc
from .evaluate_cv import majority, make_model
from .pixel_features import (OK_STATES, FeatureSet, analysis_dir, columns_for_set, plans_for, read_chunk_array,
                             run_context, track_features, wet_flags)

log = logging.getLogger(__name__)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


def write_qml(path: Path, codes: dict[str, int]) -> None:
    entries = "\n".join(f'        <paletteEntry value="{code}" color="{PALETTE[i % len(PALETTE)]}" alpha="255" label="{name}"/>'
                        for i, (name, code) in enumerate(sorted(codes.items(), key=lambda kv: kv[1])))
    run_meta.atomic_write_text(path, f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
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


def write_class_rasters(cls_path: Path, prob_path: Path, cls: np.ndarray, prob: np.ndarray, crs, transform) -> None:
    """Class codes (uint8, nodata 0) and probability percent (uint8, nodata 255) as two files."""
    base = dict(driver="GTiff", width=cls.shape[1], height=cls.shape[0], count=1, dtype="uint8", crs=crs,
                transform=transform, tiled=True, blockxsize=512, blockysize=512, compress="deflate")
    with rasterio.open(cls_path, "w", nodata=0, **base) as dst:
        dst.write(cls.astype("uint8"), 1)
    with rasterio.open(prob_path, "w", nodata=255, **base) as dst:
        dst.write(prob.astype("uint8"), 1)


def field_layer(oof: pd.DataFrame, target: str) -> pd.DataFrame:
    """One row per field from out-of-fold pixel predictions (columns gcp_id, label, pred)."""
    oof = oof.assign(hit=oof["pred"] == oof["label"], is_target=oof["pred"] == target)
    f = oof.groupby("gcp_id").agg(true_class=("label", "first"), pred_class=("pred", majority), n_pixels=("pred", "size"),
                                  pixel_accuracy=("hit", "mean"), target_share=("is_target", "mean")).reset_index()
    f["correct"] = f.true_class == f.pred_class
    f["error_type"] = np.where(f.correct, "correct", f.true_class + " -> " + f.pred_class)
    return f.rename(columns={"target_share": f"{slug(target)}_share"})


def cv_layers(cfg: dict, run_dir: Path, set_name: str) -> Path:
    """Fields GPKG + sparse pixel GeoTIFF from `oof_<set>.parquet` (run `evaluate --save-oof` first)."""
    target = cfg["analysis"]["target_class"]
    out_dir = analysis_dir(run_dir)
    oof = pd.read_parquet(out_dir / f"oof_{set_name}.parquet")
    codes = gcp_qc.codes_from_config(cfg)
    _, gd, *_ = run_context(run_dir)
    fields = gpd.read_file(gcp_qc.qc_copy_path(cfg)) if gcp_qc.qc_copy_path(cfg).exists() else gcp_qc.load_gcps(cfg)
    layer = fields[["gcp_id", "geometry"]].merge(field_layer(oof, target), on="gcp_id", how="inner")
    gpkg = out_dir / f"cv_fields_{set_name}.gpkg"
    layer.to_file(gpkg, driver="GPKG", layer="cv_fields")
    rows, cols = np.divmod(oof.pid.to_numpy(np.int64), gd["width"])
    cls = np.zeros((gd["height"], gd["width"]), dtype="uint8")
    ok = np.zeros_like(cls)
    cls[rows, cols] = oof.pred.map(codes).to_numpy(dtype="uint8")
    ok[rows, cols] = np.where(oof.pred == oof.label, 2, 1)
    profile = dict(driver="GTiff", width=gd["width"], height=gd["height"], count=2, dtype="uint8", crs=gd["crs"],
                   transform=Affine(*gd["transform"]), nodata=0, tiled=True, blockxsize=512, blockysize=512, compress="deflate")
    tif = out_dir / f"cv_pixels_{set_name}.tif"
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(cls, 1); dst.write(ok, 2)
        dst.set_band_description(1, "predicted class code"); dst.set_band_description(2, "1 = wrong, 2 = correct")
    write_qml(tif.with_suffix(".qml"), codes)
    log.info("CV layers: %d fields (%d correct) -> %s", len(layer), int(layer.correct.sum()), gpkg)
    return gpkg


def check_same_layout(train_run: Path, map_run: Path) -> None:
    key = ["track_id", "band_idx", "acquisition_id", "pol"]
    a = run_meta.read_band_layout(train_run)[key].reset_index(drop=True)
    b = run_meta.read_band_layout(map_run)[key].reset_index(drop=True)
    if not a.equals(b):
        raise ValueError(f"band layouts of {train_run.name} and {map_run.name} differ: features would not be comparable")


def predict_aoi(cfg: dict, train_run: Path, fs: FeatureSet, cols: list[str], map_run: Path, model,
                rules: dict) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """Apply a fitted model to every AOI pixel of `map_run`.

    `rules` = {name: function(probabilities) -> class index per row (index into model.classes_)}.
    Returns ({rule: class-code raster}, target probability raster in percent, grid).

    Rain flags come from the TRAINING run: rain is averaged over each run's AOI, so a different AOI can flag
    different dates as wet, which would make bins pick different acquisitions than the model was trained on.
    """
    t0 = time.time()
    a = cfg["analysis"]
    classes = list(model.classes_)
    codes = gcp_qc.codes_from_config(cfg)
    code_of = np.array([codes[c] for c in classes], dtype="uint8")
    t_idx = classes.index(a["target_class"])
    check_same_layout(train_run, map_run)
    _, gd, tracks, season_start, end = run_context(map_run)
    plans = {tr: plans_for(map_run, tr, [fs], wet_flags(train_run, tr, float(a.get("rain_mm_24h", 5))), season_start, end)[0]
             for tr in tracks}
    man = manifest.read_manifest(map_run)
    cls = {name: np.zeros((gd["height"], gd["width"]), dtype="uint8") for name in rules}
    prob = np.full((gd["height"], gd["width"]), 255, dtype="uint8")
    map_run_cfg = config_mod.load_run_config(map_run)
    with rasterio.open(config_mod.grid_dir(map_run_cfg) / "pixel_index.tif") as idx:
        for k, ch in enumerate(gd["chunks"], 1):
            h, w = ch["height"], ch["width"]
            rr, cc = np.indices((h, w)).reshape(2, -1)
            feats = {}
            for tr in tracks:
                row = man[(man.track_id == tr) & (man.chunk_name == ch["name"]) & man.state.isin(OK_STATES)]
                if row.empty:
                    raise ValueError(f"{tr} {ch['name']} is not verified in {map_run.name}")
                paths = [map_run / p for p in row.local_paths.iloc[0].split(";") if p]
                arr = read_chunk_array(paths, ch, gd)
                feats.update(track_features(10 ** (arr / 10), rr, cc, tr, plans[tr]))
            F = np.column_stack([feats[c] for c in cols])
            win = Window(ch["col_off"], ch["row_off"], w, h)
            good = (idx.read(3, window=win).reshape(-1) == 1) & np.isfinite(F).all(axis=1)
            if good.any():
                p = model.predict_proba(F[good])
                sl = (slice(ch["row_off"], ch["row_off"] + h), slice(ch["col_off"], ch["col_off"] + w))
                for name, rule in rules.items():
                    sub_c = np.zeros(h * w, dtype="uint8")
                    sub_c[good] = code_of[rule(p)]
                    cls[name][sl] = sub_c.reshape(h, w)
                sub_p = np.full(h * w, 255, dtype="uint8")
                sub_p[good] = np.round(100 * p[:, t_idx]).astype("uint8")
                prob[sl] = sub_p.reshape(h, w)
            log.info("  chunk %d/%d %s (%.0f s)", k, len(gd["chunks"]), ch["name"], time.time() - t0)
    return cls, prob, gd


def class_shares(cls: np.ndarray, codes: dict[str, int]) -> dict[str, float]:
    """Percent of mapped pixels per class name."""
    names = {v: k for k, v in codes.items()}
    vals, counts = np.unique(cls[cls > 0], return_counts=True)
    return {names[int(v)]: round(float(100 * n / counts.sum()), 1) for v, n in zip(vals, counts)}


def class_map(cfg: dict, train_run: Path, set_name: str, map_run: Path, features_name: str = "features") -> Path:
    """Train the fixed RF on all field pixels of `train_run`, predict every AOI pixel of `map_run` (argmax)."""
    t0 = time.time()
    target = cfg["analysis"]["target_class"]
    feats_path = analysis_dir(train_run) / f"{features_name}.parquet"
    meta = json.loads(feats_path.with_suffix(".json").read_text())
    fs = FeatureSet(**meta["sets"][set_name])
    X = pd.read_parquet(feats_path)
    cols = columns_for_set(X.columns, set_name)
    model = make_model(cfg).fit(X[cols].to_numpy(dtype="float32"), X["label"].astype(str).to_numpy())
    log.info("trained on %d pixels, %d features (%.0f s)", len(X), len(cols), time.time() - t0)
    cls, prob, gd = predict_aoi(cfg, train_run, fs, cols, map_run, model, {"argmax": lambda p: p.argmax(axis=1)})
    cls = cls["argmax"]
    codes = gcp_qc.codes_from_config(cfg)
    out_dir = analysis_dir(map_run); out_dir.mkdir(parents=True, exist_ok=True)
    cls_path = out_dir / f"rf_{set_name}_classes.tif"
    prob_path = out_dir / f"rf_{set_name}_{slug(target)}_prob.tif"
    write_class_rasters(cls_path, prob_path, cls, prob, gd["crs"], Affine(*gd["transform"]))
    write_qml(cls_path.with_suffix(".qml"), codes)
    shares = class_shares(cls, codes)
    run_meta.atomic_write_json(out_dir / f"rf_{set_name}_classes.json",
                               {"trained_on": train_run.name, "set": set_name, "class_share_percent": shares,
                                "median_target_prob_of_target_pixels": float(np.median(prob[cls == codes[target]])) if (cls == codes[target]).any() else None})
    log.info("class map -> %s | class share %% of AOI pixels: %s (%.0f s)", cls_path, shares, time.time() - t0)
    return cls_path
