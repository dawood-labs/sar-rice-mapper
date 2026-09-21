"""Ground-truth fields: loading, class codes, vector QC and the QC-labelled copy.

The original ground-truth file is never modified. QC decisions go into a copy next to it
(`<stem>_qc.gpkg`) with three extra columns:
  qc_status        ok | check_label | mixed_pixels | note_water | note_builtup
  qc_reason        plain-English reason
  use_in_analysis  False for check_label / mixed_pixels (kept in the file, excluded from analysis)
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from ..config import aoi_path, root
from ..run_meta import atomic_write_csv, replace_with_retry

log = logging.getLogger(__name__)

EXCLUDED_STATUSES = ("check_label", "mixed_pixels")


def gcps_cfg(cfg: dict) -> dict:
    g = cfg.get("gcps") or {}
    missing = [k for k in ("path", "class_field", "id_field") if not g.get(k)]
    if missing:
        raise ValueError(f"config gcps section is missing {missing}")
    return g


def gcps_path(cfg: dict) -> Path:
    return root(cfg) / gcps_cfg(cfg)["path"]


def qc_copy_path(cfg: dict) -> Path:
    p = gcps_path(cfg)
    return p.with_name(p.stem + "_qc.gpkg")


def load_gcps(cfg: dict) -> gpd.GeoDataFrame:
    """Original ground truth with a stable `gcp_id` (row order) and a `label` column (class name)."""
    g = gpd.read_file(gcps_path(cfg))
    g["gcp_id"] = np.arange(len(g))
    g["label"] = g[gcps_cfg(cfg)["class_field"]].astype(str)
    return g


def class_codes(gdf: pd.DataFrame, class_field: str, id_field: str) -> dict[str, int]:
    """{class name: numeric code} read from the data; each class must have exactly one code and vice versa."""
    pairs = gdf[[class_field, id_field]].drop_duplicates()
    if pairs[class_field].duplicated().any() or pairs[id_field].duplicated().any():
        raise ValueError(f"{class_field} and {id_field} are not one-to-one:\n{pairs.to_string(index=False)}")
    return {str(c): int(i) for c, i in zip(pairs[class_field], pairs[id_field])}


def codes_from_config(cfg: dict) -> dict[str, int]:
    g = gcps_cfg(cfg)
    return class_codes(gpd.read_file(gcps_path(cfg), ignore_geometry=True), g["class_field"], g["id_field"])


def load_fields(cfg: dict, crs: str) -> gpd.GeoDataFrame:
    """Fields for analysis in `crs`: the QC copy's use_in_analysis rows, or every field when no copy exists yet."""
    qc = qc_copy_path(cfg)
    if qc.exists():
        g = gpd.read_file(qc)
        g["label"] = g[gcps_cfg(cfg)["class_field"]].astype(str)
    else:
        log.warning("no QC copy at %s: using every ground-truth field", qc)
        g = load_gcps(cfg)
        g["qc_status"], g["qc_reason"], g["use_in_analysis"] = "ok", "", True
    n_all = len(g)
    g = g[g["use_in_analysis"].astype(bool)].to_crs(crs)
    log.info("fields used: %d of %d (QC excluded %d)", len(g), n_all, n_all - len(g))
    return g


# ---------------------------------------------------------------- vector QC
def vector_qc(cfg: dict, crs: str, res: float, border_m: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-field geometry checks and the list of overlapping field pairs."""
    g = load_gcps(cfg).to_crs(crs)
    aoi = gpd.read_file(aoi_path(cfg)).to_crs(crs).union_all()
    qc = pd.DataFrame({"gcp_id": g.gcp_id, "label": g.label})
    qc["is_valid"] = g.geometry.is_valid.values
    qc["is_empty"] = g.geometry.is_empty.values
    qc["n_parts"] = g.geometry.apply(lambda x: len(getattr(x, "geoms", [x]))).values
    qc["area_ha"] = (g.area / 1e4).round(3).values
    qc["n_pixels"] = (g.area / res**2).round(1).values
    qc["n_pure_pixels"] = (g.buffer(-border_m).area / res**2).round(1).values
    qc["compactness"] = (4 * np.pi * g.area / g.length**2).round(3).values
    qc["frac_in_aoi"] = (g.intersection(aoi).area / g.area).round(3).values
    qc["duplicate_geom"] = g.geometry.normalize().to_wkb().duplicated(keep=False).values
    tree = shapely.STRtree(g.geometry.values)
    a, b = tree.query(g.geometry.values, predicate="intersects")
    rows = []
    for i, j in zip(a, b):
        if i < j:
            inter = g.geometry.iloc[i].intersection(g.geometry.iloc[j]).area
            if inter > 1:
                rows.append(dict(gcp_a=int(i), gcp_b=int(j), label_a=g.label.iloc[i], label_b=g.label.iloc[j],
                                 overlap_m2=round(inter, 1)))
    overlaps = pd.DataFrame(rows, columns=["gcp_a", "gcp_b", "label_a", "label_b", "overlap_m2"])
    qc["overlaps_other"] = qc.gcp_id.isin(set(overlaps.gcp_a) | set(overlaps.gcp_b))
    return qc, overlaps


def write_vector_qc(cfg: dict, out_dir: Path, crs: str, res: float, border_m: float) -> Path:
    qc, overlaps = vector_qc(cfg, crs, res, border_m)
    codes_from_config(cfg)  # fails loudly if class names and codes are not one-to-one
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(qc, out_dir / "gcp_vector_qc.csv")
    atomic_write_csv(overlaps, out_dir / "gcp_overlaps.csv")
    log.info("vector QC: %d invalid, %d empty, %d overlapping pairs, %d not fully inside the AOI",
             int((~qc.is_valid).sum()), int(qc.is_empty.sum()), len(overlaps), int((qc.frac_in_aoi < 1).sum()))
    return out_dir / "gcp_vector_qc.csv"


# ---------------------------------------------------------------- QC-labelled copy
def qc_table(gcps: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    """Merge review statuses (gcp_id, qc_status, qc_reason) into all fields; unlisted fields are 'ok'."""
    out = gcps.merge(review[["gcp_id", "qc_status", "qc_reason"]], on="gcp_id", how="left")
    out["qc_status"] = out["qc_status"].fillna("ok")
    out["qc_reason"] = out["qc_reason"].fillna("")
    out["use_in_analysis"] = ~out["qc_status"].isin(EXCLUDED_STATUSES)
    return out


def write_qc_copy(cfg: dict, review_csv: Path, force: bool = False) -> Path:
    out = qc_copy_path(cfg)
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; pass --force to rebuild it from {review_csv}")
    g = qc_table(load_gcps(cfg), pd.read_csv(review_csv))
    g = g.drop(columns=[c for c in ("label",) if c in g.columns])
    tmp = out.with_name(f".{out.stem}.tmp.gpkg")
    tmp.unlink(missing_ok=True)
    g.to_file(tmp, driver="GPKG", layer="gcps_qc")
    replace_with_retry(tmp, out)
    log.info("QC copy written: %s (%s)", out, g.qc_status.value_counts().to_dict())
    return out
