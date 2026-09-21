"""Sentinel-1 analysis-ready-data (ARD) chain for one chunk and one track, built server-side in Earth Engine.

Nothing here downloads data: the functions build an `ee.Image` expression that `export.py` sends
to Earth Engine as a batch export. See docs/developer/interfaces.md §6.3.

Processing order and why
1. Border-noise mask per slice      - optional (off by default): extra swath-edge mask for old data.
2. Terrain flattening per slice     - needs the slice's own projection and look direction.
3. Mosaic the slices of one acquisition and set the default projection to the MASTER GRID, so
   neighbourhood filters are measured in 10 m grid pixels and every chunk sees the same pixels.
   Reprojection from the slice grid to the master grid is nearest neighbour (Earth Engine default).
4. Speckle filter in LINEAR power   - mono-temporal filter, optionally multi-temporal (Quegan).
5. Convert to dB and encode (float32 or scaled int16) with an explicit nodata value.

The mosaic is never clipped to the chunk: Earth Engine computes filter neighbourhoods beyond the
export region, so filtered values at a chunk edge are identical to those in the adjacent chunk.

Adapted in parts from gee_s1_ard (https://github.com/adugnag/gee_s1_ard, MIT License,
(c) 2021 Adugna Mullissa; see LICENSE_gee_s1_ard.txt).
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping

import ee
import pandas as pd

from ..errors import PipelineError
from . import border_noise, speckle, terrain_flattening
from .helper import lin_to_db, ordered_pols
from .multitemporal import quegan_series, temporal_neighbors

__all__ = [
    "ARDConfigError", "validate_config", "load_slices", "preprocess_slice", "mosaic_acquisition",
    "filter_series", "to_output", "build_chunk_image", "temporal_neighbors", "band_names",
]

DTYPES = ("float32", "int16")
OUTPUT_FORMATS = ("DB",)


class ARDConfigError(PipelineError, ValueError):
    """Invalid `ard` / `export` / `s1` configuration for the ARD chain."""


# ---------------------------------------------------------------- config validation
def validate_config(cfg: dict) -> None:
    """Raise ARDConfigError with a clear message when an ARD-related setting is invalid."""
    ard = cfg.get("ard") or {}
    exp = cfg.get("export") or {}
    s1 = cfg.get("s1") or {}
    problems = []

    try:
        ordered_pols(s1.get("pols", []))
    except ValueError as e:
        problems.append(str(e))
    if not s1.get("collection"):
        problems.append("s1.collection is required (e.g. COPERNICUS/S1_GRD_FLOAT)")
    elif s1["collection"] != "COPERNICUS/S1_GRD_FLOAT":
        problems.append("s1.collection must be COPERNICUS/S1_GRD_FLOAT (linear power); "
                        f"got {s1['collection']!r} — filtering in dB is wrong")

    for key in ("border_noise_correction", "terrain_flattening", "multitemporal"):
        if not isinstance(ard.get(key), bool):
            problems.append(f"ard.{key} must be true or false, got {ard.get(key)!r}")
    if ard.get("terrain_flattening"):
        if ard.get("terrain_flattening_model") not in terrain_flattening.MODELS:
            problems.append(f"ard.terrain_flattening_model must be one of {list(terrain_flattening.MODELS)}, "
                            f"got {ard.get('terrain_flattening_model')!r}")
        if not ard.get("dem"):
            problems.append("ard.dem is required when ard.terrain_flattening is true")
        buf = ard.get("layover_shadow_buffer_m", 0)
        if not isinstance(buf, (int, float)) or isinstance(buf, bool) or buf < 0:
            problems.append(f"ard.layover_shadow_buffer_m must be a number >= 0, got {buf!r}")

    if ard.get("speckle_filter") not in speckle.FILTERS:
        problems.append(f"ard.speckle_filter must be one of {list(speckle.FILTERS)}, got {ard.get('speckle_filter')!r}")
    k = ard.get("speckle_kernel")
    if not isinstance(k, int) or isinstance(k, bool) or k < 3 or k % 2 == 0:
        problems.append(f"ard.speckle_kernel must be an odd integer >= 3, got {k!r}")
    hw = ard.get("temporal_half_window")
    if ard.get("multitemporal") and (not isinstance(hw, int) or isinstance(hw, bool) or hw < 0):
        problems.append(f"ard.temporal_half_window must be an integer >= 0, got {hw!r}")
    if ard.get("output_format") not in OUTPUT_FORMATS:
        problems.append(f"ard.output_format must be one of {list(OUTPUT_FORMATS)}, got {ard.get('output_format')!r}")

    dtype = exp.get("dtype")
    if dtype not in DTYPES:
        problems.append(f"export.dtype must be one of {list(DTYPES)}, got {dtype!r}")
    elif dtype == "int16":
        scale = exp.get("int16_scale")
        if not isinstance(scale, (int, float)) or isinstance(scale, bool) or scale <= 0:
            problems.append(f"export.int16_scale must be a positive number, got {scale!r}")
        nd = exp.get("nodata_int16")
        if nd != -32768:
            problems.append(f"export.nodata_int16 must be -32768 (valid values are clamped to ±32767), got {nd!r}")
    else:
        nd = exp.get("nodata_float32")
        if not isinstance(nd, (int, float)) or isinstance(nd, bool):
            problems.append(f"export.nodata_float32 must be a number, got {nd!r}")

    if problems:
        raise ARDConfigError("Invalid ARD configuration:\n  - " + "\n  - ".join(problems))


# ---------------------------------------------------------------- acquisitions helpers (pure python)
def _date_token(row: Mapping) -> str:
    date = row.get("date_utc")
    if not isinstance(date, str) or len(date) < 10:
        date = str(row["datetime_utc"])
    return date[:10].replace("-", "")


def _sorted_rows(acquisitions: pd.DataFrame) -> pd.DataFrame:
    if "datetime_utc" not in acquisitions.columns:
        raise ValueError("acquisitions must have a 'datetime_utc' column")
    order = pd.to_datetime(acquisitions["datetime_utc"], utc=True).argsort(kind="stable")
    return acquisitions.iloc[order].reset_index(drop=True)


def band_names(acquisitions: pd.DataFrame, pols=("VV", "VH")) -> list[str]:
    """[VV_d1, VH_d1, VV_d2, VH_d2, ...] in ascending acquisition time.

    Normally the token is the UTC date (VV_20260705). If several acquisitions share a UTC date, the
    suffix of their acquisition id is used (the audit names the second one <track>_<date>_2, so
    its bands become VV_20260705_2); without acquisition ids the order of acquisition decides.
    """
    pols = ordered_pols(pols)
    rows = _sorted_rows(acquisitions).to_dict("records")
    dates = [_date_token(r) for r in rows]
    counts = Counter(dates)
    seen: dict[str, int] = {}
    tokens = []
    for r, d in zip(rows, dates):
        if counts[d] == 1:
            tokens.append(d)
            continue
        m = re.search(rf"{d}(_\d+)?$", str(r.get("acquisition_id") or ""))
        if m:
            tokens.append(d + (m.group(1) or ""))
        else:
            n = seen.get(d, 0) + 1
            seen[d] = n
            tokens.append(d if n == 1 else f"{d}_{n}")
    if len(set(tokens)) != len(tokens):
        dupes = sorted({t for t in tokens if tokens.count(t) > 1})
        raise ValueError(f"acquisitions {dupes} are not unique; check the acquisition ids")
    return [f"{p}_{t}" for t in tokens for p in pols]


def _system_indexes(row: Mapping) -> list[str]:
    val = row["system_indexes"]
    items = val.split(";") if isinstance(val, str) else list(val)
    items = [s.strip() for s in items if s and s.strip()]
    if not items:
        raise ValueError(f"acquisition {row.get('acquisition_id')!r} has no system_indexes")
    return items


# ---------------------------------------------------------------- Earth Engine building blocks
def load_slices(cfg: dict, system_indexes: list[str]) -> ee.ImageCollection:
    """The GRD slices of one acquisition (linear power) with bands <pols> + angle."""
    pols = ordered_pols(cfg["s1"]["pols"])
    return (
        ee.ImageCollection(cfg["s1"]["collection"])
        .filter(ee.Filter.inList("system:index", list(system_indexes)))
        .select(pols + ["angle"])
    )


def preprocess_slice(img: ee.Image, cfg: dict) -> ee.Image:
    """Border-noise mask, then terrain flattening, in the slice's native projection."""
    ard = cfg["ard"]
    pols = ordered_pols(cfg["s1"]["pols"])
    img = ee.Image(img)
    if ard["border_noise_correction"]:
        img = ee.Image(border_noise.mask_border_noise(img))
    if ard["terrain_flattening"]:
        dem = terrain_flattening.load_dem(ard["dem"])
        img = terrain_flattening.flatten_slice(
            img, pols, ard["terrain_flattening_model"], dem, float(ard.get("layover_shadow_buffer_m", 0))
        )
    return img


def mosaic_acquisition(cfg: dict, gd: dict, acquisition_row: Mapping) -> ee.Image:
    """Preprocess every slice of one acquisition, mosaic, and put it on the master grid (linear, bands <pols>)."""
    pols = ordered_pols(cfg["s1"]["pols"])
    row = dict(acquisition_row)
    slices = load_slices(cfg, _system_indexes(row))
    mosaic = slices.map(lambda im: preprocess_slice(im, cfg)).select(pols).mosaic()
    return mosaic.setDefaultProjection(crs=gd["crs"], crsTransform=list(gd["transform"])).set({
        "acquisition_id": str(row.get("acquisition_id", "")),
        "date_utc": _date_token(row),
        # bounded region of the acquisition (a mosaic's own geometry is unbounded); used by LEE SIGMA
        "footprint": slices.geometry(),
    })


def filter_series(cfg: dict, images: list[ee.Image]) -> list[ee.Image]:
    """Speckle-filter a time-ordered list of linear acquisition images of ONE track."""
    ard = cfg["ard"]
    pols = ordered_pols(cfg["s1"]["pols"])
    filtered = [speckle.mono_filter(ee.Image(im), ard["speckle_filter"], int(ard["speckle_kernel"]), pols) for im in images]
    if not ard["multitemporal"]:
        return [f.updateMask(ee.Image(im).select(pols).mask()) for f, im in zip(filtered, images)]
    return quegan_series([ee.Image(im) for im in images], filtered, pols, int(ard["temporal_half_window"]))


def to_output(cfg: dict, img: ee.Image) -> ee.Image:
    """Linear -> dB -> export encoding. Masked pixels (and pixels outside the data) become nodata."""
    pols = ordered_pols(cfg["s1"]["pols"])
    exp = cfg["export"]
    db = lin_to_db(ee.Image(img), pols)
    if exp["dtype"] == "int16":
        scale = float(exp["int16_scale"])
        # mirrors helper.encode_int16_value: floor(x*scale + 0.5), clamp ±32767; -32768 = nodata
        enc = db.multiply(scale).add(0.5).floor().clamp(-32767, 32767).toInt16()
        return enc.unmask(int(exp["nodata_int16"]), False).rename(pols)
    return db.toFloat().unmask(float(exp["nodata_float32"]), False).rename(pols)


def build_chunk_image(cfg: dict, gd: dict, chunk: Mapping, track_id: str, acquisitions: pd.DataFrame) -> ee.Image:
    """Multi-band image for one chunk and one track: [VV_d1, VH_d1, VV_d2, VH_d2, ...].

    `acquisitions` is the track's full acquisition list (audit `s1_acquisitions.csv` rows). Every
    acquisition becomes two bands, and the multi-temporal filter draws its neighbours from this same
    list. The chunk only defines what gets exported (the export region); the image is not clipped.
    """
    validate_config(cfg)
    pols = ordered_pols(cfg["s1"]["pols"])
    if "track_id" in acquisitions.columns:
        other = sorted(set(acquisitions["track_id"]) - {track_id})
        if other:
            raise ValueError(f"acquisitions contain other tracks {other}; pass only rows of {track_id}")
    if acquisitions.empty:
        raise PipelineError(f"no acquisitions for track {track_id}")
    rows = _sorted_rows(acquisitions)
    names = band_names(rows, pols)

    mosaics = [mosaic_acquisition(cfg, gd, r) for r in rows.to_dict("records")]
    filtered = filter_series(cfg, mosaics)
    outputs = [to_output(cfg, f) for f in filtered]
    image = ee.Image.cat(outputs).rename(names)
    return image.set({
        "track_id": track_id,
        "chunk_name": str(chunk.get("name", "")),
        "n_acquisitions": len(rows),
    })
