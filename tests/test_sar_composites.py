"""Tests for ``sar_composites``: the radar colour composites for looking at rice in QGIS (docs/16).

All synthetic and network-free. What they protect:

* the maths (monthly means taken in linear power, the flood as the June-July minimum, NaNs ignored),
  because a mean taken in dB or a mean in place of the minimum would quietly change every colour;
* the int16 storage (dB x 100, NaN -> -32768), because QGIS styles and the preview rely on it;
* the QGIS style file (right band numbers, the dB range multiplied by 100).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from sar_pipeline import sar_composites as sc

SHAPE = (2, 3)


def _const(db):
    return np.full(SHAPE, float(db))


def _passes(vh, vv=None):
    """``{"VH": [(Timestamp, array)], "VV": [...]}`` from ``[("2026-06-10", -25.0), ...]``."""
    to = lambda lst: [(pd.Timestamp(d), a if isinstance(a, np.ndarray) else _const(a)) for d, a in lst]  # noqa: E731
    return {"VH": to(vh), "VV": to(vv or [])}


def _db_mean(*dbs):
    return 10 * np.log10(np.mean([10 ** (d / 10) for d in dbs]))


# --- dB / power helpers --------------------------------------------------------------------------

def test_power_and_db_round_trip():
    db = np.array([-30.0, -22.5, -15.0, 0.0, 3.0])
    assert np.allclose(sc._db(sc._power(db)), db)
    assert sc._power(0.0) == pytest.approx(1.0)
    assert sc._power(-10.0) == pytest.approx(0.1)
    assert sc._db(100.0) == pytest.approx(20.0)


def test_db_of_zero_power_is_minus_inf_without_warning():
    with np.errstate(all="raise"):                      # would raise if _db let the warning through
        out = sc._db(np.array([0.0, 1.0]))
    assert out[0] == -np.inf and out[1] == 0.0


def test_scaled_stores_db_times_100_and_nodata():
    out = sc._scaled(np.array([-15.234, -26.0, np.nan, -np.inf]))
    assert out.dtype == np.int16
    assert out.tolist() == [-1523, -2600, sc.NODATA, sc.NODATA]


# --- the composite maths -------------------------------------------------------------------------

def test_monthly_mean_is_taken_in_power_not_in_db():
    monthly, *_ = sc.composite_arrays(_passes(
        vh=[("2026-04-03", -10.0), ("2026-04-15", -30.0),        # April: two passes
            ("2026-06-01", -20.0), ("2026-08-01", -15.0)],
        vv=[("2026-04-03", -8.0)]), SHAPE)
    got = monthly[("VH", "2026-04")]
    assert np.allclose(got, _db_mean(-10.0, -30.0))               # about -13 dB ...
    assert not np.allclose(got, -20.0)                            # ... not the dB average -20
    assert np.allclose(monthly[("VV", "2026-04")], -8.0)          # VV kept apart from VH
    assert set(monthly) == {(p, m) for p in ("VH", "VV") for m in sc.MONTHS}


def test_month_without_pass_is_all_nan():
    monthly, *_ = sc.composite_arrays(_passes(
        vh=[("2026-04-03", -18.0), ("2026-06-01", -20.0), ("2026-08-01", -15.0)]), SHAPE)
    assert np.isnan(monthly[("VH", "2026-05")]).all()
    assert monthly[("VH", "2026-05")].shape == SHAPE
    assert np.isnan(monthly[("VV", "2026-09")]).all()


def test_rice_rgb_bands_canopy_flood_dry():
    # A paddy's story: dry in April, flooded once in June-July, canopy in August-September.
    monthly, canopy, flood, dry = sc.composite_arrays(_passes(vh=[
        ("2026-04-05", -19.0), ("2026-04-17", -21.0),
        ("2026-06-10", -20.0), ("2026-06-22", -26.0), ("2026-07-04", -23.0),
        ("2026-08-09", -16.0), ("2026-09-02", -13.0),
        ("2026-03-28", -40.0), ("2026-10-01", -40.0),              # outside every window: ignored
    ]), SHAPE)
    assert np.allclose(flood, -26.0)                               # the darkest pass, not the mean
    assert np.allclose(canopy, _db_mean(-16.0, -13.0))            # August and September together
    assert np.allclose(dry, _db_mean(-19.0, -21.0))
    assert dry is monthly[("VH", "2026-04")]


def test_nan_pixels_are_ignored_in_means_and_minimum():
    a = _const(-24.0)
    a[0, 0] = np.nan                                               # e.g. an artefact pass on one pixel
    _, canopy, flood, dry = sc.composite_arrays(_passes(vh=[
        ("2026-04-05", -18.0), ("2026-04-17", a),
        ("2026-06-10", -20.0), ("2026-07-04", a),
        ("2026-08-09", -15.0),
    ]), SHAPE)
    assert flood[0, 0] == pytest.approx(-20.0) and flood[1, 1] == pytest.approx(-24.0)
    assert dry[0, 0] == pytest.approx(-18.0)
    assert np.isfinite(canopy).all()


def test_pixels_differ_independently():
    # Two pixels, two stories: pixel (0, 0) floods in June (paddy), pixel (0, 1) never does.
    jun = _const(-17.0)
    jun[0, 0] = -27.0
    _, canopy, flood, _ = sc.composite_arrays(_passes(vh=[
        ("2026-04-05", -19.0), ("2026-06-10", jun), ("2026-08-09", -14.0)]), SHAPE)
    assert flood[0, 0] == pytest.approx(-27.0) and flood[0, 1] == pytest.approx(-17.0)
    assert np.allclose(canopy, -14.0)


# --- QGIS style ----------------------------------------------------------------------------------

def test_qml_rgb_has_band_numbers_and_range_times_100(tmp_path):
    path = sc.qml_rgb(6, 3, 1, -26, -12, tmp_path / "style.qml")
    root = ET.parse(path).getroot()
    r = root.find("pipe/rasterrenderer")
    assert r.get("type") == "multibandcolor"
    assert (r.get("redBand"), r.get("greenBand"), r.get("blueBand")) == ("6", "3", "1")
    for tag in ("red", "green", "blue"):
        ce = r.find(f"{tag}ContrastEnhancement")
        assert ce.findtext("minValue") == "-2600"
        assert ce.findtext("maxValue") == "-1200"
        assert ce.findtext("algorithm") == "StretchToMinimumMaximum"


def test_qml_rgb_other_range(tmp_path):
    path = sc.qml_rgb(1, 2, 3, -24.5, -10, tmp_path / "s.qml")
    r = ET.parse(path).getroot().find("pipe/rasterrenderer")
    assert r.find("redContrastEnhancement/minValue").text == "-2450"
    assert r.find("blueContrastEnhancement/maxValue").text == "-1000"


# --- preview PNG ---------------------------------------------------------------------------------

def test_preview_writes_png(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import rasterio
    from affine import Affine

    h, w = 8, 10
    tr = Affine(10.0, 0, 500000.0, 0, -10.0, 5500000.0)           # neutral UTM test grid
    prof = dict(driver="GTiff", width=w, height=h, crs="EPSG:32632", transform=tr)
    out = tmp_path / "sar"
    (out / "per_aoi").mkdir(parents=True)
    rgb = np.full((3, h, w), -1800, dtype="int16")
    rgb[:, 0, 0] = sc.NODATA
    with rasterio.open(out / "per_aoi" / "aoi7_sar_rice_rgb.tif", "w", count=3, dtype="int16",
                       nodata=sc.NODATA, **prof) as ds:
        ds.write(rgb)
    final = tmp_path / "processed/_batch/s2_2026/aoi7"
    final.mkdir(parents=True)
    cls = np.ones((1, h, w), dtype="uint8")
    cls[0, :, :3] = 255
    with rasterio.open(final / "aoi7_monsoon2026_final.tif", "w", count=1, dtype="uint8", nodata=255, **prof) as ds:
        ds.write(cls)
    monkeypatch.chdir(tmp_path)                                    # preview reads the final map relative to cwd
    png = sc.preview(7, out_dir=out)
    assert png == out / "preview_aoi7.png"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
