"""Live, READ-ONLY Earth Engine tests for sar_pipeline.audit.

Run with: SAR_PIPELINE_CONFIG=config/<your config>.yaml pytest -m gee tests/gee/test_audit_live.py
They only read metadata / reduce small regions; they never start tasks or write to GCS.
"""
from __future__ import annotations

import copy
import math

import pandas as pd
import pytest
import shapely

from sar_pipeline import audit
from sar_pipeline.config import season_dir

pytestmark = pytest.mark.gee


@pytest.fixture(scope="module")
def two_week_cfg(live_cfg):
    cfg = copy.deepcopy(live_cfg)
    cfg["season"]["start"] = "2026-07-01"
    cfg["season"]["end"] = "2026-07-15"
    return cfg


@pytest.fixture(scope="module")
def slices(two_week_cfg):
    return audit.list_slices(two_week_cfg)


def test_list_slices_columns_and_platforms(slices):
    assert len(slices) > 0
    assert list(slices.drop(columns="geometry").columns) == audit.SLICE_COLUMNS
    assert set(slices["platform"]) <= {"A", "C", "D"}
    assert slices["track_id"].str.match(r"^RO\d{3}_(ASC|DSC)$").all()
    assert slices.geometry.notna().all()


def test_group_and_rain_source(two_week_cfg, slices):
    aoi = audit._aoi_union_4326(two_week_cfg)
    acq = audit.group_acquisitions(slices, aoi, two_week_cfg)
    assert len(acq) > 0 and (acq["aoi_coverage_pct"] > 0).all()
    rain = audit.rain_flags(two_week_cfg, acq.iloc[:1])
    assert len(rain) == 1
    r = rain.iloc[0]
    assert r["source"] in {two_week_cfg["rain"]["primary"], two_week_cfg["rain"]["fallback"]}
    assert math.isfinite(r["rain_6h_mm"]) and math.isfinite(r["rain_24h_mm"])
    assert 0 <= r["rain_6h_mm"] <= r["rain_24h_mm"] + 1e-6
    assert r["n_images_found"] > 0 and r["rain_gap_images"] == 0


def test_incidence_angle_range(two_week_cfg, slices):
    aoi = audit._aoi_union_4326(two_week_cfg)
    acq = audit.group_acquisitions(slices, aoi, two_week_cfg)
    angles = audit.incidence_angles(two_week_cfg, acq.iloc[:2])
    assert angles.notna().all() and ((angles > 25) & (angles < 50)).all()


def test_empty_rain_window_returns_nan_not_crash(live_cfg):
    """A timestamp before the dataset exists: the window has no images -> NaN + source none."""
    c = audit._aoi_union_4326(live_cfg).centroid
    rec = {"acquisition_id": "EMPTY_WINDOW", "datetime_utc": "1995-01-01T10:00:00Z",
           "source": "NASA/GPM_L3/IMERG_V07", "geometry": shapely.box(c.x - 0.05, c.y - 0.05, c.x + 0.05, c.y + 0.05)}
    rows = audit._rain_batch(live_cfg, [rec], "aoi", None)
    assert len(rows) == 1
    assert rows[0]["n_images_found"] == 0 and rows[0]["source"] == "none"
    assert math.isnan(rows[0]["rain_6h_mm"]) and math.isnan(rows[0]["rain_24h_mm"])


def test_rain_matches_previous_audit_within_tolerance(live_cfg):
    """Re-compute the wettest earlier-audited acquisition with the new window weighting."""
    audits = sorted((season_dir(live_cfg) / "audit").glob("[0-9]" * 8))
    prev = [a / "rain_flags.csv" for a in audits if (a / "rain_flags.csv").exists()]
    if not prev:
        pytest.skip("no earlier audit with rain_flags.csv")
    old = pd.read_csv(prev[0])
    old = old[(old["source"] == live_cfg["rain"]["primary"]) & old["rain_24h_mm"].notna()]
    if old.empty:
        pytest.skip("no earlier rain values from the primary dataset")
    row = old.sort_values("rain_24h_mm", ascending=False).iloc[0]
    t = pd.Timestamp(row["datetime_utc"])
    cfg = copy.deepcopy(live_cfg)
    cfg["season"]["start"] = (t - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cfg["season"]["end"] = (t + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    acq = audit.group_acquisitions(audit.list_slices(cfg), audit._aoi_union_4326(cfg), cfg)
    acq = acq[acq["acquisition_id"] == row["acquisition_id"]]
    assert len(acq) == 1, "the earlier acquisition must be found again"
    new = audit.rain_flags(cfg, acq).iloc[0]
    assert new["source"] == row["source"] and new["rain_gap_images"] == 0
    assert new["rain_24h_mm"] == pytest.approx(row["rain_24h_mm"], rel=0.15, abs=1.0)
    assert new["rain_6h_mm"] == pytest.approx(row["rain_6h_mm"], rel=0.25, abs=1.0)


def test_slope_stats_keys_and_value(live_cfg):
    s = audit.slope_stats(live_cfg)
    assert set(s) == {"dem", "scale_m", "percentiles_deg", "pct_area_gt_5deg", "pct_area_gt_10deg", "pct_area_gt_15deg"}
    assert set(s["percentiles_deg"]) == {"p50", "p75", "p90", "p95", "p99"}
    assert 0 <= s["percentiles_deg"]["p50"] <= s["percentiles_deg"]["p99"] < 90
    if live_cfg["aoi"]["key"] == "rnd_aoi":
        assert s["percentiles_deg"]["p50"] == pytest.approx(5.38, abs=0.2)  # earlier whole-AOI reduction
