"""Unit tests for sar_pipeline.audit (synthetic data, no network)."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import box

from sar_pipeline import audit
from sar_pipeline.config import aoi_path
from sar_pipeline.errors import AuditRequired
from sar_pipeline.grid import compute_grid_def


# ---------------------------------------------------------------- helpers
def _slices(rows):
    """rows: (system_index, ro, pass, platform, iso_time, geometry)."""
    recs, geoms = [], []
    for sid, ro, ps, plat, t, geom in rows:
        recs.append(dict(system_index=sid, track_id=audit.track_id(ro, ps), relative_orbit=ro, **{"pass": ps},
                         platform=plat, datetime_utc=t, slice_number=1, polarisations="VV;VH",
                         resolution_meters=10))
        geoms.append(geom)
    return gpd.GeoDataFrame(pd.DataFrame(recs), geometry=geoms, crs="EPSG:4326")


AOI = box(-30.3, 39.7, -30.1, 39.9)
BIG = box(-31.3, 38.7, -29.3, 40.7)
COVER_ALL = box(-30.3, 39.7, -29.7, 40.3)   # contains the synthetic project AOI (conftest)


def _acq_table(track, dates, cov=100.0, platforms="C", ro=123, ps="ASCENDING"):
    return pd.DataFrame([dict(acquisition_id=f"{track}_{d.replace('-', '')}", track_id=track, relative_orbit=ro,
                              **{"pass": ps}, datetime_utc=f"{d}T10:00:00Z", date_utc=d, date_local=d,
                              platforms=platforms, n_slices=1, system_indexes=f"x{d}",
                              aoi_coverage_pct=cov, mean_incidence_angle=35.0, coverage_basis="footprint")
                         for d in dates])


def _many_part_aoi(n_cols=60, n_rows=50, x0=-30.00, y0=40.00, width=0.20, height=0.20):
    """A MultiPolygon AOI of n_cols × n_rows small squares (3,000 parts by default)."""
    dx, dy = width / n_cols, height / n_rows
    parts = [box(x0 + i * dx, y0 + j * dy, x0 + i * dx + 0.6 * dx, y0 + j * dy + 0.6 * dy)
             for i in range(n_cols) for j in range(n_rows)]
    return shapely.MultiPolygon(parts)


def _project_with_aoi(make_project, geom, overrides=None):
    cfg = make_project(overrides)
    gpd.GeoDataFrame({"name": ["aoi"]}, geometry=[geom], crs="EPSG:4326").to_file(aoi_path(cfg), driver="GPKG")
    return cfg


# ---------------------------------------------------------------- ids and grouping
def test_track_id_format():
    assert audit.track_id(123, "ASCENDING") == "RO123_ASC"
    assert audit.track_id(3, "DESCENDING") == "RO003_DSC"


def test_group_two_slices_one_pass_and_two_tracks(make_project):
    cfg = make_project()
    s = _slices([
        ("a1", 123, "ASCENDING", "C", "2026-07-04T09:58:13Z", box(-31.3, 38.7, -29.3, 39.8)),
        ("a2", 123, "ASCENDING", "C", "2026-07-04T09:58:38Z", box(-31.3, 39.8, -29.3, 40.7)),
        ("d1", 45, "DESCENDING", "D", "2026-07-07T21:45:24Z", BIG),
    ])
    acq = audit.group_acquisitions(s, AOI, cfg)
    assert list(acq["acquisition_id"]) == ["RO123_ASC_20260704", "RO045_DSC_20260707"]
    a = acq.set_index("acquisition_id")
    assert a.loc["RO123_ASC_20260704", "n_slices"] == 2
    assert a.loc["RO123_ASC_20260704", "system_indexes"] == "a1;a2"
    assert a.loc["RO123_ASC_20260704", "aoi_coverage_pct"] == pytest.approx(100.0, abs=0.01)
    assert list(acq.columns[:-1]) == audit.ACQ_COLUMNS  # geometry is the last column


def test_coverage_basis_column_follows_border_noise_setting(make_project):
    s = _slices([("d1", 45, "DESCENDING", "D", "2026-07-07T21:45:24Z", BIG)])
    off = audit.group_acquisitions(s, AOI, make_project({"ard": {"border_noise_correction": False}}))
    on = audit.group_acquisitions(s, AOI, make_project({"ard": {"border_noise_correction": True}}))
    assert off.iloc[0]["coverage_basis"] == "footprint"
    assert on.iloc[0]["coverage_basis"] == "footprint, border-noise mask not applied"


def test_s1c_s1d_one_day_apart_are_separate_acquisitions(make_project):
    cfg = make_project()
    s = _slices([
        ("c", 123, "ASCENDING", "C", "2026-05-01T09:58:00Z", BIG),
        ("d", 123, "ASCENDING", "D", "2026-05-02T09:57:30Z", BIG),
    ])
    acq = audit.group_acquisitions(s, AOI, cfg)
    assert list(acq["acquisition_id"]) == ["RO123_ASC_20260501", "RO123_ASC_20260502"]
    assert list(acq["platforms"]) == ["C", "D"]


def test_pass_crossing_midnight_utc_stays_one_acquisition():
    s = _slices([
        ("x1", 10, "DESCENDING", "C", "2026-07-07T23:59:50Z", BIG),
        ("x2", 10, "DESCENDING", "C", "2026-07-08T00:00:15Z", BIG),
    ])
    ids = audit.assign_acquisition_ids(s)
    assert ids.nunique() == 1 and ids.iloc[0] == "RO010_DSC_20260707"


def test_local_date_crosses_midnight(make_project):
    cfg = make_project({"season": {"timezone": "Etc/GMT+5"}})  # UTC-5 (POSIX sign convention)
    s = _slices([("d1", 45, "DESCENDING", "D", "2026-07-08T02:15:24Z", BIG)])
    acq = audit.group_acquisitions(s, AOI, cfg)
    assert acq.iloc[0]["date_utc"] == "2026-07-08"
    assert acq.iloc[0]["date_local"] == "2026-07-07"


def test_coverage_is_area_share_in_equal_area_crs(make_project):
    cfg = make_project()
    west_half = box(-31.3, 38.7, -30.2, 40.7)
    s = _slices([("h", 123, "ASCENDING", "C", "2026-07-04T09:58:13Z", west_half)])
    acq = audit.group_acquisitions(s, AOI, cfg)
    assert acq.iloc[0]["aoi_coverage_pct"] == pytest.approx(50.0, abs=0.5)


def test_slices_from_features_parses_ee_payload():
    feats = [{
        "type": "Feature",
        "geometry": {"type": "LinearRing", "coordinates": [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]},
        "properties": {"system_index": "S1D_X", "time_start": 1783461924000, "relative_orbit": 45,
                       "pass": "DESCENDING", "platform": "D", "slice_number": 5,
                       "polarisations": ["VV", "VH"], "resolution_meters": 10},
    }]
    gdf = audit._slices_from_features(feats)
    assert list(gdf.drop(columns="geometry").columns) == audit.SLICE_COLUMNS
    r = gdf.iloc[0]
    assert r["track_id"] == "RO045_DSC" and r["polarisations"] == "VV;VH" and r["platform"] == "D"
    assert r["datetime_utc"].endswith("Z") and gdf.geometry.iloc[0].geom_type == "Polygon"


# ---------------------------------------------------------------- summary, gaps, recommendation
def test_track_summary_gaps_and_counts(make_project):
    cfg = make_project()
    acq = pd.concat([
        _acq_table("RO123_ASC", ["2026-06-01", "2026-06-13", "2026-06-25"]),
        _acq_table("RO045_DSC", ["2026-06-02", "2026-06-20"], cov=50.0, platforms="D", ro=45, ps="DESCENDING"),
    ])
    s = audit.track_summary(acq, cfg).set_index("track_id")
    assert s.loc["RO123_ASC", "n_acq"] == 3 and s.loc["RO123_ASC", "max_gap_days"] == 12
    assert s.loc["RO045_DSC", "n_acq_ok_coverage"] == 0 and s.loc["RO045_DSC", "max_gap_days"] == 18


def test_detect_gaps_with_events_and_edges(make_project):
    cfg = make_project({"season": {"start": "2026-06-01", "end": "2026-08-01"},
                        "audit": {"max_gap_days": 12,
                                  "constellation_events": [{"date": "2026-06-29", "note": "S1A end"}]}})
    acq = _acq_table("RO123_ASC", ["2026-06-05", "2026-06-17", "2026-07-11", "2026-07-17"])
    gaps = audit.detect_gaps(acq, cfg, now=datetime(2026, 7, 20, tzinfo=timezone.utc))
    assert list(gaps.columns) == audit.GAP_COLUMNS
    assert len(gaps) == 1  # 12-day gap not flagged, leading 4 d and trailing 3 d not flagged
    g = gaps.iloc[0]
    assert (g["gap_start"], g["gap_end"], g["gap_days"]) == ("2026-06-17", "2026-07-11", 24)
    assert "2026-06-29 S1A end" in g["events_in_gap"]

    late = audit.detect_gaps(acq, cfg, now=datetime(2026, 9, 1, tzinfo=timezone.utc))
    trailing = late[late["gap_start"] == "2026-07-17"].iloc[0]
    assert trailing["gap_end"] == "2026-07-31"  # capped at the last season day


def test_recommend_tracks_rule(make_project):
    cfg = make_project()
    summary = pd.DataFrame([
        dict(track_id="RO001_ASC", n_acq=20, n_acq_ok_coverage=18, max_gap_days=12),
        dict(track_id="RO002_DSC", n_acq=20, n_acq_ok_coverage=18, max_gap_days=6),   # tie -> smaller gap wins
        dict(track_id="RO003_ASC", n_acq=12, n_acq_ok_coverage=9, max_gap_days=12),   # exactly 50 %
        dict(track_id="RO004_DSC", n_acq=10, n_acq_ok_coverage=8, max_gap_days=12),
    ])
    rec = audit.recommend_tracks(summary, None, cfg).set_index("track_id")["role"].to_dict()
    assert rec == {"RO002_DSC": "primary", "RO001_ASC": "secondary", "RO003_ASC": "secondary", "RO004_DSC": "unused"}


# ---------------------------------------------------------------- chunk coverage
def test_chunk_track_coverage(make_project):
    cfg = make_project({"grid": {"chunk_px": 128}})
    gd = compute_grid_def(cfg)
    n_chunks = len(gd["chunks"])
    assert n_chunks > 1
    west = box(-30.3, 39.7, -29.971, 40.3)  # roughly half of the synthetic AOI
    fp = _slices([
        ("f1", 123, "ASCENDING", "C", "2026-07-04T09:58:00Z", COVER_ALL),
        ("f2", 123, "ASCENDING", "D", "2026-07-10T09:58:00Z", west),
        ("f3", 45, "DESCENDING", "D", "2026-07-07T21:45:00Z", west),
    ])
    cov = audit.chunk_track_coverage(cfg, gd, fp)
    assert list(cov.columns) == audit.COVERAGE_COLUMNS
    asc = cov[cov["track_id"] == "RO123_ASC"]
    assert asc["chunk_name"].nunique() == n_chunks                  # every chunk touched by the full footprint
    assert (asc["n_acq_full_cover"] >= 1).all()
    assert (asc["n_acq"] <= 2).all() and (asc["n_acq"] == 2).any()
    dsc = cov[cov["track_id"] == "RO045_DSC"]
    assert 0 < dsc["chunk_name"].nunique() < n_chunks                # west half only
    assert (dsc["mean_cover_pct"] <= 100).all() and (dsc["mean_cover_pct"] > 0).all()


# ---------------------------------------------------------------- bounded Earth Engine requests (H9)
def test_bounded_geometry_caps_vertices_and_keeps_area():
    aoi = _many_part_aoi()
    assert len(aoi.geoms) == 3000 and audit.n_vertices(aoi) == 15000
    small = audit.bounded_geometry(aoi, max_vertices=400)
    assert audit.n_vertices(small) <= 400
    assert shapely.intersection(small, aoi).area >= 0.95 * aoi.area
    plain = box(0, 0, 1, 1)
    assert audit.bounded_geometry(plain) is plain  # small geometries are sent unchanged


def test_plan_batches_bounds_weight_and_bytes():
    recs = [{"geometry": box(0, 0, 1, 1), "w": w} for w in [3, 3, 3, 9, 1]]
    batches = audit.plan_batches(recs, weight=lambda r: r["w"], max_weight=6, max_bytes=10**9)
    assert [[r["w"] for r in b] for b in batches] == [[3, 3], [3], [9], [1]]  # oversize record alone
    by_bytes = audit.plan_batches(recs, max_weight=100, max_bytes=2 * (audit.geojson_bytes(box(0, 0, 1, 1)) + 256))
    assert all(len(b) <= 2 for b in by_bytes) and sum(len(b) for b in by_bytes) == 5
    assert audit.plan_batches(recs) == audit.plan_batches(recs)  # deterministic (resume relies on it)


def test_slope_requests_bounded_for_3000_part_aoi(make_project, monkeypatch):
    aoi = _many_part_aoi()
    cfg = _project_with_aoi(make_project, aoi, {"grid": {"chunk_px": 64}})
    gd = compute_grid_def(cfg)
    monkeypatch.setattr(audit, "REGION_BLOCK_M", 2000)
    monkeypatch.setattr(audit, "EE_BATCH", 25)
    monkeypatch.setattr(audit, "MAX_REQUEST_BYTES", 60_000)
    monkeypatch.setattr(audit, "MAX_GEOM_VERTICES", 400)
    calls = []

    def fake_hist(cfg_, features, scale):
        calls.append(features)
        return [{"block_id": f["block_id"], "histogram": [[4.9, 10.0], [5.0, 30.0], [12.0, 10.0]]} for f in features]

    monkeypatch.setattr(audit, "_slope_hist_batch", fake_hist)
    out = audit.slope_stats(cfg, gd=gd)
    assert len(calls) >= 2
    sent = [f["geometry"] for call in calls for f in call]
    for call in calls:
        assert len(call) <= 25
        assert len(call) == 1 or sum(audit.geojson_bytes(f["geometry"]) + 256 for f in call) <= 60_000
    assert max(audit.n_vertices(g) for g in sent) <= 400                     # never the full 15,000-vertex AOI
    assert shapely.intersection(shapely.union_all(sent), aoi).area >= 0.95 * aoi.area  # regions still cover the AOI
    assert out["percentiles_deg"]["p50"] == pytest.approx(5.05, abs=0.06)
    assert out["pct_area_gt_5deg"] == pytest.approx(80.0)
    boxes = audit.region_filter_boxes(audit.region_blocks(cfg, gd))
    assert boxes and all(audit.n_vertices(b) == 5 for b in boxes)           # slice filter geometries are tiny


def test_percentiles_from_merged_histograms_match_direct():
    rng = np.random.default_rng(7)
    samples = np.clip(rng.gamma(2.0, 3.0, 200_000), 0, 89.99)
    edges = np.round(np.arange(0, 90.0001, audit.SLOPE_BIN_DEG), 6)
    hists = []
    for part in np.array_split(samples, 3):
        counts, _ = np.histogram(part, bins=edges)
        hists.append([[float(lo), float(c)] for lo, c in zip(edges[:-1], counts)])
    out = audit.aggregate_slope_histograms(hists, audit.SLOPE_BIN_DEG)
    for q in (50, 75, 90, 95, 99):
        assert out["percentiles_deg"][f"p{q}"] == pytest.approx(np.percentile(samples, q), abs=0.1)
    assert out["pct_area_gt_5deg"] == pytest.approx(100 * (samples >= 5).mean(), abs=0.2)
    assert out["pct_area_gt_15deg"] == pytest.approx(100 * (samples >= 15).mean(), abs=0.2)


# ---------------------------------------------------------------- rain
def test_rain_source_selection():
    t = pd.Timestamp("2026-09-10T10:00:00Z")
    ends = {"P": pd.Timestamp("2026-09-01T00:00:00Z"), "F": pd.Timestamp("2026-09-14T00:00:00Z")}
    assert audit.rain_source_for(t, ends, ["P", "F"]) == "F"
    assert audit.rain_source_for(pd.Timestamp("2026-08-01T00:00:00Z"), ends, ["P", "F"]) == "P"
    assert audit.rain_source_for(pd.Timestamp("2026-10-01T00:00:00Z"), ends, ["P", "F"]) is None
    assert audit.RAIN_DATASETS["NASA/GPM_L3/IMERG_V07"]["hours_per_image"] == 0.5


def _half_hour_images(t, hours_before=30):
    start = (t - pd.Timedelta(hours=hours_before)).floor("30min")
    starts = pd.date_range(start, t + pd.Timedelta(hours=1), freq="30min")
    return [(s, s + pd.Timedelta(minutes=30)) for s in starts]


@pytest.mark.parametrize("t_iso, hours, expected", [
    ("2026-06-01T10:00:00Z", 6, 12),    # window on image boundaries: 04:00 … 09:30
    ("2026-06-01T21:45:00Z", 24, 49),   # partial first (21:30 prev day) and last (21:30) images
    ("2026-06-01T09:58:11Z", 6, 13),
])
def test_rain_window_selection_and_weights(t_iso, hours, expected):
    t = pd.Timestamp(t_iso)
    images = _half_hour_images(t)
    weights = audit.rain_window_weights(t, hours, images)
    assert audit.expected_rain_images(t, hours, 0.5) == expected == sum(w > 0 for w in weights)
    # a constant 1 mm/h rate must give exactly `hours` mm: partial images are weighted by overlap
    assert sum(w * 0.5 for w in weights) == pytest.approx(hours)
    # images starting at or after t, or ending before t - hours, never count
    t_s = t.timestamp()
    for (s, e), w in zip(images, weights):
        if s.timestamp() >= t_s or e.timestamp() <= t_s - hours * 3600:
            assert w == 0


def test_rain_window_partial_weights_values():
    t = pd.Timestamp("2026-06-01T21:45:00Z")
    images = _half_hour_images(t)
    weights = dict(zip([s for s, _ in images], audit.rain_window_weights(t, 24, images)))
    assert weights[pd.Timestamp("2026-06-01T21:30:00Z")] == pytest.approx(0.5)   # last image, half inside
    assert weights[pd.Timestamp("2026-05-31T21:30:00Z")] == pytest.approx(0.5)   # first image, half inside
    assert weights[pd.Timestamp("2026-06-01T12:00:00Z")] == pytest.approx(1.0)


def test_parse_rain_features_empty_window():
    rows = audit.parse_rain_features([
        {"properties": {"acquisition_id": "A", "source": "S", "n_images_found": 0}},
        {"properties": {"acquisition_id": "B", "source": "S", "n_images_found": 48, "rain_6h_mm": 1.23456, "rain_24h_mm": 5.0}},
    ], "aoi")
    assert rows[0]["source"] == "none" and np.isnan(rows[0]["rain_6h_mm"]) and np.isnan(rows[0]["rain_24h_mm"])
    assert rows[1]["source"] == "S" and rows[1]["rain_6h_mm"] == 1.235 and rows[1]["n_images_found"] == 48


def test_rain_flags_empty_window_gives_nan_without_crash(make_project, monkeypatch):
    cfg = make_project()
    monkeypatch.setattr(audit, "_dataset_end", lambda ds, s, e: pd.Timestamp("2030-01-01T00:00:00Z"))

    def fake_batch(cfg_, records, level, gd):
        feats = [{"properties": {"acquisition_id": r["acquisition_id"], "source": r["source"], "n_images_found": 0}}
                 for r in records]
        return audit.parse_rain_features(feats, level)

    monkeypatch.setattr(audit, "_rain_batch", fake_batch)
    acq = _acq_table("RO123_ASC", ["2026-06-01", "2026-06-13"])
    out = audit.rain_flags(cfg, acq)
    assert list(out.columns) == audit.RAIN_COLUMNS and len(out) == 2
    assert (out["source"] == "none").all() and out["rain_24h_mm"].isna().all()
    assert (out["n_images_expected"] == 48).all() and (out["rain_gap_images"] == 48).all()


def test_rain_chunk_level_requests_bounded(make_project, monkeypatch):
    cfg = make_project({"grid": {"chunk_px": 64}, "rain": {"level": "chunk"}})
    gd = compute_grid_def(cfg)
    n_chunks = len(gd["chunks"])
    assert n_chunks > 25
    monkeypatch.setattr(audit, "MAX_FEATURES_PER_REQUEST", 10)
    monkeypatch.setattr(audit, "_dataset_end", lambda ds, s, e: pd.Timestamp("2030-01-01T00:00:00Z"))
    calls = []

    def fake_batch(cfg_, records, level, gd_):
        assert level == "chunk"
        calls.append(sum(len(r["chunks"]) for r in records))
        return [dict(acquisition_id=r["acquisition_id"], source=r["source"], rain_6h_mm=0.0, rain_24h_mm=2.0,
                     chunk_name=c["name"], n_images_found=48) for r in records for c in r["chunks"]]

    monkeypatch.setattr(audit, "_rain_batch", fake_batch)
    table = _acq_table("RO123_ASC", ["2026-06-01", "2026-06-13", "2026-06-25"])
    acq = gpd.GeoDataFrame(table, geometry=[COVER_ALL] * len(table), crs="EPSG:4326")
    out = audit.rain_flags(cfg, acq, gd)
    assert calls and max(calls) <= 10
    assert len(out) == 3 * n_chunks and not out.duplicated(["acquisition_id", "chunk_name"]).any()
    assert (out["level"] == "chunk").all() and (out["rain_gap_images"] == 0).all()


def test_rain_flags_disabled_returns_empty(make_project):
    cfg = make_project({"rain": {"enabled": False}})
    out = audit.rain_flags(cfg, _acq_table("RO123_ASC", ["2026-06-01"]))
    assert out.empty and list(out.columns) == audit.RAIN_COLUMNS


# ---------------------------------------------------------------- readers
def test_selected_acquisitions_errors_and_order(make_project, tmp_path):
    cfg = make_project()
    with pytest.raises(AuditRequired):
        audit.selected_acquisitions(cfg)  # empty tracks

    cfg["s1"]["tracks"] = [{"track_id": "RO123_ASC", "role": "primary"}]
    with pytest.raises(AuditRequired):
        audit.selected_acquisitions(cfg)  # no audit folder yet

    adir = tmp_path / "audit_x"
    adir.mkdir()
    acq = _acq_table("RO123_ASC", ["2026-06-25", "2026-06-01", "2026-06-13"])
    acq.to_csv(adir / "s1_acquisitions.csv", index=False)
    sel = audit.selected_acquisitions(cfg, adir)
    assert list(sel) == ["RO123_ASC"]
    assert list(sel["RO123_ASC"]["date_utc"]) == ["2026-06-01", "2026-06-13", "2026-06-25"]

    cfg["s1"]["tracks"].append({"track_id": "RO999_DSC", "role": "secondary"})
    with pytest.raises(AuditRequired, match="RO999_DSC"):
        audit.selected_acquisitions(cfg, adir)


def test_gaps_report_writes(make_project, tmp_path):
    cfg = make_project({"ard": {"border_noise_correction": True}})
    acq = _acq_table("RO123_ASC", ["2026-06-01", "2026-06-25"], cov=80.0)
    summary = audit.track_summary(acq, cfg)
    gaps = audit.detect_gaps(acq, cfg, now=datetime(2026, 7, 1, tzinfo=timezone.utc))
    rec = audit.recommend_tracks(summary, None, cfg)
    rain = pd.DataFrame([dict(acquisition_id="RO123_ASC_20260601", track_id="RO123_ASC", datetime_utc="x",
                              rain_6h_mm=0.0, rain_24h_mm=7.0, source="S", level="aoi", chunk_name="",
                              n_images_expected=48, n_images_found=40, rain_gap_images=8)])
    p = audit.write_gaps_report(tmp_path / "r.md", cfg, acq, acq, summary, gaps, rec, rain, None, None)
    text = p.read_text()
    assert "Gaps longer than" in text and "RO123_ASC" in text and "coverage below" in text
    assert "border-noise mask not applied" in text and "rain_gap_images" in text


# ---------------------------------------------------------------- resume / checkpoints (rule 9)
class _FakeEE:
    """Counts calls to every Earth Engine-touching function of audit.run_audit."""

    def __init__(self, monkeypatch, fail_rain_call: int | None = None):
        self.calls = {"list_slices": 0, "angle_batch": 0, "rain_batch": 0, "dataset_end": 0, "slope": 0}
        self.fail_rain_call = fail_rain_call
        rows = []
        for i, day in enumerate(["2026-06-01", "2026-06-13", "2026-06-25", "2026-07-07", "2026-07-19"]):
            rows.append((f"asc{i}", 123, "ASCENDING", "C", f"{day}T09:58:00Z", COVER_ALL))
            rows.append((f"dsc{i}", 45, "DESCENDING", "D", f"{day}T21:45:00Z", COVER_ALL))
        self.slices = _slices(rows)
        monkeypatch.setattr(audit, "list_slices", self._list_slices)
        monkeypatch.setattr(audit, "_angle_batch", self._angle_batch)
        monkeypatch.setattr(audit, "_rain_batch", self._rain_batch)
        monkeypatch.setattr(audit, "_dataset_end", self._dataset_end)
        monkeypatch.setattr(audit, "slope_stats", self._slope)
        monkeypatch.setattr(audit, "EE_BATCH", 3)

    def _list_slices(self, cfg, start=None, end=None, gd=None):
        self.calls["list_slices"] += 1
        return self.slices.copy()

    def _angle_batch(self, cfg, records):
        self.calls["angle_batch"] += 1
        return {r["acquisition_id"]: 38.5 for r in records}

    def _rain_batch(self, cfg, records, level, gd):
        self.calls["rain_batch"] += 1
        if self.fail_rain_call is not None and self.calls["rain_batch"] == self.fail_rain_call:
            raise RuntimeError("simulated Earth Engine timeout")
        return [dict(acquisition_id=r["acquisition_id"], source=r["source"], rain_6h_mm=0.0, rain_24h_mm=1.5,
                     chunk_name="", n_images_found=49) for r in records]

    def _dataset_end(self, ds, start, end):
        self.calls["dataset_end"] += 1
        return pd.Timestamp("2026-12-31T00:00:00Z")

    def _slope(self, cfg, gd=None):
        self.calls["slope"] += 1
        return {"dem": "x", "scale_m": 30, "percentiles_deg": {"p50": 1.0, "p75": 2.0, "p90": 3.0, "p95": 4.0, "p99": 5.0},
                "pct_area_gt_5deg": 1.0, "pct_area_gt_10deg": 0.5, "pct_area_gt_15deg": 0.1}


NOW = datetime(2026, 8, 1, tzinfo=timezone.utc)
EXPECTED_FILES = ["s1_slices.csv", "s1_footprints.gpkg", "s1_acquisitions.csv", "track_summary.csv",
                  "chunk_track_coverage.csv", "track_recommendation.csv", "gaps.csv", "gaps_report.md",
                  "rain_flags.csv", "slope_stats.json"]


def test_run_audit_second_run_makes_no_ee_calls(make_project, monkeypatch, caplog):
    cfg = make_project({"grid": {"chunk_px": 128}})
    fake = _FakeEE(monkeypatch)
    out = audit.run_audit(cfg, now=NOW, audit_date="20260801")
    for name in EXPECTED_FILES:
        assert (out / name).exists(), name
    first = dict(fake.calls)
    assert first["list_slices"] == 1 and first["angle_batch"] == 4 and first["rain_batch"] == 4 and first["slope"] == 1
    assert not (out / "_partial").exists()
    rain = pd.read_csv(out / "rain_flags.csv")
    assert (rain["n_images_expected"] == 49).all() and (rain["rain_gap_images"] == 0).all()  # xx:58 / xx:45 passes: 2 partial images

    for k in fake.calls:
        fake.calls[k] = 0
    caplog.set_level("INFO", logger="sar_pipeline.audit")
    audit.run_audit(cfg, now=NOW, audit_date="20260801")
    assert all(v == 0 for v in fake.calls.values()), fake.calls
    assert "skipped slices: cached" in caplog.text and "skipped report: cached" in caplog.text


def test_changed_max_gap_recomputes_gaps_not_slices(make_project, monkeypatch, caplog):
    cfg = make_project({"grid": {"chunk_px": 128}})
    fake = _FakeEE(monkeypatch)
    audit.run_audit(cfg, now=NOW, audit_date="20260801")
    for k in fake.calls:
        fake.calls[k] = 0
    cfg["audit"]["max_gap_days"] = 5
    caplog.set_level("INFO", logger="sar_pipeline.audit")
    out = audit.run_audit(cfg, now=NOW, audit_date="20260801")
    assert fake.calls["list_slices"] == 0 and fake.calls["angle_batch"] == 0 and fake.calls["rain_batch"] == 0
    assert "computed gaps" in caplog.text and "computed report" in caplog.text
    assert "skipped slices: cached" in caplog.text
    assert len(pd.read_csv(out / "gaps.csv")) > 0  # 12-day revisits now exceed 5 days


def test_crash_during_rain_resumes_remaining_batches(make_project, monkeypatch):
    cfg = make_project({"grid": {"chunk_px": 128}})
    fake = _FakeEE(monkeypatch, fail_rain_call=3)
    with pytest.raises(RuntimeError, match="simulated"):
        audit.run_audit(cfg, now=NOW, audit_date="20260801")
    out = audit_dir_of(cfg)
    assert list((out / "_partial").rglob("rain_aoi_*.csv"))  # batches 0 and 1 persisted
    assert not (out / "rain_flags.csv").exists()

    for k in fake.calls:
        fake.calls[k] = 0
    fake.fail_rain_call = None
    audit.run_audit(cfg, now=NOW, audit_date="20260801")
    assert fake.calls["list_slices"] == 0 and fake.calls["angle_batch"] == 0
    assert fake.calls["rain_batch"] == 2  # 4 batches total, 2 were already done
    rain = pd.read_csv(out / "rain_flags.csv")
    assert len(rain) == 10 and rain["acquisition_id"].is_unique
    assert not (out / "_partial").exists()


def test_corrupt_csv_is_recomputed(make_project, monkeypatch, caplog):
    cfg = make_project({"grid": {"chunk_px": 128}})
    _FakeEE(monkeypatch)
    out = audit.run_audit(cfg, now=NOW, audit_date="20260801")
    good = (out / "track_summary.csv").read_text()
    (out / "track_summary.csv").write_text("garbage,\n1,2\n")
    stale, fresh = out / "stale.csv.tmp", out / "fresh.csv.tmp"
    stale.write_text("half")
    fresh.write_text("another process is writing this right now")
    old = time.time() - 2 * audit.STALE_TMP_SECONDS
    os.utime(stale, (old, old))
    caplog.set_level("INFO", logger="sar_pipeline.audit")
    audit.run_audit(cfg, now=NOW, audit_date="20260801")
    assert "computed track_summary" in caplog.text
    assert (out / "track_summary.csv").read_text() == good
    assert not stale.exists()
    assert fresh.exists()  # young temp files may belong to a live writer and are kept


def audit_dir_of(cfg):
    from sar_pipeline.config import audit_dir

    return audit_dir(cfg, "20260801")


def test_bounded_geometry_repairs_invalid_input():
    bow_tie = shapely.Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    assert not bow_tie.is_valid
    out = audit.bounded_geometry(bow_tie)
    assert out.is_valid and out.geom_type in ("Polygon", "MultiPolygon")
    assert out.area == pytest.approx(0.5, rel=1e-6)
    many = _many_part_aoi(n_cols=30, n_rows=30)
    broken = shapely.GeometryCollection([*many.geoms, shapely.LineString([(0, 0), (1, 1)]), bow_tie])
    capped = audit.bounded_geometry(broken, max_vertices=200)
    assert capped.is_valid and audit.n_vertices(capped) <= 200


def _cluster(x0, y0, n=20, size=0.04):
    d = size / n
    return [box(x0 + i * d, y0 + j * d, x0 + i * d + 0.6 * d, y0 + j * d + 0.6 * d) for i in range(n) for j in range(n)]


def test_bounded_geometry_keeps_far_clusters_separate(caplog):
    """Two fragmented clusters at opposite corners of one 0.5° cell: a per-cell convex hull would cover the whole
    diagonal (>20x area); the closing step merges only neighbouring fragments."""
    aoi = shapely.MultiPolygon(_cluster(-30.48, 40.02) + _cluster(-30.06, 40.44))
    assert audit.n_vertices(aoi) == 4000
    with caplog.at_level("WARNING"):
        out = audit.bounded_geometry(aoi, max_vertices=400)
    assert out.is_valid and audit.n_vertices(out) <= 400
    assert shapely.intersection(out, aoi).area >= 0.98 * aoi.area
    inflation = out.area / aoi.area
    hull_inflation = aoi.convex_hull.area / aoi.area
    assert inflation <= 3.0 < hull_inflation
    assert any("inflated" in r.getMessage() for r in caplog.records)
