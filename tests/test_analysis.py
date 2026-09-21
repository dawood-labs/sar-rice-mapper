"""Analysis package: small synthetic checks, no network and no real data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rasterio

from sar_pipeline.analysis import gcp_qc, maps
from sar_pipeline.analysis import pixel_features as pf

T = pd.Timestamp


# ---------------------------------------------------------------- bins
def test_half_month_bins_align_to_calendar():
    fs = pf.FeatureSet(kind="half_month", start="2026-05-16")
    bins = pf.make_bins(fs, T("2026-04-01"), T("2026-07-10"))
    assert [s.strftime("%m%d") for s, _ in bins] == ["0516", "0601", "0616", "0701"]
    assert bins[0][1] == T("2026-06-01") and bins[1][1] == T("2026-06-16")
    assert bins[-1][1] == T("2026-07-10")  # last bin clipped at the end


def test_half_month_needs_day_1_or_16():
    with pytest.raises(ValueError):
        pf.FeatureSet(kind="half_month", start="2026-05-05")


def test_day_bins_anchored_at_start_and_default_season_start():
    bins = pf.make_bins(pf.FeatureSet(kind="days", days=8, start="2026-04-16"), T("2026-04-01"), T("2026-05-10"))
    assert [s for s, _ in bins] == [T("2026-04-16"), T("2026-04-24"), T("2026-05-02")]
    assert bins[-1][1] == T("2026-05-10")
    assert pf.make_bins(pf.FeatureSet(kind="days", days=8), T("2026-04-01"), T("2026-04-09")) == [(T("2026-04-01"), T("2026-04-09"))]


def test_feature_set_name_and_window_check():
    assert pf.FeatureSet(kind="half_month", window=5, start="2026-05-01").name == "hm_w5_s0501"
    assert pf.FeatureSet(kind="days", days=12, window=3).name == "d12_w3_sseason"
    with pytest.raises(ValueError):
        pf.FeatureSet(kind="days", window=4)


def test_fill_linear_interpolates_in_time_and_repeats_at_ends():
    s = [None, np.array([1.0]), None, np.array([4.0]), None]
    out = pf.fill_linear(s, [0, 10, 20, 40, 50])
    assert [float(a[0]) for a in out] == [1.0, 1.0, 2.0, 4.0, 4.0]  # 20 is 1/3 of the way from 10 to 40
    with pytest.raises(ValueError):
        pf.fill_linear([None, None], [0, 1])


def _layout():
    rows = []
    for k, (d, acq) in enumerate([("2026-05-02", "A1"), ("2026-05-08", "A2"), ("2026-05-20", "A3")]):
        rows += [dict(acquisition_id=acq, date_utc=d, pol="VV", band_idx=2 * k + 1),
                 dict(acquisition_id=acq, date_utc=d, pol="VH", band_idx=2 * k + 2)]
    return pd.DataFrame(rows)


def test_rain_rule_drops_wet_dates_only_when_a_dry_date_exists():
    bins = pf.make_bins(pf.FeatureSet(kind="half_month", start="2026-05-01"), T("2026-04-01"), T("2026-07-01"))
    plan = pf.plan_bins(_layout(), {"A1": True, "A2": False, "A3": True}, bins, T("2026-04-01"))
    assert plan[0][2:] == ((2,), (3,))      # 1-15 May: wet A1 dropped, dry A2 kept (0-based bands)
    assert plan[1][2:] == ((4,), (5,))      # 16-31 May: only a wet date, kept
    assert plan[2][2] is None and plan[3][2] is None  # June bins empty -> interpolated later


# ---------------------------------------------------------------- pixel values
def test_box_mean_ignores_nan():
    img = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
    out = pf.box_mean(img, 3)
    assert out[1, 1] == pytest.approx(5.0)                 # mean of the 8 finite neighbours
    assert out[0, 0] == pytest.approx((1 + 2 + 4) / 3)     # corner: only available pixels
    assert np.isnan(pf.box_mean(np.full((3, 3), np.nan), 3)).all()
    assert pf.box_mean(img, 1) is img


def test_track_features_columns_and_interpolation():
    lin = np.stack([np.full((4, 4), 0.1), np.full((4, 4), 0.01)]).astype("float32")  # VV, VH of one date
    bins = [(T("2026-05-01"), 7.0, (0,), (1,)), (T("2026-05-16"), 22.0, None, None)]
    feats = pf.track_features(lin, np.array([1, 2]), np.array([1, 2]), "RO123_ASC", {"hm_w3_s0501": (3, bins)})
    assert set(feats) == {f"RO123_ASC__hm_w3_s0501__{v}__{d}" for v in ("VH", "VV", "VHmVV") for d in ("0501", "0516")}
    assert feats["RO123_ASC__hm_w3_s0501__VV__0516"] == pytest.approx([-10.0, -10.0])
    assert feats["RO123_ASC__hm_w3_s0501__VHmVV__0501"] == pytest.approx([-10.0, -10.0])


def test_non_finite_rows_are_dropped_and_counted():
    df = pd.DataFrame({"a__s__VH__0501": [1.0, -np.inf, 2.0], "b__s__VV__0501": [1.0, 1.0, np.nan], "pid": [1, 2, 3]})
    out, n = pf.drop_non_finite(df, ["a__s__VH__0501", "b__s__VV__0501"])
    assert n == 2 and out.pid.tolist() == [1]


def test_pixel_cap_is_deterministic_and_order_independent():
    df = pd.DataFrame({"pid": np.arange(100), "gcp_id": np.repeat([0, 1], 50)})
    a = pf.cap_per_field(df, 10, seed=0)
    b = pf.cap_per_field(df.sample(frac=1.0, random_state=5), 10, seed=0)
    assert a.groupby("gcp_id").size().tolist() == [10, 10]
    assert sorted(a.pid) == sorted(b.pid)


def test_group_id_uses_cells():
    assert pf.group_id(4999.0, 10001.0, 5000) == "0_2"
    assert pf.group_id(5000.0, 10001.0, 5000) == "1_2"


def test_feature_set_column_selection():
    cols = ["RO123_ASC__hm_w5_s0501__VH__0501", "RO045_DSC__hm_w5_s0501__VV__0501", "RO123_ASC__hm_w5_s0401__VH__0401",
            "pid", "gcp_id", "label", "group", "border"]
    assert pf.columns_for_set(cols, "hm_w5_s0501") == cols[:2]
    assert pf.sets_in(cols) == ["hm_w5_s0401", "hm_w5_s0501"]


def test_analysis_dir_sits_next_to_runs(tmp_path):
    run = tmp_path / "processed" / "aoi" / "season" / "runs" / "v001_20260101"
    assert pf.analysis_dir(run) == tmp_path / "processed" / "aoi" / "season" / "analysis" / "v001_20260101"


# ---------------------------------------------------------------- ground truth
def test_class_codes_come_from_the_data():
    df = pd.DataFrame({"crop": ["b", "a", "b", "c"], "code": [7, 2, 7, 9]})
    assert gcp_qc.class_codes(df, "crop", "code") == {"b": 7, "a": 2, "c": 9}
    with pytest.raises(ValueError):
        gcp_qc.class_codes(pd.DataFrame({"crop": ["a", "a"], "code": [1, 2]}), "crop", "code")


def test_qc_table_marks_excluded_statuses():
    g = pd.DataFrame({"gcp_id": [0, 1, 2, 3]})
    review = pd.DataFrame({"gcp_id": [1, 2, 3], "qc_status": ["check_label", "mixed_pixels", "note_water"],
                           "qc_reason": ["x", "y", "z"]})
    out = gcp_qc.qc_table(g, review)
    assert out.qc_status.tolist() == ["ok", "check_label", "mixed_pixels", "note_water"]
    assert out.use_in_analysis.tolist() == [True, False, False, True]


# ---------------------------------------------------------------- maps
def test_class_and_probability_rasters_have_separate_nodata(tmp_path):
    cls = np.array([[0, 3], [1, 3]], dtype="uint8")
    prob = np.array([[255, 0], [12, 100]], dtype="uint8")
    tf = rasterio.transform.from_origin(0, 20, 10, 10)
    maps.write_class_rasters(tmp_path / "c.tif", tmp_path / "p.tif", cls, prob, "EPSG:3857", tf)
    with rasterio.open(tmp_path / "c.tif") as c, rasterio.open(tmp_path / "p.tif") as p:
        assert c.nodata == 0 and p.nodata == 255
        assert p.read(1, masked=True).mask.tolist() == [[True, False], [False, False]]  # 0 % stays valid


def test_cv_field_layer_error_type():
    oof = pd.DataFrame({"gcp_id": [1, 1, 1, 2, 2], "label": ["a", "a", "a", "b", "b"], "pred": ["a", "a", "b", "a", "a"]})
    f = maps.field_layer(oof, target="a").set_index("gcp_id")
    assert f.loc[1, "error_type"] == "correct" and bool(f.loc[1, "correct"])
    assert f.loc[2, "error_type"] == "b -> a"
    assert f.loc[1, "a_share"] == pytest.approx(2 / 3) and f.loc[1, "pixel_accuracy"] == pytest.approx(2 / 3)


def test_qml_lists_classes_in_code_order(tmp_path):
    maps.write_qml(tmp_path / "x.qml", {"b": 7, "a": 2})
    text = (tmp_path / "x.qml").read_text()
    assert text.index('value="2"') < text.index('value="7"') and 'label="b"' in text
