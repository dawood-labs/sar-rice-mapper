"""Unit tests for the pure-python parts of sar_pipeline.s1_ard (no network, no Earth Engine init)."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from sar_pipeline.s1_ard import ARDConfigError, band_names, temporal_neighbors, validate_config
from sar_pipeline.s1_ard.helper import decode_int16_value, encode_int16_value, ordered_pols
from sar_pipeline.s1_ard.pipeline import _system_indexes
from sar_pipeline.s1_ard.speckle import FILTERS, KERNEL_SIZE_IGNORED
from sar_pipeline.s1_ard import terrain_flattening as tf


# ---------------------------------------------------------------- temporal_neighbors
@pytest.mark.parametrize(
    "n,k,h,expected",
    [
        (10, 5, 3, [2, 3, 4, 5, 6, 7, 8]),
        (10, 0, 3, [0, 1, 2, 3]),          # start of series: clipped
        (10, 9, 3, [6, 7, 8, 9]),          # end of series: clipped
        (10, 1, 3, [0, 1, 2, 3, 4]),
        (5, 2, 0, [2]),                    # no temporal neighbours
        (3, 1, 10, [0, 1, 2]),             # window larger than series
        (1, 0, 3, [0]),                    # single acquisition
    ],
)
def test_temporal_neighbors(n, k, h, expected):
    assert temporal_neighbors(n, k, h) == expected


def test_temporal_neighbors_always_contains_k_and_is_contiguous():
    for n in range(1, 12):
        for k in range(n):
            for h in range(0, 5):
                idx = temporal_neighbors(n, k, h)
                assert k in idx
                assert idx == list(range(idx[0], idx[-1] + 1))
                assert len(idx) <= 2 * h + 1


@pytest.mark.parametrize("args", [(0, 0, 1), (5, 5, 1), (5, -1, 1), (5, 2, -1), (5, 2, 1.5), (5, 2, True)])
def test_temporal_neighbors_invalid(args):
    with pytest.raises(ValueError):
        temporal_neighbors(*args)


# ---------------------------------------------------------------- band_names
def _acq(rows):
    return pd.DataFrame(rows, columns=["acquisition_id", "track_id", "datetime_utc", "date_utc", "system_indexes"])


def test_band_names_sorted_by_time_vv_first():
    acq = _acq([
        ("RO123_ASC_20260711", "RO123_ASC", "2026-07-11T10:01:00Z", "2026-07-11", "b"),
        ("RO123_ASC_20260705", "RO123_ASC", "2026-07-05T10:01:00Z", "2026-07-05", "a"),
        ("RO123_ASC_20260717", "RO123_ASC", "2026-07-17T10:01:00Z", "2026-07-17", "c"),
    ])
    assert band_names(acq) == [
        "VV_20260705", "VH_20260705", "VV_20260711", "VH_20260711", "VV_20260717", "VH_20260717",
    ]


def test_band_names_canonical_pol_order_and_fallback_to_datetime():
    acq = pd.DataFrame({"datetime_utc": ["2026-05-02T22:10:00Z", "2026-04-26T22:10:00Z"]})
    assert band_names(acq, pols=["VH", "VV"]) == ["VV_20260426", "VH_20260426", "VV_20260502", "VH_20260502"]
    assert band_names(acq, pols=["VH"]) == ["VH_20260426", "VH_20260502"]


def test_band_names_duplicate_dates_use_acquisition_id_suffix():
    acq = _acq([
        ("RO201_ASC_20260421_2", "RO201_ASC", "2026-04-21T23:50:00Z", "2026-04-21", "b"),
        ("RO201_ASC_20260421", "RO201_ASC", "2026-04-21T00:10:00Z", "2026-04-21", "a"),
        ("RO201_ASC_20260503", "RO201_ASC", "2026-05-03T00:10:00Z", "2026-05-03", "c"),
    ])
    assert band_names(acq) == [
        "VV_20260421", "VH_20260421", "VV_20260421_2", "VH_20260421_2", "VV_20260503", "VH_20260503",
    ]


def test_band_names_duplicate_dates_without_ids_use_time_order():
    acq = pd.DataFrame({"datetime_utc": ["2026-05-02T23:50:00Z", "2026-05-02T00:10:00Z"]})
    assert band_names(acq, pols=["VH"]) == ["VH_20260502", "VH_20260502_2"]


def test_band_names_duplicate_ids_raise():
    acq = _acq([
        ("X_20260502", "X", "2026-05-02T00:10:00Z", "2026-05-02", "a"),
        ("X_20260502", "X", "2026-05-02T23:50:00Z", "2026-05-02", "b"),
    ])
    with pytest.raises(ValueError, match="not unique"):
        band_names(acq)


def test_ordered_pols_rejects_unknown():
    with pytest.raises(ValueError):
        ordered_pols(["HH"])
    with pytest.raises(ValueError):
        ordered_pols([])


def test_system_indexes_parsing():
    assert _system_indexes({"system_indexes": "a; b;;c"}) == ["a", "b", "c"]
    assert _system_indexes({"system_indexes": ["a", "b"]}) == ["a", "b"]
    with pytest.raises(ValueError):
        _system_indexes({"system_indexes": ""})


# ---------------------------------------------------------------- config validation
@pytest.fixture
def cfg(make_project):
    return make_project()


def test_example_config_is_valid(cfg):
    validate_config(cfg)


@pytest.mark.parametrize("name", FILTERS)
def test_all_filters_accepted(cfg, name):
    cfg["ard"]["speckle_filter"] = name
    validate_config(cfg)


def test_refined_lee_documented_as_ignoring_kernel():
    assert "REFINED LEE" in KERNEL_SIZE_IGNORED


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        ("ard", "speckle_filter", "MEDIAN", "speckle_filter"),
        ("ard", "speckle_kernel", 4, "speckle_kernel"),
        ("ard", "speckle_kernel", 1, "speckle_kernel"),
        ("ard", "speckle_kernel", "5", "speckle_kernel"),
        ("ard", "terrain_flattening_model", "SURFACE", "terrain_flattening_model"),
        ("ard", "layover_shadow_buffer_m", -10, "layover_shadow_buffer_m"),
        ("ard", "temporal_half_window", -1, "temporal_half_window"),
        ("ard", "output_format", "LINEAR", "output_format"),
        ("ard", "multitemporal", "yes", "multitemporal"),
        ("export", "dtype", "uint16", "export.dtype"),
        ("s1", "collection", "COPERNICUS/S1_GRD", "S1_GRD_FLOAT"),
        ("s1", "pols", ["HH", "HV"], "s1.pols"),
    ],
)
def test_invalid_config_values(cfg, section, key, value, match):
    cfg[section][key] = value
    with pytest.raises(ARDConfigError, match=match):
        validate_config(cfg)


def test_int16_config_checks(cfg):
    cfg["export"]["dtype"] = "int16"
    validate_config(cfg)
    cfg["export"]["nodata_int16"] = -9999
    with pytest.raises(ARDConfigError, match="nodata_int16"):
        validate_config(cfg)
    cfg["export"]["nodata_int16"] = -32768
    cfg["export"]["int16_scale"] = 0
    with pytest.raises(ARDConfigError, match="int16_scale"):
        validate_config(cfg)


def test_all_problems_reported_together(cfg):
    cfg["ard"]["speckle_filter"] = "X"
    cfg["export"]["dtype"] = "Y"
    with pytest.raises(ARDConfigError) as exc:
        validate_config(cfg)
    assert "speckle_filter" in str(exc.value) and "export.dtype" in str(exc.value)


def test_terrain_model_not_checked_when_flattening_disabled(cfg):
    cfg["ard"]["terrain_flattening"] = False
    cfg["ard"]["terrain_flattening_model"] = None
    validate_config(cfg)


# ---------------------------------------------------------------- int16 encoding mirror
@pytest.mark.parametrize(
    "db,expected",
    [
        (-12.345, -1234),     # -1234.5 + 0.5 = -1234.0 -> floor -1234
        (-12.3449, -1234),
        (-12.3451, -1235),
        (0.004, 0),
        (0.005, 1),           # tie rounds up (add 0.5, floor)
        (-0.005, 0),          # tie rounds up towards +inf
        (3.0, 300),
        (-400.0, -32767),     # clamped, -32768 stays reserved for nodata
        (400.0, 32767),
    ],
)
def test_encode_int16_value(db, expected):
    assert encode_int16_value(db, 100, -32768) == expected


@pytest.mark.parametrize("bad", [None, float("nan"), float("-inf")])
def test_encode_int16_nodata(bad):
    assert encode_int16_value(bad, 100, -32768) == -32768


def test_int16_roundtrip_error_bound():
    for i in range(-3000, 1000):
        db = i / 97.0
        back = decode_int16_value(encode_int16_value(db, 100, -32768), 100, -32768)
        assert abs(back - db) <= 0.005 + 1e-9
    assert math.isnan(decode_int16_value(-32768, 100, -32768))


# ---------------------------------------------------------------- terrain geometry mirrors
@pytest.mark.parametrize(
    "gx,gy,expected",
    [(1.0, 0.0, 270.0), (-1.0, 0.0, 90.0), (0.0, 1.0, 180.0), (0.0, -1.0, 0.0), (5.8e-5, 1.1e-5, 259.3)],
)
def test_toward_sensor_is_opposite_of_angle_gradient(gx, gy, expected):
    # the incidence angle grows away from the sensor, so the sensor lies against the gradient
    assert tf.toward_sensor_azimuth_deg(gx, gy) == pytest.approx(expected, abs=0.1)


def test_range_slope_sign_positive_when_facing_sensor():
    toward = 259.0
    assert tf.range_slope_deg(20, aspect_deg=toward, toward_deg=toward) == pytest.approx(20)        # downhill side points at the sensor
    assert tf.range_slope_deg(20, aspect_deg=toward - 180, toward_deg=toward) == pytest.approx(-20)
    assert tf.range_slope_deg(20, aspect_deg=toward + 90, toward_deg=toward) == pytest.approx(0, abs=1e-9)
    assert abs(tf.range_slope_deg(20, aspect_deg=toward + 45, toward_deg=toward)) < 20


def test_volume_factor_direction_and_boundaries():
    theta = 36.0
    assert tf.volume_factor(theta, 0.0) == pytest.approx(1.0)
    assert tf.volume_factor(theta, 10.0) < 1.0          # facing slope is too bright -> darkened
    assert tf.volume_factor(theta, -10.0) > 1.0         # slope facing away is too dark -> brightened
    assert tf.volume_factor(theta, theta - 1e-6) == pytest.approx(0.0, abs=1e-6)   # zero at layover boundary
    assert tf.volume_factor(theta, -(90.0 - theta) + 1e-3) > 1e3                  # pole at shadow boundary
    # monotonic: steeper towards the sensor -> smaller factor, over the whole valid range
    alphas = [a / 2 for a in range(int(-(90 - theta) * 2) + 1, int(theta * 2))]
    factors = [tf.volume_factor(theta, a) for a in alphas]
    assert all(f > 0 for f in factors)
    assert all(a > b for a, b in zip(factors, factors[1:]))


def test_direct_factor_matches_volume_on_flat_and_uses_azimuth_slope():
    theta = 36.0
    assert tf.direct_factor(theta, 0.0, 0.0) == pytest.approx(1.0)
    assert tf.direct_factor(theta, 10.0, 0.0) < 1.0 < tf.direct_factor(theta, -10.0, 0.0)
    assert tf.direct_factor(theta, 0.0, 30.0) == pytest.approx(math.cos(math.radians(30)))
    assert tf.direct_factor(theta, theta, 0.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "alpha,valid",
    [(0.0, True), (35.9, True), (36.0, False), (50.0, False), (-53.9, True), (-54.0, False), (-70.0, False)],
)
def test_layover_shadow_boundaries(alpha, valid):
    assert tf.is_valid_geometry(36.0, alpha) is valid


def test_flatten_model_name_validated():
    with pytest.raises(ValueError):
        tf.flatten_slice(None, ["VV"], "SURFACE", None)
