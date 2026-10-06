"""Tests for the merged, smoothed radar curve. Synthetic arrays only; no rasters, no network."""
import numpy as np
import pandas as pd
import pytest

from sar_pipeline.analysis import sar_curve as sc


def dates(start, n, step):
    return pd.DatetimeIndex([pd.Timestamp(start) + pd.Timedelta(days=step * i) for i in range(n)])


def test_pair_offsets_recovers_a_known_shift():
    a = dates("2025-01-01", 20, 6)
    b = dates("2025-01-03", 20, 6)          # two days behind, so every pass pairs
    cube_a = np.full((20, 4, 4), -15.0)
    cube_b = cube_a - 2.5                   # the second track reads 2.5 dB low
    offset, n_pairs = sc.pair_offsets(a, cube_a, b, cube_b)
    assert n_pairs == 20
    assert np.allclose(offset, 2.5)


def test_pair_offsets_ignores_passes_that_are_too_far_apart():
    a = dates("2025-01-01", 5, 12)
    b = dates("2025-01-07", 5, 12)          # always six days away
    cube = np.full((5, 2, 2), -15.0)
    with pytest.raises(ValueError, match="pairing window"):
        sc.pair_offsets(a, cube, b, cube, pair_days=3)


def test_pair_offsets_is_per_pixel():
    a, b = dates("2025-01-01", 10, 6), dates("2025-01-02", 10, 6)
    cube_a = np.full((10, 2, 2), -15.0)
    cube_b = cube_a.copy()
    cube_b[:, 0, 0] -= 3.0
    offset, _ = sc.pair_offsets(a, cube_a, b, cube_b)
    assert np.isclose(offset[0, 0], 3.0) and np.isclose(offset[1, 1], 0.0)


def test_windows_average_in_power_not_in_decibels():
    """The mean of decibels sits below the decibel of the mean; the difference is the whole point."""
    acq = dates("2025-03-01", 2, 1)
    cube = np.array([[[-20.0]], [[-10.0]]])
    grid = pd.DatetimeIndex([pd.Timestamp("2025-03-01")])
    out = sc.to_windows(acq, cube, grid, step_days=5)
    in_db_mean = -15.0
    expected = 10 * np.log10((10 ** -2.0 + 10 ** -1.0) / 2)
    assert np.isclose(out[0, 0, 0], expected)
    assert out[0, 0, 0] > in_db_mean


def test_a_window_with_no_acquisition_stays_empty():
    acq = dates("2025-03-01", 1, 1)
    grid = pd.DatetimeIndex([pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-06")])
    out = sc.to_windows(acq, np.array([[[-15.0]]]), grid, step_days=5)
    assert np.isfinite(out[0, 0, 0]) and np.isnan(out[1, 0, 0])


def test_whittaker_db_fits_in_power_and_comes_back_in_decibels():
    values = np.tile([-20.0, -10.0], 15)          # long enough that the fit's ends do not dominate
    fitted = sc.whittaker_db(values, lmbd=100.0)
    middle = float(np.mean(fitted[5:-5]))
    flat_in_power = 10 * np.log10((10 ** -2.0 + 10 ** -1.0) / 2)   # -12.6 dB
    assert abs(middle - flat_in_power) < 0.3
    assert middle > -15.0                          # the mean of the decibels would land here


def test_choose_lambda_prefers_smoothing_on_a_noisy_flat_series():
    rng = np.random.default_rng(0)
    series = -15.0 + rng.normal(0, 1.5, size=(40, 30))
    chosen, table = sc.choose_lambda(series, candidates=(0.05, 1.0, 20.0), max_pixels=10)
    assert chosen == 20.0
    assert table["rmse_db"].idxmin() == 2


def test_choose_lambda_prefers_a_light_touch_on_a_clean_shape():
    days = np.arange(40)
    shape = -20 + 8 * np.exp(-((days - 20) ** 2) / 50)
    series = np.tile(shape[:, None], (1, 20))
    chosen, _ = sc.choose_lambda(series, candidates=(0.05, 1.0, 20.0), max_pixels=10)
    assert chosen == 0.05


def test_the_one_se_rule_never_picks_a_larger_lambda_than_the_minimum():
    rng = np.random.default_rng(1)
    series = -15.0 + rng.normal(0, 1.2, size=(40, 30))
    lenient, table = sc.choose_lambda(series, candidates=(0.25, 1.0, 2.0, 5.0), max_pixels=10)
    strict, _ = sc.choose_lambda(series, candidates=(0.25, 1.0, 2.0, 5.0), max_pixels=10, rule="min")
    assert lenient <= strict


def test_a_fit_that_overshoots_below_zero_power_returns_nan_not_a_floor():
    """Clipping to a tiny power produced -80 dB spikes that a minimum-based feature latched onto."""
    values = np.array([-30.0, -5.0, -30.0, -5.0, -30.0, -5.0, -30.0, -5.0, -30.0])
    fitted = sc.whittaker_db(values, lmbd=0.001)
    assert not np.any(fitted < -60)
    assert np.isfinite(fitted).any()


def test_box_mean_does_not_spread_one_missing_pixel():
    """One NaN used to blank its whole row and column (issue 20); now the window just skips it."""
    from sar_pipeline.analysis.sar_curve import box_mean

    rng = np.random.default_rng(0)
    img = rng.uniform(0.01, 0.1, size=(30, 30)).astype("float32")
    plain = box_mean(img, 5)
    img[10, 12] = np.nan
    out = box_mean(img, 5)
    assert np.isfinite(out).all()
    # far from the hole the mean is the plain filter; around it, the mean of the 24 valid neighbours
    assert np.allclose(out[0:5, 0:5], plain[0:5, 0:5])
    assert abs(out[10, 12] - np.nanmean(img[8:13, 10:15])) < 1e-6
    # a 3-column strip of missing data: every column inside it (at most 2 of 5 valid) stays
    # missing, the columns beside it (3 of 5 valid) keep a value from their valid neighbours
    img[:, 20:23] = np.nan
    out = box_mean(img, 5)
    assert np.isnan(out[:, 20:23]).all()
    assert np.isfinite(out[:, [19, 23]]).all()
    assert abs(out[5, 19] - np.nanmean(img[3:8, 17:22])) < 1e-6


def test_read_track_pixels_matches_the_whole_aoi_read(tmp_path):
    """The field sheets read only a box around the field (plus half a smoothing window): same values as the whole AOI."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from sar_pipeline.analysis import sar_curve as sc

    rng = np.random.default_rng(0)
    h, w, n = 20, 24, 4
    stack = tmp_path / "stack" / "track_T1"
    stack.mkdir(parents=True)
    for pol in ("VH", "VV"):
        data = rng.normal(-15, 2, (n, h, w)).astype("float32")
        data[1, 5, 7] = -9999                                           # one missing value
        path = stack / f"stack_{pol}.vrt"
        with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=n, dtype="float32", nodata=-9999,
                           crs="EPSG:32633", transform=from_origin(500000, 5000000, 10, 10)) as ds:
            ds.write(data)
            ds.descriptions = tuple(f"x_2026060{i + 1}" for i in range(n))
    loc = {"run": str(tmp_path), "aoi": "aoiX"}
    pids = np.array([0, 5 * w + 6, 6 * w + 8, 12 * w + 23, (h - 1) * w + 3])   # corners, edges, near the gap
    d_all, all_ = sc.read_track(loc, "T1", drop_bad=False)
    d_box, box = sc.read_track_pixels(loc, "T1", pids, w, h, drop_bad=False)
    assert (d_all == d_box).all()
    for pol in ("VH", "VV"):
        np.testing.assert_allclose(box[pol], all_[pol].reshape(n, -1)[:, pids], rtol=1e-5)


def test_single_chunk_reads_the_same_values_as_the_nested_vrts(tmp_path):
    """The direct chunk read (one call, all bands) gives what the per-date VRTs give, missing values included."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from sar_pipeline.analysis import sar_curve as sc

    rng = np.random.default_rng(1)
    h, w, dates = 12, 10, ["20260601", "20260613", "20260625"]
    chunk_dir = tmp_path / "raw_chunks" / "track_T1"
    chunk_dir.mkdir(parents=True)
    data = rng.normal(-15, 2, (2 * len(dates), h, w)).astype("float32")
    data[3, 2, 2] = -9999
    chunk = chunk_dir / "chunk_r00c00.tif"
    with rasterio.open(chunk, "w", driver="GTiff", width=w, height=h, count=len(data), dtype="float32", nodata=-9999,
                       crs="EPSG:32633", transform=from_origin(500000, 5000000, 10, 10), interleave="pixel",
                       compress="lzw") as ds:
        ds.write(data)
        ds.descriptions = tuple(f"{p}_{d}" for d in dates for p in ("VV", "VH"))
    gt = "500000.0, 10.0, 0.0, 5000000.0, 0.0, -10.0"

    def source(path, band):
        return (f'<ComplexSource><SourceFilename relativeToVRT="1">{path}</SourceFilename><SourceBand>{band}</SourceBand>'
                f'<SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}" /><DstRect xOff="0" yOff="0" xSize="{w}" ySize="{h}" />'
                f'<NODATA>-9999</NODATA></ComplexSource>')

    stack = tmp_path / "stack" / "track_T1"
    for k, pol in enumerate(("VV", "VH")):
        (stack / pol).mkdir(parents=True)
        bands = []
        for i, d in enumerate(dates):
            (stack / pol / f"{pol}_{d}.vrt").write_text(
                f'<VRTDataset rasterXSize="{w}" rasterYSize="{h}"><GeoTransform>{gt}</GeoTransform>'
                f'<VRTRasterBand dataType="Float32" band="1"><NoDataValue>-9999</NoDataValue>'
                f'{source("../../../raw_chunks/track_T1/chunk_r00c00.tif", 2 * i + k + 1)}</VRTRasterBand></VRTDataset>')
            bands.append(f'<VRTRasterBand dataType="Float32" band="{i + 1}"><Description>{pol}_{d}</Description>'
                         f'<NoDataValue>-9999</NoDataValue>{source(f"{pol}/{pol}_{d}.vrt", 1)}</VRTRasterBand>')
        (stack / f"stack_{pol}.vrt").write_text(f'<VRTDataset rasterXSize="{w}" rasterYSize="{h}">'
                                                f'<GeoTransform>{gt}</GeoTransform>{"".join(bands)}</VRTDataset>')
    loc = {"run": str(tmp_path), "aoi": "aoiX"}
    assert sc.single_chunk(stack / "stack_VH.vrt")[1] == [2, 4, 6]
    d1, fast = sc._read(loc, "T1", 5, False)
    d2, slow = sc._read(loc, "T1", 5, False, fast=False)
    assert (d1 == d2).all()
    for pol in ("VH", "VV"):
        np.testing.assert_allclose(fast[pol], slow[pol], rtol=1e-6, equal_nan=True)


def test_disk_cache_key_changes_with_the_bad_pass_list(tmp_path):
    from sar_pipeline.analysis import sar_curve
    (tmp_path / "stack_VV.vrt").write_text("<VRTDataset/>")
    a = sar_curve._cache_file(tmp_path, "VV", 5, True, [3])
    b = sar_curve._cache_file(tmp_path, "VV", 5, True, [3, 7])
    c = sar_curve._cache_file(tmp_path, "VV", 5, True, [3])
    assert a != b and a == c and a.parent.name == "cache"
