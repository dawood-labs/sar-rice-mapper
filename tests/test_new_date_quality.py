"""newest_imagery.date_quality: the shares reported for a newly downloaded Sentinel-2 date. No network, no files."""
import numpy as np

from sar_pipeline import newest_imagery as ni


def _bands():
    # 10 pixels: 2 outside the swath (all bands 0), 1 opaque cloud, 1 low Cloud Score+ and bright,
    # 1 hazy canopy (blue-bright, NDVI >= 0.3), 5 clear canopies with NDVI 0.6
    b2 = np.array([0, 0, 400, 400, 1500, 300, 300, 300, 300, 300], "float32")
    b3 = np.array([0, 0, 600, 600, 900, 500, 500, 500, 500, 500], "float32")
    b4 = np.array([0, 0, 500, 500, 800, 400, 400, 400, 400, 400], "float32")
    b8 = np.array([0, 0, 3000, 3000, 3000, 1600, 1600, 1600, 1600, 1600], "float32")
    qa = np.array([0, 0, 1 << 10, 0, 0, 0, 0, 0, 0, 0], "float32")
    cs = np.array([0, 0, 90, 10, 90, 90, 90, 90, 90, 90], "float32")
    return b2, b3, b4, b8, qa, cs


def test_no_data_clear_kept_and_ndvi():
    q = ni.date_quality(*_bands())
    assert q["no_data_pct"] == 20.0            # swath edge counted as missing, not cloud
    assert q["cs_clear_pct"] == 70.0           # Cloud Score+ >= 60 among the 10 pixels (includes the opaque one)
    assert q["series_kept_pct"] == 50.0        # opaque, low score and haze removed by the series mask
    assert q["median_ndvi_kept"] == 0.6


def test_missing_cloud_score_is_reported_as_unknown():
    b2, b3, b4, b8, qa, _ = _bands()
    q = ni.date_quality(b2, b3, b4, b8, qa, np.zeros(10, "float32"))
    assert q["cs_clear_pct"] is None
    # with the score unknown the M1 series mask falls back to QA60 + haze only
    assert q["series_kept_pct"] == 60.0


def test_nothing_kept_gives_no_median():
    b2, b3, b4, b8, _, cs = _bands()
    q = ni.date_quality(b2, b3, b4, b8, np.full(10, 1 << 10, "float32"), cs)
    assert q["series_kept_pct"] == 0.0 and q["median_ndvi_kept"] is None
