"""The per-test flags must agree with the delivered mask (ndvi_5day.clear_mask, hyb40) and partition the pixels."""
import numpy as np

from sar_pipeline import mask_share as ms
from sar_pipeline.analysis import ndvi_5day as nd


def test_flags_match_the_delivered_mask():
    rng = np.random.default_rng(0)
    n = 5000
    b4 = rng.integers(0, 2500, n).astype("float32")
    b8 = rng.integers(0, 4500, n).astype("float32")
    b2 = rng.integers(200, 1600, n).astype("float32")
    qa = rng.choice([0, 1 << 10, 1 << 11], n).astype("float32")
    cs = rng.integers(0, 101, n).astype("float32")
    f = ms.test_flags(b2, b4, b8, qa, cs)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where((b4 > 0) & (b8 > 0), (b8 - b4) / (b8 + b4), np.nan)
    ok = nd.clear_mask((b4 > 0) & (b8 > 0), qa, cs, b8, ndvi, 40, 1 << 10, True, True, b2, b4)
    assert (f["kept"] == ok).all()
    parts = f["no_data"].astype(int) + f["opaque"] + f["cs_fail"] + f["haze_only"] + f["kept"]
    assert (parts == 1).all()



def test_m1_water_with_b8_zero_is_data_and_missing_cloud_score_is_unknown():
    b3 = np.array([500.0, 0.0, 400.0])
    b4 = np.array([300.0, 0.0, 200.0])
    b8 = np.array([0.0, 0.0, 2500.0])
    assert nd.data_mask(b3, b4, b8).tolist() == [False, False, True]                  # the delivered rule
    assert nd.data_mask(b3, b4, b8, b8_zero_water=True).tolist() == [True, False, True]  # water kept, swath gap not
    data = np.array([True, False, True])
    assert nd.cloud_score_missing(np.array([0.0, 0.0, 0.0]), data)
    assert not nd.cloud_score_missing(np.array([0.0, 0.0, 3.0]), data)                   # a real (cloudy) score
    assert not nd.cloud_score_missing(None, data)
