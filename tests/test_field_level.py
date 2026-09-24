import numpy as np

from sar_pipeline.analysis import field_level as fl


def test_field_means_per_field_and_nan_safe():
    values = np.array([[1.0, 3.0, np.nan, 10.0], [2.0, 2.0, 2.0, np.nan]])
    idx = np.array([0, 0, 0, 1])
    out = fl.field_means(values, idx, 2)
    assert out[:, 0].tolist() == [2.0, 2.0]
    assert out[0, 1] == 10.0 and np.isnan(out[1, 1])


def test_power_round_trip():
    assert np.allclose(fl.to_db(fl.to_power(np.array([-10.0, -20.0]))), [-10.0, -20.0])
