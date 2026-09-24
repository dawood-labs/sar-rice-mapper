import numpy as np

from sar_pipeline.analysis import finalize as fz


def test_relabel_applies_only_the_listed_class():
    c = np.array([[0, 1, 3], [3, 4, 255]], dtype="uint8")
    out = fz.relabel(c, [{"from": 3, "to": 1}])
    assert out.tolist() == [[0, 1, 1], [1, 4, 255]]
    assert (fz.relabel(c, None) == c).all()
