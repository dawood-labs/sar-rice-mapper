import numpy as np

from sar_pipeline.analysis import class3_audit as ca


def test_context_counts_class1_neighbours():
    c = np.zeros((9, 9), dtype=int)
    c[:, :5] = 1                      # left part of the field is confirmed rice
    c[4, 4] = 3
    share = ca.context(c, [4 * 9 + 4, 4 * 9 + 8], size=3)
    assert share[0] > 0.5 and share[1] == 0.0
