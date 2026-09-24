import numpy as np

from sar_pipeline.analysis import rice_similarity as rs


def test_knn_distance_excludes_self_and_separates_clusters():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, (200, 3))
    near = rng.normal(0, 1, (50, 3))
    far = rng.normal(8, 1, (50, 3))
    within = rs.knn_distance(ref, ref, k=5, exclude_self=True)
    cut = np.percentile(within, 90)
    assert (rs.knn_distance(ref, near, k=5) <= cut).mean() > 0.6
    assert (rs.knn_distance(ref, far, k=5) <= cut).mean() == 0.0
