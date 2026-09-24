import numpy as np

from sar_pipeline.analysis import sieve as sv


def test_small_patch_takes_the_surrounding_class_and_nodata_stays():
    c = np.ones((7, 7), dtype="uint8")
    c[3, 3] = 3                      # one speck inside a rice field
    c[0, :] = 255                    # a no-data row
    out = sv.sieve_classes(c, size=4)
    assert out[3, 3] == 1
    assert (out[0] == 255).all()
    assert (sv.sieve_classes(c, 1) == c).all()


def test_large_patch_survives():
    c = np.zeros((10, 10), dtype="uint8")
    c[2:6, 2:6] = 1                  # 16 px
    assert (sv.sieve_classes(c, 10)[2:6, 2:6] == 1).all()
