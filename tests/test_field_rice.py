import numpy as np

from sar_pipeline.analysis import field_rice as fr


def test_counts_and_plurality_label_with_fallback():
    idx = np.array([[0, 0, 0], [1, 1, -1]])
    cls = np.array([[1, 1, 3], [0, 255, 4]])
    counts = fr.counts_per_field(idx, cls, n_fields=3)
    assert counts[0].tolist() == [0, 2, 0, 1, 0, 0]
    assert counts[1].tolist() == [1, 0, 0, 0, 0, 0]
    lab = fr.label_from_counts(counts, fallback=[9, 9, 4])
    assert lab["label"].tolist() == [1, 0, 4]          # field 2 has no pixel: takes its fallback
    assert lab["rice_share"].iloc[0] == round(2 / 3, 3)


def test_field_label_raster_keeps_unfielded_pixels_and_nodata():
    idx = np.array([[0, -1], [0, 0]])
    cls = np.array([[3, 0], [1, 255]], dtype="uint8")
    out = fr.field_label_raster(idx, np.array([1]), cls)
    assert out.tolist() == [[1, 0], [1, 255]]
