import numpy as np

from sar_pipeline.analysis import field_rice as fr


def test_counts_and_plurality_label_with_fallback():
    idx = np.array([[0, 0, 0], [1, 1, -1]])
    cls = np.array([[1, 1, 3], [0, 255, 4]])
    counts = fr.counts_per_field(idx, cls, n_fields=3)
    assert counts[0].tolist() == [0, 2, 0, 1, 0, 0, 0, 0, 0]
    assert counts[1].tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0]
    lab = fr.label_from_counts(counts, fallback=[9, 9, 4])
    assert lab["label"].tolist() == [1, 0, 4]          # field 2 has no pixel: takes its fallback
    assert lab["rice_share"].iloc[0] == round(2 / 3, 3)


def test_field_label_raster_keeps_unfielded_pixels_and_nodata():
    idx = np.array([[0, -1], [0, 0]])
    cls = np.array([[3, 0], [1, 255]], dtype="uint8")
    out = fr.field_label_raster(idx, np.array([1]), cls)
    assert out.tolist() == [[1, 0], [1, 255]]


def test_field_label_raster_leaves_pixels_of_flagged_non_fields_alone():
    import numpy as np

    from sar_pipeline.analysis import field_rice as fr

    idx = np.array([[0, 0, 1], [1, -1, 2]])
    classes = np.array([[3, 3, 0], [1, 4, 2]], dtype="uint8")
    out = fr.field_label_raster(idx, np.array([1, -1, 6]), classes)
    assert out.tolist() == [[1, 1, 0], [1, 4, 6]]       # field 1 is a flagged strip: its pixels keep 0 / 1


def test_confidence_notes_flag_late_floods_after_a_crop_and_stale_views():
    import pandas as pd

    from sar_pipeline.analysis import field_rice as fr

    fl = pd.DataFrame({"flood_date": pd.to_datetime(["2026-08-10", "2026-08-10", "2026-06-10", "NaT", "2026-08-12", "NaT"]),
                       "green_before_flood": [True, False, True, False, False, False],
                       "radar_bright_before_flood": [False, False, False, False, True, False],
                       "days_since_clear": [10, 60, 60, 5, 10, 5],
                       "field_rule_label": [1, 6, 1, 0, 1, 3]})
    notes = fr.confidence_notes(fl, [1, 6, 1, 0, 6, 1])
    assert notes.tolist() == ["late flood after an earlier crop", "no clear view in the last 45 days",
                              "no clear view in the last 45 days", "", "late flood after an earlier crop", ""]
    assert fr.confidence_notes(fl, [1, 6, 1, 0, 6, 1], relabelled_aoi=True)[5] == "rice by AOI relabel"
