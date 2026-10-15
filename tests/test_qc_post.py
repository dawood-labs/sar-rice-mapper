"""qc_post: labels from the QC file alone, parts, gap only between neighbours, exact geometry otherwise."""
import geopandas as gpd
from shapely.geometry import box

from sar_pipeline.analysis import qc_post as qp


def _qc():
    return gpd.GeoDataFrame({
        "field_id": ["aoi1_p000001", "aoi1_p000002", None, "aoi1_p000003", "aoi1_p000004", "aoi1_p000004"],
        "major_class": ["rice", "rice", None, "non-rice", "rice", "rice"],
        "sub_class": ["young rice", "rice harvested", None, "tree/orchard", "rice standing transplanted",
                      "rice standing transplanted"],
        "origin": ["delineation"] * 6},
        geometry=[box(0, 0, 50, 50), box(50, 0, 100, 50), box(200, 200, 260, 260), box(300, 300, 350, 350),
                  box(500, 500, 540, 540), box(600, 600, 640, 640)], crs=32646)


def test_labels_parts_and_gap():
    rice = qp.label(1, _qc())
    assert "aoi1_p000003" not in set(rice["field_id"])                    # non-rice dropped
    sub = dict(zip(rice["field_id"], rice["sub_class"]))
    assert sub["aoi1_p000002"] == qp.FLIPPED_SUB_CLASS and sub["aoi1_p000001"] == "young rice"
    assert sub["aoi1_qc0001"] == "rice (added in QC)"
    f = qp.clean(rice)
    parts = f[f["field_id"] == "aoi1_p000004"]["part"].tolist()
    assert sorted(parts) == [1, 2]                                         # reshaped field: one id, two parts
    c = qp.check(f)
    assert c["overlapping_pairs"] == 0 and c["touching_pairs"] == 0 and c["duplicate_ids"] == 0
    alone = f[f["field_id"] == "aoi1_qc0001"].geometry.iloc[0]
    assert alone.equals_exact(box(200, 200, 260, 260), 0)                 # no neighbour: geometry untouched
    assert set(f["predicted"]) == {7} and (f["area_ac"] == f["area_ac"].round(3)).all()


def test_no_rice_gives_an_empty_layer():
    qc = _qc()
    qc = qc[qc["major_class"] == "non-rice"]
    f = qp.clean(qp.label(1, qc))
    assert len(f) == 0 and list(f.columns) == qp.COLUMNS


def test_check_of_an_empty_layer_is_all_zero():
    qc = _qc()
    f = qp.clean(qp.label(1, qc[qc["major_class"] == "non-rice"]))
    assert qp.check(f)["polygons_out"] == 0
