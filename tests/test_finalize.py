import numpy as np

from sar_pipeline.analysis import finalize as fz


def test_relabel_applies_only_the_listed_class():
    c = np.array([[0, 1, 3], [3, 4, 255]], dtype="uint8")
    out = fz.relabel(c, [{"from": 3, "to": 1}])
    assert out.tolist() == [[0, 1, 1], [1, 4, 255]]
    assert (fz.relabel(c, None) == c).all()


def test_class3_relabel_skips_radar_against_pixels():
    import numpy as np

    from sar_pipeline.analysis import finalize as fz

    c = np.array([3, 3, 1, 0])
    rules = [{"from": 3, "to": 1}]
    assert fz.relabel(c, rules).tolist() == [1, 1, 1, 0]
    assert fz.relabel(c, rules, keep=np.array([False, True, False, False])).tolist() == [1, 3, 1, 0]


def test_class3_relabel_with_keep_to_writes_class9():
    """With the per-pixel switch the radar-against class-3 pixels become class 9, the others follow the relabel."""
    c = np.array([3, 3, 1, 0, 9])
    keep = np.array([False, True, False, False, False])
    out = fz.relabel(c, [{"from": 3, "to": 1}], keep=keep, keep_to=fz.NO_WATER_CLASS)
    assert out.tolist() == [1, 9, 1, 0, 9]
    # no class-3 rule for the AOI: nothing would have become rice, so nothing is set apart
    assert fz.relabel(c, [], keep=keep, keep_to=9).tolist() == c.tolist()


def test_finalize_aoi_per_pixel_switch_writes_class9(tmp_path, monkeypatch):
    import rasterio
    from rasterio.transform import from_origin

    from sar_pipeline.analysis import monsoon_rule as mr

    d = tmp_path / "aoi7"
    d.mkdir()
    prof = dict(driver="GTiff", width=30, height=1, count=1, dtype="uint8", crs="EPSG:32633",
                transform=from_origin(500000, 4000000, 10, 10), nodata=255)
    with rasterio.open(d / "aoi7_monsoon2026.tif", "w", **prof) as ds:
        ds.write(np.full((1, 30), 3, dtype="uint8"), 1)
    with rasterio.open(d / "aoi7_monsoon2026_c3_radar_against.tif", "w", **prof) as ds:
        ds.write((np.arange(30) >= 10).astype("uint8")[None], 1)   # 20 flagged pixels
    ov = {"aoi7": [{"from": 3, "to": 1}]}
    monkeypatch.setattr(fz, "PER_PIXEL_RELABEL", True)
    row = fz.finalize_aoi(7, ov, sieve_px=0, src_root=tmp_path)
    with rasterio.open(d / "aoi7_monsoon2026_final.tif") as ds:
        assert ds.read(1).tolist() == [[1] * 10 + [9] * 20]
    assert row["rice_like_no_water_acres_final"] == round(mr.acres(20), 1) > 0
    assert row["rice_unconfirmed_acres_rule"] == round(mr.acres(30), 1)
    monkeypatch.setattr(fz, "PER_PIXEL_RELABEL", False)
    fz.finalize_aoi(7, ov, sieve_px=0, src_root=tmp_path)
    with rasterio.open(d / "aoi7_monsoon2026_final.tif") as ds:
        assert (ds.read(1) == 1).all()
