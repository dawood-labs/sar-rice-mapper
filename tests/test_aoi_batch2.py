from sar_pipeline.analysis import aoi_batch2 as b2


def test_internal_ids_and_client_names():
    assert b2.internal_id(133) == 1133 and b2.is_batch2(1133) and not b2.is_batch2(133)
    assert b2.delivery_name(1133) == "b2_aoi133" and b2.delivery_name(19) == "aoi19"


def test_reviewed_aois_choose_their_own_rules():
    assert all(b2.rule_source(a, {})[0] == a for a in b2.REVIEW)


def test_skipped_aois_are_duplicates_or_empty():
    assert set(b2.SKIP) == {165, 169, 185, 220, 222}


def test_s2_export_starts_no_later_than_the_dry_season_the_rules_read():
    from sar_pipeline.analysis import curve_rules as cr

    assert b2.S2_START <= cr.DRY_SEASON_FROM and b2.S2_END > "2026-10-01"


def test_every_skipped_aoi_has_a_reason():
    """The merged file explains every dropped AOI, so no skip is silent."""
    from sar_pipeline.analysis import aoi_batch2 as b2

    assert set(b2.SKIP_REASONS) == set(b2.SKIP)
    assert all(r for r in b2.SKIP_REASONS.values())


def test_review_fields_writes_ids_and_keeps_an_existing_file(tmp_path, monkeypatch):
    """Field ids follow the delineation rows, and a second call never rewrites them."""
    import geopandas as gpd
    from shapely.geometry import box

    from sar_pipeline.analysis import aoi_batch2 as b2
    from sar_pipeline.analysis import field_rice as fr

    frame = gpd.GeoDataFrame({"uid": ["a", "b"], "Confidence": [0.9, 0.8], "area_acres": [1.0, 2.0]},
                             geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10)], crs=32646)
    monkeypatch.setattr(fr, "load_fields", lambda aoi: frame)
    out = b2.review_fields(1166, out_dir=str(tmp_path))
    got = gpd.read_file(out)
    assert got["field_id"].tolist() == ["aoi1166_000000", "aoi1166_000001"]
    assert got["is_field"].all()
    monkeypatch.setattr(fr, "load_fields", lambda aoi: frame.iloc[::-1])
    assert gpd.read_file(b2.review_fields(1166, out_dir=str(tmp_path)))["uid"].tolist() == ["a", "b"]
