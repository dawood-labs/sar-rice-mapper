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
