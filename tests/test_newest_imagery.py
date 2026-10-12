

def test_clear_dates_to_fetch_needs_half_clear_and_complete_scores():
    from sar_pipeline.newest_imagery import clear_dates_to_fetch

    rep = [{"aoi": "aoi1", "s2_new": [
        {"date": "2026-10-05T00:00:00", "aoi_clear_pct": 62.0, "cloud_score_complete": True},
        {"date": "2026-10-07T00:00:00", "aoi_clear_pct": 49.9, "cloud_score_complete": True},
        {"date": "2026-10-10T00:00:00", "aoi_clear_pct": 80.0, "cloud_score_complete": False},
        {"date": "2026-10-11T00:00:00", "aoi_clear_pct": None, "cloud_score_complete": True}]},
        {"aoi": "aoi2", "s2_new": []}]
    assert clear_dates_to_fetch(rep) == {"aoi1": ["2026-10-05"]}


def test_new_dates_folder_is_not_the_manifest_folder():
    """step_upload prunes s2/ to the MANIFEST's dates; the newer dates must live elsewhere."""
    from sar_pipeline.newest_imagery import NEW_DATES_FOLDER

    assert NEW_DATES_FOLDER != "s2" and "/" not in NEW_DATES_FOLDER
