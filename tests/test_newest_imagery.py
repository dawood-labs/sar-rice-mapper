

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


def test_clear_since_lists_every_date_with_clear_share_and_held(monkeypatch):
    import geopandas as gpd
    import pandas as pd

    from sar_pipeline import config as config_mod
    from sar_pipeline import newest_imagery as ni
    from sar_pipeline import optical_export as oe

    monkeypatch.setattr(ni, "season_config", lambda a, s="monsoon2026": {"aoi": {"key": f"aoi{a}"}})
    monkeypatch.setattr(config_mod, "aoi_path", lambda cfg: "x.gpkg")
    monkeypatch.setattr(gpd, "read_file", lambda p: None)
    monkeypatch.setattr(oe, "aoi_geojson_2d", lambda g: {})
    monkeypatch.setattr(oe, "scene_dates", lambda poly, s, e: {"2026-10-06": 40.0, "2026-09-11": 99.5})
    monkeypatch.setattr(ni, "aoi_clear_pct", lambda poly, dates: {"2026-09-11": 0.0, "2026-10-06": 46.9})
    monkeypatch.setattr(ni, "s2_dates_gcs", lambda cfg: [pd.Timestamp("2026-10-06")])
    t = ni.clear_since(5, "2026-09-01", "2026-10-17")
    assert list(t["date"]) == ["2026-09-11", "2026-10-06"]
    assert list(t["aoi_clear_pct"]) == [0.0, 46.9]
    assert list(t["held_gcs"]) == [False, True]
    assert set(t["aoi"]) == {"aoi5"}


def test_clear_summary_picks_clearest_overall_and_per_month():
    import pandas as pd

    from sar_pipeline import newest_imagery as ni

    t = pd.DataFrame({"aoi": ["aoi5"] * 3, "date": ["2026-09-11", "2026-09-26", "2026-10-06"],
                      "aoi_clear_pct": [10.0, 55.0, 90.0]})
    s = ni.clear_summary(t).iloc[0]
    assert (s["clearest_date"], s["clearest_pct"], s["dates_ge50"], s["dates_ge80"]) == ("2026-10-06", 90.0, 2, 1)
    assert s["best_sep"] == "2026-09-26 (55 %)"


def test_choose_follows_the_window_rules():
    import pandas as pd

    from sar_pipeline import newest_imagery as ni

    t = pd.DataFrame([
        # an AOI without a clear September view: October only
        ("aoi5", "2026-09-26", 90.0), ("aoi5", "2026-10-01", 70.0), ("aoi5", "2026-10-06", 60.0),
        # early September good enough: the early date wins over a clearer late one
        ("aoi20", "2026-09-11", 85.0), ("aoi20", "2026-09-28", 99.0),
        # early September all cloudy: the late date
        ("aoi21", "2026-09-11", 40.0), ("aoi21", "2026-09-28", 95.0), ("aoi21", "2026-10-01", 100.0),
    ], columns=["aoi", "date", "clear_pct_both"])
    c = ni.choose(t, oct_aois=(5,)).set_index("aoi")["date"].to_dict()
    assert c == {"aoi5": "2026-10-01", "aoi20": "2026-09-11", "aoi21": "2026-09-28"}
