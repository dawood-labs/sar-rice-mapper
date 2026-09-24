import pandas as pd

from sar_pipeline.analysis import sar_curve


def test_bad_pass_indices_masks_only_the_recorded_pass(monkeypatch):
    table = pd.DataFrame({"aoi": ["aoi7", "aoi7"], "track": ["T1", "T1"], "pol": ["VH", "VV"],
                          "date": ["2026-06-11", "2026-06-11"], "bad": [True, False]})
    monkeypatch.setitem(sar_curve._BAD_CACHE, "table", table)
    dates = pd.to_datetime(["2026-05-30", "2026-06-11", "2026-06-23"]).date
    assert sar_curve.bad_pass_indices({"aoi": "aoi7"}, "T1", "VH", list(dates)) == [1]
    assert sar_curve.bad_pass_indices({"aoi": "aoi7"}, "T1", "VV", list(dates)) == []
    assert sar_curve.bad_pass_indices({"aoi": "aoi8"}, "T1", "VH", list(dates)) == []


def test_screen_passes_catches_one_polarisation_and_track_wide_artefacts():
    from sar_pipeline.analysis import final_audit as fa

    rows = []
    # five AOIs on track T1; 11 Jun VH low while VV flat in three of them, one AOI has no stable ground
    for aoi, vh in (("aoi1", -2.6), ("aoi2", -2.2), ("aoi3", -7.5), ("aoi4", -0.3), ("aoi5", None)):
        for date, jvh in (("2026-06-11", vh), ("2026-06-23", 0.2 if vh is not None else None)):
            rows.append({"aoi": aoi, "track": "T1", "pol": "VH", "date": date, "stable_pixels": 0 if vh is None else 200,
                         "stable_jump_db": jvh, "bad": False})
            rows.append({"aoi": aoi, "track": "T1", "pol": "VV", "date": date, "stable_pixels": 0 if vh is None else 200,
                         "stable_jump_db": None if vh is None else 0.3, "bad": False})
    # a pass with a real 3 dB jump in both polarisations (rain on the canopy) in one AOI only
    rows.append({"aoi": "aoi4", "track": "T1", "pol": "VH", "date": "2026-07-05", "stable_pixels": 200, "stable_jump_db": 3.2, "bad": False})
    rows.append({"aoi": "aoi4", "track": "T1", "pol": "VV", "date": "2026-07-05", "stable_pixels": 200, "stable_jump_db": 3.4, "bad": False})
    # a tiny AOI (30 stable pixels) with a rainy-day jump on a date nobody else flags: not an artefact
    rows.append({"aoi": "aoi6", "track": "T1", "pol": "VH", "date": "2026-07-17", "stable_pixels": 30, "stable_jump_db": -4.5, "bad": False})
    rows.append({"aoi": "aoi6", "track": "T1", "pol": "VV", "date": "2026-07-17", "stable_pixels": 30, "stable_jump_db": -3.9, "bad": False})
    t = fa.screen_passes(pd.DataFrame(rows)).set_index(["aoi", "pol", "date"])
    assert t.loc[("aoi3", "VH", "2026-06-11"), "reason"] == "jump"
    assert t.loc[("aoi1", "VH", "2026-06-11"), "reason"] == "one_pol"
    assert t.loc[("aoi2", "VH", "2026-06-11"), "reason"] == "one_pol"
    # three AOIs bad -> the whole track on that date, VH only; the AOI without stable ground included
    assert t.loc[("aoi4", "VH", "2026-06-11"), "reason"] == "track_wide"
    assert t.loc[("aoi5", "VH", "2026-06-11"), "reason"] == "track_wide"
    assert not t.loc[("aoi1", "VV", "2026-06-11"), "bad"]
    assert not t.loc[("aoi5", "VV", "2026-06-11"), "bad"]
    # the ordinary jump rule still applies, and a single-AOI event does not spread
    assert t.loc[("aoi4", "VH", "2026-07-05"), "reason"] == "jump" and t.loc[("aoi4", "VV", "2026-07-05"), "bad"]
    assert not t.loc[("aoi1", "VH", "2026-06-23"), "bad"]
    assert not t.loc[("aoi6", "VH", "2026-07-17"), "bad"] and not t.loc[("aoi6", "VV", "2026-07-17"), "bad"]
