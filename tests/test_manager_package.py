"""The clear-date selection and the package README (no real data, no cloud)."""
import numpy as np
import pandas as pd

from sar_pipeline import manager_package as mp


def test_clear_dates_keeps_dates_with_enough_clear_area_inside_the_aoi(monkeypatch):
    dates = pd.to_datetime(["2026-04-20", "2026-05-05", "2026-06-10", "2026-07-15"])
    ok = np.zeros((4, 10), dtype=bool)
    ok[0, :] = True                 # clear but before May
    ok[1, :3] = True                # 30 % of the 10 inside pixels: kept at 25 %
    ok[2, :2] = True                # 20 %: dropped
    ok[3, :] = True                 # fully clear
    inside = np.ones(10, dtype=bool)
    from sar_pipeline.analysis import ndvi_5day as nd
    monkeypatch.setattr(nd, "load", lambda aoi_id, **kw: {"dates": dates, "ok": ok})
    monkeypatch.setattr(nd, "inside_aoi", lambda aoi_id: inside)
    monkeypatch.setattr(nd, "forget", lambda: None)
    t = mp.clear_dates(7, since="2026-05-01", min_clear_pct=25.0)
    assert t["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-05-05", "2026-07-15"]
    assert t["clear_pct"].tolist() == [30.0, 100.0] and (t["aoi"] == "aoi7").all()
    assert mp.s2_object("base", 7, pd.Timestamp("2026-05-05")) == "base/s2_dates_masks/aoi7/aoi7_2026-05_S2_2026-05-05.tif"
    text = mp.readme("pkg", [7, 8], "2026-09-21", 12, "2026-05-01", 25.0)
    assert "rasters/" in text and "25 %" in text and "aoi" not in text.split("areas:")[0]
