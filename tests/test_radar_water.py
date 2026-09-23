"""Tests for per-pixel radar dips at each pixel's own trough date. No files."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sar_pipeline.analysis import radar_water as rw


def test_dip_is_read_at_each_pixels_own_trough():
    dates = pd.date_range("2026-03-15", "2026-09-20", freq="12D")
    troughs = np.array(["2026-06-15", "2026-07-27", "NaT"], dtype="datetime64[D]")
    cube = np.full((len(dates), 3), -8.0)
    for p, t in enumerate(troughs[:2]):
        days = (dates.to_numpy().astype("datetime64[D]") - t) / np.timedelta64(1, "D")
        cube[(days >= -6) & (days <= 12), p] = -8.0 - (6.0 if p == 0 else 1.0)
    dry, flood, dip = rw.dips_for_track(dates, cube, troughs)
    assert abs(dip[0] - 6.0) < 1e-9 and abs(dip[1] - 1.0) < 1e-9
    assert np.isnan(dip[2])                       # no trough: nothing to line up on
    assert dry[0] == -8.0 and flood[0] == -14.0


def test_trough_at_the_start_of_the_run_is_not_checkable():
    dates = pd.date_range("2026-05-01", "2026-09-20", freq="12D")
    troughs = np.array(["2026-05-04"], dtype="datetime64[D]")
    _, _, dip = rw.dips_for_track(dates, np.full((len(dates), 1), -8.0), troughs)
    assert np.isnan(dip[0])                       # the dry window lies before the first date


def test_flood_search_finds_a_flood_weeks_before_the_anchor():
    import numpy as np

    from sar_pipeline.analysis import radar_water as rw

    dates = np.arange(np.datetime64("2026-04-01"), np.datetime64("2026-09-01"), 6)
    v = np.full(len(dates), -9.0)
    flood = (dates >= np.datetime64("2026-06-10")) & (dates <= np.datetime64("2026-06-25"))
    v[flood] = -15.0
    cube = v[:, None]
    anchor = np.array(["2026-07-20"], dtype="datetime64[D]")
    best, when, persist = rw.flood_search(dates, cube, anchor)
    assert best[0] == 6.0 and np.datetime64("2026-06-10") <= when[0] <= np.datetime64("2026-06-25")
    assert persist[0] >= 2
    # a steadily rising field (a dry-land crop growing) never drops below its own earlier level
    rising = np.linspace(-12, -5, len(dates))[:, None]
    best, _, persist = rw.flood_search(dates, rising, anchor)
    assert best[0] < 0 and persist[0] == 0
