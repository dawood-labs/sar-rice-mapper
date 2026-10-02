"""Factor breakdown (a check map): each pixel's sowing / water / state / cut date / peak code, and the non-crop codes."""
import numpy as np
import pandas as pd

from sar_pipeline.analysis import factor_breakdown as fb

WIN = pd.date_range("2026-04-01", "2026-09-26", freq="5D")


def curve(bare_until, peak, cut_on=None, base=0.1, rise_days=40, end=None):
    """A fitted NDVI curve: ``base`` up to ``bare_until``, a linear climb to ``peak`` over ``rise_days``, then flat, down
    to ``base`` from ``cut_on``; ``end`` overrides the last value."""
    t = (WIN - pd.Timestamp(bare_until)).days.to_numpy()
    v = base + np.clip(t / rise_days, 0, 1) * (peak - base)
    if cut_on is not None:
        v = np.where(WIN >= pd.Timestamp(cut_on), base + 0.05, v)
    if end is not None:
        v[-1] = end
    return v


def build(pixels):
    """(events, fit) for a list of (curve, extra event columns); 60 radar-confirmed rice pixels come first."""
    rows, curves = [], []
    rice = [(curve("2026-06-20", 0.80), dict(flood_ok=True, rise_z_all=6.0, flood_date="2026-06-20"))] * 60
    for c, extra in rice + pixels:
        curves.append(c)
        low_i = int(np.argmin(c[WIN >= "2026-05-01"])) + int((WIN < "2026-05-01").sum())
        trough_i = max(i for i in range(len(c)) if c[i] <= c[low_i] + 1e-9 and WIN[i] >= pd.Timestamp("2026-05-01"))
        row = dict(valid=True, trough_ndvi=c[trough_i], trough_date=WIN[trough_i], peak_after=c[trough_i:].max(),
                   last_ndvi=c[-1], ndvi_noise_own=0.02, flood_ok=False, flood_by_pattern=False, rise_z_all=0.0,
                   trough_peer_z=0.0, vh_end=-15.0, open_water_year=False, flood_date=None,
                   brightened_at_sowing=False, ndvi_at_flood_fit=c[trough_i], ndvi_max_own=c.max())
        row.update(extra)
        rows.append(row)
    ev = pd.DataFrame(rows)
    ev["flood_date"] = pd.to_datetime(ev["flood_date"])
    return ev, np.array(curves).T


def codes(pixels):
    ev, fit = build(pixels)
    return list(fb.factors(ev, fit, WIN)["code"].iloc[60:])


def test_rice_full_and_sowing_periods():
    got = codes([
        (curve("2026-05-10", 0.80), dict(flood_ok=True, rise_z_all=5.0, flood_date="2026-05-12")),
        (curve("2026-06-25", 0.80), dict(flood_ok=True, rise_z_all=5.0, flood_date="2026-06-25")),
        (curve("2026-07-25", 0.80, rise_days=30), dict(flood_ok=True, rise_z_all=5.0, flood_date="2026-07-25")),
    ])
    assert got == ["S1 W1 full P1", "S2 W1 full P1", "S3 W1 full P1"]


def test_water_not_seen_and_against():
    got = codes([(curve("2026-06-10", 0.78), {}), (curve("2026-06-10", 0.78), dict(brightened_at_sowing=True))])
    assert got == ["S2 W2 full P1", "S2 W3 full P1"]


def test_cut_dates_and_low_peak():
    got = codes([
        (curve("2026-05-15", 0.80, cut_on="2026-08-30"), dict(flood_ok=True, rise_z_all=5.0, flood_date="2026-05-15")),
        (curve("2026-05-15", 0.55, cut_on="2026-09-21"), {}),
    ])
    assert got == ["S1 W1 cut H1 P1", "S1 W2 cut H2 P2"]


def test_ripening_crop_is_not_cut():
    # yellowing from 0.80 to 0.50: still above half-way back to bare (0.45)
    assert codes([(curve("2026-06-10", 0.80, end=0.50), {})]) == ["S2 W2 full P1"]


def test_young_and_flooded():
    got = codes([
        (curve("2026-08-20", 0.80, rise_days=80), dict(flood_ok=True, rise_z_all=3.0, flood_date="2026-08-20")),
        (curve("2026-08-01", 0.10), dict(flood_ok=True, rise_z_all=0.5, flood_date="2026-08-01", ndvi_max_own=0.4)),
    ])
    assert got == ["S3 W1 young", "S3 W1 flooded"]


def test_non_crop_codes():
    got = codes([
        (np.full(len(WIN), -0.2), dict(open_water_year=True)),
        (np.full(len(WIN), 0.80), dict(trough_peer_z=9.0)),
        (np.full(len(WIN), 0.10), dict(vh_end=-14.0)),
        (np.full(len(WIN), 0.10), dict(vh_end=-24.0)),
        (curve("2026-05-01", 0.55, base=0.45), dict(trough_peer_z=9.0)),
    ])
    assert got == ["N water", "N trees", "N built", "N empty", "N veg"]


def test_ids_words_and_field_majority():
    ids = fb.ids_for(["S1 W1 full P1", "N water", "no data", "S2 W2 cut H1 P2"])
    assert ids == {"no data": 0, "S1 W1 full P1": 1, "S2 W2 cut H1 P2": 2, "N water": 3}
    assert fb.words("S2 W2 cut H1 P2") == ("sown June-20 Jul; no water seen; cut; cut 20 Aug-16 Sep; "
                                          "peak below the AOI's rice")
    maj, share = fb.field_majority(np.array([0, 0, 0, 1, -1]), np.array([2, 2, 1, 3, 1]), 3)
    assert list(maj) == [2, 3, -1]
    assert share[0] == 2 / 3 and np.isnan(share[2])


def test_table_splits_by_final_class():
    t = fb.table(["S1 W1 full P1"] * 300 + ["N water"], [1] * 200 + [3] * 100 + [0], ["N water"], ["aoi1_000001"])
    assert t.loc["S1 W1 full P1", "final_1_rice"] > t.loc["S1 W1 full P1", "final_3_rice_unconfirmed"]
    assert t.loc["N water", "example_fields"] == "aoi1_000001"


def test_water_under_canopy_and_radar_only():
    got = codes([
        # a May crop whose only "flood" is a mid-August radar dip under the 0.80 canopy: not sowing water
        (curve("2026-05-10", 0.80), dict(flood_ok=True, rise_z_all=3.0, flood_date="2026-08-17", ndvi_at_flood_fit=0.80)),
        # flooded since June, the radar roughens a little, the optical never leaves the water
        (np.r_[np.full(15, 0.2), np.full(len(WIN) - 15, -0.05)], dict(flood_ok=True, rise_z_all=3.5,
                                                                       flood_date="2026-06-15", ndvi_max_own=0.4)),
    ])
    assert got == ["S1 W4 full P1", "S2 W1 radar-only"]
