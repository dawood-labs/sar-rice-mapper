import numpy as np
import pandas as pd

from sar_pipeline import review as rv


def _ev(**cols):
    base = dict(field_id=["f"], area_acres=[1.0], label=[1], late_clear_ndvi_max=[np.nan], last_ndvi=[np.nan],
                vh_drop_end=[3.0], flood_passes=[1], dark_monsoon_passes=[3], vh_median=[-20.0],
                vh_second_darkest=[-25.0], dry_season_ndvi=[0.1], best_monsoon_drop=[8.0],
                field_rule_label=[1], days_since_clear=[5])
    base.update({k: [v] for k, v in cols.items()})
    return pd.DataFrame(base)


def test_clean_rice_is_not_suspect():
    assert rv.suspects(_ev()).empty


def test_harvested_with_standing_canopy_is_flagged():
    s = rv.suspects(_ev(label=4, late_clear_ndvi_max=0.55))
    assert "harvested_but_standing" in s["category"].tolist()


def test_not_rice_with_flood_and_canopy_is_flagged():
    s = rv.suspects(_ev(label=0, late_clear_ndvi_max=0.8))
    assert "water_and_canopy_not_rice" in s["category"].tolist()


def test_village_in_young_is_flagged():
    s = rv.suspects(_ev(label=2, vh_median=-13.0, vh_second_darkest=-16.0, flood_passes=0,
                        dark_monsoon_passes=0, best_monsoon_drop=1.0))
    assert "builtup_or_trees_in_crop_class" in s["category"].tolist()


def test_rice_judged_on_an_old_observation_is_flagged():
    s = rv.suspects(_ev(days_since_clear=45))
    assert s["category"].tolist() == ["standing_extrapolated"]


def test_stretch_ignores_zeros_and_scales_to_unit():
    a = np.array([[0.0, 100.0], [200.0, 300.0]])
    out = rv._stretch(a, 0, 100)
    assert out.min() == 0.0 and out.max() == 1.0 and out[0, 0] == 0.0
