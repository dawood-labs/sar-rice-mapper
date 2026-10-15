"""aoi_stats: shapefile names fit the 10-character limit."""
from sar_pipeline.analysis import aoi_stats as st


def test_shapefile_names_are_short_and_unique():
    names = list(st.SHP_NAMES.values())
    assert all(len(n) <= 10 for n in names) and len(set(names)) == len(names)
