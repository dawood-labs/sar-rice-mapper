"""Tests for the request tiling of the direct Earth Engine fetch (no network)."""
from sar_pipeline import direct_fetch as df


def test_tiles_cover_the_grid_and_stay_under_the_limit():
    ts = df.tiles(164, 153, 46, 4)
    covered = sum(w * h for _, _, w, h in ts)
    assert covered == 164 * 153
    assert all(w * h * 46 * 4 <= df.MAX_REQUEST_BYTES for _, _, w, h in ts)
    big = df.tiles(3000, 2000, 200, 4)
    assert all(w * h * 200 * 4 <= df.MAX_REQUEST_BYTES for _, _, w, h in big) and len(big) > 1
