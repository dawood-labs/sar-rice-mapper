"""Tests for the portfolio image export (no project data)."""
from sar_pipeline import outreach_portfolio as op


def test_save_jpg_size_and_limit(tmp_path):
    plt = op._style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([0, 1], [0, 1], color=op.GREEN)
    p = op.save_jpg(fig, tmp_path / "x.jpg")
    from PIL import Image

    im = Image.open(p)
    assert max(im.size) == 1600 and p.stat().st_size <= 400 * 1024


def _numbers():
    import numpy as np
    import pandas as pd

    d = pd.date_range("2026-05-01", periods=20, freq="8D").to_numpy()
    return {"fields": 40, "acres": 120.0, "standing": 30, "harvested": 10, "standing_acres": 90.0, "harvested_acres": 30.0,
            "age_standing": 100.0, "age_harvested": 125.0, "checked": 20, "status_wrong": 1, "sowing_off": 1, "all_right": 18,
            "sow_weeks": [("Jul week 1", 5, 6), ("Jul week 2", 12, 3), ("Jul week 5", 13, 1)],
            "example": {"radar_dates": d, "radar": -18 + np.sin(np.arange(20)), "green_dates": d,
                        "green": np.linspace(0.1, 0.8, 20), "sowing": pd.Timestamp("2026-06-10"),
                        "harvest_from": pd.Timestamp("2026-09-20"), "harvest_to": pd.Timestamp("2026-09-30")}}


def test_stage_calendar_renders(tmp_path):
    from PIL import Image

    p = op.stage_calendar(0, tmp_path, numbers=_numbers())
    assert p.name == "crop_stage_calendar.jpg"
    assert max(Image.open(p).size) == 1600 and p.stat().st_size <= 400 * 1024


def test_stage_map_renders(tmp_path):
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box

    x0, y0 = 500000.0, 5000000.0                      # neutral location inside UTM zone 32N
    layer = tmp_path / "f.gpkg"
    gpd.GeoDataFrame({"status": ["standing", "harvested"], "area_ac": [2.0, 3.0]},
                     geometry=[box(x0 + 100, y0 + 100, x0 + 300, y0 + 300), box(x0 + 400, y0 + 100, x0 + 600, y0 + 300)],
                     crs=32632).to_file(layer, driver="GPKG")
    img = tmp_path / "s2.tif"
    with rasterio.open(img, "w", driver="GTiff", width=80, height=60, count=3, dtype="uint16", crs="EPSG:32632",
                       transform=from_origin(x0, y0 + 600, 10, 10)) as ds:
        ds.write(np.random.default_rng(0).integers(100, 3000, (3, 60, 80)).astype("uint16"))
        ds.descriptions = ("B2", "B3", "B4")
    from PIL import Image

    p = op.stage_map(0, "Country", tmp_path, layer=layer, image=img)
    assert max(Image.open(p).size) == 1600 and p.stat().st_size <= 400 * 1024
