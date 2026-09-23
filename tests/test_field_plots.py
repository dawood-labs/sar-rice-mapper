"""Tests for loading and placing field plots. Local files only, neutral location."""
from __future__ import annotations

import zipfile

import geopandas as gpd
from shapely.geometry import Polygon, box, mapping

from sar_pipeline.prep import field_plots as fp

# a neutral location (central Europe, UTM 33N)
X0, Y0 = 15.00, 50.00
D = 0.001  # about 70-110 m


def _write_zipped_shapefile(folder, name, geoms):
    sub = folder / name
    sub.mkdir()
    gpd.GeoDataFrame({"id": [None] * len(geoms)}, geometry=geoms, crs=4326).to_file(sub / f"{name}.shp")
    with zipfile.ZipFile(folder / f"{name}.zip", "w") as z:
        for f in sub.iterdir():
            z.write(f, f"{name}/{f.name}")
    for f in sub.iterdir():
        f.unlink()
    sub.rmdir()


def test_load_reads_every_zip_and_tags_the_source(tmp_path):
    _write_zipped_shapefile(tmp_path, "area_a", [box(X0, Y0, X0 + D, Y0 + D)])
    _write_zipped_shapefile(tmp_path, "area_b", [box(X0 + 2 * D, Y0, X0 + 3 * D, Y0 + D)] * 2)
    plots = fp.load(tmp_path)
    assert plots["source"].tolist() == ["area_a", "area_b", "area_b"]
    assert plots["plot_id"].tolist() == [0, 1, 2]


def test_bow_tie_is_repaired_and_areas_are_in_acres():
    bow_tie = Polygon([(X0, Y0), (X0 + D, Y0 + D), (X0 + D, Y0), (X0, Y0 + D)])
    plots = gpd.GeoDataFrame({"plot_id": [0, 1], "source": ["s", "s"]},
                             geometry=[box(X0, Y0, X0 + D, Y0 + D), bow_tie], crs=4326)
    out = fp.with_area(plots)
    assert out["valid"].tolist() == [True, False]
    assert 1.5 < out.loc[0, "acres"] < 2.0          # ~72 m x 111 m = ~0.8 ha = ~2 acres
    assert out.loc[1, "acres"] > 0


def test_plots_are_placed_in_the_aoi_that_holds_most_of_them():
    plots = gpd.GeoDataFrame({"plot_id": [0, 1, 2], "source": ["s"] * 3},
                             geometry=[box(X0, Y0, X0 + D, Y0 + D),
                                       box(X0 + 0.5 * D, Y0 + 5 * D, X0 + 1.5 * D, Y0 + 6 * D),
                                       box(X0 + 20 * D, Y0, X0 + 21 * D, Y0 + D)], crs=4326)
    aois = {"aoi1": mapping(box(X0 - D, Y0 - D, X0 + 2 * D, Y0 + 2 * D)),
            "aoi2": mapping(box(X0 + D, Y0 + 4 * D, X0 + 3 * D, Y0 + 7 * D))}
    out = fp.in_aois(plots, aois).set_index("plot_id")
    assert out.loc[0, "aoi"] == "aoi1" and out.loc[0, "share_in_aoi"] == 1.0
    assert out.loc[1, "aoi"] == "aoi2" and 0.4 < out.loc[1, "share_in_aoi"] < 0.6
    assert out.loc[2, "share_in_aoi"] == 0.0
