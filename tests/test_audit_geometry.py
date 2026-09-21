"""Regression tests for the shapely -> ee.Geometry conversion in `audit._ee_geometry`.

Background
----------
Sentinel-1 footprints returned by Earth Engine carry a third, always-zero coordinate. Passing that
3D GeoJSON straight back to `ee.Geometry` fails with "Invalid GeoJSON geometry", which names the
geometry rather than its dimensionality and is therefore very hard to diagnose: the polygon is
valid, non-empty and small, and every obvious check on it passes.

These tests pin the two guarantees that make that failure impossible: Z is stripped, and an empty
geometry is reported as empty instead of being handed to Earth Engine.

`ee.Geometry` itself is stubbed, so nothing here touches the network.
"""
from __future__ import annotations

import sys
import types

import pytest
import shapely
from shapely.geometry import Polygon

from sar_pipeline import audit


@pytest.fixture
def captured_geojson(monkeypatch):
    """Replace `ee.Geometry` with a recorder, and hand back the GeoJSON it was given."""
    seen: dict = {}

    def fake_geometry(geojson, proj=None, geodesic=None):
        seen["geojson"] = geojson
        seen["proj"] = proj
        seen["geodesic"] = geodesic
        return "ee.Geometry"

    fake_ee = types.ModuleType("ee")
    fake_ee.Geometry = fake_geometry
    monkeypatch.setitem(sys.modules, "ee", fake_ee)
    return seen


def square(z: float | None = None) -> Polygon:
    ring = [(95.0, 16.0), (95.1, 16.0), (95.1, 16.1), (95.0, 16.1)]
    if z is not None:
        ring = [(x, y, z) for x, y in ring]
    return Polygon(ring)


def depth(coords) -> int:
    """Number of components in the first coordinate pair, e.g. 2 for (x, y) and 3 for (x, y, z)."""
    return len(coords[0][0])


def test_z_is_stripped_from_a_3d_polygon(captured_geojson):
    """The real-world case: an Earth Engine footprint with a zero Z on every vertex."""
    geom = square(z=0.0)
    assert geom.has_z                                   # the input really is 3D

    audit._ee_geometry(geom)

    assert depth(captured_geojson["geojson"]["coordinates"]) == 2


def test_nonzero_z_is_also_stripped(captured_geojson):
    """Z is dropped because Earth Engine has no use for it, not because it happened to be zero."""
    audit._ee_geometry(square(z=137.5))
    assert depth(captured_geojson["geojson"]["coordinates"]) == 2


def test_2d_polygon_passes_through_unchanged(captured_geojson):
    audit._ee_geometry(square())
    coords = captured_geojson["geojson"]["coordinates"]
    assert depth(coords) == 2
    assert captured_geojson["geojson"]["type"] == "Polygon"


def test_projection_and_geodesic_are_still_set(captured_geojson):
    """Stripping Z must not disturb the planar EPSG:4326 interpretation the pipeline relies on."""
    audit._ee_geometry(square(z=0.0))
    assert captured_geojson["proj"] == "EPSG:4326"
    assert captured_geojson["geodesic"] is False


def test_coordinates_are_preserved_exactly(captured_geojson):
    """Dropping Z must not move any vertex."""
    audit._ee_geometry(square(z=0.0))
    got = [tuple(pt) for pt in captured_geojson["geojson"]["coordinates"][0]]
    want = [tuple(pt[:2]) for pt in square(z=0.0).exterior.coords]
    assert got == want


def test_empty_geometry_raises_a_clear_error(captured_geojson):
    """An empty ring must not reach Earth Engine, whose error for it is indistinguishable from 3D."""
    with pytest.raises(ValueError, match="empty geometry"):
        audit._ee_geometry(shapely.Polygon())
    assert "geojson" not in captured_geojson


def test_none_geometry_raises_a_clear_error(captured_geojson):
    with pytest.raises(ValueError, match="empty geometry"):
        audit._ee_geometry(None)
    assert "geojson" not in captured_geojson
