"""Tests for the client delivery layers (no client data needed)."""
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box

from sar_pipeline.analysis import client_delivery as cd


def test_dissolve_overlaps_merges_only_overlapping_groups():
    a, b = box(0, 0, 10, 10), box(5, 0, 15, 10)          # overlap: one polygon
    c = box(20.2, 0, 30, 10)                               # apart (gap): kept as it is
    out = cd.dissolve_overlaps(np.array([a, b, c], dtype=object))
    assert len(out) == 2
    assert sorted(round(g.area) for g in out) == [98, 150]
    assert any(g.equals_exact(c, 0) for g in out)


def test_pick_takes_first_value_of_its_script():
    r = pd.Series({"VT": "NeByaw", "VT_2": "နဲ", "VTE": None, "JCompany": " Co. Ltd\n"})
    assert cd.pick(r, cd.VILLAGE_TRACT, local=True) == "နဲ"
    assert cd.pick(r, cd.VILLAGE_TRACT, local=False) == "NeByaw"
    assert cd.pick(r, cd.COMPANY) == "Co. Ltd"
    assert cd.pick(pd.Series({"Company": np.nan}), cd.COMPANY) is None


def test_drop_empty_holes_keeps_real_holes():
    shell = [(0, 0), (1, 0), (1, 1), (0, 1)]
    real = [(0.2, 0.2), (0.4, 0.2), (0.4, 0.4), (0.2, 0.4)]
    flat = [(0.6, 0.6), (0.7, 0.6), (0.6, 0.6), (0.6, 0.6)]
    p = cd.drop_empty_holes(Polygon(shell, [real, flat]))
    assert len(p.interiors) == 1


def test_client_shapefiles_reads_a_twice_sent_zip_once(tmp_path):
    for z in ("set_a", "set_a (1)", "set_b (1)"):
        d = tmp_path / z / z.split(" ")[0]
        d.mkdir(parents=True)
        (d / "x.shp").write_text("")
    got = [f"{p.parts[-3]}/{p.name}" for p in cd.client_shapefiles(tmp_path)]
    assert got == ["set_a/x.shp", "set_b (1)/x.shp"]
