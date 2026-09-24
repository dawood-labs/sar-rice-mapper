import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from sar_pipeline.analysis import compare_runs as cr
from sar_pipeline.analysis import review_verdicts as rv


def _write(path, data):
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                       dtype="uint8", crs="EPSG:32646", transform=from_origin(0, 0, 10, 10), nodata=255) as ds:
        ds.write(data, 1)


def test_transitions_count_acres_per_pair(tmp_path):
    # 40x40 blocks (1,600 px = 39.5 acres), so acres survive the one-decimal rounding
    base = np.kron(np.array([[1, 1, 3], [4, 0, 255]], dtype="uint8"), np.ones((40, 40), dtype="uint8"))
    new = np.kron(np.array([[1, 6, 1], [1, 0, 255]], dtype="uint8"), np.ones((40, 40), dtype="uint8"))
    (tmp_path / "base").mkdir()
    _write(tmp_path / "base" / "aoi7_monsoon2026_final.tif", base)
    _write(tmp_path / "new.tif", new)
    t = cr.transitions(7, tmp_path / "new.tif", tmp_path / "base")
    pairs = {(r["from"], r["to"]): r["acres"] for _, r in t.iterrows()}
    block = round(1600 * 100 / 4046.8564224, 1)
    assert set(pairs) == {(1, 1), (1, 6), (3, 1), (4, 1), (0, 0)}
    assert pairs[(1, 1)] == block
    m = cr.transition_matrix(t)
    assert m.loc["rice", "baseline"] == 2 * block and m.loc["rice", "new"] == 3 * block
    assert m.loc["rice", "net"] == block and m.loc["harvested", "net"] == -block


def test_verdict_parsing_reduces_free_text(tmp_path):
    (tmp_path / "aoi9.md").write_text(
        "# aoi9\n\n| field_id | acres | label | verdict | true class | issue |\n|---|---|---|---|---|---|\n"
        "| aoi9_000001 | 1.20 | 4 | wrong (standing) | 1 rice | 1 |\n"
        "| aoi9_000002 | 0.50 | 1 | right | 1 | - |\n"
        "| aoi9_000003 | 2.00 | 3 | uncertain | 3 or 1 | 6 |\n"
        "| aoi9_000004 | 0.80 | 1 | likely wrong | 0 dry-land crop | 9 |\n"
        "| aoi9_000005 | 0.30 | 2 | acceptable | trees/village | 5 |\n")
    t = rv.build(tmp_path, tmp_path / "v.csv")
    got = t.set_index("field_id")
    assert got.loc["aoi9_000001", "verdict"] == "wrong" and got.loc["aoi9_000001", "expected_rice"] == 1
    assert got.loc["aoi9_000002", "verdict"] == "right" and got.loc["aoi9_000002", "expected_rice"] == 1
    assert got.loc["aoi9_000003", "verdict"] == "uncertain" and pd.isna(got.loc["aoi9_000003", "expected_rice"])
    assert got.loc["aoi9_000004", "verdict"] == "wrong" and got.loc["aoi9_000004", "expected_rice"] == 0
    assert got.loc["aoi9_000005", "verdict"] == "right" and got.loc["aoi9_000005", "expected_rice"] == 0
    # a fix that flips field 1 to rice and field 4 away from rice agrees with every decidable verdict
    labels = pd.DataFrame({"field_id": [f"aoi9_00000{i}" for i in range(1, 6)], "label": [1, 1, 3, 0, 2]})
    s = rv.score(labels, t).set_index("verdict")
    assert s.loc["wrong", "agree_now"] == 2 and s.loc["wrong", "agreed_before"] == 0
    assert s.loc["right", "agree_now"] == 2 and s.loc["right", "disagree_now"] == 0
