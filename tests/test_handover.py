"""Handover folder scan: files listed with sizes, symbolic links recorded (not followed) and restored."""
import json
import os

from sar_pipeline import handover as h


def test_scan_records_links_without_following_them_and_restore_recreates_them(tmp_path):
    root = tmp_path / "proj"
    (root / "data").mkdir(parents=True)
    (root / "data" / "big.gpkg").write_bytes(b"x" * 100)
    os.symlink("big.gpkg", root / "data" / "fields.gpkg")
    os.symlink("data", root / "data_link")
    files, links = h.scan(root)
    assert files == [("data/big.gpkg", 100)]
    assert {l["path"] for l in links} == {"data/fields.gpkg", "data_link"}
    (root / h.META).mkdir()
    (root / h.META / "symlinks.json").write_text(json.dumps(links))
    os.remove(root / "data" / "fields.gpkg")
    os.remove(root / "data_link")
    assert h.restore_links(str(root)) == 2 and os.readlink(root / "data" / "fields.gpkg") == "big.gpkg"


def test_deal_spreads_big_and_small_files_evenly_over_processes():
    items = [(f"f{i}", s) for i, s in enumerate([100, 1, 50, 2, 70, 3, 90, 4])]
    parts = h._deal(items, 2)
    assert sorted(it for p in parts for it in p) == sorted(items)
    assert [p[0][1] for p in parts] == [100, 90]
    assert abs(len(parts[0]) - len(parts[1])) <= 1
