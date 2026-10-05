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


def test_to_upload_sends_new_resized_and_recently_changed_files(tmp_path):
    import os
    import time

    for name in ("same.txt", "resized.txt", "rewritten.txt", "new.txt"):
        (tmp_path / name).write_text("abcd")
    old = time.time() - 3600
    for name in ("same.txt", "resized.txt", "rewritten.txt", "new.txt"):
        os.utime(tmp_path / name, (old, old))
    cut = time.time() - 60
    os.utime(tmp_path / "rewritten.txt", None)                       # touched after the cut, same size
    files = [("same.txt", 4), ("resized.txt", 4), ("rewritten.txt", 4), ("new.txt", 4)]
    have = {"d/same.txt": 4, "d/resized.txt": 3, "d/rewritten.txt": 4}
    assert [r for r, _ in h.to_upload(tmp_path, files, have, "d")] == ["resized.txt", "new.txt"]
    assert [r for r, _ in h.to_upload(tmp_path, files, have, "d", cut)] == ["resized.txt", "rewritten.txt", "new.txt"]
