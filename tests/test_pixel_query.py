"""Tests for sar_pipeline.pixel_query (no network). The synthetic run comes from tests/_stack_fixtures.py."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import rasterio  # noqa: E402
from pyproj import Transformer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stack_fixtures import ACQS, TRACK, build_synthetic_run  # noqa: E402

from sar_pipeline import pixel_query, stack  # noqa: E402
from sar_pipeline.errors import DecisionRequired, PipelineError  # noqa: E402
from sar_pipeline.index import rowcol_to_pid, rowcol_to_xy  # noqa: E402


@pytest.fixture(params=["float32", "int16"])
def built(request, make_project):
    cfg = make_project({"resources": {"cpu_workers": 1}})
    syn = build_synthetic_run(cfg, request.param)
    with pytest.raises(DecisionRequired):
        stack.build_stack(cfg, syn["run"], TRACK)
    yield cfg, syn
    pixel_query.clear_cache()


def _direct_read(syn: dict, row: int, col: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode the pixel straight from the VERIFIED chunk files (the reference)."""
    x, y = rowcol_to_xy(syn["gd"], row, col)
    man = pd.read_csv(syn["run"] / "export_manifest.csv", dtype=str, keep_default_na=False)
    for f in stack.verified_chunk_files(syn["run"], TRACK):
        with rasterio.open(f) as src:
            b = src.bounds
            if b.left <= x < b.right and b.bottom < y <= b.top:
                r, c = src.index(x, y)
                vals = src.read()[:, r, c].astype("float64")
                vals[vals == syn["nodata"]] = np.nan
                if syn["dtype"] == "int16":
                    vals /= 100.0
                return vals[[0, 2, 4]], vals[[1, 3, 5]]
    assert len(man)
    return np.full(3, np.nan), np.full(3, np.nan)


@pytest.mark.parametrize("row,col", [(0, 1), (3, 7), (4, 0), (7, 8), (8, 7), (8, 8), (12, 4), (14, 15), (2, 3), (5, 18), (13, 19)])
def test_pixel_timeseries_equals_direct_read(built, row, col):
    cfg, syn = built
    pid = rowcol_to_pid(syn["gd"], row, col)
    df = pixel_query.pixel_timeseries(cfg, syn["run"], TRACK, pid)
    vv, vh = _direct_read(syn, row, col)
    np.testing.assert_allclose(df["VV_db"].to_numpy(), vv, rtol=0, atol=1e-5)
    np.testing.assert_allclose(df["VH_db"].to_numpy(), vh, rtol=0, atol=1e-5)
    np.testing.assert_allclose(df["VH_minus_VV_db"].to_numpy(), vh - vv, atol=1e-5)
    assert list(df["valid"]) == list(np.isfinite(vv) & np.isfinite(vh))
    assert list(df["acquisition_id"]) == [a for a, _, _ in ACQS]
    assert list(df.columns) == pixel_query.COLUMNS
    assert df.attrs["pid"] == pid


def test_known_values_and_nodata(built):
    cfg, syn = built
    gd = syn["gd"]
    pid = rowcol_to_pid(gd, 12, 4)
    df = pixel_query.pixel_timeseries(cfg, syn["run"], TRACK, pid)
    exp_vv = np.array([-10 - b - pid / 1000 for b in (1, 3, 5)])
    tol = 1e-5 if syn["dtype"] == "float32" else 0.006
    np.testing.assert_allclose(df["VV_db"], exp_vv, atol=tol)
    df2 = pixel_query.pixel_timeseries(cfg, syn["run"], TRACK, rowcol_to_pid(gd, 5, 18))
    assert df2["VV_db"].isna().all() and not df2["valid"].any()
    assert df["rain_24h_mm"].iloc[1] == pytest.approx(7.2)


def test_lonlat_variant(built):
    cfg, syn = built
    x, y = rowcol_to_xy(syn["gd"], 9, 10)
    lon, lat = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True).transform(x, y)
    a = pixel_query.pixel_timeseries_lonlat(cfg, syn["run"], TRACK, lon, lat)
    b = pixel_query.pixel_timeseries(cfg, syn["run"], TRACK, rowcol_to_pid(syn["gd"], 9, 10))
    pd.testing.assert_frame_equal(a, b, check_like=False)


def test_repeated_queries_reuse_open_datasets_and_metadata(built, monkeypatch):
    cfg, syn = built
    pixel_query.clear_cache()
    opens = {"n": 0}
    metas = {"n": 0}
    orig_open, orig_meta = rasterio.open, pixel_query.read_export_meta
    monkeypatch.setattr(pixel_query.rasterio, "open", lambda *a, **k: (opens.__setitem__("n", opens["n"] + 1), orig_open(*a, **k))[1])
    monkeypatch.setattr(pixel_query, "read_export_meta", lambda *a, **k: (metas.__setitem__("n", metas["n"] + 1), orig_meta(*a, **k))[1])
    for r, c in [(1, 1), (2, 2), (12, 4), (9, 10)]:
        pixel_query.pixel_timeseries(cfg, syn["run"], TRACK, rowcol_to_pid(syn["gd"], r, c))
    assert opens["n"] == 2   # stack_VV.vrt + stack_VH.vrt, once each
    assert metas["n"] == 1


def test_rebuilt_stack_is_picked_up(built):
    cfg, syn = built
    pid = rowcol_to_pid(syn["gd"], 5, 18)
    assert pixel_query.pixel_timeseries(cfg, syn["run"], TRACK, pid)["VV_db"].isna().all()
    man_path = syn["run"] / "export_manifest.csv"
    man = pd.read_csv(man_path, dtype=str, keep_default_na=False)
    man.loc[man["chunk_name"] == "chunk_r00c02", "state"] = "VERIFIED"
    man.to_csv(man_path, index=False)
    with pytest.raises(DecisionRequired):
        stack.build_stack(cfg, syn["run"], TRACK)
    df = pixel_query.pixel_timeseries(cfg, syn["run"], TRACK, pid)
    # dates 1-2 now come from the newly verified chunk; date 3 is nodata in rows 0-9 by construction
    assert list(df["VV_db"].notna()) == [True, True, False]


def test_single_polarisation_layout(make_project):
    cfg = make_project({"resources": {"cpu_workers": 1}})
    syn = build_synthetic_run(cfg)
    path = syn["run"] / "band_layout.csv"
    bl = pd.read_csv(path)
    bl[bl["pol"] == "VH"].to_csv(path, index=False)
    with pytest.raises(DecisionRequired):
        stack.build_stack(cfg, syn["run"], TRACK)
    df = pixel_query.pixel_timeseries(cfg, syn["run"], TRACK, rowcol_to_pid(syn["gd"], 12, 4))
    assert list(df.columns) == ["date_utc", "datetime_utc", "acquisition_id", "platforms", "VH_db",
                                "rain_6h_mm", "rain_24h_mm", "valid"]
    ax = pixel_query.plot_timeseries(df, cfg)
    assert "VH (dB)" in [t.get_text() for t in ax.get_legend().get_texts()]
    pixel_query.clear_cache()


def test_plot_timeseries(built):
    cfg, syn = built
    df = pixel_query.pixel_timeseries(cfg, syn["run"], TRACK, rowcol_to_pid(syn["gd"], 12, 4))
    cfg2 = dict(cfg, audit={"max_gap_days": 12, "constellation_events": [{"date": "2026-04-17", "note": "S1D"}]})
    ax = pixel_query.plot_timeseries(df, cfg2)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "VV (dB)" in labels and "VH (dB)" in labels and any("rain" in l for l in labels)
    assert ax.get_title()


def test_missing_stack_raises(make_project):
    cfg = make_project()
    syn = build_synthetic_run(cfg)
    with pytest.raises(PipelineError, match="dates.csv"):
        pixel_query.pixel_timeseries(cfg, syn["run"], TRACK, 0)


def test_concurrent_queries_with_evictions(built, monkeypatch):
    """8 threads x 200 queries while the reader cache holds fewer datasets than are in use -> no errors, exact values."""
    import threading

    cfg, syn = built
    gd = syn["gd"]
    rng = np.random.default_rng(7)
    pids = [int(p) for p in rng.integers(0, gd["width"] * gd["height"], size=40)]
    pixel_query.clear_cache()
    reference = {p: pixel_query.pixel_timeseries(cfg, syn["run"], TRACK, p)[["VV_db", "VH_db"]].to_numpy() for p in pids}
    pixel_query.clear_cache()
    monkeypatch.setattr(pixel_query, "_MAX_READERS", 1)  # 2 VRTs in use (VV, VH) -> constant evictions
    errors: list[BaseException] = []

    def worker(seed: int) -> None:
        local = np.random.default_rng(seed)
        try:
            for _ in range(200):
                p = pids[int(local.integers(len(pids)))]
                got = pixel_query.pixel_timeseries(cfg, syn["run"], TRACK, p)[["VV_db", "VH_db"]].to_numpy()
                np.testing.assert_array_equal(np.isnan(got), np.isnan(reference[p]))
                np.testing.assert_allclose(got[~np.isnan(got)], reference[p][~np.isnan(reference[p])], atol=1e-9)
        except BaseException as exc:  # noqa: BLE001 - collected and re-raised in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0]
    assert len(pixel_query._READERS) <= 1
    pixel_query.clear_cache()
    assert not pixel_query._READERS
