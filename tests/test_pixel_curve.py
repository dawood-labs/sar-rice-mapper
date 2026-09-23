"""Tests for the pixel curve viewer. A synthetic cube is injected; no files and no network."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from sar_pipeline.analysis import optical_phenology as op
from sar_pipeline.analysis import pixel_curve as pc


@pytest.fixture
def fake_cube(monkeypatch):
    """Two pixels over one year: one carrying a crop, one flat. Every third date is cloudy."""
    dates = pd.date_range("2025-03-01", "2026-01-31", freq="3D")
    days = (dates - dates[0]).days.to_numpy().astype("float64")
    crop = np.interp(days, [0, 70, 100, 170, 200, 365], [0.15, 0.15, 0.35, 0.80, 0.20, 0.18])
    flat = np.full(len(dates), 0.22)
    ndvi = np.stack([crop, flat], axis=1).reshape(len(dates), 1, 2)
    ok = np.ones((len(dates), 1, 2), dtype=bool)
    ok[::3] = False
    loc = {"aoi": "aoiTest", "grid": {"width": 2, "height": 1, "crs": "EPSG:32646",
                                      "x0": 0, "y0": 100, "res": 10}}

    def _read_cube(aoi_id, clear_min=75, cache_root=None, folder=None):
        return dates, ndvi, ndvi * 0 - 0.5, np.full_like(ndvi, np.nan), ok, loc

    monkeypatch.setattr(op, "read_cube", _read_cube)
    monkeypatch.setattr(pc, "pick_folder", lambda *a, **k: ("s2_reference", "data/s2_reference"))
    pc.forget()
    yield
    pc.forget()


def test_raw_holds_only_clear_dates_and_smooth_sits_on_the_five_day_grid(fake_cube):
    total = len(pd.date_range("2025-03-01", "2026-01-31", freq="3D"))
    d = pc.series("x", 0)
    assert d["n_clear"] == len(d["raw"]) < total          # the cloudy dates are gone
    assert set(np.diff(d["smooth"]["date"]).astype("timedelta64[D]").astype(int)) == {5}


def test_the_smoothed_curve_is_the_one_the_detector_reads(fake_cube):
    """The picture is worthless if it is smoothed differently from the detector."""
    d = pc.series("x", 0)
    again = op.pixel_cycles(pd.DatetimeIndex(d["smooth"]["date"]), d["smooth"]["ndvi"].to_numpy(),
                            d["smooth"]["ndwi"].to_numpy())
    assert [c["peak_date"] for c in again] == [c["peak_date"] for c in d["cycles"]]


def test_the_cropped_pixel_gets_a_cycle_and_the_flat_one_does_not(fake_cube):
    assert pc.series("x", 0)["main"] is not None
    assert pc.series("x", 1)["main"] is None


def test_the_cube_is_read_once_per_aoi(fake_cube):
    calls = []
    original = op.read_cube

    def counting(*a, **k):
        calls.append(1)
        return original(*a, **k)

    import sar_pipeline.analysis.optical_phenology as mod
    mod.read_cube = counting
    try:
        pc.forget()
        pc.series("x", 0)
        pc.series("x", 1)
        pc.series("x", 0)
    finally:
        mod.read_cube = original
    assert len(calls) == 1


def test_a_pid_outside_the_grid_is_rejected(fake_cube):
    with pytest.raises(ValueError, match="outside"):
        pc.series("x", 99)


def test_cycles_returns_a_frame(fake_cube):
    frame = pc.cycles("x", 0)
    assert not frame.empty and "peak_date" in frame.columns
    assert pc.cycles("x", 1).empty


def test_show_draws_without_error(fake_cube):
    fig = pc.show("x", 0)
    assert len(fig.axes) == 1          # no LSWI in this cube, so one panel
    fig = pc.show("x", 0, panels=("ndvi", "ndwi"))
    assert len(fig.axes) == 2


def test_pick_folder_prefers_swir_when_it_is_cached(tmp_path, monkeypatch):
    from sar_pipeline import config as config_mod

    monkeypatch.setattr(config_mod, "repo_root", lambda *a, **k: tmp_path)
    (tmp_path / "data" / "s2_reference" / "aoi7").mkdir(parents=True)
    assert pc.pick_folder(7)[0] == "s2_reference"
    (tmp_path / "data" / "s2_reference_swir" / "aoi7").mkdir(parents=True)
    assert pc.pick_folder(7)[0] == "s2_reference_swir"
    assert pc.pick_folder(7, folder="s2_reference")[0] == "s2_reference"


def test_the_cache_is_found_from_any_working_directory(tmp_path, monkeypatch):
    """Regression: running from notebooks/ made every repo-relative path resolve to notebooks/."""
    from sar_pipeline import config as config_mod

    monkeypatch.setattr(config_mod, "repo_root", lambda *a, **k: tmp_path)
    (tmp_path / "data" / "s2_reference_swir" / "aoi7").mkdir(parents=True)
    elsewhere = tmp_path / "notebooks"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert pc.pick_folder(7)[0] == "s2_reference_swir"


def test_plotly_labels_each_date_once_not_once_per_panel(fake_cube):
    """Regression: add_vline annotates every subplot it spans, so the labels piled up on each other."""
    pytest.importorskip("plotly")
    fig = pc.show("x", 0, backend="plotly", panels=("ndvi", "ndwi"))
    main = pc.series("x", 0)["main"]
    expected = sum(1 for key, _, _ in pc.MARKS if main.get(key) is not None)
    assert len(fig.layout.annotations) == expected


def test_plotly_puts_the_panel_name_on_the_axis_not_over_the_date_labels(fake_cube):
    pytest.importorskip("plotly")
    fig = pc.show("x", 0, backend="plotly", panels=("ndvi", "ndwi"))
    titles = [a.text for a in fig.layout.annotations]
    assert "NDVI" not in titles                       # no centred subplot title to collide with
    assert fig.layout.yaxis.title.text == pc.LABELS["ndvi"]
