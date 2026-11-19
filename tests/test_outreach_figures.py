"""The outreach figures render from plain arrays and carry no place or id (synthetic data only)."""
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sar_pipeline import outreach_figures as of  # noqa: E402


def _texts(fig):
    return " ".join(t.get_text() for ax in fig.axes for t in ax.texts) + " ".join(t.get_text() for t in fig.texts)


def test_radar_story_draws_phases_and_no_ids(tmp_path):
    w = pd.date_range("2026-03-15", "2026-09-20", freq="5D")
    ndvi = np.where(w < "2026-07-05", 0.7, np.nan)
    r = pd.date_range("2026-03-16", "2026-09-20", freq="12D")
    curves = {"windows": w, "ndvi": ndvi, "radar_dates": r, "vv": np.full(len(r), -9.0), "vh": np.full(len(r), -16.0)}
    fig = of.radar_story(curves, [("2026-06-29", "2026-07-22", "Flooded for transplanting", "#9cc7ef")],
                         out=str(tmp_path / "a.png"))
    text = _texts(fig)
    assert "Flooded for transplanting" in text and "only the radar" in text
    assert "aoi" not in text.lower() and (tmp_path / "a.png").exists()


def test_clouds_diagram_renders(tmp_path):
    fig = of.clouds_diagram(str(tmp_path / "b.png"))
    assert "Radar satellite" in _texts(fig) and (tmp_path / "b.png").exists()


def _curves():
    w = pd.date_range("2026-03-15", "2026-09-20", freq="5D")
    ndvi = np.where(w < "2026-07-05", 0.7, np.nan)
    r = pd.date_range("2026-03-16", "2026-09-20", freq="12D")
    vh = np.where(r < "2026-06-20", -14.0, np.where(r < "2026-07-20", -24.0, -16.0))
    return {"windows": w, "ndvi": ndvi, "radar_dates": r, "vv": vh + 8, "vh": vh}


def test_season_events_are_read_from_the_curves():
    e = of.season_events(_curves())
    assert e["flood"][1] == -24.0 and e["before"][1] == -14.0 and e["after"][1] == -16.0
    assert e["last_clear"] < pd.Timestamp("2026-07-05")


def test_infographic_and_pipeline_render_without_ids(tmp_path):
    c = _curves()
    ev = {"phases": [("2026-06-20", "2026-07-20", "Flooded")], **of.season_events(c)}
    monthly = pd.DataFrame({"month": pd.period_range("2026-01", "2026-09", freq="M"), "kept": np.linspace(0.9, 0.1, 9)})
    fig = of.season_infographic(c, monthly, {"Region A": 99.0}, [("99%", "plots")], ev, out=str(tmp_path / "i.png"))
    assert "aoi" not in _texts(fig).lower() and (tmp_path / "i.png").exists()
    fig = of.pipeline_figure(str(tmp_path / "p.png"))
    assert "Find the\nwater" in _texts(fig) and (tmp_path / "p.png").exists()


def test_patterns_figures_render(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    import pandas as pd

    from sar_pipeline import outreach_figures as of

    days = pd.date_range("2026-03-05", "2026-09-21", freq="5D")
    s = pd.Series(np.linspace(0.2, 0.8, len(days)), index=days)
    of.patterns_infographic([("rice", s, of.SLOT_3, 0.05), ("not rice", s * 0.5, of.TEXT_3)],
                            [("Rise", "NDVI", 0.65, 0.29)], [("2,950", "plots")], str(tmp_path / "a.png"))
    of.patterns_flow(str(tmp_path / "b.png"))
    assert (tmp_path / "a.png").exists() and (tmp_path / "b.png").exists()


def test_pattern_medians(tmp_path):
    import numpy as np
    import pandas as pd

    from sar_pipeline import outreach_figures as of

    both = pd.DataFrame({"ndvi_2026-05-01": [0.1, 0.3, 0.5], "ndvi_2026-05-06": [0.2, 0.4, 0.6]})
    both.to_parquet(tmp_path / "c.parquet")
    pd.DataFrame({"group": [1, 1, 2]}).to_csv(tmp_path / "g.csv", index=False)
    m = of.pattern_medians(str(tmp_path / "c.parquet"), str(tmp_path / "g.csv"), (1, 2))
    assert np.allclose(m[1].values, [0.2, 0.3]) and np.allclose(m[2].values, [0.5, 0.6])


def test_delivery_figures_render_from_summary_files(tmp_path):
    import json

    from sar_pipeline import outreach_figures as of

    d = tmp_path / "delivery" / "aoi5"
    d.mkdir(parents=True)
    (d / "MANIFEST.json").write_text("{}")
    (d / "aoi5_too_young.json").write_text(json.dumps({"rice_acres": 100.0, "too_young_acres": 2.0, "too_young_fields": 3}))
    (d / "aoi5_trees_cut.json").write_text(json.dumps({"rice_fields": 40, "rice_acres_before": 98.0, "rice_acres_after": 95.0,
                                                      "fields_cut": 4, "slivers_dropped": 2}))
    chosen = tmp_path / "chosen.json"
    chosen.write_text(json.dumps({"5": {"right_pct": 61.0}}))
    n = of.delivery_numbers(str(tmp_path / "delivery"), str(chosen), str(tmp_path / "review"))
    assert n["areas"] == 1 and n["rice_acres_final"] == 95.0 and n["too_young_fields"] == 3
    assert of.delivery_results_figure(n, tmp_path / "r.png").stat().st_size > 10_000
    assert of.delivery_method_figure(n, tmp_path / "m.png").stat().st_size > 10_000


def test_weak_area_figures_render_from_search_records(tmp_path):
    import json

    from sar_pipeline import outreach_figures as of

    w = tmp_path / "weak"
    (w / "round2").mkdir(parents=True)
    (w / "aoi5.json").write_text(json.dumps({"current_check": 60.0, "found_check": 70.0, "adopt": True}))
    (w / "round2" / "aoi5.json").write_text(json.dumps({"found": "x+rice_needs_empty_field", "found_check": 75.0}))
    (w / "aoi6.json").write_text(json.dumps({"current_check": 80.0, "found_check": 78.0, "adopt": False}))
    (w / "round2" / "aoi6.json").write_text(json.dumps({"found": "x", "found_check": 90.0}))
    loss = tmp_path / "loss.csv"
    loss.write_text("km,loss_pts\n22,4\n70,30\n")
    n = of.weak_numbers(str(w), str(loss), str(tmp_path / "review"), new_areas=3)
    assert n["check"].tolist() == [[60.0, 70.0, 75.0], [80.0, 80.0, 80.0]]   # no new switch in aoi6: kept as it was
    assert of.weak_results_figure(n, tmp_path / "r.png").stat().st_size > 10_000
    assert of.weak_method_figure(n, tmp_path / "m.png").stat().st_size > 10_000


def test_crop_stage_figures_render_from_stage_outputs(tmp_path):
    import json

    d = tmp_path / "aoi9"
    d.mkdir()
    ids = [f"x_p{i:06d}" for i in range(12)]
    cut = [i < 6 for i in range(12)]
    pd.DataFrame({"field_id": ids, "acres": 1.0}).to_csv(d / "fields.csv", index=False)
    pd.DataFrame({"field_id": ids, "status": ["harvested" if c else "standing" for c in cut],
                  "sowing_median": ["2026-07-03" if c else "2026-07-31" for c in cut],
                  "harvest_from": ["2026-11-02" if c else None for c in cut],
                  "harvest_to": ["2026-11-05" if c else None for c in cut],
                  "interior_pixels": 9, "source": "rule"}).to_csv(d / "field_stage.csv", index=False)
    rd = pd.to_datetime(["2026-06-21", "2026-07-03", "2026-07-15", "2026-08-08", "2026-10-07", "2026-11-12"])
    gd = pd.to_datetime(["2026-06-05", "2026-07-25", "2026-09-06", "2026-10-16", "2026-11-02", "2026-11-05", "2026-11-15"])
    rows = []
    for f in ids:
        rows += [(f, "VH", "T1", a, v) for a, v in zip(rd, [-15, -20, -17, -16, -14, -13])]
        rows += [(f, "NDVI", "S2", a, v) for a, v in zip(gd, [0.1, 0.2, 0.6, 0.8, 0.75, 0.4, 0.38])]
    pd.DataFrame(rows, columns=["field_id", "source", "track", "date", "value"]).to_parquet(d / "field_curves.parquet")
    (d / "field_check.json").write_text(json.dumps({"fields_checked": 12, "status_wrong": 1, "sowing_slightly_off": 2}))
    n = of.stage_numbers(9, str(tmp_path))
    assert (n["fields"], n["standing"], n["harvested"], n["all_right"]) == (12, 6, 6, 9)
    assert n["age_harvested"] == 125 and [w for w, _s, _h in n["sow_weeks"]] == ["Jul week 1", "Jul week 5"]
    assert of.stage_results_figure(n, tmp_path / "r.png").stat().st_size > 10_000
    assert of.stage_method_figure(n, tmp_path / "m.png").stat().st_size > 10_000
