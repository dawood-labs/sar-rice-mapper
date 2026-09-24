import pandas as pd

from sar_pipeline.analysis import mask_experiment as me


def test_variants_are_distinct_folders_and_summary_reads():
    roots = {me.root(v) for v in me.VARIANTS}
    assert len(roots) == len(me.VARIANTS) and all(r != me.BASE for r in roots)
    assert me.VARIANTS["qa60"]["cs_min"] is None and me.VARIANTS["hyb60"]["keep_dark"]
    refs = pd.DataFrame([
        {"variant": "qa60", "aoi": "aoi1", "set": "rice_plot_interior", "region": "R", "pixels": 100, "delivered_pct": 90.0},
        {"variant": "qa60", "aoi": "aoi2", "set": "rice_plot_interior", "region": "R", "pixels": 300, "delivered_pct": 98.0},
        {"variant": "hyb60", "aoi": "aoi1", "set": "rice_plot_interior", "region": "R", "pixels": 100, "delivered_pct": 96.0},
        {"variant": "hyb60", "aoi": "aoi2", "set": "rice_plot_interior", "region": "R", "pixels": 300, "delivered_pct": 98.0}])
    text = me.summarise(refs, pd.DataFrame())
    assert "96.0" in text and "97.5" in text      # pixel-weighted: (100*96 + 300*98) / 400 = 97.5
