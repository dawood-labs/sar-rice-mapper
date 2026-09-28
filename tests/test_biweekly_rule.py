"""The colleague's 15-day radar rule on synthetic composites (no real data)."""
import numpy as np

from sar_pipeline.analysis import biweekly_rule as br


def _composites(vh_series, vv_series, shape=(12, 12)):
    vh = np.stack([np.full(shape, v, dtype="float32") for v in vh_series])
    vv = np.stack([np.full(shape, v, dtype="float32") for v in vv_series])
    return {"VH": vh, "VV": vv, "passes_per_period": [1] * 10, "shape": shape}


def test_rule_tests_and_kernels(monkeypatch):
    # a paddy: VV dark in June, VH low then high, still high in late September
    paddy = _composites([-16, -17, -23, -22, -18, -15, -13, -13, -13, -13], [-10, -11, -18, -17, -12, -8, -7, -7, -7, -7])
    monkeypatch.setattr(br, "period_composites", lambda aoi_id, **kw: paddy)
    res = br.biweekly_rule(0, smooth=False)
    assert res["rice"].all() and res["flooded"].all() and res["seasonal"].all() and res["not_harvested"].all()
    # the same field cut in early September: VH falls 5 dB from its peak -> not standing
    cut = _composites([-16, -17, -23, -22, -18, -15, -13, -13, -18, -18], [-10, -11, -18, -17, -12, -8, -7, -7, -12, -12])
    monkeypatch.setattr(br, "period_composites", lambda aoi_id, **kw: cut)
    assert not br.biweekly_rule(0, smooth=False)["rice"].any()
    # trees: VV never below -12 and VH flat -> not rice
    trees = _composites([-12] * 10, [-7] * 10)
    monkeypatch.setattr(br, "period_composites", lambda aoi_id, **kw: trees)
    assert not br.biweekly_rule(0, smooth=False)["rice"].any()
    # the kernels: a one-pixel hole inside a field closes, an isolated pixel disappears
    mask = np.ones((20, 20), dtype=bool)
    mask[10, 10] = False
    mask[:, :6] = False
    mask[2, 2] = True
    out = br.closing_then_majority(mask)
    assert out[10, 10] and not out[2, 2] and out[10, 15]
