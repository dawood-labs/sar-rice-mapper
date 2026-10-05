"""Does one rule set generalise? Every AOI's rule set scored on every AOI's labelled pixels.

Why
---
User (5 Oct 2026): "from all the labelled pixels of several AOIs, see whether a generalisation is happening". The rule
(``curve_rules``) is tuned AOI by AOI (``AOI_OVERRIDES``); if one set scores as well on the other AOIs' labels as their
own sets do, that set is a candidate default for the AOIs not labelled yet. This module builds the cross table: rows =
the AOI whose labels are judged, columns = the AOI whose rule set judges them, cell = share of labels the set gets
right. Two scores: the full class (rice standing direct seeded / transplanted, rice harvested, young rice, flooded /
bare, other vegetation, tree/orchard) and the coarse class (establishment ignored), because direct seeded vs
transplanted is the hardest and least sure part. Read-only: no map, no locked output is written.

Use::

    python -m sar_pipeline.analysis.rule_generalization                 # all labelled AOIs x all rule sets
    python -m sar_pipeline.analysis.rule_generalization --out report.csv --misses misses.csv
    python -m sar_pipeline.analysis.rule_generalization --candidates general_v1   # also a candidate general set
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

#: Label -> map class name (``curve_rules.MAP_CLASSES``); rice needs its state and establishment.
SIMPLE = {"flooded": "flooded / bare", "bare": "flooded / bare", "young rice": "young rice",
          "other vegetation": "other vegetation", "tree/orchard": "tree/orchard"}


def truth(label: str, state: str = "", establishment: str = "") -> str:
    """The map class a user label stands for."""
    if label == "rice":
        if state == "harvested":
            return "rice harvested"
        return "rice standing transplanted" if str(establishment).startswith("transpl") else \
            "rice standing direct seeded"
    return SIMPLE.get(label, label)


def coarse(name: str) -> str:
    """The class with establishment ignored (both standing-rice classes -> "rice standing")."""
    return "rice standing" if str(name).startswith("rice standing") else name


#: Candidate general rule sets (not used by any map; tried with ``--candidates``). "general_v1" (5 Oct): the aoi116 set
#: (which scored best on the other AOIs) plus aoi39's water / pond switches, without aoi39's "one wet pass is a
#: transplanting" (WATER_SPELL_MIN_PASSES 1), which turned direct-seeded and young rice of aoi160 into transplanted.
def _general_v1() -> dict:
    from . import curve_rules as cr

    s = dict(cr.AOI_OVERRIDES[116])
    s.update({k: cr.AOI_OVERRIDES[39][k] for k in ("POND_IF_DRY_WATER", "GROWN_NOT_IF_WATER_VIEW",
                                                  "SECOND_CROP_BY_RADAR", "RADAR_DECIDES_WITHOUT_OPTICAL",
                                                  "YOUNG_RADAR_LOW_NEEDS_RISE")})
    return s


def _without(*keys):
    return lambda: {k: v for k, v in _general_v1().items() if k not in keys}


#: v2: without SOWING_FROM_RADAR (sowing at the END of the water spell made standing rice look young: aoi39 21856,
#: 24716, aoi72 27550); v3: also without YOUNG_WHILE_RADAR_LOW; v4: without AGE_FROM_WATER instead.
CANDIDATES = {"general_v1": _general_v1, "general_v2": _without("SOWING_FROM_RADAR"),
              "general_v3": _without("SOWING_FROM_RADAR", "YOUNG_WHILE_RADAR_LOW"),
              "general_v4": _without("AGE_FROM_WATER")}


def _context(source):
    """Rule context for an AOI number, None (plain defaults) or a candidate name / dict of switches."""
    from contextlib import contextmanager, nullcontext

    from . import curve_rules as cr

    if source is None:
        return nullcontext()
    if isinstance(source, (int, np.integer)):
        return cr.rules_for(int(source))
    rules = CANDIDATES[source]() if isinstance(source, str) else dict(source)

    @contextmanager
    def ctx():
        key = -999                                  # a slot no AOI uses, removed afterwards
        cr.AOI_OVERRIDES[key] = rules
        try:
            with cr.rules_for(key):
                yield
        finally:
            cr.AOI_OVERRIDES.pop(key, None)
    return ctx()


def judge(target: int, pixels, source) -> pd.Series:
    """Classes a rule set gives these pixels of AOI ``target`` (``source``: an AOI number = that AOI's set, ``None`` =
    the plain defaults i.e. the aoi160 rules, or a ``CANDIDATES`` name), read on the target's own pinned inputs.
    Features are computed inside the rule context, since several switches change the features (age, water spell)."""
    from . import curve_rules as cr

    with _context(source):
        f = cr.own_range_features(target, np.asarray(pixels, dtype=int))
        return pd.Series(cr.classify_relative(f).to_numpy(), index=f["pixel"].to_numpy())


def cross_table(labels: pd.DataFrame | None = None, sources=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(scores, per-pixel results). ``sources``: rule sets to try, default the defaults (aoi160) plus every AOI in
    ``AOI_OVERRIDES``. Scores: one row per (target AOI, rule set) with n, full and coarse share right."""
    from . import curve_rules as cr
    from . import ndvi_5day as nd
    from .curve_labels import load

    lab = load() if labels is None else labels
    sources = [None] + sorted(cr.AOI_OVERRIDES) if sources is None else list(sources)
    rows, per = [], []
    for target in sorted(lab["aoi"].unique()):
        t = lab[lab["aoi"] == target]
        want = pd.Series([truth(r.label, r.state, r.establishment) for r in t.itertuples()], index=t["pixel"].to_numpy())
        for src in sources:
            got = judge(int(target), t["pixel"].to_numpy(), src).reindex(want.index)
            name = "defaults (aoi160)" if src is None else (src if isinstance(src, str) else f"aoi{src} rules")
            full = (got == want)
            crs_ = got.map(coarse) == want.map(coarse)
            rows.append({"labels_of": f"aoi{target}", "rule_set": name, "n": len(want),
                         "full_right": int(full.sum()), "coarse_right": int(crs_.sum())})
            per.append(pd.DataFrame({"aoi": target, "pixel": want.index, "rule_set": name, "truth": want.to_numpy(),
                                     "rule": got.to_numpy(), "full_ok": full.to_numpy(), "coarse_ok": crs_.to_numpy()}))
        nd.forget()
    scores = pd.DataFrame(rows)
    return scores, pd.concat(per, ignore_index=True)


def summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Per rule set: right / total over ALL labelled AOIs, and over the AOIs other than its own (the honest score)."""
    out = []
    for rs, g in scores.groupby("rule_set"):
        own = g["labels_of"] == rs.replace(" rules", "")
        other = g[~own] if own.any() else g[g["labels_of"] != "aoi160"] if rs.startswith("defaults") else g
        out.append({"rule_set": rs, "labels": int(g["n"].sum()),
                    "full_%": round(100 * g["full_right"].sum() / g["n"].sum(), 1),
                    "coarse_%": round(100 * g["coarse_right"].sum() / g["n"].sum(), 1),
                    "other_aois_labels": int(other["n"].sum()),
                    "other_full_%": round(100 * other["full_right"].sum() / max(other["n"].sum(), 1), 1),
                    "other_coarse_%": round(100 * other["coarse_right"].sum() / max(other["n"].sum(), 1), 1)})
    return pd.DataFrame(out).sort_values("full_%", ascending=False)


def plots_vs_trials(aoi: int, fresh: str = "processed/_batch/s2_2026/rice_fresh", inner_buffer_m: float = 10.0) -> pd.DataFrame:
    """Share of the ground-survey plots' interior pixels (plots reported as standing rice on 2 Sep 2026, shrunk by
    ``inner_buffer_m`` so edge pixels do not count) in each class, per rule set tried by ``curve_rules.try_rules``
    (``rice_fresh/aoi<N>/rule_trials/rules_<set>/``). Why (5 Oct): an AOI with survey plots can rank the rule sets
    before any pixel label, as an independent check."""
    import rasterio

    from . import curve_rules as cr
    from . import ndvi_5day as nd
    from .compare_runs import load_plots
    from .plot_curves import plot_pixels

    plots = load_plots()
    plots = plots[plots["aoi"] == f"aoi{aoi}"]
    grid = nd.load(aoi, out_root=nd.analysis_series_root(aoi))["loc"]["grid"]
    pix = plot_pixels(plots, grid, inner_buffer_m=inner_buffer_m)
    idx = np.unique(np.concatenate(list(pix.values()))) if pix else np.array([], int)
    names = {k: v[0] for k, v in cr.MAP_CLASSES.items()}
    rows = []
    for d in sorted((Path(fresh) / f"aoi{aoi}" / "rule_trials").glob("rules_*")):
        with rasterio.open(d / f"aoi{aoi}" / f"aoi{aoi}_rel_class.tif") as ds:
            v = ds.read(1).ravel()[idx]
        share = pd.Series(v).map(names).value_counts(normalize=True).mul(100).round(1)
        rows.append({"rule_set": d.name.replace("rules_", "") + " rules", "plot_pixels": len(idx), **share.to_dict()})
    out = pd.DataFrame(rows).fillna(0)
    rice = [c for c in out.columns if str(c).startswith("rice") or c == "young rice"]
    out.insert(2, "any rice %", out[rice].sum(axis=1).round(1))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", help="write the per-AOI scores here (CSV)")
    p.add_argument("--misses", help="write every pixel a rule set gets wrong here (CSV)")
    p.add_argument("--candidates", nargs="*", default=[], help=f"also try these general sets: {list(CANDIDATES)}")
    a = p.parse_args(argv)
    from . import curve_rules as cr

    scores, per = cross_table(sources=[None] + sorted(cr.AOI_OVERRIDES) + list(a.candidates))
    pd.set_option("display.width", 220)
    full = scores.assign(pct=lambda d: (100 * d.full_right / d.n).round(0)).pivot(
        index="labels_of", columns="rule_set", values="pct")
    crs_ = scores.assign(pct=lambda d: (100 * d.coarse_right / d.n).round(0)).pivot(
        index="labels_of", columns="rule_set", values="pct")
    print("full class right (%):\n", full.to_string(), "\n\ncoarse class right (%):\n", crs_.to_string())
    print("\n", summary(scores).to_string(index=False))
    if a.out:
        scores.to_csv(a.out, index=False)
    if a.misses:
        per[~per["full_ok"]].to_csv(a.misses, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
