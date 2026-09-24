"""The delivered map: the rule's classes, the user's relabels, then the minimum mapping unit.

Why
---
Three things decide a delivered pixel, and they must happen in a fixed, recorded order:

1. **the rule** (``monsoon_rule``, ``<aoi>_monsoon2026.tif``), which is never edited by hand;
2. **relabels the user decided** after looking at the evidence (for example: in one AOI the "water
   unconfirmed" fields are the same crop as its confirmed rice). They live in a local config,
   ``config/class_overrides_<season>.yaml``, each with its reason and date, so a map can always be
   traced back to a decision;
3. **the sieve** (``analysis/sieve``): patches smaller than the smallest real field take the class
   that surrounds them. The size comes from the field plots (docs/13).

Output: ``<aoi>_monsoon2026_final.tif``; the delivery package is built from it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import monsoon_rule as mr
from .sieve import NODATA, sieve_classes

SRC = "processed/_batch/s2_2026"
SIEVE_PX = 4            # ~0.1 acre: 10 % of the field plots are smaller (docs/13)


def load_overrides(path="config/class_overrides_monsoon2026.yaml") -> dict:
    import yaml

    p = Path(path)
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def relabel(classes, rules) -> np.ndarray:
    """Apply ``[{from, to}, ...]`` in order."""
    out = np.asarray(classes).copy()
    for r in rules or []:
        out[np.asarray(classes) == int(r["from"])] = int(r["to"])
    return out


def finalize_aoi(aoi_id: int, overrides: dict | None = None, sieve_px: int = SIEVE_PX, src_root=SRC) -> dict:
    """Write ``<aoi>_monsoon2026_final.tif``; return acres per class before and after."""
    import rasterio

    overrides = load_overrides() if overrides is None else overrides
    src = Path(src_root) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026.tif"
    with rasterio.open(src) as ds:
        c = ds.read(1)
        profile = ds.profile.copy()
    r = relabel(c, overrides.get(f"aoi{aoi_id}"))
    f = sieve_classes(r, sieve_px)
    with rasterio.open(src.with_name(f"aoi{aoi_id}_monsoon2026_final.tif"), "w", **profile) as dst:
        dst.write(f, 1)
    row = {"aoi": f"aoi{aoi_id}", "overrides": len(overrides.get(f"aoi{aoi_id}") or []), "sieve_px": sieve_px}
    for tag, arr in (("rule", c), ("final", f)):
        counts = np.bincount(arr[arr != NODATA], minlength=mr.N_CLASSES)
        for k in range(mr.N_CLASSES):
            row[f"{mr.CLASSES[k]}_acres_{tag}"] = round(mr.acres(int(counts[k])), 1)
    return row


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis.finalize", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", nargs="*", type=int, default=None, help="default: every AOI with a rule map")
    p.add_argument("--sieve-px", type=int, default=SIEVE_PX)
    p.add_argument("--out", default=f"{SRC}/monsoon2026_final_acres.csv")
    args = p.parse_args(argv)
    ids = args.ids or sorted(int(x.parent.name[3:]) for x in Path(SRC).glob("aoi*/aoi*_monsoon2026.tif"))
    ov = load_overrides()
    t = pd.DataFrame([finalize_aoi(a, ov, args.sieve_px) for a in ids])
    t.to_csv(args.out, index=False)
    cols = [c for c in t.columns if c.endswith("_rule") or c.endswith("_final")]
    print(t[cols].sum().round(0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
