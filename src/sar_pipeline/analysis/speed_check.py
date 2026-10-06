"""Speed check: the parallel versions of the slow steps give exactly what the one-process versions give.

Why
---
User (5 Oct 2026): audit every slow step, make it fast, and test it before going on. The rule-set comparison
(``curve_rules.try_rules``) and the variant scoring on reviewed fields (``field_review.try_variants``) now run one
process per rule set; this check runs both ways on one AOI, times them and compares the tables (and the rule-trial
rasters) cell by cell. Run it after any change to those functions. Read-only for the AOI: trials go to ``--scratch``.

Use::

    python -m sar_pipeline.analysis.speed_check --aoi 125 --scratch /tmp/speed125
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def check_trials(aoi: int, sources, scratch: Path) -> dict:
    import rasterio

    from . import curve_rules as cr

    t = time.time()
    a = cr.try_rules(aoi, sources=sources, out=str(scratch / "serial"), jobs=1)
    ts = time.time() - t
    t = time.time()
    b = cr.try_rules(aoi, sources=sources, out=str(scratch / "parallel"))
    tp = time.time() - t
    same = all(np.array_equal(rasterio.open(scratch / "serial" / f"rules_aoi{s}" / f"aoi{aoi}" /
                                            f"aoi{aoi}_rel_class.tif").read(1),
                              rasterio.open(scratch / "parallel" / f"rules_aoi{s}" / f"aoi{aoi}" /
                                            f"aoi{aoi}_rel_class.tif").read(1)) for s in sources)
    return {"step": "try_rules", "serial_s": round(ts), "parallel_s": round(tp), "tables_equal": a.equals(b),
            "rasters_equal": same}


def check_variants(aoi: int, sets) -> dict:
    from . import curve_rules as cr
    from . import field_review as fr

    variants = {"young_while_low_off": {"YOUNG_WHILE_RADAR_LOW": False}, "sowing_from_radar": {"SOWING_FROM_RADAR": True}}
    full = {f"aoi{s}": cr.AOI_OVERRIDES.get(s, {}) for s in sets}
    t = time.time()
    o1, _ = fr.try_variants(aoi, variants, sets=full, jobs=1)
    ts = time.time() - t
    t = time.time()
    o2, _ = fr.try_variants(aoi, variants, sets=full)
    tp = time.time() - t
    return {"step": "try_variants", "serial_s": round(ts), "parallel_s": round(tp), "tables_equal": o1.equals(o2)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--aoi", type=int, required=True)
    p.add_argument("--scratch", required=True)
    p.add_argument("--sources", type=int, nargs="*", default=[13, 83, 118, 116, 72, 33, 160, 28, 20])
    p.add_argument("--sets", type=int, nargs="*", default=[13, 83, 116])
    a = p.parse_args(argv)
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    rows = [check_trials(a.aoi, a.sources, scratch)]
    print(rows[-1], flush=True)
    rows.append(check_variants(a.aoi, a.sets))
    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(scratch / "speed_check.csv", index=False)
    return 0 if all(r.get("tables_equal") and r.get("rasters_equal", True) for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
