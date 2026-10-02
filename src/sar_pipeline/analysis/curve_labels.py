"""Curve labels, second fresh start (1 Oct 2026): the user labels single pixels' curves from scratch.

Why
---
User (1 Oct): "we are going to label the curves from scratch, both rice and non-rice; I give the AOI, the pixel and the
label, you store them; a completely new try". The earlier labels (plot groups, AOI groups, ``aoi<N>_pixel_verdicts.csv``)
are left untouched and NOT read here, so nothing old leaks into the new set.

Storage: one CSV, ``rice_fresh/labels_v2/curve_labels.csv``, columns ``aoi, pixel, label, note, added``. Adding the same
(aoi, pixel) again replaces its label (the user's latest word counts).

Use::

    python -m sar_pipeline.analysis.curve_labels add --aoi 39 --pixel 24703 --label rice [--note "..."]
    python -m sar_pipeline.analysis.curve_labels list
    python -m sar_pipeline.analysis.curve_labels describe --aoi 160 --pixel 31179   # the numbers behind a label
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

STORE = Path("processed/_batch/s2_2026/rice_fresh/labels_v2/curve_labels.csv")
#: ``sowing`` = the sowing / transplanting date the user confirmed (blank if not confirmed); ``establishment`` =
#: "transplanted (water)", "direct seeded" or blank (user, 1 Oct: step 2 picked 9 May for aoi160 pixel 78199, the radar
#: flood and the user say mid-June: the confirmed dates will be the reference for a better sowing rule).
#: ``state`` = "standing" / "harvested" on the newest image (user, 1 Oct: one "rice" label, details as attributes).
COLUMNS = ["aoi", "pixel", "label", "state", "sowing", "establishment", "note", "added"]
#: User (1 Oct): rice at most this many days old on the newest Sentinel-2 image is "young rice" (it replaces "late
#: rice"); every other rice is "rice", with standing / harvested, sowing and establishment as attributes.
YOUNG_MAX_DAYS = 50


def load(store: Path = STORE) -> pd.DataFrame:
    """All labels so far (empty table if none); columns added later come back blank."""
    t = pd.read_csv(store, dtype=str, keep_default_na=False) if Path(store).exists() else pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in t:
            t[c] = ""
    return t[COLUMNS].astype({"aoi": int, "pixel": int}) if len(t) else t[COLUMNS]


def add(rows, store: Path = STORE, sowing: str = "", establishment: str = "", state: str = "") -> pd.DataFrame:
    """Add labels: ``rows`` = iterable of (aoi, pixel, label[, note]). A repeated (aoi, pixel) takes the new label;
    a blank ``sowing`` / ``establishment`` keeps what was stored for that pixel."""
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    old = load(store)
    prev = {(int(a), int(p)): (s_, e, st) for a, p, s_, e, st in
            old[["aoi", "pixel", "sowing", "establishment", "state"]].itertuples(index=False)} if len(old) else {}
    new = pd.DataFrame([{"aoi": int(r[0]), "pixel": int(r[1]), "label": str(r[2]).strip(),
                         "sowing": sowing or prev.get((int(r[0]), int(r[1])), ("", "", ""))[0],
                         "establishment": establishment or prev.get((int(r[0]), int(r[1])), ("", "", ""))[1],
                         "state": state or prev.get((int(r[0]), int(r[1])), ("", "", ""))[2],
                         "note": (r[3] if len(r) > 3 else ""), "added": now} for r in rows], columns=COLUMNS)
    t = pd.concat([old, new], ignore_index=True)
    t = t.drop_duplicates(["aoi", "pixel"], keep="last").sort_values(["aoi", "pixel"]).reset_index(drop=True)
    Path(store).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(store, index=False)
    return t


def describe(aoi: int, pixel: int, series_root: str = "processed/_batch/s2_2026_hyb40m1late",
             start: str = "2026-04-01") -> dict:
    """The numbers the user and Claude read before a label (1 Oct): the fitted NDVI per 5-day window (``*`` = a clear
    view that window), every clear view with NDVI and LSWI (water: NDVI near or below 0 and LSWI above NDVI), every
    radar pass per track with VV / VH (dB), and the fresh-start steps' outputs at the pixel."""
    import numpy as np
    import rasterio

    from . import ndvi_5day as nd
    from . import radar_water as rw

    out = {"aoi": aoi, "pixel": pixel}
    fresh = Path("processed/_batch/s2_2026/rice_fresh") / f"aoi{aoi}"
    for name, f in (("step1 (1 crop, 2 trees)", "step1_cover"), ("step2 sowing", "step2_sowing"),
                    ("step3t class", "step3t_class")):
        path = fresh / f"aoi{aoi}_{f}.tif"
        if path.exists():
            with rasterio.open(path) as ds:
                v = int(ds.read(1).ravel()[pixel])
            out[name] = str((pd.Timestamp("2026-01-01") + pd.Timedelta(days=v - 1)).date()) if f == "step2_sowing" and v \
                else v
    d = nd.load(aoi, out_root=series_root)
    w = pd.DatetimeIndex(d["windows"])
    f = d["ndvi5d"].reshape(len(w), -1)[:, pixel]
    r = d["ndvi5d_raw"].reshape(len(w), -1)[:, pixel]
    keep = w >= pd.Timestamp(start)
    out["ndvi (fit, * = clear view)"] = " ".join(f"{x:%m-%d}:{f[i]:.2f}{'*' if np.isfinite(r[i]) else ''}"
                                                 for i, x in enumerate(w) if keep[i])
    dt = pd.DatetimeIndex(d["dates"])
    nv, lv, ok = (d[k].reshape(len(dt), -1)[:, pixel] for k in ("ndvi", "lswi", "ok"))
    out["clear views ndvi/lswi"] = " ".join(f"{x:%m-%d}:{nv[i]:.2f}/{lv[i]:.2f}" for i, x in enumerate(dt)
                                            if ok[i] and x >= pd.Timestamp(start))
    nd.forget()
    for j, (dd, flat) in enumerate(rw.read_series(aoi)):
        dd = pd.DatetimeIndex(dd)
        out[f"radar track {j + 1} VV/VH"] = " ".join(
            f"{x:%m-%d}:{float(flat['VV'][i, pixel]):.1f}/{float(flat['VH'][i, pixel]):.1f}"
            for i, x in enumerate(dd) if x >= pd.Timestamp(start))
    return out


def colours(aoi: int, pixels, out_png: str | None = None, half: int = 7, start: str = "2026-06-01",
            clear_min: float = 60) -> pd.DataFrame:
    """5-3-2 colour of each pixel on every clear date since ``start`` (B5, B3, B2 reflectance; the pixel's own Cloud
    Score+ >= ``clear_min``), and optionally a PNG of the chips side by side (one row per pixel, the pixel in the middle).
    Why (user, 1 Oct, aoi160 pixel 31179): rice changes colour month by month in 5-3-2 (green, dark green, then lighter,
    brown, yellow); a crop whose colour hardly changes from July to late September is suspect."""
    import numpy as np

    from .pixel_2026 import GAMMA, RGB, chip_stack

    rows, panels = [], []
    for px in pixels:
        table, chips, clear = chip_stack(aoi, int(px), half=half)
        keep = (table["date"] >= pd.Timestamp(start)) & (table["pixel_clear"] >= clear_min)
        for i in np.flatnonzero(keep.to_numpy()):
            v = chips[i][:, half, half]
            rows.append({"pixel": px, "date": table.loc[i, "date"].date(), **{b: int(x) for b, x in zip(RGB, v)},
                         "clear": int(table.loc[i, "pixel_clear"])})
        panels.append((px, table[keep].reset_index(drop=True), chips[keep.to_numpy()]))
    out = pd.DataFrame(rows)
    if out_png and rows:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ncol = max(len(t) for _, t, _ in panels)
        fig, axes = plt.subplots(len(panels), ncol, figsize=(2.2 * ncol, 2.4 * len(panels)), squeeze=False)
        lo, hi = 0.0, float(np.percentile(np.concatenate([c.ravel() for _, _, c in panels if len(c)]), 98))
        for r, (px, t, c) in enumerate(panels):
            for j in range(ncol):
                ax = axes[r, j]
                ax.axis("off")
                if j < len(t):
                    img = np.clip((np.moveaxis(c[j], 0, -1) - lo) / (hi - lo), 0, 1) ** GAMMA
                    ax.imshow(img, interpolation="nearest")
                    ax.plot(half, half, "s", mfc="none", mec="red", ms=6)
                    ax.set_title(f"{px}  {t.loc[j, 'date']:%d %b}", fontsize=8)
        fig.suptitle(f"aoi{aoi}: 5-3-2 chips, same stretch for all (pixel in the red box)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["add", "list", "describe", "colours"])
    p.add_argument("--png", help="colours: write the chip sheet here")
    p.add_argument("--aoi", type=int)
    p.add_argument("--pixel", type=int, nargs="+")
    p.add_argument("--label")
    p.add_argument("--note", default="")
    p.add_argument("--sowing", default="", help="sowing / transplanting date the user confirmed, YYYY-MM-DD")
    p.add_argument("--establishment", default="", help='"transplanted (water)" or "direct seeded"')
    p.add_argument("--state", default="", help='"standing" or "harvested" on the newest image')
    args = p.parse_args(argv)
    if args.step == "colours":
        print(colours(args.aoi, args.pixel, args.png).to_string(index=False))
        return 0
    if args.step == "describe":
        for px in args.pixel:
            for k, v in describe(args.aoi, px).items():
                print(f"{k}: {v}")
        return 0
    if args.step == "add":
        t = add([(args.aoi, px, args.label, args.note) for px in args.pixel], sowing=args.sowing,
                establishment=args.establishment, state=args.state)
    else:
        t = load()
    print(t.to_string(index=False) if len(t) else "no labels yet")
    if len(t):
        print(t.groupby("label").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
