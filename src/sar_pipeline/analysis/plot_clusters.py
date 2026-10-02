"""Group the surveyed plots by the shape of their NDVI curve, so the user can say which groups are rice.

Why
---
The ground data (plots reported as standing rice on 2 Sep 2026) is not all rice: the user judged 7 of 10 checked one region
plot pixels not rice (30 Sep). Learning "the rice curve" from all plots therefore mixes rice with other crops. Instead
(user's idea, 30 Sep): cluster the plots' curves into their distinct patterns, show each pattern, and let the user
label every pattern rice or not; the rice patterns then become the reference.

What
----
* One curve per plot: the median smoothed NDVI of its interior pixels (all its pixels when it has no interior), one
  value per 5-day window from ``START`` to the series end; beside it the median VH of the plot per window (all tracks,
  the pass nearest each window within ``REVISIT_DAYS``), for looking only.
* k-means on the NDVI curves (``K`` groups): plots with similar curves, in level and timing, fall together.
* A sheet with one panel per group: median curve, the 10-90 % band, a few member plots, the median VH, the number of
  plots and their regions; and a table plot -> group.

Use::

    python -m sar_pipeline.analysis.plot_clusters --k 20
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import ndvi_5day as nd

START = "2026-03-01"
K = 20
REVISIT_DAYS = 6
OUT = "processed/_batch/s2_2026/rice_fresh/plot_clusters"


def plot_curves(plots=None, series_root: str = "processed/_batch/s2_2026") -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """``(info, ndvi, vh)``: one row per plot (``plot_id``, ``aoi``, ``region``, ``pixels``), its NDVI curve and its VH
    curve (columns = window dates)."""
    from . import radar_water as rw
    from . import validation as va
    from .compare_runs import load_plots
    from .plot_curves import interior_pixels, plot_pixels

    plots = load_plots() if plots is None else plots
    info, ndvi_rows, vh_rows, windows = [], [], [], None
    for aoi in sorted(plots["aoi"].unique(), key=lambda a: int(a[3:])):
        aoi_id = int(aoi[3:])
        d = nd.load(aoi_id, out_root=series_root)
        g = d["loc"]["grid"]
        w = pd.DatetimeIndex(d["windows"])
        keep = w >= pd.Timestamp(START)
        windows = w[keep]
        fit = d["ndvi5d"].reshape(len(w), -1)[keep]
        pix = plot_pixels(plots[plots["aoi"] == aoi], g)
        inner = interior_pixels(pix, int(g["width"]))
        series = rw.read_series(aoi_id)
        wd = windows.to_numpy().astype("datetime64[D]")
        for pid, p in pix.items():
            use = p[inner[pid]] if inner[pid].any() else p
            if len(use) == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ndvi_rows.append(np.nanmedian(fit[:, use], axis=1))
                per_track = []
                for dates, flat in series:
                    dd = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
                    vh = np.nanmedian(np.asarray(flat["VH"])[:, use], axis=1)
                    j = np.abs(dd[:, None] - wd[None, :]).argmin(axis=0)
                    near = np.abs(dd[j] - wd) <= np.timedelta64(REVISIT_DAYS, "D")
                    per_track.append(np.where(near, vh[j], np.nan))
                vh_rows.append(np.nanmedian(np.vstack(per_track), axis=0) if per_track else np.full(len(wd), np.nan))
            info.append({"plot_id": pid, "aoi": aoi, "region": va.REGION.get(aoi, aoi), "pixels": int(len(use))})
        nd.forget()
        print(f"{aoi}: {len(pix)} plots", flush=True)
    cols = [f"{x:%Y-%m-%d}" for x in windows]
    return pd.DataFrame(info), pd.DataFrame(ndvi_rows, columns=cols), pd.DataFrame(vh_rows, columns=cols)


def cluster(ndvi: pd.DataFrame, k: int = K, seed: int = 0) -> np.ndarray:
    """k-means groups of the NDVI curves (rows with a gap get -1), numbered 1..k in the order of each group's lowest
    point in time (earliest trough first), so neighbouring numbers look alike."""
    from sklearn.cluster import KMeans

    x = ndvi.to_numpy(dtype=float)
    ok = np.isfinite(x).all(axis=1)
    labels = np.full(len(x), -1)
    if ok.sum() < k:
        return labels
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x[ok])
    order = np.argsort(km.cluster_centers_.argmin(axis=1) + 1e-3 * km.cluster_centers_.min(axis=1))
    rank = np.empty(k, dtype=int)
    rank[order] = np.arange(1, k + 1)
    labels[ok] = rank[km.labels_]
    return labels


def sheet(info: pd.DataFrame, ndvi: pd.DataFrame, vh: pd.DataFrame, labels, out: Path, seed: int = 0,
          title: str | None = None, size=None, sowing=None) -> Path:
    """One panel per group: median NDVI, 10-90 % band, six member plots, median VH (right axis), size and regions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted(set(labels) - {-1})
    days = pd.to_datetime(ndvi.columns)
    cols = 4
    rows = int(np.ceil(len(groups) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.3 * rows), squeeze=False, sharex=True)
    rng = np.random.default_rng(seed)
    x = ndvi.to_numpy(dtype=float)
    v = vh.to_numpy(dtype=float)
    for ax, gid in zip(axes.ravel(), groups):
        m = np.asarray(labels) == gid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, med, hi = np.nanpercentile(x[m], [10, 50, 90], axis=0)
            vmed = np.nanmedian(v[m], axis=0)
        ax.fill_between(days, lo, hi, color="#2a78d6", alpha=0.18)
        for i in rng.choice(np.flatnonzero(m), min(6, m.sum()), replace=False):
            ax.plot(days, x[i], color="#2a78d6", lw=0.5, alpha=0.6)
        ax.plot(days, med, color="#0b3d91", lw=2.2)
        if sowing is not None:
            # the group's sowing (step 2): median (dashed) and the middle half of its fields (shaded)
            sw = pd.to_datetime(pd.Series(np.asarray(sowing)[m])).dropna()
            if len(sw):
                q1, q2, q3 = (sw.quantile(q) for q in (0.25, 0.5, 0.75))
                ax.axvspan(q1, q3, color="#b45f06", alpha=0.12)
                ax.axvline(q2, color="#b45f06", ls="--", lw=1.4)
                ax.text(q2, 0.93, f"sowing {q2:%d %b}", color="#b45f06", fontsize=7, ha="center")
        ax.set_ylim(-0.3, 1.0)
        a2 = ax.twinx()
        a2.plot(days, vmed, ":", color="#b45f06", lw=1.5)
        a2.set_ylim(-28, -8)
        a2.tick_params(labelsize=7, colors="#b45f06")
        reg = info.loc[m, "region"].value_counts(normalize=True)
        regs = ", ".join(f"{r} {100 * f:.0f}%" for r, f in reg.head(3).items())
        what = size(int(m.sum())) if size else f"{int(m.sum())} plots"
        ax.set_title(f"group {gid}: {what} ({regs})", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[len(groups):]:
        ax.axis("off")
    fig.autofmt_xdate()
    fig.suptitle(title or "Surveyed plots grouped by NDVI curve (blue: median, band 10-90 %, thin: sample plots; "
                 "orange dotted: median VH, right axis, dB)", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


#: The share of a group's own plots that must lie inside its circle (``radii``): the circle is read from the group.
RADIUS_SHARE = 0.95
MATCH = {1: "rice (matches a rice pattern)", 2: "not rice (matches a not-rice pattern)", 3: "no matching pattern"}


def rms(a, b) -> np.ndarray:
    """Root-mean-square difference between the rows of ``a`` (n, t) and one curve or matching rows ``b``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.sqrt(np.nanmean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2, axis=-1))


def centroids(ndvi, labels, min_members: int = 10) -> dict:
    """Group -> (median curve, circle): the circle holds ``RADIUS_SHARE`` of the group's own plots."""
    x = np.asarray(ndvi, dtype=float)
    out = {}
    for gid in sorted(set(np.asarray(labels)) - {-1}):
        m = np.asarray(labels) == gid
        if m.sum() < min_members:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            c = np.nanmedian(x[m], axis=0)
        out[gid] = (c, float(np.quantile(rms(x[m], c), RADIUS_SHARE)))
    return out


def shifted_rms(x, c, shift: int) -> np.ndarray:
    """RMS difference of every row of ``x`` moved by ``shift`` windows (positive: the field is later than the pattern)
    against the pattern ``c``, over the windows both cover."""
    x = np.asarray(x, dtype=float)
    c = np.asarray(c, dtype=float)
    t = len(c)
    if shift >= 0:
        a, b = x[:, shift:], c[:t - shift]
    else:
        a, b = x[:, :t + shift], c[-shift:]
    return rms(a, b)


def assign(curves, cents: dict, rice_groups, max_shift: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(result, nearest group, distance)`` per curve (rows): result 1 rice / 2 not rice / 3 no matching pattern.
    ``max_shift`` (windows): a curve may be moved up to that many windows earlier or later before it is compared (the
    same crop sown a few weeks apart; user, 30 Sep: rice is sown from May to July); the best move counts."""
    x = np.asarray(curves, dtype=float)
    gids = list(cents)
    d = np.stack([np.min(np.stack([shifted_rms(x, cents[g][0], s) for s in range(-max_shift, max_shift + 1)]), axis=0)
                  for g in gids], axis=1)
    d = np.where(np.isfinite(d), d, np.inf)
    near = d.argmin(axis=1)
    dist = d[np.arange(len(x)), near]
    group = np.array(gids)[near]
    inside = dist <= np.array([cents[g][1] for g in gids])[near]
    rice = np.isin(group, list(rice_groups))
    result = np.where(~inside, 3, np.where(rice, 1, 2)).astype("uint8")
    return result, group, dist


def check_patterns(info: pd.DataFrame, ndvi: pd.DataFrame, labels, verdicts: pd.DataFrame, max_shift: int = 0) -> pd.DataFrame:
    """Leave one region out: the group curves and circles come from the other regions' plots only."""
    rice_groups = set(verdicts.loc[verdicts["rice"].astype(bool), "group"])
    lab = np.asarray(labels)
    rows = []
    for region in sorted(info["region"].unique()):
        held = (info["region"] == region).to_numpy()
        cents = centroids(ndvi.to_numpy()[~held], lab[~held])
        res, _, _ = assign(ndvi.to_numpy()[held], cents, rice_groups, max_shift)
        truth = np.isin(lab[held], list(rice_groups))
        for kind, m in (("rice groups", truth), ("not-rice groups", ~truth & (lab[held] > 0))):
            if m.sum():
                rows.append({"shift_days": max_shift * 5, "region": region, "plots": kind, "n": int(m.sum()),
                             **{MATCH[c]: round(100 * float((res[m] == c).mean()), 1) for c in MATCH}})
    return pd.DataFrame(rows)


def load_cache(out: Path = Path(OUT)) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    both = pd.read_parquet(out / "plot_curves.parquet")
    info = both[["plot_id", "aoi", "region", "pixels"]]
    return (info, both.filter(like="ndvi_").rename(columns=lambda c: c[5:]),
            both.filter(like="vh_").rename(columns=lambda c: c[3:]))


def run_aoi(aoi_id: int, k: int = K, series_root: str = "processed/_batch/s2_2026_hyb40m1late",
            fresh: str = "processed/_batch/s2_2026/rice_fresh", max_shift: int = 0) -> pd.DataFrame:
    """Step 3 by pattern: every crop-ground pixel (step 1) matched to the user-labelled plot patterns. Writes
    ``aoi<N>_step3p_class.tif`` (0 not vegetation, 1 rice, 2 not rice, 3 no matching pattern, 5 trees) + ``.qml``,
    ``aoi<N>_step3p_group.tif`` (nearest group) and the acre table."""
    import rasterio

    from .optical_phenology import acres

    info, ndvi, _ = load_cache()
    labels = pd.read_csv(Path(OUT) / f"plot_groups_k{k}.csv")["group"].to_numpy()
    verdicts = pd.read_csv(Path(OUT) / f"group_verdicts_k{k}.csv")
    cents = centroids(ndvi.to_numpy(), labels)
    stem = Path(fresh) / f"aoi{aoi_id}" / f"aoi{aoi_id}_step"
    with rasterio.open(f"{stem}1_cover.tif") as ds:
        cover = ds.read(1).ravel()
        profile = ds.profile
    d = nd.load(aoi_id, out_root=series_root)
    w = pd.DatetimeIndex(d["windows"])
    cols = pd.to_datetime(ndvi.columns)
    idx = np.searchsorted(w, cols)
    ok = (idx < len(w)) & (w[np.clip(idx, 0, len(w) - 1)] == cols)
    fit = d["ndvi5d"].reshape(len(w), -1)
    crop = cover == 1
    sub = fit[idx[ok]][:, crop].T
    tmp = np.full((int(crop.sum()), len(cols)), np.nan, dtype="float32")
    tmp[:, ok] = sub
    nd.forget()
    res, group, dist = assign(tmp, cents, set(verdicts.loc[verdicts["rice"].astype(bool), "group"]), max_shift)
    out = np.where(cover == 255, 255, np.where(cover == 2, 5, 0)).astype("uint8")
    out[crop] = res
    grp = np.zeros(cover.size, dtype="uint8")
    grp[crop] = group
    shape = (profile["height"], profile["width"])
    with rasterio.open(f"{stem}3p_class.tif", "w", **dict(profile, dtype="uint8", nodata=255, count=1)) as ds:
        ds.write(out.reshape(shape)[None])
    with rasterio.open(f"{stem}3p_group.tif", "w", **dict(profile, dtype="uint8", nodata=0, count=1)) as ds:
        ds.write(grp.reshape(shape)[None])
    colours = {0: "#d9d9d9", 1: "#1a9850", 2: "#e3a21a", 3: "#9ecae1", 5: "#1b5e20"}
    names = {0: "not vegetation in September", **MATCH, 5: "trees / orchards (step 1)"}
    entries = "\n".join(f'        <paletteEntry value="{c}" color="{col}" alpha="255" label="{c} {names[c]}"/>'
                         for c, col in colours.items())
    Path(f"{stem}3p_class.qml").write_text(f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" nodataColor="">
      <colorPalette>
{entries}
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
""")
    t = pd.DataFrame([{"class": c, "name": names[c], "acres": round(acres(int((out == c).sum())), 1)} for c in colours])
    t.to_csv(f"{stem}3p.csv", index=False)
    g = pd.Series(group[res < 3]).value_counts().sort_index()
    t.attrs["nearest_groups_acres"] = {int(k2): round(acres(int(v)), 1) for k2, v in g.items()}
    return t


def aoi_groups(aoi_id: int, k: int = K, series_root: str = "processed/_batch/s2_2026_hyb40m1late",
               fresh: str = "processed/_batch/s2_2026/rice_fresh", sample: int = 20000, seed: int = 0,
               force: bool = False) -> Path:
    """Group the crop-ground pixels (step 1) of one AOI by their NDVI curve, the same way as the plots, for the user to
    label; the sheet titles show what the learned trait tree (``rice_features``, step 3t) calls each group.
    Writes ``aoi<N>_groups_k<k>.tif`` + ``.qml`` (group per pixel) and ``aoi<N>_groups_k<k>.png``. Why (30 Sep): the tree
    only knows the kinds of non-rice it was shown; the AOI's own patterns, labelled, add the missing kinds."""
    import matplotlib
    import rasterio
    from sklearn.cluster import KMeans

    from .rice_features import pixel_vh

    labelled = Path(fresh) / f"aoi{aoi_id}" / f"aoi{aoi_id}_group_verdicts_k{k}.csv"
    if labelled.exists() and not force:
        # the user's verdicts name these group numbers: new groups would silently re-point them (1 Oct, aoi13)
        print(f"aoi{aoi_id}: groups kept, they are labelled ({labelled.name}); force=True to rebuild")
        return Path(fresh) / f"aoi{aoi_id}" / f"aoi{aoi_id}_groups_k{k}.png"

    both = pd.read_parquet(Path(OUT) / "plot_curves.parquet")
    dates = pd.to_datetime([c[5:] for c in both.filter(like="ndvi_").columns])
    stem = Path(fresh) / f"aoi{aoi_id}" / f"aoi{aoi_id}_"
    with rasterio.open(f"{stem}step1_cover.tif") as ds:
        cover = ds.read(1).ravel()
        profile = ds.profile
    with rasterio.open(f"{stem}step3t_class.tif") as ds:
        tree_cls = ds.read(1).ravel()
    crop = np.flatnonzero(cover == 1)
    d = nd.load(aoi_id, out_root=series_root)
    w = pd.DatetimeIndex(d["windows"])
    x = d["ndvi5d"].reshape(len(w), -1)[np.searchsorted(w, dates)][:, crop].T.astype("float32")
    nd.forget()
    vh = pixel_vh(aoi_id, dates)[crop]
    ok = np.isfinite(x).all(axis=1)
    rng = np.random.default_rng(seed)
    fit_rows = np.flatnonzero(ok)
    fit_rows = rng.choice(fit_rows, min(sample, len(fit_rows)), replace=False)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x[fit_rows])
    order = np.argsort(km.cluster_centers_.argmin(axis=1) + 1e-3 * km.cluster_centers_.min(axis=1))
    rank = np.empty(k, dtype=int)
    rank[order] = np.arange(1, k + 1)
    labels = np.full(len(crop), -1)
    labels[ok] = rank[km.predict(x[ok])]
    names = pd.Series(tree_cls[crop]).map({1: "monsoon rice", 6: "late rice", 2: "not rice", 7: "flooded",
                                            8: "weak"}).fillna("other").to_numpy()
    with rasterio.open(f"{stem}step2_sowing.tif") as ds:
        doy = ds.read(1).ravel()[crop]
    sowing = pd.Series(pd.Timestamp("2026-01-01") + pd.to_timedelta(doy.astype(int) - 1, "D")).where(doy > 0)
    info = pd.DataFrame({"region": names})
    cols = [f"{t:%Y-%m-%d}" for t in dates]
    from .optical_phenology import acres

    sheet(info, pd.DataFrame(x, columns=cols), pd.DataFrame(vh, columns=cols), labels, Path(f"{stem}groups_k{k}.png"),
          title=f"aoi{aoi_id} crop ground grouped by NDVI curve (blue: median, band 10-90 %, thin: sample pixels; "
                "orange dotted: median VH, right axis, dB; brown: sowing, median and middle half)",
          size=lambda n: f"{acres(n):.0f} ac", sowing=sowing)
    grp = np.zeros(cover.size, dtype="uint8")
    grp[crop] = np.clip(labels, 0, 255)
    shape = (profile["height"], profile["width"])
    with rasterio.open(f"{stem}groups_k{k}.tif", "w", **dict(profile, dtype="uint8", nodata=0, count=1)) as ds:
        ds.write(grp.reshape(shape)[None])
    cmap = matplotlib.colormaps["tab20"]
    entries = "\n".join(f'        <paletteEntry value="{g}" color="{matplotlib.colors.to_hex(cmap((g - 1) % 20))}" '
                         f'alpha="255" label="group {g}"/>' for g in range(1, k + 1))
    Path(f"{stem}groups_k{k}.qml").write_text(f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" nodataColor="">
      <colorPalette>
{entries}
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
""")
    return Path(f"{stem}groups_k{k}.png")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--series-root", default="processed/_batch/s2_2026")
    p.add_argument("--check", action="store_true", help="leave-one-region-out check of the pattern matching")
    p.add_argument("--aoi", type=int, help="step 3 by pattern for this AOI")
    p.add_argument("--shifts", nargs="+", type=int, default=[0], help="max shifts to try, in days (multiples of 5)")
    p.add_argument("--aoi-groups", type=int, metavar="AOI", help="group this AOI's crop-ground pixels for labelling")
    args = p.parse_args(argv)
    if args.aoi_groups:
        print(aoi_groups(args.aoi_groups, args.k))
        return 0
    if args.check or args.aoi:
        info, ndvi, _ = load_cache()
        labels = pd.read_csv(Path(OUT) / f"plot_groups_k{args.k}.csv")["group"].to_numpy()
        verdicts = pd.read_csv(Path(OUT) / f"group_verdicts_k{args.k}.csv")
        if args.check:
            t = pd.concat([check_patterns(info, ndvi, labels, verdicts, sh // 5) for sh in args.shifts])
            t.to_csv(Path(OUT) / f"pattern_check_k{args.k}.csv", index=False)
            summary = t[t["region"] != "aoi148"].groupby(["shift_days", "plots"]).apply(
                lambda g: pd.Series({c: round(float(np.average(g[c], weights=g["n"])), 1) for c in MATCH.values()}))
            print(t.to_string(index=False))
            print(summary.to_string())
        if args.aoi:
            t = run_aoi(args.aoi, args.k, max_shift=args.shifts[-1] // 5)
            print(t.to_string(index=False))
            print("acres by nearest group (matched pixels):", t.attrs["nearest_groups_acres"])
        return 0
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "plot_curves.parquet"
    if cache.exists():
        both = pd.read_parquet(cache)
        info = both[["plot_id", "aoi", "region", "pixels"]]
        ndvi = both.filter(like="ndvi_").rename(columns=lambda c: c[5:])
        vh = both.filter(like="vh_").rename(columns=lambda c: c[3:])
    else:
        info, ndvi, vh = plot_curves(series_root=args.series_root)
        pd.concat([info, ndvi.add_prefix("ndvi_"), vh.add_prefix("vh_")], axis=1).to_parquet(cache)
    labels = cluster(ndvi, args.k)
    info.assign(group=labels).to_csv(out / f"plot_groups_k{args.k}.csv", index=False)
    print(sheet(info, ndvi, vh, labels, out / f"plot_groups_k{args.k}.png"))
    print(pd.Series(labels).value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
