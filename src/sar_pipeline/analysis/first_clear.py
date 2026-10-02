"""Per pixel, the FIRST cloud-free Sentinel-2 view from a start date on, and one image made of those views.

Why
---
Fresh start (user, 30 Sep 2026): instead of a long chain of rules, begin from what can be seen. From the start of
September, walk through the scenes in date order and, for every pixel, keep the first view that is clear. The result is
one gap-free-as-possible image of the AOI "as first seen clearly in September", plus a raster saying which date each
pixel came from, so the user can check it in QGIS against the single scenes.

"Clear" is the mask the current series uses (the M1 mask, ``mask_experiment.VARIANTS["hyb40m1"]``): data present
(B8 = 0 over water counts as data), no QA60 opaque-cloud bit, Cloud Score+ ``clear`` at or above 40 unless the pixel is
dark (water / wet soil), no blue-band haze, and a date whose Cloud Score+ was not produced yet is judged without it.

Outputs, in ``<out_root>/first_clear/aoi<N>/`` (``<tag>`` = e.g. ``2026-09-01_2026-09-30``):

* ``aoi<N>_first_clear_<tag>.tif``: bands B2, B3, B4, B5, B8, B11, B12 (reflectance x 10000, uint16, 0 = never clear)
  + ``.qml`` (true colour 4-3-2);
* ``aoi<N>_first_clear_<tag>_date.tif``: day of the month of the chosen scene (0 = never clear) + ``.qml``;
* ``aoi<N>_first_clear_<tag>.csv``: per scene, the pixels and acres it filled and its own clear share;
* ``aoi<N>_first_clear_<tag>.png``: 4-3-2, 8-3-2, 5-3-2 and the date map side by side;
* with ``--scenes``: ``aoi<N>_scenes_<start>_<end>.png``, every scene with its Cloud Score+ and the mask;
* with ``--vegetation`` (on an existing image): the image with only its vegetated pixels (``..._vegetation.tif``,
  mask ``..._vegetation_mask.tif``, PNG); vegetated = NDVI above the image's own Otsu split (no fixed NDVI); pixels never
  clear in the month are decided by the radar (``radar_vegetation``), recorded in ``..._vegetation_source.tif``.

Use::

    python -m sar_pipeline.analysis.first_clear --aoi 160 --start 2026-09-01 --end 2026-09-30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import ndvi_5day as nd
from .optical_phenology import acres

BANDS = ("B2", "B3", "B4", "B5", "B8", "B11", "B12")
#: The mask of the current series (fix round 3, M1).
MASK = {"qa60_mode": "opaque", "cs_min": 40, "keep_dark": True, "drop_haze": True, "cs_missing_unknown": True,
        "b8_zero_water": True}


def first_clear(ok) -> np.ndarray:
    """Index of the first True along axis 0 of ``ok`` (dates, ...) per pixel; -1 where never True."""
    ok = np.asarray(ok, dtype=bool)
    return np.where(ok.any(axis=0), ok.argmax(axis=0), -1)


def compose(stack, index) -> np.ndarray:
    """(bands, rows, cols) taken per pixel from ``stack`` (dates, bands, rows, cols) at ``index`` (rows, cols); 0 where
    ``index`` is -1."""
    stack = np.asarray(stack)
    idx = np.clip(index, 0, None)[None, None]
    out = np.take_along_axis(stack, np.broadcast_to(idx, (1,) + stack.shape[1:]), axis=0)[0]
    return np.where(index[None] >= 0, out, 0).astype(stack.dtype)


def relative_clear(scores, base_ok, missing, inside, k: float | None = None) -> np.ndarray:
    """(dates, rows, cols): the views that count as clear under the RELATIVE Cloud Score+ rule (user, 30 Sep).

    A view is clear when it passes ``base_ok`` (data, no opaque-cloud bit, no haze), its scene had a Cloud Score+
    (``missing`` per date = unknown, never taken) and its score is not ``k`` x ``s`` below the pixel's own best score of
    the period. ``s`` = how much the score of truly clear pixels varies: the robust spread (1.4826 x MAD) of the scores
    of the AOI pixels whose best view is that same scene, taken on that scene. No fixed score: a hazy 50 is not clear
    where the same pixel reached 90 on another date; where no date is good, the pixel's best view is still taken."""
    from .radar_water import WATER_K

    k = WATER_K if k is None else k
    cs = np.asarray(scores, dtype="float32")
    valid = np.asarray(base_ok, dtype=bool) & ~np.asarray(missing, dtype=bool)[:, None, None]
    masked = np.where(valid, cs, -np.inf)
    best = masked.max(axis=0)
    best_idx = masked.argmax(axis=0)
    has = np.isfinite(best) & np.asarray(inside, dtype=bool)
    spread = np.zeros(cs.shape[0], dtype="float32")
    for j in range(cs.shape[0]):
        v = best[has & (best_idx == j)]
        if len(v):
            spread[j] = 1.4826 * np.median(np.abs(v - np.median(v)))
    tol = k * spread[best_idx]
    return valid & has[None] & (cs >= (best - tol)[None])


def clear_of(ds, mask=MASK) -> tuple[np.ndarray, dict]:
    """(clear bool (rows, cols), bands dict) of one exported scene under ``mask`` (see the module docstring). The bands
    dict also carries ``clear`` (Cloud Score+) and ``cs_missing`` (the scene had no Cloud Score+ yet)."""
    from ..optical_export import band_index

    read = lambda n: ds.read(band_index(ds, n)).astype("float32")  # noqa: E731
    b = {n: read(n) for n in BANDS}
    qa, cs = read("QA60"), read("clear")
    data = nd.data_mask(b["B3"], b["B4"], b["B8"], mask["b8_zero_water"])
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(data, (b["B8"] - b["B4"]) / (b["B8"] + b["B4"]), np.nan)
    cs_min = None if (mask["cs_missing_unknown"] and nd.cloud_score_missing(cs, data)) else mask["cs_min"]
    bits = (1 << 10) | (1 << 11) if mask["qa60_mode"] == "both" else (1 << 10)
    ok = nd.clear_mask(data, qa, cs, b["B8"], ndvi, cs_min, bits, mask["keep_dark"], mask["drop_haze"], b["B2"], b["B4"])
    b["clear"], b["cs_missing"] = cs, nd.cloud_score_missing(cs, data)
    return ok, b


def run(aoi_id: int, start: str = "2026-09-01", end: str = "2026-09-30", out_root: str = "processed/_batch/s2_2026",
        folder: str = nd.FOLDER, relative: bool = True) -> pd.DataFrame:
    """Build the first-clear image of one AOI between ``start`` and ``end`` (inclusive); return the per-scene table.
    ``relative`` (default, user 30 Sep): Cloud Score+ judged against each pixel's own best score of the period
    (:func:`relative_clear`) instead of the mask's fixed 40."""
    import rasterio

    paths = sorted(Path(f"data/{folder}/aoi{aoi_id}").glob("*.tif"))
    day = lambda p: pd.Timestamp(p.stem.rsplit("_S2_", 1)[1])  # noqa: E731
    paths = sorted((p for p in paths if pd.Timestamp(start) <= day(p) <= pd.Timestamp(end)), key=day)
    if not paths:
        raise FileNotFoundError(f"no scenes for aoi{aoi_id} between {start} and {end}")
    inside = nd.inside_aoi(aoi_id)
    nd.forget()
    oks, stacks, dates, scores, missing = [], [], [], [], []
    for p in paths:
        with rasterio.open(p) as ds:
            ok, b = clear_of(ds, dict(MASK, cs_min=None) if relative else MASK)
            profile = ds.profile
        oks.append(ok)
        stacks.append(np.stack([b[n] for n in BANDS]).astype("uint16"))
        scores.append(b["clear"])
        missing.append(b["cs_missing"])
        dates.append(day(p))
    shape = oks[0].shape
    inside = inside.reshape(shape)
    ok = np.stack(oks) & inside[None]
    if relative:
        ok = relative_clear(np.stack(scores), ok, np.array(missing), inside)
    index = first_clear(ok)
    image = compose(np.stack(stacks), index)
    dom = np.where(index >= 0, np.array([d.day for d in dates])[np.clip(index, 0, None)], 0).astype("uint8")

    tag = f"{pd.Timestamp(start):%Y-%m-%d}_{pd.Timestamp(end):%Y-%m-%d}"
    out = Path(out_root) / "first_clear" / f"aoi{aoi_id}"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"aoi{aoi_id}_first_clear_{tag}"
    prof = dict(profile, count=len(BANDS), dtype="uint16", nodata=0, compress="deflate")
    with rasterio.open(f"{stem}.tif", "w", **prof) as ds:
        ds.write(image)
        ds.descriptions = BANDS
    with rasterio.open(f"{stem}_date.tif", "w", **dict(profile, count=1, dtype="uint8", nodata=0, compress="deflate")) as ds:
        ds.write(dom[None])
    n_in = int(inside.sum())
    table = pd.DataFrame({
        "date": [d.date() for d in dates],
        "clear_share_of_aoi": [round(float((o & inside).sum()) / max(n_in, 1), 3) for o in ok],
        "first_clear_pixels": [int((index == i).sum()) for i in range(len(dates))],
        "cloud_score_missing": missing,
        # Cloud Score+ of the pixels this scene gave to the image (a low value = taken although the score says cloud:
        # the dark-pixel exception or a missing score let it through)
        "cloud_score_taken_p10": [float(np.percentile(sc[index == i], 10)) if (index == i).any() else np.nan
                                  for i, sc in enumerate(scores)],

    })
    table["first_clear_acres"] = table["first_clear_pixels"].map(lambda v: round(acres(v), 1))
    never = int(((index < 0) & inside).sum())
    table.loc[len(table)] = {"date": "never clear", "first_clear_pixels": never, "first_clear_acres": round(acres(never), 1)}
    table.to_csv(f"{stem}.csv", index=False)
    write_qml(Path(f"{stem}.qml"), Path(f"{stem}_date.qml"), dates)
    preview(image, dom, inside, dates, Path(f"{stem}.png"), f"aoi{aoi_id}: first clear view per pixel, {tag}")
    return table


def write_qml(rgb_path: Path, date_path: Path, dates) -> None:
    """QGIS styles: true colour 4-3-2 (bands 3, 2, 1 of the file) and a colour per scene date."""
    band = lambda tag: (f"<{tag}ContrastEnhancement><minValue>0</minValue><maxValue>2000</maxValue>"  # noqa: E731
                        f"<algorithm>StretchToMinimumMaximum</algorithm></{tag}ContrastEnhancement>")
    rgb_path.write_text(f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="multibandcolor" redBand="3" greenBand="2" blueBand="1" opacity="1">
      {band("red")}{band("green")}{band("blue")}
    </rasterrenderer>
  </pipe>
</qgis>
""")
    colours = date_colours(len(dates))
    entries = "\n".join(f'        <paletteEntry value="{d.day}" color="{c}" alpha="255" label="{d:%d %b}"/>'
                        for d, c in zip(dates, colours))
    date_path.write_text(f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
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


def date_colours(n: int) -> list[str]:
    """``n`` colours from early (light) to late (dark) on one hue ramp, so later dates read darker."""
    import matplotlib

    cmap = matplotlib.colormaps["viridis_r"]
    return [matplotlib.colors.to_hex(cmap(0.1 + 0.8 * i / max(n - 1, 1))) for i in range(n)]


def stretch(rgb, inside) -> np.ndarray:
    """(rows, cols, 3) in 0-1: each band stretched between its 2nd and 98th percentile inside the AOI."""
    out = np.zeros(rgb.shape[1:] + (3,), dtype="float32")
    for i in range(3):
        v = rgb[i].astype("float32")
        good = inside & (v > 0)
        if not good.any():
            continue
        lo, hi = np.percentile(v[good], (2, 98))
        out[..., i] = np.clip((v - lo) / max(hi - lo, 1), 0, 1)
    out[~inside] = 1.0
    return out


def preview(image, dom, inside, dates, path: Path, title: str) -> None:
    """A PNG with the three band combinations and the date map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    k = {n: i for i, n in enumerate(BANDS)}
    combos = (("True colour 4-3-2", ("B4", "B3", "B2")), ("False colour 8-3-2", ("B8", "B3", "B2")),
              ("5-3-2", ("B5", "B3", "B2")))
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.2))
    never = inside & (dom == 0)
    for ax, (name, bands) in zip(axes, combos):
        rgb = stretch(np.stack([image[k[b]] for b in bands]), inside)
        rgb[never] = (1.0, 0.0, 1.0)                       # magenta: never clear in the period
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")
    colours = date_colours(len(dates))
    lut = np.full((32, 4), 1.0)
    for d, c in zip(dates, colours):
        lut[d.day] = matplotlib.colors.to_rgba(c)
    lut[0] = (1.0, 0.0, 1.0, 1.0)
    img = lut[dom]
    img[~inside] = (1, 1, 1, 1)
    axes[3].imshow(img, interpolation="nearest")
    axes[3].set_title("Date of the view used")
    axes[3].axis("off")
    n_in = max(int(inside.sum()), 1)
    handles = [Patch(color=c, label=f"{d:%d %b}: {100 * ((dom == d.day) & inside).sum() / n_in:.0f} %")
               for d, c in zip(dates, colours) if ((dom == d.day) & inside).any()]
    handles.append(Patch(color=(1, 0, 1), label=f"never clear: {100 * never.sum() / n_in:.1f} %"))
    axes[3].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def scenes_sheet(aoi_id: int, start: str = "2026-09-01", end: str = "2026-09-30", out_root: str = "processed/_batch/s2_2026",
                 folder: str = nd.FOLDER) -> Path:
    """Every scene of the period in true colour (fixed stretch 0-2000, comparable between dates) above its Cloud Score+
    (0-100), inside the AOI. Why: to see with one's own eyes whether a view the mask called clear really is."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio

    from ..optical_export import band_index

    paths = sorted(Path(f"data/{folder}/aoi{aoi_id}").glob("*.tif"))
    day = lambda p: pd.Timestamp(p.stem.rsplit("_S2_", 1)[1])  # noqa: E731
    paths = sorted((p for p in paths if pd.Timestamp(start) <= day(p) <= pd.Timestamp(end)), key=day)
    inside = nd.inside_aoi(aoi_id)
    nd.forget()
    fig, axes = plt.subplots(3, len(paths), figsize=(3.2 * len(paths), 9.6), squeeze=False)
    for j, p in enumerate(paths):
        with rasterio.open(p) as ds:
            rgb = np.stack([ds.read(band_index(ds, n)).astype("float32") for n in ("B4", "B3", "B2")])
            cs = ds.read(band_index(ds, "clear")).astype("float32")
            ok, _ = clear_of(ds)
        ins = inside.reshape(cs.shape)
        img = np.clip(np.moveaxis(rgb, 0, -1) / 2000.0, 0, 1)
        img[~ins] = 1.0
        axes[0, j].imshow(img, interpolation="nearest")
        axes[0, j].set_title(f"{day(p):%d %b}")
        axes[1, j].imshow(np.where(ins, cs, np.nan), vmin=0, vmax=100, cmap="viridis", interpolation="nearest")
        axes[1, j].set_title("Cloud Score+ (0 cloud - 100 clear)", fontsize=8)
        axes[2, j].imshow(np.where(ins, ok, np.nan), vmin=0, vmax=1, cmap="Greys_r", interpolation="nearest")
        axes[2, j].set_title(f"series mask (CS>=40) clear: {100 * (ok & ins).sum() / max(ins.sum(), 1):.0f} %", fontsize=8)
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle(f"aoi{aoi_id}: every scene {start} to {end} (true colour, Cloud Score+, current mask)")
    fig.tight_layout()
    out = Path(out_root) / "first_clear" / f"aoi{aoi_id}" / f"aoi{aoi_id}_scenes_{start}_{end}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def otsu(values, bins: int = 256) -> float:
    """The value that splits ``values`` into two groups with the smallest spread inside each group (Otsu's method).
    Why: the boundary between bare / water and vegetation is read from the image itself, not fixed in advance."""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    hist, edges = np.histogram(v, bins=bins)
    centre = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * centre)
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = s0 / w0
        m1 = (s0[-1] - s0) / w1
        between = w0 * w1 * (m0 - m1) ** 2
    between = np.nan_to_num(between[:-1], nan=-1.0)
    # two well separated groups give a flat top across the empty gap: take the middle of it
    top = np.flatnonzero(between >= between.max() * (1 - 1e-9))
    return float(centre[top].mean())


def radar_vegetation(series, since: str = "2026-05-01", month_start: str = "2026-09-01", k: float | None = None) -> np.ndarray:
    """Per pixel: does the radar show a canopy in September? On some track and polarisation (VH or VV), the median of its passes from
    ``month_start`` lies ``k`` x its own pass-to-pass wobble above its own second-darkest pass since ``since`` (the crop
    grew since the field was bare or flooded). Why (user, 1 Oct): where September had no clear optical view at all
    (aoi13), the radar decides. A tree that never went low reads "no rise" here: such pixels are not crop ground."""
    from .radar_water import WATER_K, pass_noise

    k = WATER_K if k is None else k
    out = None
    # VV too (user, 1 Oct, aoi13 pixel 10120: VH noisy, VV up 10 dB from July to September, the double bounce of a
    # transplanted paddy)
    for dates, flat, pol in ((dt_, fl, p) for dt_, fl in series for p in ("VH", "VV") if p in fl):
        d = pd.DatetimeIndex(dates)
        x = np.asarray(flat[pol], dtype="float32")[d >= pd.Timestamp(since)]
        dd = d[d >= pd.Timestamp(since)]
        if len(dd) < 3 or not (dd >= pd.Timestamp(month_start)).any():
            continue
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            low = np.partition(np.where(np.isfinite(x), x, np.inf), 1, axis=0)[1]
            end = np.nanmedian(x[dd >= pd.Timestamp(month_start)], axis=0)
            noise = pass_noise(x)
        with np.errstate(invalid="ignore"):
            rose = np.isfinite(low) & (end >= low + k * noise)
        out = rose if out is None else (out | rose)
    return out


def radar_overrule(seen, veg, inside, rv, ndvi):
    """Pixels whose one clear view says "no vegetation" but whose radar shows a canopy: the radar wins (user, 1 Oct,
    aoi13 pixel 16244: NDVI 0.10 on its only clear view, VH and VV both up ~8 dB since the July water; a young paddy
    also reads low) -- but never over open water: NDVI below 0 is water, which haze cannot fake, while wind on water
    lifts the radar a few dB (user, 1 Oct, aoi39 pixel 7636: NDVI -0.37 on 11 Sep, VH +2.6 dB; 78 ac of water became
    rice)."""
    with np.errstate(invalid="ignore"):
        water = np.asarray(ndvi) < 0
    return seen & ~veg & inside & rv & ~water


def vegetation_only(aoi_id: int, start: str = "2026-09-01", end: str = "2026-09-30",
                    out_root: str = "processed/_batch/s2_2026") -> dict:
    """Keep only the vegetated pixels of the first-clear image (user, 30 Sep: any vegetation, crops, orchards or trees;
    everything else masked). Vegetated = NDVI of the pixel's view above the image's own Otsu split of NDVI inside the
    AOI. Writes ``aoi<N>_first_clear_<tag>_vegetation.tif`` (image, 0 = masked), ``..._vegetation_mask.tif`` (1 / 0),
    and a PNG."""
    import rasterio

    tag = f"{pd.Timestamp(start):%Y-%m-%d}_{pd.Timestamp(end):%Y-%m-%d}"
    stem = Path(out_root) / "first_clear" / f"aoi{aoi_id}" / f"aoi{aoi_id}_first_clear_{tag}"
    with rasterio.open(f"{stem}.tif") as ds:
        image = ds.read()
        profile = ds.profile
    k = {n: i for i, n in enumerate(BANDS)}
    b4, b8 = image[k["B4"]].astype("float32"), image[k["B8"]].astype("float32")
    seen = image[k["B3"]] > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(seen & (b8 + b4 > 0), (b8 - b4) / (b8 + b4), np.nan)
    split = otsu(ndvi[seen]) if seen.any() else np.nan
    # never above the ground data's bare level (~0.30): where nearly everything is green Otsu cuts the green in two
    # (aoi28: split 0.61, 29 ac of green fields "not vegetation")
    from .vegetation_types import ground_bare_level

    bare = ground_bare_level()
    if np.isfinite(bare) and np.isfinite(split):
        split = min(split, bare)
    veg = seen & (ndvi > split)
    # never seen clearly in the month: the radar decides (user, 1 Oct)
    inside = nd.inside_aoi(aoi_id).reshape(seen.shape)
    nd.forget()
    unseen = inside & ~seen
    source = np.where(seen, 1, 0).astype("uint8")
    from . import radar_water as rw

    rv = radar_vegetation(rw.read_series(aoi_id), month_start=start)
    if rv is not None:
        rv = rv.reshape(seen.shape)
        veg = veg | (unseen & rv)
        source[unseen] = 2
        # one clear view says "no vegetation", the radar shows a canopy: the radar wins (user, 1 Oct, aoi13 pixel 16244:
        # NDVI 0.10 on its only clear view, VH and VV both up ~8 dB since the July water; a young paddy also reads low)
        overrule = radar_overrule(seen, veg, inside, rv, ndvi)
        veg = veg | overrule
        source[overrule] = 3
    decided = seen | (source == 2)
    with rasterio.open(f"{stem}_vegetation.tif", "w", **profile) as ds:
        ds.write(np.where(veg[None], image, 0).astype(image.dtype))
        ds.descriptions = BANDS
    with rasterio.open(f"{stem}_vegetation_mask.tif", "w", **dict(profile, count=1, dtype="uint8", nodata=255)) as ds:
        ds.write(np.where(decided, veg, 255).astype("uint8")[None])
    with rasterio.open(f"{stem}_vegetation_source.tif", "w", **dict(profile, count=1, dtype="uint8", nodata=0)) as ds:
        ds.write(source[None])          # 1 = optical view, 2 = radar (no clear view), 3 = radar canopy overruled the view
    Path(f"{stem}_vegetation.qml").write_text(Path(f"{stem}.qml").read_text())
    if seen.any():
        vegetation_preview(image, ndvi, seen, veg, split, Path(f"{stem}_vegetation.png"), f"aoi{aoi_id}, {tag}")
    n = int(decided.sum())
    return {"ndvi_split": round(float(split), 3), "vegetation_acres": round(acres(int(veg.sum())), 1),
            "masked_acres": round(acres(n - int(veg.sum())), 1), "vegetation_share": round(veg.sum() / max(n, 1), 3),
            "decided_by_radar_acres": round(acres(int((source == 2).sum())), 1),
            "radar_overruled_view_acres": round(acres(int((source == 3).sum())), 1),
            "radar_vegetation_acres": round(acres(int((veg & (source == 2)).sum())), 1)}


def vegetation_preview(image, ndvi, seen, veg, split, path: Path, title: str) -> None:
    """True colour before / after the vegetation mask and the NDVI histogram with the split."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = {n: i for i, n in enumerate(BANDS)}
    rgb = stretch(np.stack([image[k[b]] for b in ("B4", "B3", "B2")]), seen)
    after = rgb.copy()
    after[seen & ~veg] = 1.0
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    axes[0].imshow(rgb, interpolation="nearest")
    axes[0].set_title("First clear view, true colour")
    axes[1].imshow(after, interpolation="nearest")
    axes[1].set_title(f"Vegetation only ({100 * veg.sum() / max(seen.sum(), 1):.0f} % kept, white = masked)")
    for a in axes[:2]:
        a.axis("off")
    axes[2].hist(ndvi[seen], bins=120, color="#4a7d4a")
    axes[2].axvline(split, color="#b45f06", lw=2, label=f"split from this image: NDVI {split:.2f}")
    axes[2].set_xlabel("NDVI of the pixel's view")
    axes[2].set_ylabel("pixels")
    axes[2].legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--aoi", type=int, required=True)
    p.add_argument("--start", default="2026-09-01")
    p.add_argument("--end", default="2026-09-30")
    p.add_argument("--out-root", default="processed/_batch/s2_2026")
    p.add_argument("--scenes", action="store_true", help="also draw every scene of the period with its Cloud Score+")
    p.add_argument("--vegetation", action="store_true",
                   help="only mask the existing image to its vegetated pixels (Otsu split of its own NDVI)")
    args = p.parse_args(argv)
    if args.vegetation:
        print(vegetation_only(args.aoi, args.start, args.end, args.out_root))
        return 0
    if args.scenes:
        print(scenes_sheet(args.aoi, args.start, args.end, args.out_root))
    print(run(args.aoi, args.start, args.end, args.out_root).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
