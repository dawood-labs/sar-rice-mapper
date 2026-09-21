"""Live Earth Engine tests for the ARD chain. READ-ONLY: getInfo + computePixels on a 64x64 px window.

Run with: pytest -m gee tests/gee/test_ard_live.py -s   (-s shows the printed statistics)
No coordinates are stored here: the test window is centred on the configured AOI at runtime.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.gee

WIN = 64
N_ACQ = 3


@pytest.fixture(scope="module")
def setup(live_cfg):
    import ee

    from sar_pipeline.config import aoi_path
    from sar_pipeline.grid import compute_grid_def

    cfg = copy.deepcopy(live_cfg)
    gd = compute_grid_def(cfg)
    res = gd["res"]

    # 64x64 px window on the master grid, centred on the AOI centroid
    centroid = gpd.read_file(aoi_path(cfg)).to_crs(gd["crs"]).union_all().centroid
    col = int((centroid.x - gd["x0"]) // res) - WIN // 2
    row = int((gd["y0"] - centroid.y) // res) - WIN // 2
    xmin, ymax = gd["x0"] + col * res, gd["y0"] - row * res
    chunk = dict(chunk_id=1, name="test_window", grid_row=0, grid_col=0, row_off=row, col_off=col,
                 width=WIN, height=WIN, xmin=xmin, ymin=ymax - WIN * res, xmax=xmin + WIN * res, ymax=ymax, aoi_frac=1.0)

    # Real acquisitions of the most frequent track over the window centre (metadata only)
    lon, lat = gpd.GeoSeries([centroid], crs=gd["crs"]).to_crs(4326).iloc[0].coords[0]
    end = min(cfg["season"]["end"], datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    s1 = (ee.ImageCollection(cfg["s1"]["collection"])
          .filterBounds(ee.Geometry.Point([lon, lat]))
          .filterDate(cfg["season"]["start"], end)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    meta = ee.Dictionary({
        "idx": s1.aggregate_array("system:index"),
        "ro": s1.aggregate_array("relativeOrbitNumber_start"),
        "pass": s1.aggregate_array("orbitProperties_pass"),
        "t": s1.aggregate_array("system:time_start"),
    }).getInfo()
    df = pd.DataFrame(meta)
    assert len(df) >= N_ACQ, "not enough Sentinel-1 slices over the AOI centre"
    df["track_id"] = [f"RO{int(r):03d}_{'ASC' if p == 'ASCENDING' else 'DSC'}" for r, p in zip(df["ro"], df["pass"])]
    df["dt"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.sort_values(["track_id", "dt"])
    df["acq"] = df.groupby("track_id")["dt"].transform(lambda s: (s.diff() > pd.Timedelta(minutes=15)).cumsum())
    acqs = (df.groupby(["track_id", "acq"])
              .agg(datetime=("dt", "min"), system_indexes=("idx", lambda s: ";".join(s)))
              .reset_index())
    track_id = acqs["track_id"].value_counts().idxmax()
    tr = acqs[acqs["track_id"] == track_id].sort_values("datetime").reset_index(drop=True)
    mid = len(tr) // 2
    tr = tr.iloc[max(0, mid - 1): max(0, mid - 1) + N_ACQ].reset_index(drop=True)
    acquisitions = pd.DataFrame({
        "acquisition_id": [f"{track_id}_{d:%Y%m%d}" for d in tr["datetime"]],
        "track_id": track_id,
        "datetime_utc": [d.strftime("%Y-%m-%dT%H:%M:%SZ") for d in tr["datetime"]],
        "date_utc": [d.strftime("%Y-%m-%d") for d in tr["datetime"]],
        "system_indexes": tr["system_indexes"],
    })
    print(f"\n[setup] track {track_id}, acquisitions {list(acquisitions['date_utc'])}")
    return cfg, gd, chunk, track_id, acquisitions


def _fetch(image, chunk, gd) -> dict[str, np.ndarray]:
    import ee

    arr = ee.data.computePixels({
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": chunk["width"], "height": chunk["height"]},
            "affineTransform": {"scaleX": gd["res"], "shearX": 0, "translateX": chunk["xmin"],
                                "shearY": 0, "scaleY": -gd["res"], "translateY": chunk["ymax"]},
            "crsCode": gd["crs"],
        },
    })
    return {name: np.asarray(arr[name]) for name in arr.dtype.names}


def _local_std(a: np.ndarray) -> np.ndarray:
    """3x3 standard deviation (NaN-aware) for interior pixels."""
    from numpy.lib.stride_tricks import sliding_window_view

    return np.nanstd(sliding_window_view(a, (3, 3)), axis=(-1, -2))


@pytest.fixture(scope="module")
def float_result(setup):
    import ee

    from sar_pipeline.s1_ard import band_names, build_chunk_image, filter_series, mosaic_acquisition, to_output

    cfg, gd, chunk, track_id, acquisitions = setup
    names = band_names(acquisitions)
    filtered = build_chunk_image(cfg, gd, chunk, track_id, acquisitions)
    mid_row = acquisitions.iloc[1].to_dict()
    mid_mosaic = mosaic_acquisition(cfg, gd, mid_row)
    unfiltered = to_output(cfg, mid_mosaic).rename(["VV_raw", "VH_raw"])
    cfg_mono = copy.deepcopy(cfg)
    cfg_mono["ard"]["multitemporal"] = False
    mono = to_output(cfg_mono, filter_series(cfg_mono, [mid_mosaic])[0]).rename(["VV_mono", "VH_mono"])
    cfg_no_tf = copy.deepcopy(cfg)
    cfg_no_tf["ard"]["terrain_flattening"] = False
    unflattened = to_output(cfg_no_tf, mosaic_acquisition(cfg_no_tf, gd, mid_row)).rename(["VV_notf", "VH_notf"])
    data = _fetch(ee.Image.cat([filtered, unfiltered, mono, unflattened]), chunk, gd)
    return names, filtered, data


def test_band_names_and_count(setup, float_result):
    names, filtered, data = float_result
    assert filtered.bandNames().getInfo() == names
    assert len(names) == 2 * N_ACQ
    assert all(n in data for n in names)


def test_values_plausible_db(setup, float_result):
    cfg, *_ = setup
    names, _, data = float_result
    nodata = cfg["export"]["nodata_float32"]
    for name in names:
        a = data[name].astype("float64")
        valid = a[a != nodata]
        frac = valid.size / a.size
        p5, p50, p95 = np.percentile(valid, [5, 50, 95])
        print(f"[dB] {name}: valid {frac:.0%}, p5 {p5:.2f}, median {p50:.2f}, p95 {p95:.2f}")
        assert frac > 0.5, f"{name}: only {frac:.0%} valid pixels"
        if name.startswith("VH"):
            assert -30 <= p50 <= -5 and p5 > -35 and p95 < 0
        else:
            assert -25 <= p50 <= 5 and p5 > -30 and p95 < 10


def test_speckle_filter_reduces_local_std(setup, float_result):
    """Mono filter must smooth clearly; the multi-temporal output must also be smoother than raw.

    Note: the Quegan filter keeps full spatial resolution. Its residual noise comes from the per-pixel
    ratios I_i/F(I_i) averaged over N dates (ENL gain ~N), so with only N=3 dates it is expected to be
    rougher than the mono-temporal filter alone.
    """
    cfg, _, _, _, acquisitions = setup
    _, _, data = float_result
    nodata = cfg["export"]["nodata_float32"]
    date = acquisitions.iloc[1]["date_utc"].replace("-", "")

    def arr(name):
        return np.where(data[name] == nodata, np.nan, data[name]).astype("float64")

    for pol in ("VV", "VH"):
        s_raw = np.nanmedian(_local_std(arr(f"{pol}_raw")))
        s_mono = np.nanmedian(_local_std(arr(f"{pol}_mono")))
        s_mt = np.nanmedian(_local_std(arr(f"{pol}_{date}")))
        print(f"[speckle] {pol}: median 3x3 std raw {s_raw:.2f} dB, mono {s_mono:.2f} dB (x{s_mono / s_raw:.2f}), "
              f"multitemporal N={N_ACQ} {s_mt:.2f} dB (x{s_mt / s_raw:.2f})")
        assert s_mono < 0.8 * s_raw
        assert s_mt < 0.95 * s_raw


def test_terrain_flattening_changes_values(setup, float_result):
    cfg, *_ = setup
    _, _, data = float_result
    nodata = cfg["export"]["nodata_float32"]
    for pol in ("VV", "VH"):
        a, b = data[f"{pol}_raw"], data[f"{pol}_notf"]
        both = (a != nodata) & (b != nodata)
        diff = np.abs(a[both].astype("float64") - b[both])
        print(f"[terrain] {pol}: |gamma0_flat - sigma0| median {np.median(diff):.2f} dB, max {diff.max():.2f} dB")
        assert both.mean() > 0.5
        assert diff.max() > 0.1


def test_int16_encoding_matches_float(setup, float_result):
    from sar_pipeline.s1_ard import build_chunk_image

    cfg, gd, chunk, track_id, acquisitions = setup
    names, _, fdata = float_result
    cfg16 = copy.deepcopy(cfg)
    cfg16["export"]["dtype"] = "int16"
    data = _fetch(build_chunk_image(cfg16, gd, chunk, track_id, acquisitions), chunk, gd)
    scale, nd16, ndf = cfg16["export"]["int16_scale"], cfg16["export"]["nodata_int16"], cfg["export"]["nodata_float32"]
    for name in names:
        assert data[name].dtype == np.int16, data[name].dtype
        i, f = data[name], fdata[name]
        both = (i != nd16) & (f != ndf)
        assert ((i == nd16) == (f == ndf)).all(), f"{name}: nodata masks differ"
        err = np.abs(i[both] / scale - f[both].astype("float64"))
        assert err.max() <= 0.5 / scale + 1e-4, f"{name}: max decode error {err.max()}"


# ---------------------------------------------------------------- terrain flattening on steep terrain
D2R = np.pi / 180
SLOPE_BINS = [-30, -20, -10, -3, 3, 10, 20, 30]  # degrees of range slope, + = facing the sensor


@pytest.fixture(scope="module")
def hilly(live_cfg):
    """A steep 0.3° box near the AOI (found at runtime from the DEM) plus one ASC and one DSC slice covering it."""
    import ee

    from sar_pipeline.config import aoi_path
    from sar_pipeline.s1_ard import terrain_flattening as tf

    cfg = live_cfg
    dem = tf.load_dem(cfg["ard"]["dem"])
    minx, miny, maxx, maxy = gpd.read_file(aoi_path(cfg)).to_crs(4326).total_bounds
    cell, margin = 0.3, 0.6
    xs = np.arange(minx - margin, maxx + margin, cell)
    ys = np.arange(miny - margin, maxy + margin, cell)
    cells = ee.FeatureCollection([ee.Feature(ee.Geometry.Rectangle([float(x), float(y), float(x + cell), float(y + cell)]),
                                             {"x": float(x), "y": float(y)}) for x in xs for y in ys])
    stats = ee.Terrain.slope(dem).reduceRegions(cells, ee.Reducer.mean(), scale=300).getInfo()["features"]
    ranked = sorted((f["properties"] for f in stats if f["properties"].get("mean") is not None),
                    key=lambda p: -p["mean"])
    end = min(cfg["season"]["end"], datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    base = (ee.ImageCollection(cfg["s1"]["collection"]).filterDate(cfg["season"]["start"], end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    for p in ranked[:6]:
        box = ee.Geometry.Rectangle([p["x"], p["y"], p["x"] + cell, p["y"] + cell])
        covering = base.filter(ee.Filter.contains(leftField=".geo", rightValue=box))
        imgs = {}
        for pas in ("ASCENDING", "DESCENDING"):
            c = covering.filter(ee.Filter.eq("orbitProperties_pass", pas))
            if c.size().getInfo() > 0:
                imgs[pas] = ee.Image(c.first()).select(["VV", "VH", "angle"])
        if len(imgs) == 2:
            centre = (p["x"] + cell / 2, p["y"] + cell / 2)
            print(f"\n[hilly] box mean slope {p['mean']:.1f} deg")
            return cfg, dem, box, centre, imgs
    pytest.skip("no steep box near the AOI covered by both passes")


def _iso_toward_deg(img, lon0: float, lat0: float, dist_m: float = 20000.0) -> float:
    """Independent look direction: azimuth in which the (bilinear) angle drops most over `dist_m`."""
    import ee

    ang = img.select("angle").resample("bilinear")

    def change(azimuths):
        pts = {}
        for az in azimuths:
            dlat = dist_m * np.cos(np.radians(az)) / 111320
            dlon = dist_m * np.sin(np.radians(az)) / (111320 * np.cos(np.radians(lat0)))
            pts[str(az)] = ang.reduceRegion(ee.Reducer.first(), ee.Geometry.Point([lon0 + dlon, lat0 + dlat]), 10).get("angle")
        pts["base"] = ang.reduceRegion(ee.Reducer.first(), ee.Geometry.Point([lon0, lat0]), 10).get("angle")
        r = ee.Dictionary(pts).getInfo()
        b = r.pop("base")
        return {float(k): v - b for k, v in r.items() if v is not None}

    coarse = change(range(0, 360, 10))
    best = min(coarse, key=coarse.get)
    fine = change([(best + d) % 360 for d in np.arange(-9, 10, 1.0)])
    return min(fine, key=fine.get)


@pytest.fixture(scope="module")
def slope_samples(hilly):
    import ee

    from sar_pipeline.s1_ard import terrain_flattening as tf

    cfg, dem, box, (lon0, lat0), imgs = hilly
    out = {}
    for pas, img in imgs.items():
        geo = tf.slope_geometry(img, dem)
        toward_ind = _iso_toward_deg(img, lon0, lat0)
        toward_code = geo["look_direction_deg"].getInfo()
        elev = dem.resample("bilinear").reproject(img.select("VV").projection(), None, 10)
        slope = ee.Terrain.slope(elev)
        rel = ee.Terrain.aspect(elev).subtract(toward_ind).multiply(D2R)
        ar_ind = slope.multiply(D2R).tan().multiply(rel.cos()).atan().divide(D2R)
        valid = tf.layover_shadow_mask(geo["alpha_r"], geo["theta_i"])
        flat = tf.flatten_slice(img, ["VV", "VH"], "VOLUME", dem, 0).select("VH")
        theta = geo["theta_i"].divide(D2R)
        stack = ee.Image.cat([geo["alpha_r"].divide(D2R).rename("ar_code"), ar_ind.rename("ar_ind"), theta.rename("theta"),
                              valid.rename("valid"), flat.unmask(-1).rename("vh_flat"), img.select("VH").rename("s0"),
                              slope.rename("slope")])
        cols = ["ar_code", "ar_ind", "theta", "valid", "vh_flat", "s0", "slope"]
        fc = stack.sample(region=box, scale=30, numPixels=30000, seed=3, geometries=False, tileScale=4)
        rows = fc.reduceColumns(ee.Reducer.toList(len(cols)), cols).get("list").getInfo()
        data = np.array([r for r in rows if None not in r], dtype=float)
        # layover share over the whole box (layover is rare, a sample holds too few pixels)
        true_layover = ar_ind.gte(theta.add(2))  # 2 degrees margin against small heading differences
        lay = valid.eq(0).rename("masked").updateMask(true_layover).reduceRegion(
            ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True), box, 30, maxPixels=1e9, tileScale=8).getInfo()
        out[pas] = dict(data={c: data[:, i] for i, c in enumerate(cols)}, toward_code=toward_code,
                        toward_ind=toward_ind, layover=lay)
        print(f"[hilly] {pas}: look direction code {toward_code:.1f} deg, independent {toward_ind:.1f} deg, "
              f"n={len(data)}, layover px {lay.get('masked_count')}, masked share {lay.get('masked_mean')}")
    return out


def test_range_slope_sign_matches_independent_estimate(slope_samples):
    for pas, s in slope_samples.items():
        d = s["data"]
        steep = d["slope"] > 3
        r = np.corrcoef(d["ar_code"][steep], d["ar_ind"][steep])[0, 1]
        diff = (s["toward_code"] - s["toward_ind"] + 180) % 360 - 180
        print(f"[sign] {pas}: corr(alpha_r code, independent) = {r:.3f}, look direction difference {diff:.1f} deg")
        assert r > 0.9
        # The code uses ONE look direction per slice (mean over the whole swath, as in Vollrath et al.
        # 2020); the independent estimate is local to the box. They differ by a few degrees along a
        # 250 km swath. A heading error d changes alpha_r by at most ~tan(slope)*sin(d), i.e. < 0.2 deg
        # for a 10 deg slope at d = 10 deg, so 10 deg is a safe tolerance.
        assert abs(diff) < 10


def test_volume_flattening_removes_slope_effect(slope_samples):
    """Slopes facing the sensor and slopes facing away must look alike after flattening.

    Why pairs of opposite slopes instead of "sloped vs flat": in a hilly box the flat pixels are
    valley floors (rivers, paddies, fields) while slopes are mostly forest, so the flat bin differs by
    land cover, not by geometry (live: flat sigma0 was ~2 dB darker than gentle slopes on BOTH sides).
    Opposite slopes of the same steepness share land cover on average, so their difference isolates
    the radiometric slope effect that flattening must remove.
    """
    for pas, s in slope_samples.items():
        d = s["data"]
        ok = (d["valid"] == 1) & (d["vh_flat"] > 0) & (d["s0"] > 0)
        medians = {}
        for lo, hi in zip(SLOPE_BINS[:-1], SLOPE_BINS[1:]):
            m = ok & (d["ar_ind"] >= lo) & (d["ar_ind"] < hi)
            if m.sum() >= 50:
                medians[(lo, hi)] = (int(m.sum()), float(np.median(10 * np.log10(d["s0"][m]))),
                                     float(np.median(10 * np.log10(d["vh_flat"][m]))))
        print(f"[flatten] {pas}: bin  n  sigma0_dB  flattened_dB")
        for k, (n, s0, fl) in medians.items():
            print(f"[flatten] {pas}: {k} {n} {s0:.2f} {fl:.2f}")
        pairs = [((-h, -l), (l, h)) for (l, h) in zip(SLOPE_BINS[len(SLOPE_BINS) // 2:-1], SLOPE_BINS[len(SLOPE_BINS) // 2 + 1:])]
        pairs = [(a, b) for a, b in pairs if a in medians and b in medians]
        assert len(pairs) >= 2, f"{pas}: too few paired slope bins with data: {list(medians)}"
        for away, facing in pairs:
            raw_gap = medians[facing][1] - medians[away][1]
            flat_gap = medians[facing][2] - medians[away][2]
            print(f"[flatten] {pas}: facing {facing} - away {away}: sigma0 {raw_gap:+.2f} dB -> flattened {flat_gap:+.2f} dB")
            assert abs(flat_gap) <= 2.0, f"{pas} {facing} vs {away}: {flat_gap:+.2f} dB after flattening"
        steepest = pairs[-1]
        raw = medians[steepest[1]][1] - medians[steepest[0]][1]
        flat = medians[steepest[1]][2] - medians[steepest[0]][2]
        assert abs(flat) < 0.3 * abs(raw), f"{pas}: flattening removed too little of the slope effect ({raw:+.2f} -> {flat:+.2f} dB)"


def test_true_layover_is_masked(slope_samples):
    evaluated = 0
    for pas, s in slope_samples.items():
        lay = s["layover"]
        if (lay.get("masked_count") or 0) < 20:
            print(f"[layover] {pas}: only {lay.get('masked_count')} layover pixels, not evaluated")
            continue
        evaluated += 1
        assert lay["masked_mean"] >= 0.95, f"{pas}: only {lay['masked_mean']:.1%} of true layover masked"
    if not evaluated:
        pytest.skip("no pass had enough true-layover pixels in the box")


def test_lee_sigma_chunk_computes(setup):
    from sar_pipeline.s1_ard import build_chunk_image

    cfg, gd, chunk, track_id, acquisitions = setup
    cfg_ls = copy.deepcopy(cfg)
    cfg_ls["ard"]["speckle_filter"] = "LEE SIGMA"
    data = _fetch(build_chunk_image(cfg_ls, gd, chunk, track_id, acquisitions), chunk, gd)
    nodata = cfg["export"]["nodata_float32"]
    for name, a in data.items():
        frac = float((a != nodata).mean())
        print(f"[lee sigma] {name}: valid {frac:.0%}, median {np.median(a[a != nodata]):.2f} dB")
        assert frac > 0.5
