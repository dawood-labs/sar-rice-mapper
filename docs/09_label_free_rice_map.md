# 09 — A rice map without ground truth (optical first, then radar)

This page describes the route the project took once it was clear that **no labelled fields exist**
([08](08_analysis.md) cannot start without them). It is written for someone who has never seen the
code: read it top to bottom once, then use the command blocks as a runbook.

> **Every number this route produces is agreement between two measurements, never accuracy.** The
> optical map below is itself a measurement, not ground truth. A domain expert still has the final
> word on what is rice.

---

## 1. Why the route changed

The first radar-only draft map was anchored to a fixed calendar ("monsoon season" = the months
where VH is high). A pixel-by-pixel comparison with Sentinel-2 showed that the NDVI canopy peaked
and the crop was **cut weeks before** VH reached its seasonal maximum. In other words the radar
feature the model leaned on was measuring harvested fields, not the growing canopy. That check lives
in `analysis/phase_check.py` (section 4).

So the order was reversed:

1. **Build the map from optical first** — cloud-masked Sentinel-2, where "is there a crop, and when"
   can be read directly off the NDVI curve.
2. **Then ask what the radar can reproduce**, using the optical map as the reference, one date and
   one feature at a time, before any model is fitted.

Cloud is the obvious objection to step 1. It is handled by compositing (section 2): every 5-day
window keeps whatever clear ground any acquisition in it saw, and the windows nobody saw are
**reported as gaps**, never invented.

---

## 2. The steps in order

```
 Earth Engine                     local (this machine)
 ─────────────                    ─────────────────────────────────────────────────────────
 s2_windows  ──► GCS ──sync──►  optical_phenology ──► optical_rice_map ──► rice map per AOI
                                      │                       │
                                      ├── pixel_curve  (look at one pixel)
                                      └── chips        (look at the 5-3-2 imagery)

 stacked Sentinel-1 (pipeline stages 1–5) ──► sar_curve ──► separability ──► sar_rice_map
                                                               ▲
                                        optical rice map ──────┘  (used as the reference)
```

| Step | Module | Input | Output |
|---|---|---|---|
| 2.1 Export composites | `sar_pipeline.s2_windows` | AOI configs, date range | one GeoTIFF per AOI per 5-day window on GCS |
| 2.2 Measure cycles | `analysis.optical_phenology` | the composites (synced to `data/s2_windows5d/`) | one row per growth cycle per pixel (DataFrame) |
| 2.3 Classify | `analysis.optical_rice_map` | the cycles | `<aoi>_rice_optical.tif`, sieved copy, `aoi_rice_acres.csv` |
| 2.4 Check by eye | `analysis.pixel_curve`, `analysis.chips` | composites + cycles | plots, 5-3-2 contact sheets |
| 2.5 Current-season series | `optical_export` + `analysis.ndvi_5day` | per-date exports with QA60/SCL | gap-free 5-day NDVI and LSWI stacks + `gap_days` |
| 3.1 Radar curves | `analysis.sar_curve` | stacked Sentinel-1, both tracks | smoothed VH, VV, VH−VV per pixel on the same 5-day grid |
| 3.2 Separability | `analysis.separability` | radar curves + optical map | AUC per feature per date |
| 3.3 Radar map | `analysis.sar_rice_map` | radar curves + the AOI's sowing window | two-stage rice mask + agreement table |
| — Phase check | `analysis.phase_check` | an earlier class map + S2 + S1 | class-mean curves and harvest → VH-max lag |

None of the analysis modules has a command-line entry point yet: they are called from Python (a
notebook, or `python -c`). Run everything from the repository root with the project's own virtual
environment (`.venv/bin/python`, or the **"sar-rice-mapper (.venv)"** Jupyter kernel). The system
Python on some machines carries an older GDAL whose libtiff cannot write 64-bit predictor-2 TIFFs,
and one test fails for that reason alone.

### 2.1 Export 5-day cloud-masked composites — `s2_windows.py`

**Why.** One image per acquisition date leaves two problems: haze that the per-pixel cloud score
misses, and irregular dates that have to be interpolated before any derivative can be taken. A fixed
5-day window fixes both: each acquisition is masked with **s2cloudless** (probability > 70 %, plus
the cloud's projected shadow, plus a 50 m buffer) *before* the median is taken, so a cloud in one
pass is replaced by clear ground from another; and the result already sits on a regular grid.

Bands: B2 B3 B4 B5 B8 **B11** B12, plus **`n_obs`** — how many acquisitions survived the mask at
that pixel. `n_obs` is the only thing that says whether a pixel was seen: masked reflectance is
written as 0, and a 0 is not dark ground.

```python
from sar_pipeline import auth, config, s2_windows as sw
auth.init_ee(config.load_config("config/<any>.yaml"))
rows = sw.export_all_windows(["config/<aoi>.yaml"], "2025-03-01", "2026-02-04",
                             bucket="<bucket>", prefix="<base_folder>/s2_windows5d")
```

Windows already on GCS are skipped, so the call can be re-run after a failure. Each window's
`task.start()` is tried up to 5 times with a growing pause; a window that still fails gets its
message in the returned row's `error` field and the run carries on with the rest. Re-run the same
call to pick those windows up. **This starts Earth Engine
exports — confirm with the project lead before running it.**

Output: `gs://<bucket>/<base_folder>/s2_windows5d/<aoi>/<aoi>_W5_<window start>.tif`, on the AOI's
own Sentinel-1 grid (same CRS, origin and 10 m pixels), so it overlays the radar pixel for pixel.

### 2.2 Measure every growth cycle — `analysis/optical_phenology.py`

**Why.** A threshold such as "NDVI peak above 0.55" is the analyst's number, not the data's, and it
does not transfer between hazy, mixed and clean pixels. So every quantity is measured **against the
pixel's own curve**: amplitude from its own trough, length at half its own amplitude, green-up and
senescence as rates.

How a pixel's year is cut into cycles:

1. NDVI, NDWI and LSWI per window, valid where `n_obs >= 1`.
2. **Whittaker smoothing**, λ = 0.5, second differences, directly on the 5-day grid (no
   interpolation step — the data is already regular). Stretches nobody observed for more than
   35 days are left empty and **reported** (`observed_fraction`, `longest_gap_days`), not filled in.
3. The curve is split at its own **local minima**. A dip only ends a crop if the field stays down for
   **20 days** — otherwise a short dip (a cloud residue, a weeding) would split one crop into two.
4. **Emergence** (`start_date`) is where the rise *accelerates* hardest (maximum of the second
   derivative), not where NDVI crosses some level. The field's start (`transplant_date`) is
   `GREENUP_LAG_DAYS` (15) earlier.

```python
from sar_pipeline.analysis import optical_phenology as op
cycles = op.aoi_cycles(146)          # syncs data/s2_windows5d/aoi146/ from GCS on first use
main = op.main_cycle(cycles)         # one row per pixel: its strongest cycle
op.distributions(main)               # percentiles, to look at before choosing anything
```

**Why NDWI is not a water test.** Measured over ~900,000 clear pixel-dates, NDWI (green, NIR)
correlates with NDVI at r = −0.97: both are driven by the same NIR band, so "NDWI rises at sowing"
mostly restates "NDVI is low". Wet versus dry bare soil needs **SWIR**, which is why B11 is exported
and **LSWI = (B8 − B11) / (B8 + B11)** is used instead.

### 2.3 Turn cycles into a rice map — `analysis/optical_rice_map.py`

**Why a separate module.** `optical_phenology` measures and never decides; this module decides. The
decision stays in one visible place.

The rule, per AOI:

1. **The AOI's own season.** Take every strong, fully observed cycle in the AOI and read the median
   of their peak dates. A cycle is *in season* when its peak is within **±45 days** of that date.
   The season is measured, not written down, because it shifts across the country.
2. **Per pixel, rice when it has a cycle that is all of:**
   - in season (the strongest cycle *in season*, not the strongest overall — a weed flush or a
     second crop must not hide the rice crop underneath);
   - **complete** — not cut off by the start or end of the series;
   - a **real canopy** — its climb is at or above this AOI's own **Otsu** cut;
   - **60 to 165 days** from trough to harvest;
   - **wet at sowing** — NDVI drops *below* the field's own pre-season bare level while LSWI moves
     *the other way*. Drying soil moves both down together; opposite directions mean standing water.
3. Pixels where no cycle could be measured are **undecided**, not "not rice".

```python
from sar_pipeline.analysis import optical_rice_map as orm
table = orm.run_batch([146, 36, 114])      # default out_root: processed/_batch/optical_v3
```

Outputs in `processed/_batch/optical_v3/`:

| File | Content |
|---|---|
| `<aoi>/<aoi>_rice_optical.tif` | 1 = rice, 0 = not rice, 255 = undecided |
| `<aoi>/<aoi>_rice_optical_sieved_0p5ac.tif` | same, with patches under 0.5 acre (21 pixels) merged into their neighbour |
| `aoi_rice_acres.csv` | one row per AOI: total, rice and undecided **acres**, rice % of decided, season window, the cuts used. **Merged** on every run: the AOIs just run replace their own rows, all others are kept |

One failing AOI is logged and the batch carries on; its row carries an `error` column.

### 2.4 Check the result by eye — `pixel_curve.py` and `chips.py`

**Why.** The whole map rests on one claim: that the detected emergence and harvest dates are where
the field's crop actually starts and ends. That has to be looked at, not assumed.

```python
from sar_pipeline.analysis import pixel_curve as pc
pc.show(146, 4043)                     # raw clear points, the smoothed curve, the detected dates
pc.show(146, 4043, backend="plotly")   # hover for exact dates and values
pc.cycles(146, 4043)                   # the numbers behind the lines
```

The first call for an AOI reads its files; later pixels of the same AOI are instant. After
re-exporting, call `pc.forget()`. `notebooks/06_pixel_curve.ipynb` wraps this.

`chips.contact_sheet(aoi_id, pid, cycle)` draws one pixel's NDVI curve above a strip of **5-3-2**
(red edge, green, blue) image chips across its cycle; `cycle` is one row of the cycles table as a
dict (`row.to_dict()`). The chips come from the per-date reference exports (`data/s2_reference/`,
see the README's *Sentinel-2 reference images*), because a person needs to see real acquisitions,
not medians. In 5-3-2 a paddy goes bare/flooded → light green → dark green →
light green → yellow → cut; if the dashed dates do not sit at the ends of that sequence, the detector
is wrong for that pixel. `chips.stratified_sample(...)` picks pixels from every corner of the
distribution, not only typical ones. The colour stretch is computed **once over the whole series,
on clear pixels only** — per-image stretching would erase the colour progression the check depends on.

---

### 2.5 The current-season series: light mask, a value every 5 days — `ndvi_5day.py`

**Why a second optical route.** Looking at the curves from 2.1–2.2 showed two problems. The
s2cloudless + shadow + buffer mask threw away many clear observations, mostly in the sowing and
monsoon months that matter most. And because that mask is baked into the export, trying a different
mask meant exporting again. The curve was also blanked across long gaps, so some pixels had no value
exactly when the classifier needed one.

This route fixes both:

1. **Export every acquisition date unmasked, with the product's own mask bands** (`QA60`, `SCL`),
   so the mask is chosen locally and can change without a new export:

   ```python
   from sar_pipeline import auth, config, optical_export as ox
   cfg = config.load_config("config/<aoi>.yaml"); auth.init_ee(cfg)
   poly = ox.aoi_geojson_2d(__import__("geopandas").read_file(config.aoi_path(cfg)))
   ox.mask_summary(poly, "2025-09-01", "2026-09-24")      # read-only: is QA60 populated? QA60 vs SCL
   rows = ox.export_all_dates(["config/<aoi>.yaml"], "2025-09-01", "2026-09-24",
                              bucket=cfg["gcs"]["bucket"],
                              prefix=cfg["gcs"]["base_folder"] + "/s2_dates_masks",
                              bands=ox.BANDS_WITH_MASKS)   # starts exports: confirm first
   ox.wait_for_tasks([r["task_id"] for r in rows])        # blocks until every task has finished
   ```

   Bands, by name: B2 B3 B4 B5 B8 B11 B12 QA60 SCL clear. Each date is tried up to 5 times; a date
   that still fails is recorded in its row's `error` and the rest carry on. Re-running skips files
   already on GCS.

2. **Build the gap-free series** (`analysis/ndvi_5day.py`):

   ```python
   from sar_pipeline.analysis import ndvi_5day as nd
   nd.build(143)          # syncs data/s2_dates_masks/aoi143/, writes processed/_batch/s2_2026/aoi143/
   ```

   - **Mask:** only QA60 bit 10 (opaque cloud) and bit 11 (cirrus). There is no shadow mask on
     purpose, because a flooded paddy is dark and shadow masks tend to delete it. SCL is reported,
     not applied.
   - **5-day maximum-value composite:** of the clear dates in a window, the highest NDVI is kept,
     and LSWI and NDWI come from that same date. Haze only lowers NDVI, so the maximum is the
     cleanest value.
   - **Upper-envelope Whittaker** (Chen et al. 2004): points below the fit are down-weighted
     over 3 passes, so haze dips that got past the light mask do not pull the curve down. Missing
     windows are filled, so **every pixel has a value in every window**.
   - **`gap_days`:** for each value, how many days away the nearest real observation was. The
     value is always there, but a reader can see which values were measured and which were
     filled in.

   | Output (`processed/_batch/s2_2026/<aoi>/`) | Content, one band per 5-day window |
   |---|---|
   | `<aoi>_ndvi5d.tif` | fitted NDVI |
   | `<aoi>_lswi5d.tif` | fitted LSWI (water signal) |
   | `<aoi>_ndvi5d_raw.tif` | the composited observation before fitting; nodata where none |
   | `<aoi>_gapdays5d.tif` | days to the nearest observed window |

   Band descriptions are the window start dates, so QGIS's Temporal/Spectral Profile Tool shows a
   pixel's curve with real dates.

3. **Look at any pixel** — `notebooks/07_pixel_2026.ipynb` (logic in `analysis/pixel_2026.py`):
   set `AOI` and `PID`, Run All. It shows the pixel's info (location, dates kept by QA60, longest
   gap, last season's map class), its crop cycles (sowing, emergence, green-up, peak, harvest), the
   curve with every raw date and the fit, and **5-3-2 chips centred on the pixel** (61 x 61 px,
   ~610 m) for the clearest `PER_MONTH` dates of every month, ranked by Cloud Score+ over the whole
   chip. Each chip gets its own 2-98 % stretch on clear pixels plus gamma 0.7, for the sharpest
   colours; `STRETCH = "series"` shares one stretch so colours can be compared between dates.

   **Transplanting, not sowing.** Rice is transplanted (or direct-seeded) into a flooded field; the
   nursery sowing before it happens elsewhere and is invisible. The field's start is the trough (the
   field as a pool of water with small seedlings, NDVI near zero); NDVI only starts to climb once
   the plants have tillered enough to cover the water, **15-20 days later** as measured on the field
   plots (`GREENUP_LAG_DAYS`). `transplant_date` is the green-up onset minus that lag; the table also
   shows `trough_date`. Names: `greenup_onset` (was "emergence"), `transplant_date` (was "sowing").

4. **Is there a crop in a given season at all?** — `analysis/aoi_profile.py` summarises **every pixel
   inside the AOI on every date**, under two independent masks (QA60 and Cloud Score+ `clear` >= 60):
   NDVI p10 / median / p90, the share with a canopy (NDVI > 0.5) and the share that looks flooded
   (LSWI > NDVI). Sample pixels can miss a crop that covers part of an AOI, and one mask can hide a
   whole season; this cannot.

   ```python
   from sar_pipeline.analysis import aoi_profile as ap
   prof = ap.date_profile(143)          # one row per date and mask
   ap.plot(prof, out_path="processed/_batch/s2_2026/aoi143/figures/aoi143_date_profile.png")
   ```

5. **Which AOIs carry a crop this season?** — `sar_pipeline/season_screen.py`, one read-only Earth
   Engine request over every AOI, nothing exported (about 30 s for 132 AOIs). A pixel counts as
   **seasonal canopy** when its maximum NDVI in the season window is above 0.5 *and* at least 0.3
   above its median in the pre-season window; trees are green in both windows and are reported
   separately as `evergreen`. Only Cloud Score+ clear pixels are used, and `seen_pct` says how much
   of each AOI was observed in both windows. Areas in acres, sorted by seasonal area.

   ```python
   from sar_pipeline import auth, config, season_screen as ss
   auth.init_ee(config.load_config("config/<any>.yaml"))
   table = ss.screen(ss.aoi_polygons(), bare=("2026-04-01", "2026-06-01"),
                     season=("2026-08-01", "2026-09-24"))
   ```

   It is a screen for choosing where to work, not a classification: a seasonal canopy can be any crop.

6. **Field plots** — `sar_pipeline/prep/field_plots.py`: loads every zipped shapefile in a delivery
   folder (kept outside the repository), tags each plot with its `source` file, measures acres per
   file in that file's own UTM zone, repairs invalid geometries, counts records with no geometry,
   and finds the AOI each plot falls in.

   ```python
   from sar_pipeline.prep import field_plots as fp
   from sar_pipeline import season_screen as ss
   plots = fp.load("<field data folder>")
   fp.summary(plots)                                  # per file: plots, acres, invalid, no geometry
   fp.attribute_overview(plots)                       # what the attribute columns actually contain
   where = fp.in_aois(plots, ss.aoi_polygons())       # plot_id, aoi, share_in_aoi
   ```

   First lesson from them: the season screen (step 5) assumes fields are bare in the pre-season
   window. Where a dry-season crop is still green then and the monsoon crop follows straight after,
   the field is green in both windows and the screen calls it `evergreen`. Plots reported as
   standing rice fell almost entirely into `evergreen` in three AOIs.

7. **One curve per field plot** — `analysis/plot_curves.py`: the median fitted NDVI and LSWI over
   the pixels whose centre lies inside each plot (`inner_buffer_m` shrinks plots first to drop edge
   pixels), a figure per AOI (all curves, their median, a plots x time heatmap), and per-plot
   season events (trough date and value, LSWI at the trough, onset of the climb, last value)
   summarised per AOI.

   ```python
   from sar_pipeline.prep import field_plots as fp
   from sar_pipeline.analysis import plot_curves as pc
   plots = fp.repair(fp.load("<field data folder>"))
   where = fp.in_aois(plots, ss.aoi_polygons())
   allc = pc.run_aois([20, 24, 25], plots, where)         # builds the series, writes CSV + PNG per AOI
   pc.aoi_event_summary(pc.season_events(allc))          # trough, wetness, onset, last NDVI per AOI
   ```

   What the first delivery showed (plots reported as standing rice in September): every region
   grows **two rice crops** (a dry-season crop peaking in February to May, then the monsoon crop);
   the monsoon trough falls between late June and late August depending on the region; the climb
   starts **15-20 days after the trough** in most regions, which is the lead the detector should use
   instead of the assumed 10; and in one region LSWI never rose above NDVI at the trough, so a
   wet-at-sowing test cannot be a hard requirement.

8. **The current-season rule** — `analysis/monsoon_rule.py`. Per pixel, inside the season window
   (1 June to today): the NDVI **trough**, the **climb** after it, and the canopy reached. Thresholds
   read off the field plots: trough <= 0.40, climb >= 0.30 and canopy >= 0.50 for rice by
   phenology; climb >= 0.15 and canopy >= 0.30 for **young** (a crop has started, not confirmable
   yet). The canopy floor exists because a control AOI with no crop went from water (-0.35) to bare
   soil (0.2), a "climb" that is not a canopy. **Water is then confirmed by radar**
   (`analysis/radar_water.py`): each phenology-rice pixel's Sentinel-1 series is lined up on its own
   trough and the dip below its dry level (40-15 days before) is read; a dip of 3 dB in VV or VH
   makes it **rice** (1), otherwise **rice, water unconfirmed** (3). The optical LSWI is only
   reported, because it missed the shallow water of the capital region that the radar saw.

   ```python
   from sar_pipeline.analysis import monsoon_rule as mr
   mr.run_aoi(114)                 # writes <aoi>_monsoon2026.tif: 0 not rice, 1 rice, 2 young, 255 no data
   mr.plot_recall(20, plots)       # share of field-plot pixels called rice / young / not rice
   ```

   Checked on 2,950 field plots: 88-100 % of plot pixels are called rice where the crop was more
   than six weeks old; 9-44 % where it had just been transplanted (the rest are `young` or still
   under water), which is the limit of optical data at that date. The control AOI came out at
   0.1 acres of rice.

9. **Radar check of the water** — `analysis/sar_water_check.py`. For every field plot, the plot's
   VV and VH series (median over its pixels of the 5x5-mean dB stacks of a season run, per track)
   lined up on the plot's optical transplant date; the **dip** = median level 60-20 days before
   minus the minimum 10 days before to 15 days after, in dB. VV is the more water-sensitive
   polarisation (a flooded field mirrors the signal away and VV drops 5-15 dB); VH drops less and
   then climbs with the canopy. A VV dip where the optical LSWI saw no water is the evidence that
   the water was there and the optical test was what failed. Needs the Sentinel-1 pipeline
   (stages 1-5) run for that season, e.g. season key `monsoon2026`.

   ```python
   from sar_pipeline.analysis import sar_water_check as sw
   r = sw.run(29, plots, events)     # events: plot_curves.season_events output; writes CSV + PNG
   r["summary"]                      # per track: median VV/VH dip, % plots with a VV dip >= 3 dB
   ```

   Result on the first delivery (three AOIs, 601 plots reported as standing rice): in every AOI
   and on every track, the radar's lowest point of the season sits **on the optical transplant
   date** (median 1-5 days after the NDVI trough). In the delta AOI the drop below the field's own
   dry level was 8 dB in both VV and VH (86 % of plots on the better track), and flooding began
   about 25 days before the trough — deep, prolonged water. In the two AOIs where the optical LSWI
   had seen almost no water (0-13 % of plots), VH still dropped 5 dB in 91 % of plots and VV 3-4 dB
   in 67-76 %: shallow, short-lived water that the optical test missed. Two independent sensors
   agreeing on the date is the evidence that the trough is transplanting, and that the water was
   there.

## 3. Re-examining the radar against the optical map

### 3.1 One smoothed radar curve per pixel — `analysis/sar_curve.py`

**Why.** To ask whether the radar sees what the optical sees, it has to be on the same footing: the
same 5-day grid, the same kind of fit.

- **Both orbit tracks are used.** Together they roughly halve the revisit gap compared with one
  track. Each track's offset is measured **per pixel from near-coincident passes** (within 3 days)
  and removed; on terrain-flattened γ⁰ it turned out to be close to zero, but it is measured, not
  assumed.
- **Everything is averaged and smoothed in linear power**, then converted to dB. The mean of
  decibels sits systematically low.
- **λ is chosen by leave-one-out cross-validation** (`choose_lambda`), not guessed. Radar speckle
  makes the right λ roughly ten times the optical one.
- A fitted power that comes out ≤ 0 is set to **NaN**, never clipped to a tiny number: clipping once
  produced fake −80 dB values that a minimum-based feature latched onto.

```python
from sar_pipeline.analysis import sar_curve as sc
curves = sc.load(146)          # needs pipeline stages 1–5 run for both tracks of that AOI
sc.pixel(146, 4043)            # observed and smoothed VH, VV, VH−VV at one pixel
```

### 3.2 Can the radar separate the classes at all? — `analysis/separability.py`

**Why.** A classifier always reports a score; that score says how well it reproduces its labels,
not whether the sensor carries the information. So before any model, measure each feature on each
date: the **AUC** is the probability that a random rice pixel scores higher than a random non-rice
pixel. 0.5 = nothing; `separation = |AUC − 0.5| × 2` runs from 0 to 1 either way round.

- `per_step` — AUC of every feature at every date. Shows *when* in the season the radar sees a
  difference.
- `summary_features` / `rank_features` — whole-season summaries, the kind a classifier would get.
- `align_cube` + `earliest_call` — re-cut each pixel's curve relative to **its own** sowing date, and
  find the first day the separation passes a target. This is the operational question: how early in
  a season can the radar call it?

Rule of thumb: if nothing passes about 0.7, the honest result is that the radar cannot make this map
alone.

### 3.3 A two-stage radar map — `analysis/sar_rice_map.py`

**Why two stages.** "Is there a crop here" and "is it on the rice calendar" are two different
physical questions with two different best moments in the year.

- **Stage 1 — seasonal cropland or permanent cover**, from **dry-season VH** (Nov–Jan). The direction
  is the non-obvious one: harvested cropland is bare and smooth then, so its VH is **low**; trees and
  settlements stay high. Getting this backwards once gave a negative kappa, so the direction is a
  parameter (`cropland_is_low`) with a test.
- **Stage 2 — on the calendar or not**, from VH **around the AOI's own sowing window** (±15 days): a
  field at transplanting is flat and wet and its VH collapses.

Each stage's *window* comes from agronomy; each stage's *threshold* comes from that AOI's own
histogram (Otsu). `agreement(...)` scores the result against the optical map;
`split_halves(...)` fits a cut on one spatial half and tests it on the other.

**Known limits.** Stage 2 is weak with a single threshold — radar finds a confident *subset* of the
rice, not all of it — and the sowing window is still taken from the optical map. A fitted model over
several weak features, with spatial block cross-validation, and a sowing window found from the radar
itself, are the next steps.

---

## 4. Checking a class map's timing — `analysis/phase_check.py`

**Why.** It is how the first draft's error was found, and it is the check to repeat on any future
radar map. For the pixels of one class it averages NDVI per clear Sentinel-2 date and VH/VV per
acquisition (in linear power), and reports green-up, NDVI peak, **harvest** (half-fall after the
peak), the VH maximum, and **the lag from harvest to VH maximum**. For a real canopy signal that lag
is near zero or negative; a large positive lag means the radar maximum happens on a harvested field.

The class map it reads is set by `maps_dir` (default: the first draft's maps,
`processed/_batch/model_v2/maps/`); point it at any other map's folder to check that map instead.

```python
from sar_pipeline.analysis import phase_check as ph
ndvi = ph.class_ndvi(146, class_value=1)
sar = ph.class_sar(146, class_value=1)
ph.timing(ndvi, sar)
```

---

## 5. Traps already hit — do not hit them again

- **A physical claim is not a fact until checked on this data.** The stage 1 direction above.
- **Never clip a fitted power.** Set it to NaN (section 3.1).
- **Amplitude measured shoulder to shoulder is wrong on double-cropped fields**: the next crop starts
  before this one comes down. Measure the climb from the trough, or from the pre-season baseline when
  the trough falls in a cloudy gap.
- **`importlib.reload(pixel_curve)` does not reload `optical_phenology` behind it.** Reload both, in
  that order (notebook 06 does).
- **Earth Engine rejects 3D GeoJSON** — drop the Z coordinate first (`shapely.force_2d`).
- **Jupyter autosaves outputs into notebooks.** Strip them in the same command as `git add`; the
  hygiene test fails otherwise, and an output can leak private values into a public repository:

  ```bash
  jupyter nbconvert --clear-output --inplace notebooks/*.ipynb && git add notebooks/
  ```
