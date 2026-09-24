# 16 — Looking at the radar by eye: rice colour composites for QGIS

Code: `src/sar_pipeline/sar_composites.py` (tests: `tests/test_sar_composites.py`).

The final map (docs/13-15) comes from a rule. This doc is about a picture that lets a person check
that map, field by field, without any rule at all: a colour image made from the radar, in which
paddy rice has its own colour.

## 1. Why radar composites

* **Optical is cloudy.** In the monsoon a Sentinel-2 date shows on average only 18 % of an AOI clear
  (docs/15). A true-colour image of the season often does not exist.
* **The radar sees through cloud** every 6-12 days (Sentinel-1, all tracks together), so every
  month has several passes over every field.
* **Paddy tells a radar story that no other land cover tells in the same order** (docs/01):
  1. **dry** field before the season: VH moderate;
  2. **standing water** at transplanting: VH very dark, because calm water reflects the radar
     away from the satellite like a mirror;
  3. **growing canopy**: VH bright again, the leaves and stems scatter the signal back.

  Rain-sown dry-land crops do 1 and 3 but never 2. Trees and villages are bright all the time.
  Permanent water is dark all the time.

Put three moments of that story into the red, green and blue channels of an image and the story
becomes a colour. That is all the composite does.

## 2. The two products

### 2.1 Rice RGB (3 bands): `sar_rice_rgb_mosaic.tif`

| band | colour | what it holds | the moment |
|---|---|---|---|
| 1 | **R**ed | VH, mean of all August-September passes | canopy |
| 2 | **G**reen | VH, the **darkest** pass of June-July | flood |
| 3 | **B**lue | VH, mean of all April passes | dry season |

Why the darkest pass for green and not the mean: the flood in a paddy lasts only a few weeks and
starts on a different date in every field. A two-month mean would wash it out; the minimum catches
it whenever it happened.

Why VH and not VV: VH (cross-polarised) reacts most strongly to vegetation volume, so the jump from
water to canopy is largest in VH (docs/01, sections 4-5).

What each kind of ground looks like (bright band = high value = lots of that colour):

| ground | R (canopy) | G (flood) | B (dry) | looks |
|---|---|---|---|---|
| paddy rice | high | very low | low | **red / magenta** |
| dry-land crop (rain-sown) | high | high | low | **yellow / orange** |
| trees, villages | high | high | high | **white / light grey** |
| permanent water (rivers, ponds) | low | low | low | **black** |
| bare / fallow all season | low-mid | mid | low | **dark blue-grey** |

Red plus green light makes yellow, red plus blue makes magenta, all three make white. A paddy has
red (canopy) but no green (the flood made June-July dark), hence red; if its April was a bit
brighter it leans to magenta. A dry-land crop has red **and** green (no flood, so June-July stayed
bright), hence yellow.

Other colours you will meet:

* **dark red / maroon**: rice whose canopy is still thin (young rice, late transplanting);
* **green-tinted or olive**: canopy in June-July that fell again by August-September, e.g. a crop
  already harvested (class 4);
* **blue-tinted**: bright in April, dark later, e.g. an early-season crop harvested before the
  monsoon, or a field that stayed flooded to September (late-flooded fields, still open water).

### 2.2 Monthly stack (12 bands): `sar_monthly_mosaic.tif`

Monthly means of every pass of every track, VH first, then VV:

| band | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| holds | VH Apr | VH May | VH Jun | VH Jul | VH Aug | VH Sep | VV Apr | VV May | VV Jun | VV Jul | VV Aug | VV Sep |

The months run April to September 2026. September holds only the passes up to the end of the radar
run (late September). The band names are also written into the file (QGIS shows them in the band
drop-down, e.g. `VH_2026-06 (dB x 100)`).

Use it to build your own combination of any three months (section 4.2).

Why the means are taken in **linear power** and not in dB: dB is a logarithmic scale. Averaging
-10 dB and -30 dB in dB gives -20 dB, but the real average power is -13 dB. The code converts each
pass to power, averages, and converts back (`composite_arrays` in the module).

## 3. How to build them

Run from the repository root. The radar runs (`monsoon2026`) must already be stacked for the AOIs
(docs/04); the GDAL command-line tools (`gdalbuildvrt`, `gdal_translate`, `gdaladdo`) must be on the
PATH (the module adds `/opt/gis/bin`).

```bash
# 1. Per AOI: reads every pass of every track (artefact passes dropped, docs/15)
.venv/bin/python -m sar_pipeline.sar_composites aoi --ids 11 106 116

# 2. One mosaic per product over every AOI built so far, plus the QGIS styles
.venv/bin/python -m sar_pipeline.sar_composites mosaic

# 3. Optional: a PNG of one AOI, the rice RGB next to the final class map
.venv/bin/python -m sar_pipeline.sar_composites preview --ids 116
```

Step 1 takes about 10 seconds for a small AOI and grows with AOI size and the number of passes; run
it for the AOIs you want (it can be split over several terminals). Step 2 mosaics whatever per-AOI
files exist, so re-run it after adding AOIs. Step 3 needs the AOI's final map
(`processed/_batch/s2_2026/aoi<N>/aoi<N>_monsoon2026_final.tif`, docs/13).

Outputs, all under `processed/_batch/s2_2026/qgis_review/sar/` (gitignored):

| file | what it is |
|---|---|
| `sar_rice_rgb_mosaic.tif` | rice RGB, all AOIs, 3 bands, with overviews (fast zooming) |
| `sar_monthly_mosaic.tif` | monthly stack, all AOIs, 12 bands, with overviews |
| `sar_rice_rgb.qml` | QGIS style for the rice RGB (R 1, G 2, B 3, -26..-12 dB) |
| `sar_monthly_VH_Sep-Jun-Apr.qml` | QGIS style for the monthly stack: R = VH Sep (6), G = VH Jun (3), B = VH Apr (1) |
| `per_aoi/aoi<N>_sar_rice_rgb.tif`, `per_aoi/aoi<N>_sar_monthly.tif` | the same, one AOI each, on the AOI's 10 m grid |
| `preview_aoi<N>.png` | step 3: the composite and the final map side by side |
| `sar_rice_rgb.vrt`, `sar_monthly.vrt`, `*_files.txt` | intermediate files of the mosaic step; can be ignored |

## 4. How to look at it in QGIS

### 4.1 The rice RGB, step by step

1. **Layer > Add Layer > Add Raster Layer...**, pick `sar_rice_rgb_mosaic.tif`.
2. Right-click the layer > **Properties... > Symbology**.
3. At the bottom: **Style > Load Style...**, pick `sar_rice_rgb.qml`, **OK**.

   Or set it by hand:
   * **Render type**: Multiband color;
   * **Red band** = Band 1, **Green band** = Band 2, **Blue band** = Band 3;
   * for each band **Min** = `-2600`, **Max** = `-1200` (the values are dB x 100, so this is
     -26 to -12 dB);
   * **Contrast enhancement**: Stretch to MinMax.

   Why a fixed range and not QGIS's automatic one: the automatic stretch is computed separately for
   each band and each view, so the same field changes colour as you pan. One fixed range for all three
   bands keeps "red" meaning the same everywhere.
4. Put the class map on top (the class mosaic and `classes.qml` from `qgis_review build`, see the
   README) and the field polygons as outlines only (Symbology > Simple fill > Fill style: No Brush).
   Toggle the class map on and off: standing rice (class 1, 6) should sit on red / magenta, class 3
   mostly on yellow / orange (docs/12), class 5 on white.
5. **Zoom to fields, not to pixels.** Judge a colour over a block of pixels in the middle of a
   field, never a single pixel or the field edge (speckle, section 5).

Tip: if the image looks too dark or too washed out in an area, change Min/Max for all three bands
together (e.g. -2800..-1300). Changing one band only shifts every colour.

### 4.2 Your own combinations from the monthly stack

Load `sar_monthly_mosaic.tif`, Symbology > Multiband color, pick three bands (table in 2.2), set
Min/Max per band. Or load `sar_monthly_VH_Sep-Jun-Apr.qml` as a starting point and change the bands.

| idea | R | G | B | Min..Max (dB x 100) | read it as |
|---|---|---|---|---|---|
| the default: canopy / flood / dry | 6 VH Sep | 3 VH Jun | 1 VH Apr | -2600..-1200 | like the rice RGB, but June is a mean, not the darkest pass, so the flood is weaker |
| late transplanting | 6 VH Sep | 4 VH Jul | 2 VH May | -2600..-1200 | fields flooded in July instead of June turn red |
| early transplanting | 5 VH Aug | 2 VH May | 1 VH Apr | -2600..-1200 | fields flooded in May turn red |
| the same story in VV | 12 VV Sep | 9 VV Jun | 7 VV Apr | -2000..-500 | VV is brighter than VH, hence the higher range; VV sees water well but canopy less |

Rule of thumb: put the **canopy** month in red, the **flood** month in green and a **dry** month in
blue, and paddy comes out red.

### 4.3 One month in VV / VH / VV-VH

A different style of picture: one date, three polarisation views. R = VV, G = VH, B = VV minus VH
(the "ratio" in dB). The ratio band is not in the file, so:

1. **Raster > Raster Calculator...** three times, each saved as a GeoTIFF (no-data pixels stay
   no data):
   * `vv_sep.tif`: `"sar_monthly_mosaic@12"`
   * `vh_sep.tif`: `"sar_monthly_mosaic@6"`
   * `ratio_sep.tif`: `"sar_monthly_mosaic@12" - "sar_monthly_mosaic@6"`
2. **Raster > Miscellaneous > Build Virtual Raster...**, inputs in this order: `vv_sep.tif`,
   `vh_sep.tif`, `ratio_sep.tif`; tick **Place each input file into a separate band**.
3. Multiband color, R = 1, G = 2, B = 3. Starting ranges: VV -2000..-500, VH -2600..-1200,
   VV-VH 400..1400.

Reading it: a full canopy has a strong VH, so a low ratio (little blue, looks green-yellow); open
water and smooth bare soil have almost no VH, so a high ratio (blue); buildings are very bright in VV
(red-white). It is a snapshot of one month, not the season story; use it next to the rice RGB, not
instead of it.

## 5. Caveats

* **Values are dB x 100, int16.** `-1523` means -15.23 dB. No data is `-32768` (set as the file's
  no-data value, so QGIS hides it). Why int16: a quarter of the size of float64, with 0.01 dB steps,
  far finer than anything the radar can tell apart.
* **Tracks differ.** Most AOIs are seen by more than one track (orbits at different angles). The
  monthly means mix all passes of all tracks. A steeper angle gives a brighter VH, about 1-2 dB for
  the same field (docs/01, section 7), so where two neighbouring AOIs are covered by different
  tracks a faint seam can appear at the AOI border in the mosaic. Compare fields with their
  neighbours, not with fields in another AOI across a seam. Where AOIs overlap, the mosaic shows one
  AOI's values (the one listed later in `*_files.txt`).
* **Speckle.** The composites are built from single 10 m pixels (no spatial smoothing, so field
  edges stay sharp). A single pixel on a single pass wobbles by about 2 dB (docs/01, section 9);
  the monthly means average several passes and so are calmer. The **darkest** pass (green band of
  the rice RGB) is the exception: taking the minimum of noisy values picks the darkest wobble, so
  green is a bit darker than the true level everywhere and grainy. That is why a single dark-green
  pixel means nothing; judge a block of pixels in the field interior. At 5 m or finer screen zoom
  you are looking at speckle, not at the field.
* **Artefact passes are already excluded.** The 15 passes found broken in the final audit (docs/15)
  are dropped when the radar is read (`sar_curve.read_track`), so they cannot paint a false flood.
* **Colours are relative.** The table in 2.1 is how things usually look, not a classifier. A field
  flooded only after the canopy grew (late floods) or a field sown very late will not be red. The map
  decides; the composite is for checking it by eye and for spotting places where the map may be wrong.
* **The overviews** (the lower-resolution copies QGIS uses when zoomed out) are averaged in dB, which
  is fine for looking but not for measuring. Read values at full zoom (Identify Features tool).
