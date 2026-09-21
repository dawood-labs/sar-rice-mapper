# 08 — Ground-truth analysis

> ### ⛔ Read this first: **this project has no ground truth**
>
> Every step on this page starts from **labelled ground-truth polygons** — fields somebody visited or
> otherwise confirmed, with a crop name and a crop code, configured under `gcps`. This project has
> **no such polygons**. Nothing on this page can be run as written until labels exist.
>
> This is not a gap in the code. The `analysis` package is carried over unchanged from a sibling
> project that had a labelled field campaign, and it works. What is missing is its input.
>
> **What to do instead**, in order — the full recipe is
> [README → *Working without ground truth*](../README.md#working-without-ground-truth):
>
> 1. **Look at real curves** with the pixel explorer ([04 Runbook step 8](04_runbook.md#step-8--pixel-explorer--notebook-04)),
>    grouped by AOI polygon rather than by class.
> 2. **Apply the label-free rice recipe**: VH minimum, maximum and variance over a season window that
>    contains the flooding minimum *and* the harvest drop
>    ([01 SAR basics §5](01_sar_basics.md#5-how-crops-look-over-a-season)). This yields *candidates*, not labels.
> 3. **Get a human verdict** from a domain expert, cross-checked against any cloud-free optical imagery.
> 4. **Only then** write the confirmed signatures out in the shape `gcps` expects — and this page
>    applies from its first line, unchanged.
>
> Never skip step 3. Labels derived purely from thresholds train a model that reproduces the
> thresholds, and its scores look excellent while meaning nothing. In particular, every score
> described in sections 4 and 5 below (cross-validation F1, the holdout test, the bootstrap
> intervals) measures agreement with the labels you supply — so with synthesised labels it measures
> agreement with your own thresholds, not with the ground.
>
> Read the rest of this page now anyway: it tells you which columns the labels must have (section 1),
> and knowing that up front stops the label set from having to be rebuilt later.

Once labelled fields exist, then after a run has been downloaded and stacked the
`sar_pipeline.analysis` package answers three questions with labelled fields (ground-truth polygons):

1. **Are the labels usable?** Geometry checks and a time-series review of every field.
2. **Which features separate the classes?** Per-pixel features compared with spatial cross-validation.
3. **What does a map look like?** An exploratory class map of an AOI.
4. **How good is a tuned model on fields it never saw?** The model phase: holdout test and a validated map.

Nothing here talks to Earth Engine: every step reads local run outputs only.

---

## 1. Setup

Add the ground truth and an `analysis:` section to your config (see `config/pipeline.example.yaml`):

```yaml
gcps:
  path: "data/gcps/<folder>/<polygons>.shp"   # local, gitignored
  class_field: <class-name-field>              # e.g. crop name
  id_field: <class-code-field>                 # numeric code per class (used in rasters)
analysis:
  target_class: "<class-name>"
  bins: {kind: half_month, days: 15, start: null}
  ...
```

Class names and codes are always read from the ground-truth file; nothing is hard-coded.

---

## 2. Steps

All steps: `python -m sar_pipeline.analysis <step> --config config/<aoi>_<season>.yaml --run <run_id>`
(`--run` defaults to the newest run). Outputs go to `processed/<aoi>/<season>/analysis/<run_id>/`.

| Step | What it does | Main outputs |
|---|---|---|
| `gcp-qc` | geometry checks: validity, overlaps, area, pure pixels, share inside the AOI | `gcp_vector_qc.csv`, `gcp_overlaps.csv` |
| `gcp-eda` | per-field mean VV/VH per date, class curves, outlier flags, review list | `gcp_timeseries.csv`, `class_curves.png`, `gcp_outlier_flags.csv`, `gcp_review.csv` |
| `gcp-qc --write-copy` | copies the ground truth with `qc_status`, `qc_reason`, `use_in_analysis` | `<polygons>_qc.gpkg` next to the original |
| `features` | per-pixel features of the analysed fields | `<name>.parquet` + `<name>.json` |
| `evaluate` | spatial CV of every feature set in the file | `cv_summary_<name>.csv`, `confusion_<set>.csv`, optional `oof_<set>.parquet` |
| `cv-layers --set S` | CV results for QGIS | `cv_fields_<set>.gpkg`, `cv_pixels_<set>.tif` + `.qml` |
| `class-map --set S --map-config C` | class map of the AOI of config `C` | `rf_<set>_classes.tif` + `.qml`, `rf_<set>_<target>_prob.tif`, `rf_<set>_classes.json` |
| `model [--map-config C]` | holdout, tuning, threshold, holdout test, validated map (section 5) | `model_<set>/` folder; map `model_<rf\|xgb>_<set>_*` in the analysis folder of `C` |
| `final-models` | refit the `model` winner on all fields as variants for missing data + a field-level model (section 5) | `model_<set>/final_models/*.joblib` + `model_card.json` |
| `predict --map-config C` | map of the AOI of `C` with the final models, best available model per pixel (section 5) | `<name>_classes.tif`, `_classes_argmax.tif`, `_<target>_prob.tif`, `_model_used.tif`, `_summary.json` |
| `field-labels --delineation D --class-raster R` | clean a field delineation and label every polygon from the class map (section 6) | `fields/<name>/fields_labelled.gpkg` + `.parquet` + `summary.json` |

A typical session:

```bash
python -m sar_pipeline.analysis gcp-qc   --config config/my_aoi.yaml
python -m sar_pipeline.analysis gcp-eda  --config config/my_aoi.yaml
# look at gcp_review.csv (and the fields in QGIS), edit it if needed, then:
python -m sar_pipeline.analysis gcp-qc   --config config/my_aoi.yaml --write-copy
python -m sar_pipeline.analysis features --config config/my_aoi.yaml
python -m sar_pipeline.analysis evaluate --config config/my_aoi.yaml --save-oof
python -m sar_pipeline.analysis cv-layers --config config/my_aoi.yaml --set hm_w5_s0501
python -m sar_pipeline.analysis class-map --config config/my_aoi.yaml --set hm_w5_s0501 --map-config config/other_aoi.yaml
python -m sar_pipeline.analysis model     --config config/my_aoi.yaml --map-config config/other_aoi.yaml
```

### Field QC never deletes anything

The review flags are suggestions. The original file is never modified; the copy keeps every field and only
marks `check_label` and `mixed_pixels` as `use_in_analysis = False`. `note_water` / `note_builtup` are kept
(they are real members of a catch-all class). Check flagged fields on optical imagery before relabelling.

Manual review decisions are made in the QC copy, never in the original file. Keep an audit trail:
- back up the copy first;
- keep the old class in `crop_original`;
- record each change in `review_status` (for example `relabelled`, `returned_label_ok`, `excluded_...`) and
  `review_note`, with the evidence;
- when relabelling, change the class field *and* the class code field together (the pair must stay one-to-one).

The analysis only reads the class field and `use_in_analysis`, so the extra columns are for people. Do not run
`gcp-qc --write-copy --force` after a manual review: it rebuilds the copy from the automatic flags and discards
those decisions.

After the labels change, the `model` step refuses to reuse the old `model_<set>/` folder, because the holdout
split no longer matches. Move that folder away, together with the old `model_*` map files. Then rerun with a new
`--name`, so the features are extracted again with the new labels.

### Comparing settings in one pass

`features` accepts several values at once, and every combination becomes a *feature set*:

```bash
python -m sar_pipeline.analysis features --config config/my_aoi.yaml \
    --bins half_month 8 12 --starts 2026-04-01 2026-05-01 --windows 5 --name bin_test
python -m sar_pipeline.analysis evaluate --config config/my_aoi.yaml --name bin_test
```

Set names read `<bins>_w<window>_s<start MMDD>`, e.g. `hm_w5_s0501` = half-month bins, 5×5 window, from 1 May.
All sets share exactly the same pixels, so their scores are directly comparable.

---

## 3. How the features are built

For each pixel, track and time bin: **VH**, **VV** and **VH−VV** (dB).

- **Time bins.** Acquisitions are grouped into calendar half-months (1–15, 16–end). One track revisits every
  ~12 days, so a 15-day bin almost always holds one acquisition per track and every field gets the same
  columns. Shorter bins leave many bins empty; longer bins blur fast crop changes.
- **Averaging in linear power.** Values are averaged as power and converted to dB at the end. Averaging dB
  values would bias the mean low.
- **5×5 window.** Speckle remains even after filtering. Averaging a 5×5 window (50 m) around each pixel
  steadies the value; nodata pixels in the window are ignored.
- **Rain rule.** Rain wets soil and leaves and raises backscatter for a day or two without any crop change. An
  acquisition with ≥ `rain_mm_24h` in the previous 24 h is dropped if its bin also has a dry one.
- **Empty bins** are linearly interpolated in time (the first/last value is repeated at the ends); the number
  of interpolated bins per set and track is written to the `.json` file.
- **Non-finite values** (for example log of zero power) are never silently kept or dropped: rows with any
  non-finite value are removed and counted in the log and the `.json` file.
- **Pixel cap.** At most `max_pixels_per_field` pixels per field (fixed seed), so a few very large fields do not
  dominate training.

### Choosing the first date of the feature window

> **The `s0501` (1 May) set name used as an example throughout this page is exactly that — an example.**
> It comes from the sibling project this code was carried over from, whose crop and crop calendar were
> different. **Do not adopt it for rice without checking.** That project measured that starting later
> than mid-May cost it about 0.08 F1 on its target crop, while adding April gained nothing measurable.
> That finding is about *its* crop in *its* region, not a property of this pipeline.

The example config therefore keeps `start: null` (= `season.start`), because the right first date
follows from the crop calendar of the area you are mapping.

**For rice the rule is set by the flooding minimum.** The features must begin **before** the fields are
flooded, so that the minimum has a dry baseline to be a minimum *against*; a window that starts after
transplanting throws away the one feature that separates rice most cleanly, and the season's VH
variance collapses with it. So choose the first half-month **before** the earliest expected flooding
date in the AOI, and make sure the window also reaches past the harvest drop
([01 SAR basics §5](01_sar_basics.md#5-how-crops-look-over-a-season)).

Do not guess: `python -m sar_pipeline.prep s1-availability --config $CFG --start … --end …` prices a
candidate window in seconds before anything is exported
([04 Runbook step 0b](04_runbook.md#0b--does-sentinel-1-cover-the-aoi-often-enough)), and once a run
is stacked, `features --starts …` accepts several candidate start dates at once so their scores are
directly comparable (see *Comparing settings in one pass* above).

---

## 4. How the evaluation works

- **Spatial cross-validation.** Neighbouring fields share soil, weather and management, so they look alike. A
  random split would test on near-copies of training fields and inflate scores. Fields are grouped into
  `group_cell_m` cells (5 km by default), and all fields of a cell are always in the same fold.
- **Metrics.** Precision, recall and F1 of the target class (per pixel and by majority vote per field), per-class
  F1, accuracy and macro F1. Use target-class F1 as the headline number: accuracy is dominated by the largest
  classes.
- **Why a fixed Random Forest.** The goal is to compare *features*, so the model and its settings stay fixed
  (`analysis.rf`). RF needs no scaling, handles correlated features and is hard to overfit badly, which makes
  it a stable diagnostic baseline. It is **not** a tuned production model; tuning is the model phase
  (section 5).

---

## 5. Model phase

The `model` step turns the chosen feature set (the one in `analysis.bins` / `window_px`) into a tested model
and a map. It reuses `--name <file>.parquet` when that file holds the same feature set, otherwise it extracts
the features once. Every stage writes its result to `analysis/<run_id>/model_<set>/` and is skipped on a rerun.

**1. Lock away a holdout.** About 20 % of the fields are set aside *before* anything is tuned. Whole 5 km
blocks (`group`) go to the holdout, never single fields, so holdout fields have no neighbours in training.
The block choice is a seeded search: many random block orders are tried and the one is kept whose class
shares are closest to the full data, with at least 15 holdout fields per class. `holdout_split.csv` lists
every field's block and split; `holdout_composition.csv` gives fields and pixels per class. The code raises
an error if a holdout field ever reaches tuning or threshold selection.

**2. Tune on the rest (dev).** Two model families, each with a small grid, scored with grouped 5-fold CV on the
dev fields only:

| Model | Grid |
|---|---|
| Random Forest | `n_estimators` 200/500, `max_features` sqrt/0.3, `min_samples_leaf` 1/2/5, `class_weight` balanced/none |
| XGBoost (`hist`) | `max_depth` 4/6/8, `learning_rate` 0.05/0.1, `n_estimators` 300/600, `subsample` 0.8, `colsample_bytree` 0.6/0.9, balanced sample weights on/off |

The winner has the best target-class F1; ties (to 3 decimals) go to the better macro F1. All grid scores are
in `tuning_results.csv`. To save time, settings that differ only in `n_estimators` are fitted once with the
larger number and scored with the first N trees (RF) or boosting rounds (XGBoost), which gives exactly the
same predictions as fitting the smaller model separately.

**3. Pick a target threshold.** By default a pixel gets the class with the highest probability (*argmax*).
When the target class is often confused with one neighbour, a lower or higher bar for the target trades
precision against recall. The rule: *predict the target when its probability is ≥ t, otherwise the most
likely other class*. `t` from 0.20 to 0.80 (step 0.05) is scored on the winner's dev out-of-fold
probabilities and the t with the best target F1 is kept (`threshold_curve.csv`, `winner.json`).

**4. Test once on the holdout.** The winner is fitted on all dev fields and scored on the holdout with both
rules (argmax and threshold): per-class precision/recall/F1, confusion matrices, field majority-vote accuracy
and target F1, and a 95 % interval of target precision/recall/F1 from 1000 bootstrap resamples of *fields*
(pixels of one field are not independent). Files: `holdout_metrics.json`, `holdout_per_class.csv`,
`holdout_confusion_<rule>.csv`. This is the number to quote. It is computed once: if `holdout_metrics.json`
exists it is not recomputed, so the holdout cannot be tuned against by rerunning.

**5. Map.** The winner is refitted on *all* fields (dev + holdout: more data, same settings) and applied to
the AOI of `--map-config` with the threshold rule. Rain flags come from the training run, as in `class-map`.
Outputs in that AOI's analysis folder: `model_<rf|xgb>_<set>_classes.tif` (+ `.qml`, nodata 0),
`model_<rf|xgb>_<set>_<target>_prob.tif` (percent, nodata 255) and a `.json` with class area shares under
both rules.

**6. Final models (`final-models`).** A large AOI has pixels where a track is missing: outside the secondary
track's swath, or on a date where one image slice is absent. A model that needs every feature cannot classify them.
`final-models` refits the winner (same settings and threshold) on all fields as variants: `full`, `only_<primary
track>`, and optionally `no_<track>_<MMDD>` per `--without-bin TRACK:MMDD`. Each variant gets a grouped 5-fold CV
score on all fields in `model_card.json`, so the price of a fallback is known. It also fits `field_level`: the same
model on the mean of each field's pixel features, with its own threshold, used by `field-labels` for the second label.

**7. Map a large AOI (`predict`).** Every AOI pixel gets the variant with the most features whose features are all
present; `<name>_model_used.tif` records which (the legend and acres per variant are in `<name>_summary.json`).
Chunks are processed in parallel (about 1 GB per worker is budgeted). Both rules are written: the threshold rule as
`<name>_classes.tif` and argmax as `<name>_classes_argmax.tif`, with acres per class in the summary. The band layouts
of the training run and the map run must be identical.

To start over (new fields, new features), move the `model_<set>/` folder away: the step refuses to reuse it
when the holdout split no longer matches the data.

---

## 6. Field labels

A class map answers *what grows here* pixel by pixel; a client wants *which crop is in this field*. The
`field-labels` step joins the two, given a delineation — field outlines traced on high-resolution imagery, by
hand or by a segmentation model. One rule decides every question in it:

> **the delineation owns the geometry, the class map owns only the label.**

A 10 m raster edge is never allowed to become an output edge, except where the delineation has no polygon at
all.

```bash
python -m sar_pipeline.analysis field-labels --config config/my_aoi.yaml \
    --delineation data/fields/<layer>.shp \
    --class-raster processed/<aoi>/<season>/analysis/<run>/model_xgb_<set>_classes.tif \
    --prob-raster  processed/<aoi>/<season>/analysis/<run>/model_xgb_<set>_<target>_prob.tif \
    --model .../final_models/xgb_field_level.joblib \
    --train-config config/<training aoi>.yaml --train-run <run> --name fields_v001
```

**0. Simplify.** Outlines vectorised from a sub-metre mask carry a vertex every few decimetres: a pixel
staircase, not a boundary anyone drew, and every step below pays for each vertex. They are simplified by
`--simplify-m` metres first (default 0.5). On a traced layer this kept 11 % of the vertices, changed the area by
0.13 % and moved the median boundary by 0.5 m, far below the 10 m pixel that decides the label, and made
buffering about 100 times faster. Each polygon is simplified on its own; the few square decimetres of overlap
that can create between neighbours are resolved in the next step. `--simplify-m 0` keeps every vertex.

**1. Repair.** `make_valid`, then explode until nothing is nested. This matters more than it sounds: `make_valid`
returns GeometryCollections, a collection can hold a MultiPolygon, and one explode followed by a
`geom_type == "Polygon"` filter silently drops those — real acres that come back later as ragged blobs. Then
overlapping ground is given to the **smaller** polygon, which keeps the finest tracing intact and trims the
coarser one around it; the topological slivers that fall out (under 10 m²) are dropped.

**2. Despike.** Traced layers carry whiskers: strips a metre or two wide running several metres out of an
otherwise sensible field. The test that catches every one of them is *a shape that disappears when eroded by
half a field's width was never a field*. Eroding and dilating alone would round every corner, so the opened body
is dilated back with mitred joins and intersected with the original: true edges and corners survive, the tails
stay outside. What is left of a line rather than a field then goes: compactness below 0.10, or a body under 0.05
acres. There is deliberately no "skip it if it would lose more than X %" clause — the polygons that lose most are
precisely the ones that are mostly tail.

**3. Label.** Class counts per polygon give the majority class and the *purity* (its share of the polygon's
classified pixels). Below 10 classified pixels a share is not a measurement: the polygon keeps its geometry, gets
the label, and is flagged `too few pixels to judge`.

**4. Cut mixed polygons.** A polygon under 0.85 purity usually holds more than one field. It gets one straight
cut along its own orientation — from its minimum rotated rectangle, because subdivisions run parallel to a
field's edges — at the offset that best separates the classes. Two guards: the cut has to raise purity by at
least 0.08, and both sides have to still look like a field. A cut that adds nothing invents a boundary the ground
does not have, and a cut running along the parent's own edge just shaves a needle off it.

**5. Derive.** Classified ground no polygon claims gets a polygon of its own: an opening first (a repair, not a
rejection — tentacles come off, a solid core survives), then the 0.15-acre floor, then shape gates on how much of
its rotated rectangle it fills and how much perimeter it carries. The rectangle-fill floor sits at 0.45 on
purpose: a triangle fills exactly half its rectangle and triangular fields are real. These polygons are tagged
`origin = derived` so nobody mistakes a raster edge for traced geometry.

**6. Second label.** If `--model` is given, each polygon also gets an independent label from the field-level
model applied to the **mean** of its pixel features (the same features the pixel model uses). `agree` says
whether the two labels match; where they do not, the field is worth a look.

**7. Tidy.** Validity, no overlapping area and no lines, in the CRS the file is written in — cleaning in metres
and reprojecting afterwards reopens what was just closed, because converting to degrees rounds coordinates.
Traced geometry is placed first and derived polygons last, so a boundary the class map invented can never cut one
that was drawn on imagery.

**Output** in `processed/<aoi>/<season>/fields/<name>/`: `fields_labelled.gpkg` (and `.parquet`), plus
`summary.json` with the counts and acres of every step. Columns worth filtering on in QGIS:

| Column | Meaning |
|---|---|
| `origin` | `delineation`, `split` (a piece of a cut polygon) or `derived` (from the class map) |
| `decision` | why this polygon looks the way it does |
| `majority_class`, `majority_share` | the label and how pure the polygon is |
| `field_model_class`, `field_model_target_prob` | the second, independent label |
| `agree` | do the two labels match? |
| `n_classified`, `classified_share` | how much of the polygon the class map covered |
| `area_acres`, `compactness` | size and shape |

**Areas are in acres throughout**, including the summary.

---

## 7. Caveats — read before quoting numbers

- **Scores describe field interiors.** Ground-truth polygons are drawn inside fields. Real maps also contain
  field edges, roads, trees and mixed pixels, where accuracy is lower.
- **Class balance.** Training uses `class_weight: balanced`; the class shares in the ground truth are not the
  shares in the landscape. Precision in a real map depends on how common each class really is.
- **The class map is not a validated product.** It is trained on all fields and applied to every AOI pixel,
  including places the CV never tested. Use it to spot patterns and errors, not to report areas. The model
  phase map has a holdout score behind it, but that score still describes field interiors (above), and the
  holdout is small: read its bootstrap interval, not just the point value.
- **The threshold is tuned for the ground-truth class mix.** Fields are sampled evenly per class, not in
  landscape proportions, so a threshold that maximises F1 on them can over- or under-map the target in a real
  AOI. Compare the argmax and threshold area shares in the map `.json`.
- **Rain flags of a map come from the training run.** Rain is averaged over each run's AOI, so a different AOI
  could flag different dates as wet. `class-map` uses the training run's flags so every bin picks the same
  acquisitions the model was trained on, and it stops if the two runs have different band layouts.
- **Probability raster.** `rf_<set>_<target>_prob.tif` stores 0–100 % with nodata 255, separately from the
  class raster (nodata 0), so a 0 % probability is not confused with "no data".
