# 00 — Step-by-step guide (start here)

This guide is for someone who **does not write code**. It walks through the whole project in order: setting up
the machine, getting Sentinel-1 radar data for an area, looking at the radar curve of a single pixel, training a
crop model, making a crop map of a large area, improving the model with your own corrections, and turning the map
into labelled field boundaries.

You only need to **copy commands, change the parts in `<angle brackets>`, and read the output**. Every command
here is safe to run again: if something stops halfway, run the same command again and it continues.

> **Areas are always in acres.** One 10 m pixel = 0.025 acre. The 5×5 pixel window the model looks at = 0.62 acre.

> ### ⛔ Before you start: Parts C–G need ground truth, and this project has none
>
> **Parts A and B work today.** They get the radar data and let you look at it. Start there.
>
> **Parts C to G all begin from labelled ground-truth polygons** — fields somebody confirmed on the
> ground, each carrying a crop name and a crop code. **This project does not have any.** The code is
> carried over from a sibling project that ran a field campaign, and it works; what is missing is its
> input. So do not be surprised when Part C asks for a file you have never been given: nobody has it.
>
> **The route from here to there** is in
> [README → *Working without ground truth*](../README.md#working-without-ground-truth). In short:
> look at real VH curves with Part B, find the rice pattern (a deep dip when the field is flooded,
> then a long rise, then a sharp drop) without any labels, have a **person who knows the area confirm**
> which of those are really rice, and only then write those confirmed fields into a file in the shape
> Part C expects. From that moment Parts C–G work exactly as written below.
>
> **Do not shortcut the confirmation step.** If you turn a rule like "the dip is below −20 dB" straight
> into labels and train on them, the model just learns your rule back, and every score in Part D will
> look excellent while telling you nothing about the ground.
>
> Read Parts C–G anyway before collecting anything: Part C (7.1) says which columns the labelled file
> must have, and knowing that now saves rebuilding the file later.

---

## Contents

1. [The big picture](#1-the-big-picture)
2. [Words you will see](#2-words-you-will-see)
3. [Working in a terminal](#3-working-in-a-terminal)
4. [One-time setup](#4-one-time-setup)
5. [Part A — Get radar data for an area](#5-part-a--get-radar-data-for-an-area)
6. [Part B — Look at the radar curve of one pixel](#6-part-b--look-at-the-radar-curve-of-one-pixel)
7. [Part C — Check the ground truth](#7-part-c--check-the-ground-truth) *(needs ground truth)*
8. [Part D — Train the model](#8-part-d--train-the-model)
9. [Part E — Make a crop map of a large area](#9-part-e--make-a-crop-map-of-a-large-area)
10. [Part F — Improve the model with your corrections](#10-part-f--improve-the-model-with-your-corrections)
11. [Part G — Field boundaries with crop labels](#11-part-g--field-boundaries-with-crop-labels)
12. [Part H — Share results through cloud storage](#12-part-h--share-results-through-cloud-storage)
13. [When something goes wrong](#13-when-something-goes-wrong)
14. [Cheat sheet: every command on one page](#14-cheat-sheet-every-command-on-one-page)

---

## 1. The big picture

```
 AOI polygon ──► A. radar data ──► B. pixel curves (look)
                     │
 ground-truth ──► C. check labels ──► D. train model ──► E. crop map of a large area
 polygons                                   ▲                    │
                                            │                    ▼
                                  F. your corrections ◄── you review the map in QGIS
                                                                 │
                                   field boundaries ──► G. one crop label per field
```

| Part | What happens | Needs ground truth? | Where it runs | Typical time |
|---|---|---|---|---|
| 0 | Check the AOI file and the radar coverage (5.2) | no | your machine + EE metadata | seconds |
| A | Google Earth Engine cleans the radar images and saves them to cloud storage; you download them | no | Earth Engine + your machine | hours (export), minutes to an hour (download) |
| B | Plot how the radar signal of one spot changed through the season | no | your machine | seconds |
| C | Check the labelled fields (ground truth) for mistakes | **yes** | your machine | minutes |
| D | Train and test the crop model | **yes** | your machine | 30–90 minutes |
| E | Classify every pixel of an area | **yes** (via D) | your machine | minutes |
| F | Add polygons where the map is wrong, retrain, map again | **yes** | your machine | as D + E |
| G | Give every field polygon a crop label | **yes** (via D/E) | your machine | minutes |

Nothing in Parts B–G talks to Earth Engine. Only Part A costs Earth Engine quota.

---

## 2. Words you will see

| Word | Meaning |
|---|---|
| **AOI** | *Area of interest*: the polygon of the area you want to map. |
| **Sentinel-1** | A radar satellite. Radar sees through clouds, so it works in the rainy season. |
| **VV, VH** | The two radar measurements per image. VH reacts strongly to leaves and stems, so it rises as a crop grows. Values are in **dB** (negative numbers, e.g. −18). |
| **Track** | A fixed satellite path, named like `RO123_ASC` (ascending, evening pass) or `RO045_DSC` (descending, morning pass). The same track sees the area from the same angle every 12 days or so. |
| **Config** | A text file (`config/<name>.yaml`) with all settings of one area and season. |
| **Grid** | The fixed 10 m pixel layout of an AOI. Built once, never changed. |
| **Chunk** | The grid is cut into square tiles (chunks) so each export stays small. |
| **Pixel id (pid)** | A number that identifies one pixel of the grid. |
| **Run** | One export of one area, in its own folder `runs/v001_<date>/`. Never overwritten. |
| **Export** | Earth Engine processing the images and writing GeoTIFF files to cloud storage. |
| **Stack** | The downloaded chunks joined into one virtual raster per date. |
| **Ground truth / GCP polygons** | Fields someone checked and labelled with their real crop. The model learns from these. **This project has none yet** — see the warning above. |
| **Feature** | One number the model looks at, e.g. "VH of the descending track in the second half of July". |
| **Spatial cross-validation (CV)** | Testing the model on fields from *other* 5 km blocks than the ones it trained on, so the score is honest. |
| **F1** | One score between 0 and 1 that combines "how many rice pixels were found" (recall) and "how many pixels called rice really are rice" (precision). 0.90 is good. |
| **Threshold** | The rice probability a pixel needs before it is called rice (e.g. 0.75). |
| **Delineation** | Field boundary polygons, drawn by hand or by a segmentation model. |

---

## 3. Working in a terminal

All commands are typed in a **terminal**:

- **JupyterHub:** *File → New → Terminal*.
- **Laptop (Linux / WSL):** open the Ubuntu or terminal app.

Rules that save a lot of trouble:

1. **Always start in the project folder** and with the Python environment switched on:
   ```bash
   cd ~/sar-rice-mapper          # or wherever you cloned it
   source .venv/bin/activate      # the prompt now starts with (.venv)
   ```
2. **Copy a command exactly**, then change only the `<...>` parts (remove the angle brackets too).
3. A line ending with `\` continues on the next line: copy all lines together.
4. **Short commands** (seconds to a few minutes): just run them.
5. **Long commands** (anything that says *long* in this guide): run them in the background with a log file, so they
   keep running when you close the browser tab:
   ```bash
   mkdir -p logs
   nohup <the command> > logs/<some_name>.log 2>&1 &
   ```
   Then watch the log:
   ```bash
   tail -f logs/<some_name>.log     # Ctrl+C stops WATCHING; the job keeps running
   ```
   Is it still running?
   ```bash
   ps aux | grep sar_pipeline | grep -v grep
   ```
   A background job stops only if the JupyterHub server itself is shut down. If that happens, run the same command
   again: finished work is skipped.
6. **Commands that start big work never start it without `--yes`.** Without `--yes` they only print what *would*
   happen (size, number of tasks) and exit. Read that, then run again with `--yes`.

To save typing, set the config once per terminal:

```bash
CFG=config/<my_area>.yaml
```

Then `$CFG` in a command means that file.

---

## 4. One-time setup

### 4.1 Get the code and install it

```bash
git clone git@github.com:<account>/sar-rice-mapper.git
cd sar-rice-mapper
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

`pip install -e` makes the commands `python -m sar_pipeline ...` work from the project folder. If you ever see
`No module named 'sar_pipeline'`, the environment is not switched on (`source .venv/bin/activate`) or this step was
skipped.

### 4.2 The key file

Earth Engine and cloud storage need a **service-account key** (a `.json` file). Get it from the project owner
through a safe channel, never by chat or email.

- JupyterHub: in the file browser, open the project folder, create a folder `secrets`, and upload the key there.
- Then protect it:
  ```bash
  chmod 600 secrets/*.json
  ```

**Never** paste the key into a notebook and never commit it. `secrets/` is ignored by git.

### 4.3 The private-words list

Create `secrets/private_terms.txt` with one word per line: the client name, place names and anything else that
must never appear in the public repository. A test checks every file git would publish against this list.

### 4.4 Data you already have (moving to a new machine)

If a project is moving from another machine, the owner may have uploaded the local state (configs, AOI and
ground-truth files, grids, run bookkeeping, trained models) to cloud storage. Copy it **into the project folder**,
keeping its folders:

```bash
gcloud storage cp -r "gs://<bucket>/<project-folder>/handover/<bundle>/*" .
```

If `gcloud` is not installed, see [Part H](#12-part-h--share-results-through-cloud-storage) for a Python way.

### 4.5 Check the installation

```bash
pytest -q                                            # a few hundred tests, all should pass
python -m sar_pipeline --config $CFG resources       # needs a config, see 5.1
```

`resources` prints the CPUs, memory and free disk the pipeline will use. On a shared or small machine, jobs can be
killed for lack of memory; then cap the workers in your config:

```yaml
resources:
  cpu_workers: 4        # instead of auto
```

---

## 5. Part A — Get radar data for an area

This part is needed once per area and season. If the data of your area is already exported and downloaded, skip
to [Part B](#6-part-b--look-at-the-radar-curve-of-one-pixel).

### 5.1 Make a config

```bash
cp config/pipeline.example.yaml config/<my_area>_<season>.yaml
```

Open the copy in the JupyterHub editor (double-click it) and fill every `<placeholder>`:

| Setting | What to write |
|---|---|
| `aoi.key` | a short name without spaces, e.g. `north_block` |
| `aoi.path` | path of the AOI polygon, e.g. `data/aoi/north_block/north_block.gpkg` |
| `season.key` | e.g. `ws2026` |
| `season.start`, `season.end` | first day and the day **after** the last day, e.g. `"2026-04-01"`, `"2026-12-01"` |
| `auth.key_file`, `auth.project` | the key file path and the Earth Engine project name |
| `gcs.bucket`, `gcs.base_folder` | the storage bucket and the project folder inside it |
| `gcps` | the ground-truth file and its class-name and class-code columns (needed from Part C on) |
| `grid.crs` | one metric CRS for the whole area, e.g. `EPSG:326<zone>` (UTM north) |
| `analysis.target_class` | the crop of interest, spelled exactly as in the ground truth, e.g. `Rice` |

Leave `s1.tracks: []` for now. Config files stay on your machine; git ignores them.

> **Mapping an area with an existing model?** The new area must use the **same tracks and the same dates** as the
> run the model was trained on, otherwise the features do not line up. Copy the training config, change only
> `aoi.key` and `aoi.path`, and set `season.end` to the day after the training run's last image.

### 5.2 Prep checks — do these before anything else

Two checks that cost nothing and write nothing. They exist because the next step (the grid) silently
trusts two things nobody has verified: that your AOI file is the complete, correct AOI, and that
Sentinel-1 covers it often enough in your season. Getting either wrong wastes hours.

```bash
python -m sar_pipeline.prep aoi-qc --config $CFG
```

Reads your AOI file only — no internet, no Earth Engine. Check three things in the output:

- **`features` vs `distinct shapes`.** If they differ, the file contains duplicate polygons. The area
  total already ignores the duplicates, so it is not double-counted.
- **`area (acres)`.** Does the total match what you were told the area is? A factor of 2.47 out means
  somebody's number was in hectares.
- **the `largest` list.** Sanity-check the biggest AOIs. Polygons well under an acre (~40 pixels) will
  later be measured mostly from their neighbours.

If somebody split the AOI into one file per polygon and dropped duplicates, check that nothing was
lost on the way:

```bash
python -m sar_pipeline.prep aoi-qc --config $CFG --split-dir <that folder>
```

The last line must say `OK - faithful deduplication`. If it does not, it names exactly which polygons
are missing, which are in the split but not in the original, and which files were renumbered — then
**stop and ask**, do not continue. (Counting files cannot catch this: 160 polygons becoming 132 files
tells you nothing about *which* 28 went.)

```bash
python -m sar_pipeline.prep s1-availability --config $CFG
```

Asks Earth Engine which radar images exist. It reads **only the image descriptions**, never the images,
so it takes seconds and costs nothing. Check:

- **`tracks`** — how many satellite paths cover the area. If there is only one, track choice in 5.4 is
  not a real decision.
- **`distinct dates / month`** — roughly 2–3 per month per track is normal. A month far below the
  others is a gap that will blur the crop curve there.

Trying out a different season window is free, so do it here rather than after exporting:

```bash
python -m sar_pipeline.prep s1-availability --config $CFG --start 2026-05-01 --end 2027-02-01
```

> **For rice, the window must contain the flooding dip *and* the harvest drop**, plus some dry weeks
> before flooding. If it stops before harvest, rice stops being recognisable at all. Take the dates
> from a crop calendar for the area — see [01 SAR basics §5](01_sar_basics.md#5-how-crops-look-over-a-season).

Both commands are safe to run as often as you like. They never export, never upload, never download
and never create a folder. **The track list for the config comes from the audit (5.4), not from
here** — these counts are over a rectangle around the AOI, not the AOI itself.

### 5.3 Grid

```bash
python -m sar_pipeline --config $CFG grid
```

Prints the grid size and the number of chunks. It writes `processed/<aoi>/<season>/grid/`, including
`chunks.gpkg` (open it in QGIS over the AOI) and `pixel_index.tif` (pixel ids, used in Part B).

### 5.4 Audit: which satellite passes exist?

```bash
python -m sar_pipeline --config $CFG audit
```

Reads only image metadata; it costs nothing. Open `processed/<aoi>/<season>/audit/<date>/`:

| File | Look for |
|---|---|
| `track_summary.csv` | tracks with many dates across the whole season and high AOI coverage |
| `track_recommendation.csv` | the suggested primary and secondary track and why |
| `gaps_report.md` | long gaps (more than about 18 days in the growing season is a problem) |
| `chunk_track_coverage.csv` | chunks a track never covers |

Write your choice into the config:

```yaml
s1:
  tracks:
    - {track_id: RO123_DSC, role: primary}
    - {track_id: RO045_ASC, role: secondary}
```

Two tracks (one ascending, one descending) usually map crops clearly better than one.

### 5.5 Create a run

```bash
python -m sar_pipeline --config $CFG new-run
```

Creates `processed/<aoi>/<season>/runs/v001_<date>/` and freezes a copy of the config inside. Later edits of your
config do not change this run; for new settings, make a new run.

### 5.6 Pilot export (one chunk first)

Pick one chunk fully inside the AOI from `chunks.gpkg` in QGIS (column `name`, e.g. `chunk_r02c03`).

```bash
python -m sar_pipeline --config $CFG export --pilot chunk_r02c03          # shows the plan only
python -m sar_pipeline --config $CFG export --pilot chunk_r02c03 --yes    # starts it
python -m sar_pipeline --config $CFG monitor --yes                        # waits until done
python -m sar_pipeline --config $CFG download --yes
python -m sar_pipeline --config $CFG stack
```

Open a date file from `runs/<run>/stack/track_<id>/VH/` in QGIS over a satellite basemap. Check: fields line up
with the basemap, values are roughly −25 to −10 dB for VH, field edges look sharp. Only then export everything.

### 5.7 Full export (*long*)

```bash
python -m sar_pipeline --config $CFG export --all            # read: number of tasks, chunks not covered
python -m sar_pipeline --config $CFG export --all --yes
nohup python -m sar_pipeline --config $CFG monitor --yes > logs/monitor.log 2>&1 &
tail -f logs/monitor.log
```

`monitor` resubmits failed tasks by itself and splits chunks that are too heavy. "Not covered" chunks are chunks a
track never flies over; that is normal at the edge of a swath. It finishes with a count per state; everything
should be `COMPLETED`. If `runs/<run>/failed_chunks.csv` is not empty:

```bash
python -m sar_pipeline --config $CFG export --retry-failed --yes
nohup python -m sar_pipeline --config $CFG monitor --yes > logs/monitor.log 2>&1 &
```

### 5.8 Download (*long*)

```bash
python -m sar_pipeline --config $CFG download                # prints Files, Size, Disk free, Fits
nohup python -m sar_pipeline --config $CFG download --yes > logs/download.log 2>&1 &
```

Every file is checked after download. If the machine or network stopped, run the same command again. If
`Fits: False`, free disk space first.

### 5.9 Stack and quality check

```bash
python -m sar_pipeline --config $CFG stack
```

If it stops with **DecisionRequired**, open `runs/<run>/decisions_required.md`. It lists dates with too few
valid pixels, missing images or failed chunks. To accept an item, copy its id into your config and run `stack`
again:

```yaml
qa:
  acknowledged_issues:
    - "LOW_VALID:RO045_ASC:20260809"
```

The run is now ready for Parts B–G.

---

## 6. Part B — Look at the radar curve of one pixel

You need a run that has been **downloaded and stacked** (Part A).

### 6.1 Find the pixel

Either way works:

- **Longitude / latitude:** in QGIS, right-click the map → *Copy Coordinate* → choose `EPSG:4326`. That gives
  `longitude, latitude`. (Google Maps gives `latitude, longitude`: the other way round!)
- **Pixel id:** open `processed/<aoi>/<season>/grid/pixel_index.tif` in QGIS, click the spot with the *Identify*
  tool; **band 1** is the pixel id.

### 6.2 Print and plot the curve

```bash
python -m sar_pipeline --config $CFG pixel --track <RO123_DSC> --lon <longitude> --lat <latitude> --plot logs/pixel.png
python -m sar_pipeline --config $CFG pixel --track <RO123_DSC> --pid <pixel id> --plot logs/pixel_<pixel id>.png
```

The table lists every date with VV, VH, VH−VV (dB), rain and satellite. Open the `.png` in JupyterHub. Run it once
per track.

For an interactive plot (zoom, hover), open `notebooks/04_pixel_explorer.ipynb`, set `CONFIG_PATH` and `PID` in the
first cells, and *Run → Run All Cells*. Clear outputs before sharing the notebook: it shows locations.

### 6.3 Reading the curve

- **VH rises for weeks, then falls:** a crop growing and being harvested. The length of the cycle and
  whether it starts with a flooding dip say which crop it is: a **deep dip first, then ~4–5 months of
  rise and fall → rice**; ~4 months with **no** dip → a dryland crop such as maize; a high plateau
  lasting 10–12 months → sugarcane.
- **Very low VV and VH early in the season** (below about −20 dB VV): standing water, typical for rice.
- **A one-date jump right after rain:** wet soil and leaves, not growth.
- Flat all season: bare land, water, buildings or trees, depending on the level.

### 6.4 Average curves of all ground-truth fields

`gcp-eda` in Part C writes `class_curves.png`: the typical curve of every class, the fastest way to see how the
crops differ.

---

## 7. Part C — Check the ground truth

> **This project has no ground-truth file yet** (see the warning at the top of this page). Read 7.1
> now, so that whatever is eventually collected has the right columns from the start; the commands in
> 7.2 have nothing to read until then.

The model can only be as good as the labelled fields it learns from.

### 7.1 What the ground-truth file needs

- Polygons (SHP or GPKG) with a **class name** column (e.g. `Crop`) and a **class code** column (e.g. `id`).
  One code per name, always the same pair.
- Polygons drawn **inside** fields, away from edges, trees and roads. A field smaller than about 0.62 acre (the
  model's window) is mostly measured from its neighbours.
- The file path and the two column names go into the config under `gcps`.

### 7.2 Automatic checks

```bash
python -m sar_pipeline.analysis gcp-qc  --config $CFG
python -m sar_pipeline.analysis gcp-eda --config $CFG
python -m sar_pipeline.analysis gcp-qc  --config $CFG --write-copy
```

Outputs in `processed/<aoi>/<season>/analysis/<run>/`:

| File | Meaning |
|---|---|
| `gcp_vector_qc.csv` | broken shapes, overlaps, tiny fields |
| `class_curves.png` | typical radar curve per class |
| `gcp_outlier_flags.csv`, `gcp_review.csv` | fields whose curve does not look like their class |

`--write-copy` creates `<ground-truth>_qc.gpkg` next to the original. **From now on the analysis reads this copy.**
The original file is never changed.

### 7.3 Your manual review

Open the `_qc.gpkg` in QGIS with a satellite basemap. For every flagged field, look at the imagery and the pixel
curve (Part B) and decide. Record decisions **in the copy**:

1. Make a backup first: copy the file to `<name>_qc_before_review<N>_<date>.gpkg`.
2. In QGIS: select the layer → *Toggle Editing* (pencil) → *Open Attribute Table*.
3. For a wrong label: keep the old name in `crop_original`, change **both** the class name and the class code.
4. To leave a field out: set `use_in_analysis` to false. Nothing is deleted.
5. Write what you did in `review_status` (e.g. `relabelled`, `excluded_trees_in_polygon`) and why in `review_note`.
6. *Save Edits*, then *Toggle Editing* off.

Do **not** run `gcp-qc --write-copy --force` after a manual review: it rebuilds the copy from the automatic flags
and your decisions are lost.

---

## 8. Part D — Train the model

The run used for training must contain the ground-truth fields and be downloaded.

### 8.1 Features

```bash
nohup python -m sar_pipeline.analysis features --config $CFG --name features_<round> > logs/features.log 2>&1 &
```

For every pixel inside a ground-truth field this computes, per track and half-month from the first date in
`analysis.bins.start`, the average VH, VV and VH−VV in a 5×5 window (0.62 acre). Output:
`analysis/<run>/features_<round>.parquet`.

### 8.2 Tune and test (*long*)

```bash
nohup python -m sar_pipeline.analysis model --config $CFG --name features_<round> > logs/model.log 2>&1 &
```

What it does, in `analysis/<run>/model_<set>/` (the set name looks like `hm_w5_s0501`: half-month bins, 5×5
window, from 1 May):

1. Locks away about 20 % of the fields as a **holdout**: whole 5 km blocks the model never sees during tuning.
2. Tries Random Forest and XGBoost settings with spatial cross-validation on the rest.
3. Picks the rice probability threshold.
4. Tests the winner **once** on the holdout.

Read these files:

| File | What to look at |
|---|---|
| `winner.json` | chosen model and threshold; `dev_oof_at_threshold.f1` = the cross-validation F1 of the target crop |
| `holdout_metrics.json` | under `threshold`: `target_f1`, its 95 % interval `target_ci95`, and `field_vote_accuracy` |
| `holdout_confusion_threshold.csv` | which classes are confused with which (rows = truth, columns = prediction) |
| `tuning_results.csv` | every setting that was tried |

If the step says the `model_<set>` folder does not match the data (because the fields changed), rename the old
folder first (see [Part F](#10-part-f--improve-the-model-with-your-corrections)).

### 8.3 Final models (*long*)

```bash
nohup python -m sar_pipeline.analysis final-models --config $CFG --name features_<round> \
    > logs/final_models.log 2>&1 &
```

Fits the winner on **all** fields and saves, in `model_<set>/final_models/`:

| File | Used for |
|---|---|
| `xgb_full.joblib` | pixels with every feature (almost all pixels) |
| `xgb_only_<primary track>.joblib` | pixels the secondary track never covered (edges of its swath) |
| `xgb_field_level.joblib` | Part G: one label per field from the field's average features |
| `model_card.json` | settings, threshold, and a cross-validation score for every model |

If one track misses a single date over part of the area (for example one missing image slice), add a model without
that time bin: `--without-bin <RO045_ASC>:<0801>` (track, then the bin's start as month and day). The audit's
`chunk_track_coverage.csv` and the stack's `dates.csv` show such gaps.

---

## 9. Part E — Make a crop map of a large area

You need:

- the **training run** with its `final_models/` (Part D), and
- the **area's run**, downloaded, with the same tracks and dates (see the note in 5.1).

```bash
TRAIN=config/<training_area>.yaml
MAP=config/<large_area>.yaml
nohup python -m sar_pipeline.analysis predict --config $TRAIN --map-config $MAP --name <map_name> \
    > logs/predict.log 2>&1 &
```

If the model was trained on the large area's own run (Part F), `--config` and `--map-config` are the same file.

Every AOI pixel gets the model with the most features its data allows. Outputs in
`processed/<large area>/<season>/analysis/<run>/`:

| File | Content |
|---|---|
| `<map_name>_classes.tif` (+ `.qml`) | crop class, **main map** (rice when its probability ≥ the threshold) |
| `<map_name>_classes_argmax.tif` | crop class with the highest probability, for comparison |
| `<map_name>_Rice_prob.tif` | rice probability 0–100 % (255 = no data) |
| `<map_name>_model_used.tif` | which model classified the pixel (numbers explained in the summary) |
| `<map_name>_summary.json` | acres per class, acres per model, time taken |

### 9.1 Look at the map in QGIS

1. Add a satellite basemap: *Browser → XYZ Tiles → New Connection*, URL
   `https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}`.
2. Drag in `<map_name>_classes.tif`: the colours come from the `.qml` automatically.
3. Drag in `<map_name>_Rice_prob.tif`: *Properties → Symbology → Singleband pseudocolor*, min 0, max 100.
4. Toggle `model_used` to see where fallback models were used; they are less accurate.

Treat the class acres as the **model's estimate**, not a survey: accuracy was measured inside labelled fields, and
field edges, roads and trees are harder.

---

## 10. Part F — Improve the model with your corrections

Repeat this loop until the map is good enough.

### 10.1 Mark the mistakes

1. **Backup** the ground-truth QC copy (`<name>_qc.gpkg`) with the round number and date in the file name.
2. Open the QC copy in QGIS together with the map and the basemap.
3. *Toggle Editing* → *Add Polygon Feature*. Draw a polygon **inside** a field where the map is wrong, away from
   edges and trees, ideally larger than 1 acre.
4. Fill the attributes: the **correct** class name **and** its code, `use_in_analysis` = true,
   `review_status` = e.g. `added_round4`, `review_note` = what the map said and why you are sure.
5. Also add some fields where the map is **right**, in the same areas. Adding only mistakes skews the class mix.
6. *Save Edits*.

Check each new polygon with the pixel curve (Part B) if in doubt. One wrong label can undo several good ones.

### 10.2 Retrain on the run that contains all polygons

New polygons outside the original training area are only in the large area's run. Its config must point to the
same ground-truth file (`gcps.path`); the analysis automatically reads the `_qc.gpkg` copy next to it.

```bash
CFG=config/<large_area>.yaml
RUN=processed/<large area>/<season>/analysis/<run>

# keep the previous round: the model step refuses a folder built from other fields
mv $RUN/model_hm_w5_s0501 $RUN/model_hm_w5_s0501_round<N-1>       # only if it exists

nohup sh -c "python -m sar_pipeline.analysis features --config $CFG --name features_round<N> && \
             python -m sar_pipeline.analysis model --config $CFG --name features_round<N> && \
             python -m sar_pipeline.analysis final-models --config $CFG --name features_round<N>" \
    > logs/round<N>.log 2>&1 &
```

### 10.3 Compare with the previous round

| Compare | Files |
|---|---|
| cross-validation F1 of the target crop | `winner.json` → `dev_oof_at_threshold.f1`, old vs new folder |
| per-model scores on all fields | `final_models/model_card.json` |
| holdout | `holdout_metrics.json` (the holdout changes when the fields change, so compare with care) |

Keep the new model only if it is at least as good and the map looks better where you drew corrections.

### 10.4 Map again

```bash
nohup python -m sar_pipeline.analysis predict --config $CFG --map-config $CFG --name map_round<N> \
    > logs/predict_round<N>.log 2>&1 &
```

Open the new map next to the old one in QGIS and look at the places you corrected, and at a few places you did not.

---

## 11. Part G — Field boundaries with crop labels

A crop map says what grows in each 10 m pixel. Clients usually want **one crop per field**. This part takes field
polygons and labels each one from the map.

### 11.1 Get field polygons

**Option 1 — FTW (free, ready-made).** The *Fields of The World* project publishes predicted field boundaries.
The companion repository `geospatial-field-delineation` downloads them for your AOI:

```bash
git clone git@github.com:<account>/geospatial-field-delineation.git
cd geospatial-field-delineation
pip install -r requirements.txt
cp config.example.json my_config.json
```

Edit `my_config.json`: `aoi_path` (your AOI polygon), `country_code` (2 letters), `year` (e.g. 2025),
`out_dir`. Then:

```bash
python run_ftw.py my_config.json --dry-run      # a few seconds: are country and year right?
python run_ftw.py my_config.json                # the whole area
```

The polygons are `<out_dir>/ftw_<country>_<year>_<aoi>.gpkg`. That repository's README explains every setting.
FTW boundaries come from the year you choose, not necessarily this season.

**Option 2 — your own delineation.** Any polygon file works, for example boundaries traced by a segmentation model
on high-resolution imagery. Put it under `data/fields/`.

### 11.2 Label the fields

Back in this project (`cd` back, `source .venv/bin/activate`):

```bash
A=processed/<area>/<season>/analysis/<run>
M=processed/<training area>/<season>/analysis/<training run>/model_hm_w5_s0501/final_models
nohup python -m sar_pipeline.analysis field-labels --config $MAP \
    --delineation <path/to/fields.gpkg> \
    --class-raster $A/<map_name>_classes.tif \
    --prob-raster  $A/<map_name>_Rice_prob.tif \
    --model $M/xgb_field_level.joblib \
    --train-config $TRAIN --train-run <training run id> \
    --name <fields_name> > logs/field_labels.log 2>&1 &
```

- `--config` is the config of the area the map belongs to.
- `--model`, `--train-config`, `--train-run` are optional: they add a second, independent label per field.
  If the model was trained on the same run as the map, leave out `--train-config` and `--train-run`.

What it does to the polygons, in order: simplifies pixel-staircase outlines, repairs broken shapes and overlaps,
cuts off thin tails, labels each field by majority of its pixels, cuts fields that clearly hold two crops, adds
fields where the map shows a crop but no polygon exists, and tidies everything. Details:
[08 Ground-truth analysis §6](08_analysis.md#6-field-labels).

Output in `processed/<area>/<season>/fields/<fields_name>/`:

| File | Content |
|---|---|
| `fields_labelled.gpkg` | the labelled fields, open in QGIS |
| `summary.json` | polygons and acres after every step, acres per class |

Useful columns and QGIS filters (*right-click the layer → Filter…*):

| Column | Meaning | Filter example |
|---|---|---|
| `majority_class` | the label | `"majority_class" = 'Rice'` |
| `majority_share` | how pure the field is (1.0 = all pixels agree) | `"majority_share" < 0.6` weak labels |
| `agree` | the two labels match | `"agree" = 0` fields worth a look |
| `origin` | `delineation`, `split` (cut in two) or `derived` (drawn from the map) | `"origin" = 'split'` |
| `decision` | why the polygon looks as it does | `"decision" = 'too few pixels to judge'` |
| `area_acres` | size | `"area_acres" >= 1` |

Check a sample over the basemap before using the numbers: are the cuts along real boundaries, did the tail removal
cut real field parts?

---

## 12. Part H — Share results through cloud storage

QGIS cannot open files directly from a bucket, so results are shared as folders people download.

**Upload a folder:**

```bash
gcloud storage cp -r processed/<area>/<season>/fields/<fields_name> gs://<bucket>/<project-folder>/<where>/
```

**Download a folder:**

```bash
gcloud storage cp -r gs://<bucket>/<project-folder>/<where>/<folder> .
```

**Without `gcloud`**, with the key file (run from the project folder):

```bash
python - <<'EOF'
from pathlib import Path
from google.cloud import storage
client = storage.Client.from_service_account_json("secrets/<key>.json")
bucket = client.bucket("<bucket>")
prefix = "<project-folder>/<where>/<folder>/"           # the cloud folder, ending with /
for blob in client.list_blobs(bucket, prefix=prefix):   # download
    target = Path(blob.name[len(prefix):])
    target.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(target)
    print(target)
EOF
```

Upload the other way round: `bucket.blob(prefix + name).upload_from_filename(local_path)` for each file.

A shapefile is several files with the same name (`.shp .shx .dbf .prj .cpg`): always copy all of them.

---

## 13. When something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `No module named 'sar_pipeline'` | environment off or not installed | `source .venv/bin/activate`; if still failing, `pip install -e ".[dev]"` |
| Command prints a summary and `Nothing was started` | it needs confirmation | read the summary, add `--yes` |
| `Killed`, or the log just stops and the job is gone | out of memory | set `resources.cpu_workers` lower in your config (e.g. 4), run the same command again |
| `GridMismatch` | a grid with other settings exists for this AOI and season | do not delete it; restore the settings or use a new `aoi.key` / `season.key` |
| `AuditRequired` | no tracks chosen | fill `s1.tracks` (5.4) |
| `WARNING: your config differs from the frozen config of run ...` | you changed a setting after `new-run` | fine if intended for the *next* run; this run keeps its settings |
| `DecisionRequired` from `stack` | missing dates or chunks | read `decisions_required.md` (5.9) |
| `band layouts of ... differ` | map run and training run have different tracks or dates | make the map run with the training run's tracks and `season.end` (5.1) |
| `... differs from the split of the current data: move the model_... folder away` | fields changed since the last model | rename the old `model_<set>` folder (10.2) |
| `... already holds models: move that folder away` | final models exist | rename `final_models` (or the whole `model_<set>` folder) first |
| `... is missing: run the model step first` | `final-models` before `model` | run 8.2 first |
| `no pixel-level .joblib models in ...` | wrong `--models` folder or final models not saved | check the path; run 8.3 |
| Many `FAILED` tasks in `monitor.log` | Earth Engine errors or quota | wait, then `export --retry-failed --yes` and `monitor --yes`; see [06 Troubleshooting](06_troubleshooting.md) |
| `pytest` fails in `test_repo_hygiene` | a private word is in a file git would publish | remove it from that file (the message names file and word) |

More: [06 Troubleshooting](06_troubleshooting.md).

---

## 14. Cheat sheet: every command on one page

```bash
source .venv/bin/activate
CFG=config/<area>.yaml

# ---- 0. prep checks (read-only, no exports, nothing written) ----
python -m sar_pipeline.prep aoi-qc          --config $CFG [--split-dir <folder>] [--json]
python -m sar_pipeline.prep s1-availability --config $CFG [--start <YYYY-MM-DD> --end <YYYY-MM-DD>] [--json]

# ---- A. radar data (Earth Engine) ----
python -m sar_pipeline --config $CFG resources
python -m sar_pipeline --config $CFG grid
python -m sar_pipeline --config $CFG audit                    # then fill s1.tracks
python -m sar_pipeline --config $CFG new-run
python -m sar_pipeline --config $CFG export --pilot <chunk> --yes
python -m sar_pipeline --config $CFG export --all --yes
python -m sar_pipeline --config $CFG monitor --yes            # long
python -m sar_pipeline --config $CFG export --retry-failed --yes
python -m sar_pipeline --config $CFG download --yes           # long
python -m sar_pipeline --config $CFG stack

# ---- B. pixel curve ----
python -m sar_pipeline --config $CFG pixel --track <track> --lon <lon> --lat <lat> --plot logs/pixel.png
python -m sar_pipeline --config $CFG pixel --track <track> --pid <pid> --plot logs/pixel.png

# ---- C. ground truth ----
python -m sar_pipeline.analysis gcp-qc  --config $CFG
python -m sar_pipeline.analysis gcp-eda --config $CFG
python -m sar_pipeline.analysis gcp-qc  --config $CFG --write-copy

# ---- D. model ----
python -m sar_pipeline.analysis features     --config $CFG --name features_<round>
python -m sar_pipeline.analysis model        --config $CFG --name features_<round>        # long
python -m sar_pipeline.analysis final-models --config $CFG --name features_<round> [--without-bin <track>:<MMDD>]

# ---- E. map ----
python -m sar_pipeline.analysis predict --config <training config> --map-config <area config> --name <map_name>

# ---- G. field labels ----
python -m sar_pipeline.analysis field-labels --config <area config> --delineation <fields file> \
    --class-raster <map>_classes.tif --prob-raster <map>_Rice_prob.tif \
    [--model <final_models>/xgb_field_level.joblib --train-config <training config> --train-run <run>] --name <fields_name>

# run-scoped commands (export, monitor, download, stack, pixel and every analysis step):
#   add --run <run id> to use an older run instead of the newest.
#   grid, audit, new-run, resources and the prep steps have no --run: they are not tied to a run.
# long commands: nohup <command> > logs/<name>.log 2>&1 &    then    tail -f logs/<name>.log
```

### Where the outputs are

```
processed/<aoi>/<season>/
├─ grid/                        grid_def.json, chunks.gpkg, pixel_index.tif
├─ audit/<date>/                track_summary.csv, gaps_report.md, ...
├─ runs/<run>/                  run_config.yaml, export_manifest.csv, failed_chunks.csv,
│  ├─ raw_chunks/                 downloaded GeoTIFFs
│  └─ stack/                      per-date rasters, dates.csv
├─ analysis/<run>/              features_*.parquet, class_curves.png, gcp_*.csv,
│  ├─ model_<set>/                winner.json, holdout_metrics.json, ...
│  │  └─ final_models/            *.joblib, model_card.json
│  └─ <map_name>_*.tif            maps from predict
└─ fields/<fields_name>/        fields_labelled.gpkg, summary.json
```
