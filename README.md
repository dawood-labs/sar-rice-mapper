# sar-rice-mapper

A **Sentinel-1 SAR preprocessing and time-series feature pipeline** for mapping **paddy rice**,
built on **Google Earth Engine (GEE)**.

Radar sees through clouds. In monsoon regions the rice-growing season is exactly the cloudiest part
of the year, so optical satellites (Sentinel-2, Landsat) can miss the whole crop cycle, while
Sentinel-1 records the ground every 6–12 days regardless of weather. This repository turns raw
Sentinel-1 GRD scenes into a **clean, analysis-ready, multi-date VV + VH raster stack** plus a
**pixel index raster**, so the time series of any pixel or field can be pulled out and compared.

> **New to this project? Start with [docs/00_step_by_step_guide.md](docs/00_step_by_step_guide.md).**
> It walks the entire workflow command by command and assumes no prior radar or Python knowledge.
> Then read [docs/01_sar_basics.md](docs/01_sar_basics.md) for *why* each step exists.

---

## Why rice is a good fit for SAR (and how you recognise it)

Rice has the most distinctive radar signature of any major crop, because of one thing no other crop
does: **the field is deliberately flooded before planting.**

```
VH backscatter (dB) over one season

  high │                      ....────────....
       │                  ....                ....
       │              ....                        ..
       │          ....                              ╲     ← harvest: sharp drop
       │      ....                                    ╲
   low │  ──╲                                          ────
       │     ╲__╱   ← transplanting: the MINIMUM        
       └──────────────────────────────────────────────────────→ time
         dry    flood      vegetative growth      mature  harvested
```

1. **Before planting** the field is dry soil — moderate backscatter.
2. **Flooding / transplanting** turns the field into a shallow water surface. Water reflects the
   radar beam away from the satellite (specular reflection), so backscatter collapses to a
   **pronounced minimum**. This minimum is the single most reliable marker of paddy rice, and its
   date is effectively the transplanting date.
3. **Vegetative growth** builds a dense canopy that scatters the beam back strongly, so backscatter
   **rises steeply for weeks**.
4. **Harvest** removes the canopy and backscatter **drops sharply** again.

**VH is the polarisation to use.** Cross-polarised VH responds to the flooding phase more strongly
than VV, and the VH minimum is what published rice calendars key on.

This gives a **label-free** way to find rice, which matters when you have no ground truth. Over a
full season, three statistics of the VH time series separate rice from everything else:

| Feature | Rice | Open water | Other vegetation / built-up |
|---|---|---|---|
| **minimum** VH | **low** (the flood) | low | higher |
| **maximum** VH | **high** (dense canopy) | low | moderate |
| **variance** of VH | **high** (big seasonal swing) | low | low |

In one sentence: **rice is the class that swings from looking like water to looking like a dense
canopy and back again.** Water stays low; forest and settlements stay high and flat.

> **This only works if the window contains both anchors.** The flooding minimum *and* the harvest
> drop must both be inside `season.start`..`season.end`, plus some dry baseline before flooding to
> measure the minimum against. A window that stops before harvest destroys the maximum and the
> variance, and rice stops being separable. Choose the window from a crop calendar **before**
> exporting anything — see [docs/01_sar_basics.md](docs/01_sar_basics.md).

---

## What the pipeline does

```mermaid
flowchart LR
    Z[Prep<br/>AOI QC, S1 availability<br/>read-only] --> A[AOI polygon]
    A --> G[Stage 0 · Grid<br/>master grid, chunks,<br/>pixel index]
    G --> AU[Stage 1 · Audit<br/>which Sentinel-1 tracks,<br/>dates, gaps, rain]
    AU -->|you choose tracks| R[new-run<br/>immutable run folder]
    R --> E[Stages 2-3 · Export<br/>plan → confirm →<br/>GEE preprocessing<br/>per chunk x track]
    E -->|GeoTIFFs on GCS| D[Stage 4 · Download<br/>+ integrity checks]
    D --> S[Stage 5 · Stack<br/>per-date VRTs,<br/>QA, dates.csv]
    S --> P[Stage 6 · Pixel explorer<br/>time series of any pixel]
```

Inside Earth Engine, every image goes through (details in [docs/03_pipeline_overview.md](docs/03_pipeline_overview.md)):

1. optional extra **border-noise** masking (off by default; current GRD data is already cleaned),
2. **radiometric terrain flattening** (σ⁰ → γ⁰, layover/shadow masked),
3. **mosaicking** of slices of the same pass onto the master grid,
4. **speckle filtering** in linear power (multi-temporal, same track),
5. conversion to **dB** and export.

Safety rails built in:

- exports and bulk downloads **never start without explicit confirmation** (`--yes` / `confirmed=True`);
  a dry run only *plans* tasks, and only the confirmed scope is ever submitted;
- missing dates, failed chunks and corrupt files are **reported, never silently skipped**;
- every stage is **crash-safe**: re-running the same command resumes; two processes working on the
  same run cannot corrupt each other's bookkeeping (file locks, append-only journal);
- every run is a new immutable folder; the grid never changes once built;
- CPU, RAM and thread counts are **detected at runtime** (container-aware), nothing is hard-coded.

---

## Quickstart

```bash
# 1. Environment (Python >= 3.10; rasterio wheels include GDAL, no system GDAL needed)
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Credentials: put the service-account JSON key in secrets/ (gitignored)
mkdir -p secrets && cp /path/to/key.json secrets/ && chmod 600 secrets/*.json

# 3. Config: copy the example into config/ and fill every <placeholder>
cp config/pipeline.example.yaml config/my_aoi_season.yaml

# 4. Tests
pytest                                                   # unit tests, no network
SAR_PIPELINE_CONFIG=config/my_aoi_season.yaml pytest -m gee   # live, read-only Earth Engine tests

# 5. Check what the machine offers
python -m sar_pipeline --config config/my_aoi_season.yaml resources
```

### Step 0 — prep checks (do these first, they are free)

Before building a grid, verify the AOI is what you think it is and that Sentinel-1 actually covers
it in your window. Both steps are **read-only**: no exports, no uploads, no run folder.

```bash
CFG=config/my_aoi_season.yaml

# Duplicate detection + area table in acres.
python -m sar_pipeline.prep aoi-qc --config $CFG

# Also cross-check a deduplicated per-AOI folder against the original file.
python -m sar_pipeline.prep aoi-qc --config $CFG --split-dir data/aoi/dedup
#   -> exits non-zero if a polygon was lost, added, or silently renumbered

# Which tracks and how many acquisition dates exist in the season window?
python -m sar_pipeline.prep s1-availability --config $CFG
python -m sar_pipeline.prep s1-availability --config $CFG --start 2026-05-01 --end 2027-02-01
```

**Why `aoi-qc` exists.** AOIs usually arrive as one multi-feature file which someone then splits
into one file per AOI, dropping duplicates. Comparing counts is not enough: 160 features becoming
132 files tells you nothing about *which* 28 went. `aoi-qc` normalises every geometry to canonical
WKB, so shapes compare equal regardless of vertex order, and then reports exactly which features
were dropped, which kept polygon each was a duplicate of, and whether anything was lost or added.

**Why `s1-availability` exists.** It is the five-second version of `audit`, meant for use while you
are still *choosing* the window. It needs no grid. Counts are over the AOI **bounding box**, so use
`audit` — not this — to decide `s1.tracks`.

### Typical run

```bash
CFG=config/my_aoi_season.yaml
python -m sar_pipeline --config $CFG grid
python -m sar_pipeline --config $CFG audit            # then review the audit and fill s1.tracks
python -m sar_pipeline --config $CFG new-run
python -m sar_pipeline --config $CFG export --pilot chunk_r01c01          # dry run: plans, shows the plan
python -m sar_pipeline --config $CFG export --pilot chunk_r01c01 --yes    # confirms + starts the pilot
python -m sar_pipeline --config $CFG monitor --yes
python -m sar_pipeline --config $CFG download          # shows size + free disk
python -m sar_pipeline --config $CFG download --yes
python -m sar_pipeline --config $CFG stack
python -m sar_pipeline --config $CFG pixel --track RO123_ASC --lon <lon> --lat <lat> --plot logs/px.png
```

The full procedure, including every checkpoint where a human must look before continuing, is in
[docs/04_runbook.md](docs/04_runbook.md). The same steps are available as notebooks in `notebooks/`.

---

## Running many AOIs

When the AOIs are many separate polygons, run **one config per AOI** rather than one merged AOI.
A merged AOI gets one grid over the whole bounding box; for scattered fields that is mostly empty
pixels (in one real case about a billion, against nine million for per-AOI grids). Per-AOI configs
also let every AOI use the tracks that cover it best.

```bash
# 1. One config per AOI, all copied from one working config (tracks left empty)
python -m sar_pipeline.prep batch-configs --template config/<working>.yaml \
    --split-dir data/aoi/<split_folder> --season-key year2025 --start 2025-05-01 --end 2026-05-01

# 2. Grid + audit for every AOI (here four at a time)
ls config/*_year2025.yaml | xargs -P 4 -I{} sh -c \
    'python -m sar_pipeline --config {} grid && python -m sar_pipeline --config {} audit'

# 3. Choose each AOI's tracks by rule, and record why
python -m sar_pipeline.prep batch-tracks --season-key year2025 \
    --flood-start 2025-05-01 --flood-end 2025-08-31
#   -> reasons for every choice: processed/_batch/year2025_track_choice.csv

# 4. Then new-run, export (dry run first!), monitor, download, stack per config, as for one AOI
```

**The track rule** (`prep/batch.py`, `choose_tracks`), so choices are the same everywhere and can be
checked afterwards: a track must cover >= 90% of the AOI and have at least half the best track's
acquisitions; it is **excluded** if any gap longer than 20 days overlaps the flooding window (the
flooded period lasts about three weeks, so a longer blind spot can miss it entirely); the best
remaining track is primary and the next is secondary, at most two.

**Name AOI files distinctively** (`batch-configs` zero-pads the id to three digits). The hygiene
test treats every AOI file name in a local config as private, and a short unpadded name made of the
prefix plus a single digit also matches ordinary code, such as the EPSG:4326 helpers.

---

## Sentinel-2 reference images for checking a map by eye

`sar_pipeline.optical_export` exports, for each AOI and month, the **one date whose Sentinel-2
mosaic is clearest over the AOI itself** (scored with Cloud Score+ on the AOI's own pixels, not the
whole tile's cloud percentage). It uses exactly the AOI's SAR grid (same CRS, origin and 10 m
pixels), so the image lies pixel-for-pixel on the class map in QGIS.

```python
from sar_pipeline import auth, config, optical_export as ox
auth.init_ee(config.load_config("config/<any>.yaml"))
rows = ox.plan_and_export(["config/<aoi>.yaml"], "2025-01", "2026-05",
                          bucket="<bucket>", prefix="<base_folder>/s2_reference",
                          submit=False)        # plan only; submit=True starts the exports
```

Each file has six bands, **in this order**: 1 = B2 blue, 2 = B3 green, 3 = B4 red, 4 = B5 red edge,
5 = B8 NIR, 6 = clear score (0-100, Cloud Score+). In QGIS use *Multiband color* with:

| View | Red | Green | Blue |
|---|---|---|---|
| True colour (4-3-2) | band 3 | band 2 | band 1 |
| False colour (8-3-2): vegetation bright red | band 5 | band 2 | band 1 |
| Red edge (5-3-2): separates crop types and stages | band 4 | band 2 | band 1 |

Reflectance is stored x 10000; a stretch of about 0-3000 suits most scenes. In monsoon months the
clearest date can still be partly cloudy: check band 6, where low values mean cloud.

---

## Looking at any pixel's time series

**In QGIS (click and see).** `sar_pipeline.analysis.qgis_package.build` writes, per AOI,
`<aoi>_VH_halfmonth.tif`, `<aoi>_VV_halfmonth.tif` and `<aoi>_VHmVV_halfmonth.tif`: one band per
calendar half-month (band 1 = 1 May, band 24 = 16 April), 5x5 linear-power means in dB, exactly
what the classifier sees. It also writes the class map, the probabilities, the pixel index and a
`BANDS.txt` listing band number to date. Install the QGIS plugin **Temporal/Spectral Profile Tool**,
select a `_halfmonth.tif` layer and click a pixel: the plugin plots its 24 values. The plugin's x-axis
shows band numbers, which `BANDS.txt` translates to dates.

**From the pipeline (every acquisition, with rain).** Read the pixel id (`pid`) from
`pixel_index.tif` with QGIS *Identify*, then:

```bash
python -m sar_pipeline --config config/<aoi>.yaml pixel --track <primary track> --pid <pid> --plot px.png
# or --lon <lon> --lat <lat> instead of --pid
```

This prints every acquisition date (not binned), VV, VH, VH-VV, rain in the previous 6 and 24 hours
and the platform, and saves a plot.

---

## Working without ground truth

The `analysis` package ([docs/08_analysis.md](docs/08_analysis.md)) assumes you have **labelled
ground-truth polygons** (`gcps` in the config): it QCs them, extracts per-pixel features, and runs
supervised models with spatial cross-validation.

**When you have no labels**, that workflow cannot start at step one. The route is:

1. **Explore first.** Use `pixel` (Stage 6) and `gcp-eda`-style per-field curves on a few large AOIs
   to look at real VH/VV time series. Group by AOI polygon rather than by class.
2. **Apply the label-free recipe.** Compute VH minimum, maximum and variance over the season and
   look for the rice swing described above. This produces *candidates*, not labels.
3. **Get a human verdict.** A domain expert confirms which candidate signatures really are rice,
   cross-checking against any cloud-free optical imagery available (true colour, false colour, and
   a SWIR/NIR/red combination for crop differences).
4. **Only then synthesise labels** from the confirmed signatures, write them in the same shape the
   `gcps` config expects, and the whole supervised `analysis` workflow becomes available unchanged.

Never skip step 3. A label set derived purely from thresholds will train a model that reproduces the
thresholds, and its scores will look good while meaning nothing.

---

## Repository layout

```
config/            pipeline.example.yaml (tracked); your own *.yaml configs (gitignored)
docs/              concepts, setup, runbook, outputs, troubleshooting, scaling, glossary
docs/developer/    interface contract (interfaces.md) and testing guide
notebooks/         01_grid_and_audit, 02_export, 03_download_and_stack, 04_pixel_explorer
src/sar_pipeline/  the Python package
  config.py        config loading + processed/ folder conventions + run versioning
  auth.py          Earth Engine + Cloud Storage authentication
  resources.py     runtime CPU/RAM/disk detection (cgroup aware)
  run_meta.py      crash-safe file writes + run metadata readers
  locking.py       cross-process file locks
  prep/            pre-flight checks: AOI cross-check, Sentinel-1 availability (read-only)
    aoi_qc.py        duplicate detection, split/original cross-check, areas in acres
    s1_availability.py  tracks, platforms and acquisition dates from GEE metadata
  grid.py          master grid + chunks
  index.py         pixel index raster + pixel-id conversions
  audit.py         Sentinel-1 availability, gaps, rain flags, slope statistics
  s1_ard/          Earth Engine preprocessing (border noise, terrain flattening, speckle)
  manifest.py      export state machine (CSV snapshot + append-only journal)
  export.py        chunked exports, monitoring, retry policy
  download.py      GCS download + integrity checks
  stack.py         VRT stacks + QA
  pixel_query.py   pixel time series + plots
  analysis/        ground-truth QC, features, spatial CV, models, maps, field labels
  cli.py           command-line interface
tests/             unit tests (no network) and live read-only GEE tests (-m gee)
data/  secrets/  processed/  reference/  logs/     local only, gitignored
```

What each output file contains: [docs/05_outputs_and_folders.md](docs/05_outputs_and_folders.md).

### Units

**Areas are always reported in acres**, never hectares or m². At the 10 m pixel size the pipeline
uses, 1 pixel = 0.025 acre (so ~40 pixels per acre) and the 5×5 analysis window = 0.62 acre. The
factor is the same named constant `SQM_PER_ACRE` everywhere it is needed
(`prep/aoi_qc.py`, `analysis/field_labels.py`, `analysis/final_models.py`); never write a bare
`/ 4046.86` into new code.

---

## Security (this repository is public)

- **Never commit** credentials, client data, client-specific configs, outputs, logs, figures or vector
  files. `.gitignore` already excludes `secrets/`, `data/`, `processed/`, `reference/`, `logs/`,
  `config/*.yaml` (except the example), rasters, `*.png`, `*.log` and vector formats.
- No client, region, country, bucket, project or track names, no coordinates and no email addresses
  in tracked files — including in test fixtures.
- `tests/test_repo_hygiene.py` fails if any file git would track contains a private value: bucket,
  project id, service-account email, AOI file name (read from your local config and key) or any word
  listed in `secrets/private_terms.txt`. Run `pytest` before every commit.
- Keep the service-account key readable only by you (`chmod 600 secrets/*.json`).
- If a key is ever pushed by mistake, **revoke it in Google Cloud immediately**; deleting the
  commit is not enough.

---

## Documentation

| Doc | Read it when |
|---|---|
| [00 Step-by-step guide](docs/00_step_by_step_guide.md) | **start here**: the whole workflow, command by command, no coding needed |
| [01 SAR basics](docs/01_sar_basics.md) | you are new to radar or want to know *why* each step exists |
| [02 Setup](docs/02_setup.md) | installing on a laptop or JupyterHub |
| [03 Pipeline overview](docs/03_pipeline_overview.md) | you want to understand the stages and their order |
| [04 Runbook](docs/04_runbook.md) | you are running the pipeline |
| [05 Outputs and folders](docs/05_outputs_and_folders.md) | you are looking for a file or a column |
| [06 Troubleshooting](docs/06_troubleshooting.md) | something failed |
| [07 Scaling to large AOIs](docs/07_scaling_to_large_aois.md) | processing a country-scale AOI or many scattered AOIs |
| [08 Ground-truth analysis](docs/08_analysis.md) | checking labelled fields, comparing features, making a first class map — **assumes labels exist; see [Working without ground truth](#working-without-ground-truth)** |
| [Glossary](docs/glossary.md) | a term is unclear |
| [Developer: interfaces](docs/developer/interfaces.md) | changing code |
| [Developer: testing](docs/developer/testing.md) | writing or running tests |

---

## Credits

The pipeline stages (grid, audit, export, download, stack, pixel explorer) and the Earth Engine
preprocessing chain are carried over from a sibling project that applied the same approach to
maize; this repository adapts the crop-specific parts to rice.

Earth Engine preprocessing in `src/sar_pipeline/s1_ard/` is adapted from
[adugnag/gee_s1_ard](https://github.com/adugnag/gee_s1_ard) (MIT licence, © 2021 Adugna Mullissa),
described in Mullissa et al. (2021), *Remote Sensing* 13(10), 1954, and Vollrath et al. (2020),
*Remote Sensing* 12(11), 1867. The terrain-flattening sign convention and the layover/shadow masks
were corrected and validated against real data (see
[docs/01_sar_basics.md §10](docs/01_sar_basics.md#10-terrain-correction-two-different-things)).

The rice detection approach summarised above follows published Sentinel-1 work: the VH minimum as
the transplanting marker (Monsoon Asia Rice Calendar, *Earth Syst. Sci. Data* 16, 3893, 2024) and
the minimum/maximum/variance feature triple for paddy classification (*Earth Syst. Sci. Data* 15,
1501, 2023).
