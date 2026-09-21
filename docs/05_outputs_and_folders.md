# 05 — Outputs and folders

Everything the pipeline writes lives under `processed/` (gitignored). This page lists every file,
what is inside, and how to use it. Column-level contracts are in
[developer/interfaces.md](developer/interfaces.md).

> **The prep steps write nothing.** `python -m sar_pipeline.prep aoi-qc` and
> `python -m sar_pipeline.prep s1-availability` print their report to **stdout** (or to JSON with
> `--json`) and create no file, no folder under `processed/`, no run folder and no Cloud Storage
> object. That is deliberate: they are read-only checks you re-run freely while choosing the AOI and
> the season window, so there is no output to keep in step with anything. If you want to keep a
> report, redirect it yourself — into a gitignored folder, because it names the AOI:
> `python -m sar_pipeline.prep aoi-qc --config $CFG --json > logs/aoi_qc.json`.
> `aoi-qc` also signals through its **exit code**: non-zero when a `--split-dir` cross-check fails.

---

## 1. Local tree

```
processed/<aoi.key>/<season.key>/
├─ grid/                          built once, never overwritten with different content
│  ├─ grid_def.json
│  ├─ chunks.gpkg
│  └─ pixel_index.tif             (while building: a temporary build file + pixel_index.progress.json)
├─ audit/<YYYYMMDD>/              one folder per audit day (UTC date)
│  ├─ s1_slices.csv
│  ├─ s1_footprints.gpkg
│  ├─ s1_acquisitions.csv
│  ├─ track_summary.csv
│  ├─ chunk_track_coverage.csv
│  ├─ track_recommendation.csv
│  ├─ gaps.csv
│  ├─ gaps_report.md
│  ├─ rain_flags.csv
│  ├─ slope_stats.json
│  ├─ _cache_keys.json            resume bookkeeping: input fingerprint of every step
│  └─ _partial/                   resume bookkeeping: per-batch results, removed when a step completes
├─ runs/<run_id>/                 one immutable folder per run: v<NNN>_<YYYYMMDD>
│  ├─ run_config.yaml
│  ├─ run_uid.txt                 globally unique run id, e.g. v002_20260915_a1b2c3
│  ├─ export_meta.json
│  ├─ band_layout.csv
│  ├─ export_manifest.csv         snapshot of the manifest
│  ├─ export_manifest.journal.jsonl   changes since the snapshot (append-only)
│  ├─ download_index.jsonl        verified local files (size, mtime, checksum)
│  ├─ failed_chunks.csv
│  ├─ decisions_required.md / .csv
│  ├─ .manifest.lock / .monitor.lock / .download.lock    only while a process holds them
│  ├─ raw_chunks/track_<track_id>/<chunk_name>[-<yoff>-<xoff>].tif
│  └─ stack/track_<track_id>/
│     ├─ VV/VV_<YYYYMMDD>.vrt     (+ .sources.sha256 sidecar per VRT)
│     ├─ VH/VH_<YYYYMMDD>.vrt
│     ├─ stack_VV.vrt
│     ├─ stack_VH.vrt
│     ├─ dates.csv
│     ├─ _valid_cache/            resume cache of valid-pixel counts (per-batch part files)
│     └─ _aoi_cache.json
└─ LATEST.txt                     id of the most recent run
```

**Versioning rules**
- `grid/` is shared by all runs of a season, so pixel ids stay stable over time.
- A new run never overwrites an old one. Compare runs side by side.
- Each run carries its own frozen `run_config.yaml`, which is the config used to process and read that run.
- `LATEST.txt` is a plain text pointer (not a symlink, which is unreliable on Windows). If it ever disagrees with
  the newest complete run folder (e.g. after a crash), the newest complete run is used and a warning is logged.
- Files beginning with `_` or `.` are bookkeeping for resume and locking. Do not edit them; deleting them only
  costs recomputation (never delete a `.lock` file while a process is running).

Log files of long commands and plots belong in a gitignored folder such as `logs/` at the project root.

---

## 2. Cloud Storage tree

```
gs://<bucket>/<gcs.base_folder>/<aoi.key>/<season.key>/runs/<run_uid>/raw_chunks/track_<track_id>/<chunk_name>.tif
```

`<run_uid>` is the content of `run_uid.txt` (runs created before this file existed use the run folder name).
Earth Engine may split a large file into pieces named `<chunk_name>-<yoff>-<xoff>.tif`; the pipeline
handles both forms.

---

## 3. Grid files

### `grid_def.json`
The master grid: `crs`, `res` (m), `x0`/`y0` (top-left corner), `width`/`height` (pixels), `chunk_px`,
`n_chunk_rows`/`n_chunk_cols`, `transform` (GDAL order), `pid_dtype` (`int32` or `int64`), `aoi_file`,
and the list of `chunks`.

### `chunks.gpkg`
One polygon per chunk (layer `chunks`), with `chunk_id`, `name`, `grid_row`, `grid_col`, `row_off`,
`col_off`, `width`, `height`, extent, and `aoi_frac` (share of the chunk inside the AOI). Use it in QGIS to
pick a pilot chunk.

### `pixel_index.tif`
Same grid as the stacks. Three bands:

| Band | Name | Value |
|---|---|---|
| 1 | `pid` | pixel id = `row × width + col` |
| 2 | `chunk_id` | chunk the pixel belongs to |
| 3 | `aoi_mask` | 1 inside the AOI, 0 outside |

Nodata = −1 (areas outside AOI chunks; the file is sparse, so they take no disk space).

**Why a pixel id?** It is a stable "address" for every 10 m pixel. From a pid:

```
row, col = divmod(pid, width)
x = x0 + (col + 0.5) × res        y = y0 − (row + 0.5) × res      (pixel centre)
```

So a pixel's time series is a 1 × 1 read from the stack, with no need to load the full raster. In
Python: `index.pid_to_lonlat(gd, pid)`, `index.lonlat_to_pid(gd, lon, lat)`.

---

## 4. Audit files

| File | One row per | Use it for |
|---|---|---|
| `s1_slices.csv` | Earth Engine image (slice) | raw inventory: platform, track, time |
| `s1_footprints.gpkg` | slice footprint polygon | see swath coverage in QGIS |
| `s1_acquisitions.csv` | acquisition (slices of one pass grouped) | dates, platforms, AOI coverage, incidence angle, `coverage_basis` |
| `track_summary.csv` | track | choosing tracks |
| `chunk_track_coverage.csv` | chunk × track | does a track cover every chunk (large AOIs); chunks with zero acquisitions become `NOT_COVERED` at export |
| `track_recommendation.csv` | track | suggested role + reason |
| `gaps.csv` / `gaps_report.md` | gap | missing periods, constellation events, rain-window warnings |
| `rain_flags.csv` | acquisition (or acquisition × chunk) | rain in 6 h / 24 h before the pass |
| `slope_stats.json` | – | terrain steepness percentiles |

`coverage_basis` says how AOI coverage was computed: from the image footprint; if the optional border-noise
mask is enabled, coverage is still footprint-based and slightly optimistic.

`rain_flags.csv` columns: `acquisition_id, track_id, datetime_utc, rain_6h_mm, rain_24h_mm, source, level,
chunk_name, n_images_expected, n_images_found, rain_gap_images`. Images partly inside a window are weighted
by their overlap; `rain_gap_images > 0` means dataset images are missing in the 24 h window, so the sum
under-counts. `source = none` (and NaN rain) means no dataset covers that time yet.

---

## 5. Run files

### `run_config.yaml`
Frozen copy of your config at `new-run`. It is **the** config of the run: `export`, `monitor`,
`download`, `stack` and `pixel` (CLI and notebooks 02–04) read their settings from this file, not from
your editable config.

Why: the rasters of a run can only be read correctly with the settings that produced them (e.g.
`export.dtype`, nodata, scale, band layout, tracks, filters). Editing your config later must not change
how an existing run is interpreted.

- Never edit `run_config.yaml` by hand. To change settings, edit your config and create a **new run**.
- If your config differs from the frozen one in `aoi, season, grid, gcs, s1, ard, export` or `qa`, the
  commands print a warning listing the differing keys and keep using the frozen values. `auth.project` is also
  taken from the run.
- **Read from your config instead:** `qa.acknowledged_issues` (accepting QA issues happens after the run
  exists), `auth.key_file` and `resources` (the same run can be continued on a different machine).

### `run_uid.txt`
`v<NNN>_<YYYYMMDD>_<6 hex characters>`. Run folder names are only unique on one machine; the uid keeps Cloud
Storage prefixes and Earth Engine task descriptions unique across machines.

### `export_meta.json`
Frozen export facts: `run_id`, `run_uid`, `ee_project`, `dtype`, `nodata`, `int16_scale`, `grid_crs`,
`grid_transform`, `grid_fingerprint` (hash of `grid_def.json`), `audit_dir`, `tracks`, `config_hash`,
`created_utc`. Later stages check the grid fingerprint and the project before reading anything.

### `band_layout.csv`
For each track: which band of the chunk files holds which acquisition and polarisation. Bands are
ordered by date, with VV then VH for each date:

| band | 1 | 2 | 3 | 4 | … |
|---|---|---|---|---|---|
| content | VV date 1 | VH date 1 | VV date 2 | VH date 2 | … |

### `export_manifest.csv` + `export_manifest.journal.jsonl`
One row per export task (`task_key` = `<track_id>__<chunk_name>`): state, Earth Engine task id, attempts,
error class and message, local file paths, and `adopted` (1 when the task was taken over from an earlier
attempt). Always read it with `sar_pipeline.manifest.read_manifest(run_dir)`: the CSV alone misses changes
still in the journal.

| State | Meaning |
|---|---|
| `PLANNED` | planned by a dry run; never submitted until confirmed |
| `PENDING` | confirmed, waiting for queue headroom |
| `SUBMITTING` | about to start / start outcome unknown (looked up, never duplicated) |
| `SUBMITTED`, `READY`, `RUNNING` | Earth Engine task exists |
| `COMPLETED` | output written to Cloud Storage |
| `FAILED` | export failed after retries (see `error_class`) |
| `SPLIT` | replaced by 4 sub-chunk rows after a memory/timeout failure |
| `NOT_COVERED` | the track never covers this chunk; never exported |
| `DOWNLOADED` | local file present, not yet verified |
| `VERIFIED` | local file checked against grid, dtype and checksum |
| `VERIFIED_EMPTY` | verified, but entirely nodata |
| `FAILED_DOWNLOAD` | download or verification failed after retries |

`error_class`: `memory, timeout, quota, transient, cancelled, lost, verify, io, other`.

### `download_index.jsonl`
One line per verified chunk with each file's `local_path`, `size`, `mtime_ns` and `crc32c`. A re-run trusts
files whose size and modification time are unchanged (use `download --deep` to re-hash).

### `failed_chunks.csv`
`task_key, track_id, chunk_name, state, attempts, error_class, last_error, gcs_prefix` for FAILED and
FAILED_DOWNLOAD rows.

### `decisions_required.md` / `.csv`
QA issues waiting for a human decision (see [runbook step 7](04_runbook.md#step-7--stack-and-qa--notebook-03)).

### `raw_chunks/`
Downloaded GeoTIFFs: one per track × chunk (or several pieces), master-grid aligned. These are the
only files holding pixel data. **Do not rename or move them**, because the VRTs reference them.

### `stack/track_<id>/`

| File | What it is |
|---|---|
| `VV/VV_<date>.vrt`, `VH/VH_<date>.vrt` | one virtual raster per date and polarisation over the **whole grid**; `<date>` gets an acquisition suffix (e.g. `_2`) if two acquisitions share a UTC date |
| `stack_VV.vrt`, `stack_VH.vrt` | all dates as bands, in date order (band *n* = row with `stack_band = n` in `dates.csv`) |
| `dates.csv` | see below |

`dates.csv` columns:

| Column | Meaning |
|---|---|
| `stack_band` | band number in `stack_<POL>.vrt` |
| `acquisition_id`, `date_utc`, `datetime_utc`, `date_local` | which acquisition |
| `platforms`, `n_slices` | satellites and Earth Engine images that formed it |
| `rain_6h_mm`, `rain_24h_mm` | rain before the pass (from the audit) |
| `valid_pct_<POL>` | % valid pixels inside the AOI part of the planned scope, per polarisation |
| `aoi_planned_pct` | % of the whole AOI that this run planned to export (small for a pilot) |
| `n_temporal_neighbors` | acquisitions used by the multi-temporal speckle filter for this date (fewer at series ends = noisier) |
| `flags` | `LOW_VALID_<POL>`, `RAIN_24H` (≥ 5 mm), `GAP_BEFORE` (gap > `audit.max_gap_days`) |

Values are **γ⁰ in dB**, terrain-flattened and speckle-filtered. Nodata pixels (outside swath,
layover/shadow, masked border) hold the nodata value from `export_meta.json`.

If `export.dtype` is `int16`, stored values are `round(dB × int16_scale)`: divide by the scale
(`pixel_query` does this for you).

---

## 6. Opening outputs

**QGIS:** drag a `.vrt` into QGIS. Use a single-band grey style with min/max around −25 … −5 dB for VH.
Drag `stack_VH.vrt` and use the *Temporal/Value tool* or the identify tool to see all dates at a point.

**Python (windowed read, never the full raster):**

```python
import rasterio
from rasterio.windows import Window

with rasterio.open("processed/.../stack/track_RO123_ASC/stack_VH.vrt") as src:
    block = src.read(window=Window(col_off=1000, row_off=800, width=256, height=256))  # (dates, 256, 256)
```

**Pixel time series:** `sar_pipeline.pixel_query.pixel_timeseries_lonlat(cfg, run_dir, track_id, lon, lat)`
returns `date_utc, datetime_utc, acquisition_id, platforms, <POL>_db…, VH_minus_VV_db` (when both
polarisations exist)`, rain_6h_mm, rain_24h_mm, valid`.
