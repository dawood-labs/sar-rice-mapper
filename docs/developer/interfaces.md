# Developer interface contract

This document is the **single source of truth** for how the modules of `sar_pipeline` fit together:
module responsibilities, function signatures, file names, CSV columns and naming rules.
If code and this document disagree, one of them is a bug. Change both together.

---

## 1. Ground rules

1. **Public repository.** Never commit or write into tracked files: credentials, client names,
   bucket names, Earth Engine project ids, service-account emails, AOI coordinates or client data.
   Tests use synthetic AOIs and fake bucket/project names. `tests/test_repo_hygiene.py` enforces this.
2. **English** for code, comments, docstrings and docs. Docstrings explain *why*, not only *what*,
   because a junior analyst will maintain this code.
3. **No hard-coded machine resources.** Thread counts, process counts, block sizes and cache sizes
   come from `sar_pipeline.resources` (cgroup/container aware). The same code must run on a
   laptop and on a large JupyterHub instance.
4. **Scale.** Every step must work for a country-scale AOI at 10 m (grids of 10⁹–10¹⁰ pixels, tens of
   thousands of export tasks):
   - never materialise a full-AOI array in memory, always read/write windowed;
   - never make one Earth Engine request whose size grows with the AOI (bounded geometries, batches, paging);
   - pixel ids are `int64` when `width*height >= 2**31-1`.
5. **Nothing is silently skipped.** Missing dates, low-valid dates, failed chunks and corrupt
   files are recorded and surfaced (see §9).
6. **Immutable outputs.** `grid/` is never overwritten with different geometry. Each run lives in
   its own `runs/v<NNN>_<YYYYMMDD>/` folder. Code never deletes a previous run.
7. **Earth Engine exports and bulk downloads only after explicit confirmation.** Library functions
   that start exports or bulk downloads require a `confirmed=True` argument and raise
   `ConfirmationRequired` otherwise. Planning writes `PLANNED` rows only; a scope must be confirmed
   (`export.confirm_exports`) before anything in it is submitted. The CLI maps this to `--yes`.
8. **Tests.** Unit tests have no network access. Live Earth Engine tests are marked
   `@pytest.mark.gee`, are read-only (`getInfo`, `ee.data.computePixels`, `computeFeatures` on small
   regions) and never start tasks.
9. **Resume, never redo.** Every stage can crash at any point and be re-run with the same call:
   - work already done is detected and skipped, and the skip is *logged with counts*;
   - "done" is decided by *verification* (file opens, expected shape/bands/checksum, or a state row), never by file existence alone;
   - every file is written atomically with `run_meta.atomic_write_*` (unique temp name, fsync, rename with retry);
     temp files are removed by `run_meta.clean_stale_tmp` only when older than 6 h (a younger one may belong to a live process);
   - long loops persist progress incrementally (manifest journal, per-batch cache files), not only at the end;
   - cached results are keyed by the inputs that produced them; a changed input invalidates the cache;
   - each stage function takes `force: bool = False` to recompute derived files inside the current run/audit folder.
     `force` never touches `grid/` and never deletes other runs.
10. **Several processes may touch one run.** Read-modify-write of shared files happens under `locking.FileLock`;
    long stages hold `locking.run_lock` (one `monitor`, one `download` per run).
11. **No location hints in tracked files.** Besides client names and coordinates, do not name countries, provinces,
    time zones or UTM zones of real project areas, real track numbers over them, or sizes of the real AOIs in code,
    tests, docs or example configs.

---

## 2. Shared modules

| Module | Purpose |
|---|---|
| `config.py` | `load_config(path, project_root=None)` (root = `project_root` argument, else config key `project_root` relative to the config file, else parent of `config/`; otherwise `PipelineError`), `root(cfg)`, `season_dir(cfg)`, `grid_dir(cfg)`, `utc_today()`, `audit_dir(cfg, audit_date=None)` (UTC date), `latest_audit_dir(cfg)`, `aoi_path(cfg)`, `gcs_prefix(cfg, *parts)`, `list_runs(cfg)` (complete runs only), `new_run(cfg)` (writes `run_config.yaml` + `run_uid.txt`, atomic folder rename, updates `LATEST.txt`), `run_dir(cfg, run_id=None)` (newest complete run; warns if `LATEST.txt` disagrees), `load_run_config(run_path)` |
| `auth.py` | `credentials(cfg)`, `init_ee(cfg, high_volume=False)`, `gcs_bucket(cfg)` |
| `resources.py` | `detect_resources(cfg) -> Resources` (affinity ∩ cgroup CPU quota; tightest memory limit over the own cgroup and all ancestors, v1 and v2; reclaimable `inactive_file` not counted as used), `memory_budget_bytes`, `io_threads`, `cpu_workers`, `submit_threads`, `block_rows`, `gdal_cache_bytes`, `disk_free_bytes` |
| `run_meta.py` | `atomic_write_bytes/text/csv/json`, `replace_with_retry`, `clean_stale_tmp(folder, recursive=False, min_age_seconds=6h)`, `sha256_file`, `grid_fingerprint(cfg)`, `config_hash(cfg)`, `run_uid(run_dir)`, `relative_to_root(cfg, path)`, `read_export_meta(cfg, run_dir, check_grid=True)` (validates keys, grid fingerprint and `ee_project`), `read_band_layout(run_dir, track_id=None)`; constants `EXPORT_META`, `BAND_LAYOUT`, `RUN_UID_FILE`, `EXPORT_META_KEYS`, `OPTIONAL_META_KEYS`, `BAND_LAYOUT_COLUMNS`, `STALE_TMP_SECONDS` |
| `locking.py` | `FileLock(path, timeout=600, stale_after=300, poll=0.05)` with `acquire/release/touch/holder` (cross-process via O_EXCL lock file, re-entrant per process, broken when the holder process on the same host is dead or the file is not touched for `stale_after`); `LockTimeout`; `run_lock(run_dir, name, wait=False)` (stale after 15 min) |
| `errors.py` | exceptions below |
| `tests/conftest.py` | fixtures `make_project(overrides, aoi_lonlat)` (synthetic project in tmp) and `live_cfg` (config from env `SAR_PIPELINE_CONFIG`, default `config/live.yaml`, + EE init, for `gee` tests) |

```python
class PipelineError(Exception): ...
class ConfirmationRequired(PipelineError): ...    # export / bulk download without confirmed=True
class DecisionRequired(PipelineError): ...        # QA found issues the user must decide on
class GridMismatch(PipelineError): ...            # existing grid geometry differs / run made on another grid
class AuditRequired(PipelineError): ...           # tracks not selected / audit missing
```

---

## 3. Config schema

See `config/pipeline.example.yaml` (fully commented). Sections:
`aoi, season (key, timezone, start, end[exclusive]), auth (key_file, project), gcs, grid (crs, res, buffer_m, chunk_px),
resources, s1 (collection, instrument_mode, pols, tracks), audit, rain, ard, export, qa`.

`s1.tracks` is a list of `{track_id: "RO123_ASC", role: primary|secondary}`; it is empty until the
user has reviewed the audit. Any stage after the audit raises `AuditRequired` when it is empty.

`ard.border_noise_correction` defaults to `false` (current GRD data is already border-noise free; the extra mask
removed ~5 % valid far-range pixels in tests).

Machine-specific vs run-defining settings (see §6.8): `auth.key_file` and `resources` are machine-specific;
everything else, including `auth.project`, defines a run.

---

## 4. Identifiers and naming

| Thing | Format | Example |
|---|---|---|
| Track id | `RO{relative_orbit:03d}_{ASC|DSC}` | `RO123_ASC` |
| Acquisition id | `{track_id}_{YYYYMMDD}` (UTC date of first slice); a second acquisition on the same date gets `_2` | `RO123_ASC_20260705` |
| Chunk name | `chunk_r{grid_row:02d}c{grid_col:02d}` (3–4+ digits automatically when the grid has ≥100 / ≥1000 chunk rows or cols) | `chunk_r03c01` |
| Sub-chunk (after split) | `{chunk_name}_s{0..3}` | `chunk_r03c01_s2` |
| Task key | `{track_id}__{chunk_name}` | `RO123_ASC__chunk_r03c01` |
| Run id (folder) | `v{NNN}_{YYYYMMDD}` (UTC) | `v001_20260915` |
| Run uid | `{run_id}_{6 hex}` from `run_uid.txt`; legacy runs: the run id | `v002_20260915_a1b2c3` |
| Task description | `sanitise({aoi.key}_{season.key}_{run_uid}_{task_key})`, ≤ 100 chars | – |
| Band name in chunk files | `{POL}_{YYYYMMDD}`, plus acquisition suffix when two acquisitions share a UTC date | `VH_20260705`, `VH_20260705_2` |

**Slices vs acquisitions.** Earth Engine stores each ~170 km along-track slice as its own image.
One *acquisition* = all slices of the same track whose start times are within 15 minutes of each
other (`audit.ACQ_GAP_MINUTES`). Group by time proximity, not by date string.

**Dates.** All dates are UTC. A `date_local` column (from `season.timezone`) is added for humans.

---

## 5. Folder layout

### Local
```
processed/<aoi.key>/<season.key>/
├─ grid/
│  ├─ grid_def.json
│  ├─ chunks.gpkg                 (layer "chunks")
│  ├─ pixel_index.tif             (b1 pid, b2 chunk_id, b3 aoi_mask; tiled 512, sparse, deflate)
│  └─ pixel_index.progress.json   (+ temporary build file, only while building)
├─ audit/<YYYYMMDD>/              (UTC date)
│  ├─ s1_slices.csv, s1_footprints.gpkg, s1_acquisitions.csv, track_summary.csv,
│  │  chunk_track_coverage.csv, track_recommendation.csv, gaps.csv, gaps_report.md,
│  │  rain_flags.csv, slope_stats.json
│  ├─ _cache_keys.json            (resume: input key per step)
│  └─ _partial/                   (resume: per-batch partial results, removed when a step completes)
├─ runs/<run_id>/
│  ├─ run_config.yaml, run_uid.txt, export_meta.json, band_layout.csv
│  ├─ export_manifest.csv + export_manifest.journal.jsonl
│  ├─ download_index.jsonl        (legacy download_index.csv is read and folded in)
│  ├─ failed_chunks.csv, decisions_required.md / decisions_required.csv
│  ├─ .manifest.lock, .monitor.lock, .download.lock    (only while held)
│  ├─ raw_chunks/track_<track_id>/<chunk_name>[-<yoff>-<xoff>].tif
│  └─ stack/track_<track_id>/
│     ├─ VV/VV_<date>.vrt, VH/VH_<date>.vrt   (each with <name>.vrt.sources.sha256)
│     ├─ stack_VV.vrt, stack_VH.vrt
│     ├─ dates.csv
│     ├─ _valid_cache/            (per-batch part files, compacted at the end)
│     └─ _aoi_cache.json
└─ LATEST.txt
```

### Google Cloud Storage
`gs://<bucket>/` + `config.gcs_prefix(cfg, "runs", run_meta.run_uid(run_dir), "raw_chunks", f"track_{track_id}", chunk_name)`.
Earth Engine appends `.tif`, or `-<yoff>-<xoff>.tif` when it splits a file.

---

## 6. Module contracts

### 6.0 `prep/` (pre-flight checks, no exports)

```
python -m sar_pipeline.prep aoi-qc          --config <yaml> [--split-dir DIR] [--pattern GLOB] [--id-field F]
                                            [--area-column C] [--top N] [--json]
python -m sar_pipeline.prep s1-availability --config <yaml> [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                                            [--min-dates N] [--json]
```

**Contract.** Both steps are **read-only and side-effect free**: no file, no folder under
`processed/`, no run folder, no Earth Engine task, no Cloud Storage object. `aoi-qc` is local only;
`s1-availability` makes exactly one Earth Engine *metadata* request. Output goes to stdout (text, or
JSON with `--json`). Exit codes: `0` ok, `1` when an `aoi-qc --split-dir` cross-check fails — the
only step in the repository that signals a verdict through its exit code, so it can gate a script.

**Why the package exists.** Stage 0 (`grid`) trusts two unverified assumptions: that `aoi.path` is
the complete, correctly deduplicated AOI, and that Sentinel-1 covers it often enough in the season
window. Both are cheap to check and expensive to get wrong. Per ground rule 1 of §1 ("no throwaway
scripts") these checks are first-class tested steps, not terminal one-liners.

`prep/aoi_qc.py` — local geometry QC:
```python
SQM_PER_ACRE = 4046.8564224                        # exact; 1 px (10 m) = 0.025 acre
sqm_to_acres(sqm) -> float | pd.Series
find_duplicates(frame) -> DuplicateReport          # n_features, n_unique, group_sizes, n_duplicate_rows, is_consistent()
read_split_aois(paths, id_field="id") -> pd.DataFrame        # columns: file, number, id, wkb; raises on a multi-feature file
crosscheck(original, split, id_field="id") -> CrosscheckReport
area_table(frame, crs, id_field="id") -> pd.DataFrame        # columns: id, acres; largest first
compare_area_column(frame, crs, column="area") -> dict
```
- **Canonical shape comparison.** Every geometry is reduced to `geometry.normalize().to_wkb()`, so
  two polygons describing the same shape with a different vertex order or ring direction compare
  equal. All duplicate and cross-check maths is set arithmetic on those byte strings. Never compare
  geometries by feature count, id or WKT.
- `CrosscheckReport` fields: `duplicates`, `n_split_files`, `missing_from_split` (data loss),
  `unexpected_in_split` (the split was edited), `dropped_to_kept` (`dropped id -> kept id` per removed
  duplicate), `orphaned` (a dropped id whose shape no kept file reproduces — must be empty),
  `id_mismatches` (`(filename number, id inside the file)`), `ok` (all four empty) and `summary()`.
  Shape lists are truncated WKB hex digests, never coordinates: this report may be pasted anywhere.
- `area_table` **refuses a geographic CRS** (`ValueError`) instead of returning degree-squared
  nonsense. Pass the same `grid.crs` the pipeline will use, so areas match the grid's own.
- `compare_area_column` treats a supplied area column as metadata, not truth: it reports
  `ratio_to_computed_acres`, `likely_unit` (`acres | hectares | m2 | unknown`) and the max/mean
  disagreement in percent.
- **Areas are in acres everywhere** (see §1 ground rules and the README "Units" section). The factor
  is the named constant `SQM_PER_ACRE`; never write a bare `/ 4046.86`.

`prep/s1_availability.py` — Earth Engine metadata:
```python
SceneRow = tuple[int, int, str, str]               # (epoch ms, relative orbit, pass, platform letter)
fetch_rows(bounds, start, end, collection="COPERNICUS/S1_GRD", instrument_mode="IW") -> list[SceneRow]
build_report(rows, start, end) -> AvailabilityReport                # pure, no Earth Engine
availability(bounds, start, end, fetch=fetch_rows, **kwargs) -> AvailabilityReport
```
- `Track(relative_orbit, orbit_pass, n_scenes)` with `key` = `RO{orbit}_{ASC|DSC}` — the same shape
  `s1.tracks` uses in the config, so a track can be copied across without reformatting.
- `AvailabilityReport(start, end, n_scenes, tracks, platforms, dates_per_month)` plus `n_tracks`,
  `months_below(minimum)` and `summary()`. `dates_per_month` counts **distinct dates**, not scenes.
- **Separation of concerns:** `fetch_rows` is the only part that touches Earth Engine, `build_report`
  is pure, and `availability` takes `fetch` as an injectable argument, so `tests/test_prep_s1_availability.py`
  runs the whole report path on canned metadata with no network (§1 ground rule 8).
- One `reduceColumns(ee.Reducer.toList(4), [...])` call pulls back four properties, so the payload
  stays small for thousands of scenes (§7: never a large `getInfo`).
- `bounds` is the AOI's **bounding box** in EPSG:4326, and a scene touching the box is counted. This
  is documented as a deliberate limitation: the step sizes the problem, `audit.py` decides
  `s1.tracks`. `__main__` also strips a `_FLOAT` suffix from `s1.collection`, because the float and
  byte collections hold the same scenes and `S1_GRD` is the safer default for a metadata count.

`prep/__main__.py` builds the parser, loads the config with `config.load_config`, initialises Earth
Engine with `auth.init_ee` **only** for `s1-availability`, and deduplicates the AOI before
summarising areas so repeated polygons do not double-count the total.

### 6.1 `grid.py` (+ `index.py`)

```python
chunk_name(grid_row, grid_col, digits) -> str
read_aoi(cfg, crs) -> gpd.GeoDataFrame             # AOI in the grid CRS, exploded to single parts
compute_grid_def(cfg) -> dict
geometry_diff(old, new) -> str | None              # first geometry difference, None if identical
build_grid(cfg) -> dict                            # writes grid_def.json + chunks.gpkg atomically; GridMismatch on geometry change
load_grid(cfg) -> dict
iter_chunks(gd) -> Iterator[dict]
chunk_at(gd, grid_row, grid_col) -> dict | None    # O(1) via a thread-safe LRU lookup cache (max 4 grids)
chunk_by_name(gd, name) -> dict                    # also resolves one level of sub-chunk names "<chunk>_s<i>"
split_chunk(chunk, res) -> list[dict]
chunk_region_coords(chunk, inset_m=0.25) -> list[float]   # [xmin, ymin, xmax, ymax] in grid CRS, inset so EE selects exactly the chunk pixels
```

`grid_def.json` keys:
`crs, res, x0, y0, width, height, chunk_px, n_chunk_rows, n_chunk_cols, transform [a,b,c,d,e,f], pid_dtype, aoi_file, chunks[]`.

Chunk dict keys:
`chunk_id (int ≥1), name, grid_row, grid_col, row_off, col_off, width, height, xmin, ymin, xmax, ymax, aoi_frac`
(`aoi_frac` = share of the chunk's area inside the AOI, 0–1; sub-chunks inherit the parent's value). Only chunks
that intersect the AOI are listed.

**Immutability compares geometry only:** `GEOMETRY_KEYS = (crs, res, x0, y0, width, height, chunk_px, pid_dtype)`
and per chunk `CHUNK_GEOMETRY_KEYS = (name, row_off, col_off, width, height)`. A changed `aoi_file` path or an
`aoi_frac` drift > 1e-3 only logs a warning; the stored grid is returned unchanged.

`grid.chunk_px` should be a multiple of 512 (the GeoTIFF tile size), so a pixel-index tile never spans two chunks.

Snapping rule: `x0 = floor((minx - buffer)/res)*res`, `y0 = ceil((maxy + buffer)/res)*res`; width and height rounded up.

`index.py`:
```python
affine(gd) -> rasterio.Affine
rowcol_to_pid(gd, row, col) -> int      pid_to_rowcol(gd, pid) -> (row, col)
rowcol_to_xy(gd, row, col) -> (x, y)    # pixel centre
xy_to_rowcol(gd, x, y) -> (row, col)
pid_to_lonlat(gd, pid) -> (lon, lat)    lonlat_to_pid(gd, lon, lat) -> int
chunk_of_pid(gd, pid) -> dict | None    # O(1) via chunk_px arithmetic
build_pixel_index(cfg, gd, overwrite=False, force=False) -> str
```
`build_pixel_index` writes only AOI chunks (sparse GeoTIFF, BIGTIFF=IF_SAFER, tiled 512, deflate + predictor 2)
into a temporary build file, recording finished tiles in `pixel_index.progress.json` after each flushed session.
On resume every tile listed as done is read back and rewritten if wrong; before the final rename a spread sample
of pixels is verified. pid band dtype = `gd["pid_dtype"]`; bands 2–3 share that dtype. nodata = −1.

### 6.2 `audit.py`

Metadata only; no exports. Geometry maths runs locally in shapely/geopandas.

```python
track_id(relative_orbit, pass_) -> str
region_blocks(cfg, gd) -> gpd.GeoDataFrame            # ~50 km blocks of AOI∩chunks, simplified, EPSG:4326
region_filter_boxes(blocks, pad_deg=0.01) -> list
bounded_geometry(geom, max_vertices=None)             # ≤ MAX_GEOM_VERTICES (2000)
plan_batches(records, weight=None, max_weight=None, max_bytes=None, size=None) -> list[list[dict]]
list_slices(cfg, start=None, end=None, gd=None) -> gpd.GeoDataFrame
assign_acquisition_ids(slices, timezone_name="UTC") -> pd.Series
coverage_basis(cfg) -> str
group_acquisitions(slices, aoi_4326, cfg) -> gpd.GeoDataFrame
incidence_angles(cfg, acquisitions, partial_dir=None) -> pd.Series
track_summary(acquisitions, cfg) -> pd.DataFrame
detect_gaps(acquisitions, cfg, now=None) -> pd.DataFrame
chunk_track_coverage(cfg, gd, footprints) -> pd.DataFrame
recommend_tracks(summary, coverage, cfg) -> pd.DataFrame
rain_source_for(t, coverage_end, order) -> str | None
expected_rain_images(t, hours, step_hours) -> int
rain_window_weights(t, hours, images) -> list[float]
parse_rain_features(features, level) -> list[dict]
rain_flags(cfg, acquisitions, gd=None, partial_dir=None) -> pd.DataFrame
percentiles_from_histogram(lower_edges, counts, bin_width, qs=(50, 75, 90, 95, 99)) -> dict
aggregate_slope_histograms(histograms, bin_width=SLOPE_BIN_DEG) -> dict
slope_stats(cfg, gd=None) -> dict
write_gaps_report(path, cfg, slices, acquisitions, summary, gaps, recommendation, rain, slope, coverage) -> Path
run_audit(cfg, now=None, audit_date=None, force=False) -> Path
load_acquisitions(audit_path) -> pd.DataFrame
selected_acquisitions(cfg, audit_path=None) -> dict[str, pd.DataFrame]   # {track_id: rows sorted by datetime}; AuditRequired if s1.tracks empty
```

CSV columns:
- `s1_slices.csv`: `system_index, track_id, relative_orbit, pass, platform, datetime_utc, slice_number, polarisations, resolution_meters`
- `s1_footprints.gpkg`: layer `footprints`, columns `system_index, track_id, datetime_utc`, EPSG:4326
- `s1_acquisitions.csv`: `acquisition_id, track_id, relative_orbit, pass, datetime_utc, date_utc, date_local, platforms (";"-joined), n_slices, system_indexes (";"-joined), aoi_coverage_pct, mean_incidence_angle, coverage_basis`
- `track_summary.csv`: `track_id, relative_orbit, pass, n_acq, n_acq_ok_coverage, first_date, last_date, median_gap_days, max_gap_days, platforms, mean_aoi_coverage_pct, mean_incidence_angle`
- `chunk_track_coverage.csv`: `chunk_name, track_id, n_acq, n_acq_full_cover, mean_cover_pct`   (`full_cover` = ≥ 99 % of chunk∩AOI)
- `track_recommendation.csv`: `track_id, role (primary|secondary|unused), reason`
- `gaps.csv`: `track_id, gap_start, gap_end, gap_days, events_in_gap`
- `rain_flags.csv`: `acquisition_id, track_id, datetime_utc, rain_6h_mm, rain_24h_mm, source, level, chunk_name, n_images_expected, n_images_found, rain_gap_images`
- `slope_stats.json`: `dem, scale_m, percentiles_deg {p50,p75,p90,p95,p99}, pct_area_gt_5deg, pct_area_gt_10deg, pct_area_gt_15deg`

Coverage is computed in `EQUAL_AREA_CRS` (EPSG:6933). `coverage_basis` is `footprint`, or
`footprint, border-noise mask not applied` when the optional mask is enabled.

**Request-size rule:** every geometry sent to EE passes `bounded_geometry`; requests are split by `plan_batches`
(≤ `MAX_FEATURES_PER_REQUEST` 4000 features and ≤ `MAX_REQUEST_BYTES` 2 MB estimated payload); results are paged
with `computeFeatures`. `list_slices` filters with padded block envelopes; incidence angles filter images by
`system:index`; slope percentiles come from per-block `fixedHistogram`s (bin `SLOPE_BIN_DEG` 0.1°) merged locally.

Recommendation rule: primary = most acquisitions with `aoi_coverage_pct ≥ audit.min_aoi_coverage_pct`,
tie-break smaller `max_gap_days`; secondary = any other track with ≥ 50 % of the primary's count; rest unused.
For large AOIs, evaluate per chunk (`chunk_track_coverage`) and report it; do not auto-apply.

Rain: precipitation (mm) in the 6 h and 24 h before `datetime_utc`, mean over footprint∩AOI (or per chunk when
`rain.level == "chunk"`). Images overlapping the window `[t−h, t)` are included and weighted by their overlap
fraction (image duration from `system:time_end`; IMERG images are 0.5 h, GSMaP 1 h). Use `rain.primary` if it
covers the timestamp, else `rain.fallback`; record which in `source`. An empty window gives NaN and `source = none`
without failing. `rain_gap_images > 0` flags missing images inside the 24 h window.

DEM gotcha: `COPERNICUS/DEM/GLO30_2024_1` is an ImageCollection. Use
`.select("DEM").mosaic().setDefaultProjection(<first image projection>)` before `ee.Terrain.slope`,
otherwise slope is computed on a default 1-degree projection and is wrong.

Resume: every step's output is a checkpoint keyed in `_cache_keys.json` (config sections, AOI file hash,
hashes of upstream outputs); long batches write `_partial/`.

### 6.3 `s1_ard/` (Earth Engine server-side preprocessing)

Adapted from `adugnag/gee_s1_ard` (MIT, © 2021 Adugna Mullissa). Keep the licence notice in
`s1_ard/LICENSE_gee_s1_ard.txt` and a header in each adapted file.

```python
# s1_ard/pipeline.py
validate_config(cfg) -> None                                          # raises ARDConfigError(PipelineError, ValueError) listing all problems
band_names(acquisitions, pols=("VV", "VH")) -> list[str]             # [VV_d1, VH_d1, VV_d2, VH_d2, ...], same-date suffixes
load_slices(cfg, system_indexes) -> ee.ImageCollection               # S1_GRD_FLOAT, bands VV, VH, angle
preprocess_slice(img, cfg) -> ee.Image                                # optional border noise -> terrain flattening (native projection)
mosaic_acquisition(cfg, gd, acquisition_row) -> ee.Image              # mosaic, setDefaultProjection(grid crs, grid transform), property "footprint"
filter_series(cfg, images) -> list[ee.Image]                          # speckle filtering in LINEAR power
to_output(cfg, img) -> ee.Image                                       # dB, dtype/scale, unmask(nodata)
build_chunk_image(cfg, gd, chunk, track_id, acquisitions) -> ee.Image

# s1_ard/multitemporal.py
temporal_neighbors(n, k, half_window) -> list[int]
quegan_series(images, filtered, bands, half_window) -> list[ee.Image]

# s1_ard/terrain_flattening.py
load_dem(dem_id) -> ee.Image
look_direction_deg(image) -> ee.Number
slope_geometry(image, dem) -> dict
flattening_factor(model, theta_i, alpha_r, alpha_az) -> ee.Image
layover_shadow_mask(alpha_r, theta_i, buffer_m=0) -> ee.Image
flatten_slice(image, pols, model, dem, buffer_m=0) -> ee.Image
# pure-python mirrors (unit-tested): toward_sensor_azimuth_deg, range_slope_deg, volume_factor, direct_factor, is_valid_geometry

# s1_ard/speckle.py
FILTERS = ("BOXCAR", "LEE", "GAMMA MAP", "REFINED LEE", "LEE SIGMA"); KERNEL_SIZE_IGNORED = {"REFINED LEE"}
mono_filter(image, name, kernel_size, bands) -> ee.Image

# s1_ard/border_noise.py
mask_border_noise(image) -> ee.Image                                  # angle limits ANGLE_MIN_DEG / ANGLE_MAX_DEG
```

Processing order and why:
1. **Border-noise mask** on each slice, only if `ard.border_noise_correction` (default false). Uses the angle band
   read with bilinear resampling.
2. **Terrain flattening** (Vollrath et al. 2020, model `ard.terrain_flattening_model`, DEM `ard.dem`) on each slice
   **before mosaicking**: the look direction comes from the gradient of the slice's bilinearly resampled `angle`
   band in the slice's own projection. Output γ⁰ with layover/shadow masked.
   **Convention (validated live):** α_r (range slope) is **positive for slopes facing the sensor**.
   VOLUME: `γ⁰_f = γ⁰ · tan(θ − α_r) / tan θ`; DIRECT: `γ⁰ · cos(α_az) · sin(θ − α_r) / sin θ` (less validated).
   Layover masked where `α_r ≥ θ`, shadow where `α_r ≤ −(90° − θ)`. Validation: facing vs away-facing slopes of
   equal steepness agree within ~1.5 dB for |α_r| < 30° on ascending and descending data; ≥ 99 % of true layover
   masked; slopes facing the sensor at 30–45° remain darkened (factor → 0 near layover).
3. **Mosaic** slices of one acquisition, then `setDefaultProjection(crs=grid crs, crsTransform=grid transform)`
   so focal kernels are measured in 10 m grid pixels. The acquisition's bounded footprint is stored as image
   property `footprint` (needed by LEE SIGMA's percentile).
4. **Speckle filter** in linear power. `ard.multitemporal: true` means a Quegan-type filter over
   `temporal_neighbors(...)` of the same track: `J_k = F(I_k) · mean_i( I_i / F(I_i) )` with F the mono filter
   (`ard.speckle_filter`, `ard.speckle_kernel`), mean over valid pixels only; where only one date is valid the
   result is `F(I_k)`. Refined Lee ignores the kernel size (fixed internal windows; weights clamped to [0, 1],
   deterministic direction ties). LEE SIGMA uses a sampled 98th percentile over the footprint, a 3×3 bright-pixel
   count and weights clamped to [0, 1].
5. **dB** (`10·log10`), then output encoding:
   - float32: `toFloat()`, masked → `export.nodata_float32`;
   - int16: `floor(dB * int16_scale + 0.5)`, clamp to [−32767, 32767], masked → `export.nodata_int16`
     (mirrored in `helper.encode_int16_value` / `decode_int16_value`).

Band order of a chunk image: for each acquisition in ascending datetime: `VV_<date>`, `VH_<date>`.

### 6.4 `manifest.py`, `export.py`

```python
# manifest.py
STATES = ["PLANNED", "PENDING", "SUBMITTING", "SUBMITTED", "READY", "RUNNING", "COMPLETED", "FAILED", "SPLIT",
          "NOT_COVERED", "DOWNLOADED", "VERIFIED", "VERIFIED_EMPTY", "FAILED_DOWNLOAD"]
ERROR_CLASSES = ["", "memory", "timeout", "quota", "transient", "cancelled", "lost", "verify", "io", "other"]
COLUMNS = ["task_key", "track_id", "chunk_name", "parent_chunk", "row_off", "col_off", "width", "height",
           "n_bands", "gcs_prefix", "description", "task_id", "state", "attempts", "last_error",
           "error_class", "submitted_at", "updated_at", "local_paths", "adopted"]
ACTIVE_STATES = ["SUBMITTED", "READY", "RUNNING"]; OWN_ACTIVE_STATES = ["SUBMITTING", *ACTIVE_STATES]
manifest_path(run_dir) -> Path;  journal_path(run_dir) -> Path;  lock(run_dir) -> FileLock   # ".manifest.lock"
read_manifest(run_dir) -> pd.DataFrame                                  # snapshot + journal replay (incremental)
update_manifest(run_dir, rows, return_df=True) -> pd.DataFrame | None  # upsert by task_key; one journal append per call (batch), fsync
compact(run_dir) -> None                                                # atomic snapshot rewrite + empty journal
claim(run_dir, task_key, expected_states, values) -> bool               # compare-and-set under the manifest lock
```
Storage: `export_manifest.csv` (snapshot) + `export_manifest.journal.jsonl` (append-only; every entry sets absolute
values, so replay is idempotent; a torn last line is ignored with a warning). Compaction runs automatically when
the journal exceeds `COMPACT_MIN_LINES` (2000) lines or ~10 % of rows. Always read through `read_manifest`.

```python
# export.py
class EEBackend:                                      # real implementation; tests use fakes with the same methods
    def start_export(self, image, params: dict) -> str: ...
    def task_statuses(self, task_ids: list[str]) -> dict[str, dict]: ...   # getTaskStatus fallback, bounded per round
    def project_snapshot(self, cutoff=None, max_age=0.0) -> dict: ...      # paged listing: counts + tasks (id, description, state, error, create time)
class MonitorSession: ...                              # grid, selected acquisitions and GCS blob index loaded once per session

scan_operations(fetch_page, cutoff=None) -> dict      # pages of 500; stops after a page with a terminal op older than cutoff; full read if an active op appears after terminal ones
listing_cutoff(df, run_dir) -> datetime               # earliest submitted_at / SUBMITTING updated_at / run date, minus 1 day
task_key(track_id, chunk_name) -> str;  split_task_key(key) -> (track_id, chunk_name)
nodata_value(cfg);  sanitise_description(text);  task_description(cfg, run_uid, key)
classify_error(message) -> str                        # memory | timeout | quota | transient | cancelled | other
check_run(cfg, run_dir) -> None                       # run_meta.read_export_meta when export_meta.json exists (project, grid, keys)
plan_exports(cfg, run_dir, chunk_names=None, force=False) -> pd.DataFrame     # writes export_meta.json + band_layout.csv; PLANNED / NOT_COVERED rows
confirm_exports(cfg, run_dir, chunk_names=None) -> pd.DataFrame               # PLANNED -> PENDING for that scope only
retry_failed(run_dir, classes=None, cfg=None) -> pd.DataFrame                # FAILED -> PENDING (validates the run when cfg is given)
export_params(cfg, gd, chunk, task_key, run_uid) -> dict
queue_headroom(cfg, project_counts, own_active) -> int
median_task_seconds(df) -> float | None;  estimate_eta_seconds(df, running) -> float | None
submit_pending(cfg, run_dir, backend=None, confirmed=False, sleep=time.sleep, gcs=None, session=None, compact_after=True) -> pd.DataFrame
refresh_status(cfg, run_dir, backend=None, session=None) -> pd.DataFrame
monitor(cfg, run_dir, backend=None, confirmed=False, until_done=True, sleep=time.sleep, max_cycles=None, gcs=None) -> pd.DataFrame
write_failed_report(run_dir) -> Path                  # failed_chunks.csv: task_key, track_id, chunk_name, state, attempts, error_class, last_error, gcs_prefix
```

File schemas written by `plan_exports`:
- `export_meta.json` keys: `run_id, run_uid, ee_project, dtype ("float32"|"int16"), nodata, int16_scale, grid_crs,
  grid_transform, grid_fingerprint (sha256 of the RAW BYTES of grid/grid_def.json), audit_dir (POSIX path relative to
  project root), tracks, config_hash, created_utc`. `run_uid` and `ee_project` are optional for legacy runs.
  `config_hash` covers only settings that change the pixels: `aoi, season, grid, s1, ard` and
  `export.dtype/int16_scale/nodata_*`. A changed layout/meta raises unless `force=True`, which is refused once any
  row progressed beyond planning.
- `band_layout.csv` columns (in order): `track_id, band_idx (1-based within chunk file), acquisition_id, date_utc, datetime_utc, date_local, pol, platforms, n_slices`.
- Planning reads the run's audit `chunk_track_coverage.csv`: (track, chunk) with `n_acq == 0` → `NOT_COVERED`.

Export parameters (the grid-alignment contract):
```python
ee.batch.Export.image.toCloudStorage(
    image=..., description=task_description(...), bucket=cfg["gcs"]["bucket"],
    fileNamePrefix=<gcs prefix with run_uid>, region=ee.Geometry.Rectangle(chunk_region_coords(chunk), proj=crs, geodesic=False),
    crs=gd["crs"], crsTransform=gd["transform"],      # MASTER grid transform, never a per-chunk one
    maxPixels=1e13, fileFormat="GeoTIFF", formatOptions={"noData": nodata},
)
```

Submission and state policy:
- `submit_pending`, `refresh_status`, `monitor` and `retry_failed(cfg=...)` first call `check_run`: a run is never
  worked on with another Earth Engine project's listing (running tasks would look lost and be exported again).
- `submit_pending` holds `run_lock(run_dir, "monitor")` for the round (re-entered inside `monitor`); another process or
  thread fails immediately. The CLI takes the same lock before `confirm_exports`, so a refused `export --yes` changes nothing.
- Only `PENDING` rows are submitted. A row is **claimed** with `manifest.claim` (compare-and-set PENDING → SUBMITTING)
  which also writes its description and `submitted_at`, **before** `start()`. A row no longer PENDING is skipped.
- `start()` raising a quota error → backoff with jitter, row back to PENDING (no attempt used). Any other exception
  keeps the row `SUBMITTING` with `attempts+1`; `submitted_at` is never overwritten (the task may exist).
- `SUBMITTING` rows are resolved by stored task id, by description for tasks created at/after `submitted_at`
  (`CREATE_TIME_SKEW_SECONDS` tolerance), or by existing GCS output (`adopted = 1`). The grace period
  (`SUBMITTING_GRACE_SECONDS`, 600 s) counts from the last activity (`updated_at`); after it the row goes back to
  PENDING, or FAILED once `task_max_attempts` is used up. PENDING rows are never adopted.
- `adopted = -1` marks a **fresh export** (output failed verification; set by download, legacy rows migrated): it only
  adopts tasks with a known creation time at/after its `submitted_at` and GCS blobs whose `time_created` is at/after
  it, and keeps -1 until download verifies the new files (the claim clears the old `task_id`).
- Task states and error messages come from the project snapshot; `getTaskStatus` is used only for active rows missing
  from it (≤ `_STATUS_LOOKUP_MAX` 50 per round). Missing for `UNKNOWN_POLLS_TO_LOOKUP` (3) polls → back to `SUBMITTING`
  (`error_class = lost`, task id kept) and resolved like any interrupted submission: adopted if found, otherwise
  PENDING again after the grace period; FAILED (class `lost`) only once `task_max_attempts` is used up.
- Task `FAILED` with `memory` or `timeout` → if `split_on_resource_error` and not a sub-chunk: state `SPLIT`, 4
  `PENDING` sub-chunk rows (`parent_chunk` set). Sub-chunks failing the same way → FAILED.
- `quota` / `transient` → back to PENDING while `attempts < export.task_max_attempts`, else FAILED.
- `cancelled` / `other` → FAILED immediately. `retry_failed` re-queues FAILED rows deliberately.
- `monitor` holds `run_lock(run_dir, "monitor")`, touches it every poll, and backs off on quota/transient/timeout
  errors of the listing instead of stopping.
- GCS existence checks only for `SUBMITTING` rows; each track prefix is listed once per session (refreshed after
  `BLOB_INDEX_TTL_SECONDS` 900 s). File names are matched with `run_meta.EE_PART_SUFFIX`; listings carry `time_created`.
- Throttle, re-evaluated every round:
  1. **Project-wide queue limit:** at most `export.max_queued_tasks` (3000) queued tasks (READY + RUNNING) in the
     project, shared with other jobs. Headroom = `floor(max_queued_tasks * queue_safety_margin)` − project count.
  2. **Optional own cap:** `export.max_active_tasks` = `auto` or an int limiting this run's SUBMITTING/SUBMITTED/READY/RUNNING rows.
  Submit at most `min(pending, headroom_1[, headroom_2])`; when ≤ 0, log `queue full` and wait for the next poll.
  Each round logs `submission round: queued=… running=… submitted=… headroom=… pending_left=… eta=…`; the ETA appears
  after `_MIN_COMPLETED_FOR_ETA` (3) completed tasks. Concurrency is measured, never hard-coded.
- Submission threads: `resources.submit_threads`.

### 6.5 `download.py`

```python
class GCSBackend:
    def list_blobs(self, prefix: str) -> list[dict]: ...          # {"name", "size", "crc32c"}
    def download(self, name: str, dest: Path) -> None: ...          # unique temp file, crc32c verified, replace_with_retry

bytes_crc32c_b64(data) -> str;  file_crc32c_b64(path) -> str;  local_matches(path, size, crc32c) -> bool
read_index(run_dir) -> dict[str, dict]
group_blobs_by_prefix(blobs) -> dict[str, list[dict]];  chunk_blob_names(blobs, prefix) -> list[dict]
inspect_chunk_files(paths, gd, chunk, n_bands, nodata, dtype=None) -> tuple[str, str]   # ("ok" | "empty" | "bad", message)
verify_chunk_files(paths, gd, chunk, n_bands, nodata, dtype=None) -> tuple[bool, str]
estimate_download(cfg, run_dir, backend=None, retry_failed=False, force=False, deep=False) -> dict
    # {"n_tasks", "n_skipped_verified", "n_files", "bytes", "disk_free_bytes", "fits", "tasks_without_files"}
download_completed(cfg, run_dir, backend=None, confirmed=False, retry_failed=False, force=False, deep=False) -> pd.DataFrame
```
- Rows in `COMPLETED / DOWNLOADED / VERIFIED / VERIFIED_EMPTY` are considered; verified rows whose files still match
  the index are skipped without listing.
- Each track prefix is listed once per call (shared by estimate and download).
- Checks: CRS equals grid CRS; pixel size equals `res`; union of file extents equals the chunk extent exactly
  (origins snapped to the master grid, tolerance 1e-6 m); pixel and band count; dtype and nodata tag equal
  `run_meta.read_export_meta` (which also checks grid fingerprint and project). Entirely nodata → `VERIFIED_EMPTY`.
- Failures: io/checksum errors retried up to `IO_ATTEMPTS` (3); verification failures get one re-download, then
  `FAILED_DOWNLOAD` with `error_class = verify`; an adopted row (`adopted = 1`) failing verification goes back to
  `PENDING` (`adopted = 0`, `error_class = verify`, attempts + 1) for re-export. A file locked by another program is
  left for the next run without using an attempt.
- `runs/<run>/download_index.jsonl` (append-only, compacted at the end of a call) stores per chunk each file's
  `local_path, size, mtime_ns, crc32c`; unchanged size + mtime is trusted, `deep=True` re-hashes.
- Manifest updates are batched (`FLUSH_EVERY_N` 50 rows or `FLUSH_EVERY_S` 10 s), index before manifest.
- Holds `run_lock(run_dir, "download")`. Threads: `resources.io_threads`. Refuses to start when `fits` is False.

### 6.6 `stack.py`

```python
planned_scope(run_dir, track_id) -> pd.DataFrame        # manifest rows except SPLIT parents, NOT_COVERED, PLANNED
scope_fingerprint(scope) -> str
verified_chunk_files(run_dir, track_id) -> list[Path]   # files of VERIFIED and VERIFIED_EMPTY rows
stack_dir(run_dir, track_id) -> Path
build_date_vrts(cfg, run_dir, track_id, force=False) -> list[Path]
build_stack_vrts(cfg, run_dir, track_id, force=False) -> dict[str, Path]
compute_valid_pct(cfg, run_dir, track_id, force=False) -> pd.DataFrame    # per acquisition & pol; aoi_pixels (planned scope), aoi_pixels_total, aoi_planned_pct
temporal_neighbor_counts(n, cfg) -> list[int]
dates_table(cfg, run_dir, track_id, valid) -> pd.DataFrame
write_dates_table(cfg, run_dir, track_id, valid) -> Path
qa_issues(cfg, run_dir, track_id, valid) -> list[dict]
write_decisions_required(run_dir, issues, track_id=None, acknowledged=None) -> Path | None   # track_id may be a list; raises DecisionRequired for unacknowledged issues
build_stacks(cfg, run_dir, track_ids, force=False) -> dict[str, Path]   # all tracks first, then one decisions file / one DecisionRequired
build_stack(cfg, run_dir, track_id, force=False) -> Path                 # single-track wrapper
```
- Metadata via `run_meta.read_export_meta` / `read_band_layout`; audit folder from `export_meta.json` `audit_dir`
  (missing → `AuditRequired`); grid fingerprint mismatch → `GridMismatch`.
- **VRTs are written as XML directly** (no `osgeo`): each chunk file header is checked with rasterio first (CRS,
  pixel size, no rotation, origin snapped to the grid, inside the grid, dtype = export dtype, enough bands); a bad
  file raises `PipelineError`, nothing is dropped silently. Source paths are relative. Each VRT has a
  `.sources.sha256` sidecar (files with size + mtime, band, grid, dtype, nodata); unchanged → skipped without
  parsing. Date VRT names use the acquisition part after the track id.
- Valid-percentage reads are windowed per chunk and parallelised with `resources.cpu_workers` (spawn pool; a worker
  initializer sets the GDAL cache, counted inside the memory budget), block size from `resources.block_rows`. The
  denominator is the AOI pixels inside the planned scope; failed or missing planned chunks count as invalid.
  Counts are cached in `_valid_cache/` part files keyed per file (path, size, mtime_ns, band, index key).
- `dates.csv`: `stack_band, acquisition_id, date_utc, datetime_utc, date_local, platforms, n_slices, rain_6h_mm,
  rain_24h_mm, valid_pct_<POL>… (one per polarisation in the band layout), aoi_planned_pct, n_temporal_neighbors, flags`
  (`LOW_VALID_<POL>`, `RAIN_24H` ≥ `RAIN_FLAG_MM_24H` 5 mm, `GAP_BEFORE`).
- Issue dicts: `{"issue_id", "type", "track_id", "subject", "detail", "acknowledged"}`.
  Types (`ISSUE_TYPES`): `MISSING_ACQUISITION` (in audit but not in band layout), `LOW_VALID` (< `qa.min_valid_pct`),
  `FAILED_CHUNK` (FAILED / FAILED_DOWNLOAD), `UNVERIFIED_CHUNK` (planned, not yet verified), `EMPTY_CHUNK`
  (VERIFIED_EMPTY, informational). `issue_id` = `"{type}:{track_id}:{subject}"`. Issues listed in
  `qa.acknowledged_issues` do not raise.

### 6.7 `pixel_query.py`

```python
pixel_timeseries(cfg, run_dir, track_id, pid) -> pd.DataFrame
pixel_timeseries_lonlat(cfg, run_dir, track_id, lon, lat) -> pd.DataFrame
plot_timeseries(df, cfg=None, ax=None, title=None) -> matplotlib.axes.Axes
clear_cache() -> None
```
One 1×1 windowed read from each `stack_<POL>.vrt` (all bands at once). Decode int16 via `export_meta.json`; nodata →
NaN. Columns: `date_utc, datetime_utc, acquisition_id, platforms, <POL>_db… (per polarisation), VH_minus_VV_db (only
when both exist), rain_6h_mm, rain_24h_mm, valid`. Grid, validated metadata, `dates.csv` and open VRT readers are
cached per process, keyed by file size + mtime. The plot marks `audit.constellation_events`, shades gaps
> `audit.max_gap_days`, and flags rain ≥ 5 mm in 24 h.

### 6.8 `cli.py`, `__main__.py`, notebooks

```
python -m sar_pipeline --config <yaml> grid
python -m sar_pipeline --config <yaml> audit    [--audit-date YYYYMMDD] [--force]
python -m sar_pipeline --config <yaml> new-run
python -m sar_pipeline --config <yaml> export   [--run RUN] (--pilot CHUNK | --all | --retry-failed) [--yes] [--force]
python -m sar_pipeline --config <yaml> monitor  [--run RUN] [--yes]
python -m sar_pipeline --config <yaml> download [--run RUN] [--yes] [--force] [--retry-failed] [--deep]
python -m sar_pipeline --config <yaml> stack    [--run RUN] [--track TRACK] [--force]
python -m sar_pipeline --config <yaml> pixel    [--run RUN] --track TRACK (--pid PID | --lon LON --lat LAT) [--plot PNG]
python -m sar_pipeline --config <yaml> resources
```
- `export` without `--yes`: plans (PLANNED rows), prints "To export" and "Not covered" counts, exits 2. With `--yes`:
  plan, `confirm_exports` for that scope, `submit_pending`. `--retry-failed [--yes]`: `retry_failed` then submit.
- `monitor` without `--yes` prints Pending/Planned counts and exits 2; with `--yes` it never promotes PLANNED rows.
- `download` without `--yes` prints files, size, free disk and exits 2.
- Exit codes: 0 success, 1 pipeline error, 2 confirmation needed / bad usage.

**Which config is used.** Order: `grid` → `audit` → user fills `s1.tracks` → `new-run` (freezes the config into
`runs/<run>/run_config.yaml`; refuses with `AuditRequired` when `s1.tracks` is empty) → run-scoped commands
(`export, monitor, download, stack, pixel`). Run-scoped commands use `cli.load_run_context(user_cfg, run_id)`:
the frozen config wins; differences in `RUN_SECTIONS = (aoi, season, grid, gcs, s1, ard, export, qa)` are warned
about; `auth.project` stays the run's project (warning if the user's differs). Taken from the user's config:
`MACHINE_SECTIONS = (resources,)`, `MACHINE_KEYS = (auth.key_file,)` and `USER_DECISION_KEYS = {qa.acknowledged_issues}`.
Helpers: `load_run_context`, `config_differences`, `require_tracks`.

The prep CLI is a **separate** entry point (`python -m sar_pipeline.prep <step> --config <yaml>`, §6.0)
rather than a `sar_pipeline` subcommand, because it takes no `--run`, writes nothing and must be usable
before a grid or a run exists.

Notebooks (thin, no outputs committed): `01_grid_and_audit`, `02_export`, `03_download_and_stack`, `04_pixel_explorer`.
Notebook 02 calls `export.confirm_exports` in its confirmation cells before `submit_pending` / `monitor`.

### 6.9 `analysis/` (ground-truth analysis, local only)

```
python -m sar_pipeline.analysis gcp-qc    --config <yaml> [--run RUN] [--write-copy [--force]]
python -m sar_pipeline.analysis gcp-eda   --config <yaml> [--run RUN]
python -m sar_pipeline.analysis features  --config <yaml> [--run RUN] [--bins half_month|N ...] [--starts YYYY-MM-DD ...]
                                          [--windows W ...] [--interior-only] [--name NAME]
python -m sar_pipeline.analysis evaluate  --config <yaml> [--run RUN] [--name NAME] [--sets SET ...] [--save-oof]
python -m sar_pipeline.analysis cv-layers --config <yaml> [--run RUN] --set SET
python -m sar_pipeline.analysis class-map --config <yaml> [--run RUN] --set SET [--name NAME] --map-config <yaml> [--map-run RUN]
python -m sar_pipeline.analysis model     --config <yaml> [--run RUN] [--name NAME] [--map-config <yaml> [--map-run RUN]]
python -m sar_pipeline.analysis final-models --config <yaml> [--run RUN] [--name NAME] [--without-bin TRACK:MMDD ...]
python -m sar_pipeline.analysis predict   --config <yaml> [--run RUN] --map-config <yaml> [--map-run RUN] [--models DIR] [--name NAME]
python -m sar_pipeline.analysis field-labels --config <yaml> [--run RUN] --delineation FILE --class-raster TIF
                                             [--prob-raster TIF] [--model JOBLIB] [--train-config <yaml> [--train-run RUN]] [--name NAME] [--simplify-m M]
```
- Reads only local run outputs (`run_config.yaml`, grid, manifest `local_paths` of VERIFIED rows, `band_layout.csv`,
  `stack/track_<id>/dates.csv`). Writes to `processed/<aoi>/<season>/analysis/<run_id>/` (`pixel_features.analysis_dir`).
- Config: `gcps.{path, class_field, id_field}` and `analysis.*` (defaults in `pipeline.example.yaml`). Class names and
  codes come from the data (`gcp_qc.class_codes`, must be one-to-one); `analysis.target_class` must be one of them.
- `gcp_qc`: `load_gcps` adds `gcp_id` (row order) and `label`. `write_qc_copy` writes `<stem>_qc.gpkg` next to the
  original (never modified) from `gcp_review.csv`; `EXCLUDED_STATUSES = (check_label, mixed_pixels)` →
  `use_in_analysis = False`. `load_fields` uses the copy when it exists, else all fields (warning).
- `pixel_features`: `FeatureSet(kind, days, window, start)`, name `<hm|dN>_w<W>_s<MMDD|season>`. Bins end at the day
  after the last acquisition. Columns `<track_id>__<set>__<VH|VV|VHmVV>__<bin start MMDD>` plus `pid, gcp_id, label,
  group, border`. `columns_for_set` / `sets_in` parse names. Rows with non-finite values are dropped and counted
  (`dropped_non_finite_rows` in `<name>.json`), then capped per field (`cap_per_field`, seeded).
- `evaluate_cv`: `make_model(cfg)` (fixed RF from `analysis.rf`), `GroupKFold` on `group`. Summary columns
  `target_precision, target_recall, target_f1, target_f1_field_vote, accuracy, macro_f1, f1_<class>, n_features`.
- `maps.class_map`: requires identical band layouts in both runs (`check_same_layout`), rain flags from the training
  run, prediction only where `pixel_index.tif` band 3 (`aoi_mask`) = 1. Class raster uint8 nodata 0; target
  probability raster uint8 percent nodata 255; QML palette in class-code order. The loop is `maps.predict_aoi(cfg,
  train_run, fs, cols, map_run, fitted_model, rules)` with `rules = {name: proba -> class index}`; `class_map` uses argmax.
- `model.run`: feature set from config; outputs in `analysis/<run>/model_<set>/`: `holdout_split.csv` (gcp_id, group,
  split), `holdout_composition.csv`, `tuning_results.csv` (one row per grid point, appended; resume by `spec` JSON),
  `threshold_curve.csv`, `winner.json` (spec, threshold, dev OOF scores), `holdout_metrics.json` (written once, never
  recomputed), `holdout_per_class.csv`, `holdout_confusion_<argmax|threshold>.csv`. Map in the map run's analysis folder:
  `model_<rf|xgb>_<set>_classes.tif/.qml/.json`, `model_<rf|xgb>_<set>_<target>_prob.tif`. Grids are module constants
  (`RF_GRID`, `XGB_GRID`); `check_no_holdout` raises if holdout fields reach tuning/threshold; `XGBLabels` wraps
  XGBClassifier with class-name labels (sorted, same order as RF `classes_`).
- `final_models.save_final_models(cfg, train_run, name, without_bins)`: needs `model_<set>/winner.json`; refuses a
  `final_models/` folder that already holds `.joblib` files. Variants from `variant_columns(cols, s1.tracks,
  without_bins)`. Each bundle is a joblib dict `{model, columns, labels, target, threshold, spec, feature_set, variant}`;
  the field-level bundle adds `stat: "mean"` and has its own threshold. `model_card.json` holds grouped CV scores per
  variant (`cv_scores`).
- `final_models.predict(cfg, train_run, map_run, models_dir=None, name=None)`: `load_pixel_models` (skips bundles
  with `stat`, orders by column count, requires equal labels and threshold),
  `pick_models` (1-based variant per pixel, 0 = none), AOI mask from `pixel_index.tif` band 3, a track missing from
  the manifest for a chunk counts as missing features. Workers: `resources.cpu_workers` with `MEM_PER_WORKER`.
- `field_labels.run(cfg, map_run, delineation, class_raster, prob_raster, model_path, train_cfg, train_run, name)`:
  outputs in `processed/<aoi>/<season>/fields/<name>/` (`fields_dir`): `fields_labelled.gpkg` (layer `fields`),
  `fields_labelled.parquet`, `summary.json` (per-step counts and acres). Geometry work runs in the class raster's
  CRS, the file is written in EPSG:4326. Thresholds are module constants with the reasoning in their comments
  (`PURE`, `MIN_PIXELS`, `MIN_CUT_GAIN`, `MIN_DERIVED_ACRES`, `DESPIKE_M`, `MIN_FIELD_WIDTH_M`,
  `MIN_SPLIT_WIDTH_M`, `MIN_COMPACTNESS_ANY`, `MIN_RECTANGULARITY`, `SLIVER_SQM`, `GRID`). Building blocks:
  `repair` (make_valid + `explode_fully` + `resolve_overlaps`; `report_coverage=True` dissolves the layer for one
  log line and is off because it dominates memory on a large layer), `despike` (mitred opening intersected back
  with the original), `drop_lines`, `zonal_counts` / `collect_pixels` (windowed, `WINDOW_PX`, and the caller's
  `claimed` mask marks ground a polygon covers), `label_polygons` (majority + cutting), `best_cut` / `halves`
  (one straight cut, generalised to N classes), `derived_polygons`, `apply_field_model` (mean features per
  polygon via `field_means`, using the map run's chunk files and the TRAINING run's rain flags), `tidy`.
  `--model` is optional: without it only the majority label is written.

---

## 7. Earth Engine gotchas (read before touching `s1_ard`, `audit`, `export`)

- `COPERNICUS/S1_GRD_FLOAT` is linear power; `COPERNICUS/S1_GRD` is dB. Filter and average only in linear.
- Properties: `relativeOrbitNumber_start`, `orbitProperties_pass`, `platform_number`, `sliceNumber`,
  `instrumentMode`, `transmitterReceiverPolarisation`, `resolution_meters`.
- The `angle` band is a coarse tie-point grid: read it with `resample("bilinear")`.
- A mosaic has a default WGS84 1° projection. Set a default projection before focal/terrain operations.
- `image.geometry()` of a mosaic is unbounded: pass a bounded geometry to reductions.
- Export masked pixels explicitly (`unmask(nodata, sameFootprint=False)` + `formatOptions.noData`); otherwise they become 0.
- Never send full AOI geometries; bound vertices, batch by payload, page with `computeFeatures` (no large `getInfo`).
- Task listing: `projects().operations().list` returns active operations first, then terminal ones newest-first;
  operation states map `PENDING→READY, RUNNING→RUNNING, CANCELLING→CANCEL_REQUESTED, SUCCEEDED→COMPLETED,
  FAILED→FAILED, CANCELLED→CANCELLED`. `ee.data.getTaskStatus` sends one request per id (deprecated): use sparingly.
  The paged reader uses private `ee.data` helpers, which may change with earthengine-api upgrades.

---

## 8. Sentinel-1 constellation context (2026)

S1D user data from 2026-04-17; S1A ended 2026-06-29; S1C phasing manoeuvre in late June (~2 weeks without S1C data);
S1C + S1D give a 6-day repeat afterwards. Expect a possible sensor-change step and a gap; the audit reports both.

---

## 9. Surfacing problems

| Problem | Where it shows up | Pipeline behaviour |
|---|---|---|
| Gap > `audit.max_gap_days` in a track | `gaps.csv`, `gaps_report.md` | reported; the user decides the tracks |
| Acquisition coverage < `min_aoi_coverage_pct` | `s1_acquisitions.csv`, `gaps_report.md` | reported |
| Rain images missing in a window | `rain_flags.csv` `rain_gap_images`, `gaps_report.md` | reported |
| Track never covers a chunk | manifest `NOT_COVERED`, `export` counts | never exported |
| Export task failed / lost | `export_manifest.csv`, `failed_chunks.csv` | retried per policy, then listed; `retry_failed` on request |
| Corrupt / misaligned download | manifest `FAILED_DOWNLOAD` (`io` / `verify`) | retried, adopted tasks re-exported, then listed |
| Entirely nodata chunk | manifest `VERIFIED_EMPTY`, issue `EMPTY_CHUNK` | included, reported |
| Missing acquisition / low valid % / failed or unverified chunk | `decisions_required.md/.csv` | `DecisionRequired` until acknowledged |
| Second monitor/download on the same run | `PipelineError` naming the lock holder | refused |
