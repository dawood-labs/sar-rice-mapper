# 06 — Troubleshooting

Find the message you see, read *what it means*, then *what to do*. When in doubt: **re-running the same
command is always safe**.

---

## General

### The script crashed / the kernel died / the laptop went to sleep

**What it means:** the process stopped in the middle of a stage.

**What to do:** run the **same command or notebook cell again**. The pipeline:
- skips work that is already finished *and verified*, and logs how much it skipped;
- cleans up old temporary files (files are always written under a temporary name and renamed only when
  complete, so no half file ever looks finished);
- after an export crash, looks up tasks that were already started in Earth Engine and **adopts** them
  (manifest rows in state `SUBMITTING`), so nothing is exported twice;
- ignores a half-written last line of the manifest journal (logged as a warning).

Use `--force` only when you *want* derived files recomputed (see
[Crash-safe re-runs](03_pipeline_overview.md#crash-safe-re-runs)).

### `Another 'monitor' process is already working on this run` (or `'download'`)

**What it means:** only one `monitor` and one `download` may run per run, and another process holds the lock
(the message names its process id, host and start time).

**What to do:** let the other process finish, or stop it. If it crashed, the lock clears itself: on the same
machine as soon as the process no longer exists, otherwise after 15 minutes without activity.

### `... is not inside a 'config/' folder and has no 'project_root' key`

Move the config into `<project>/config/`, or add `project_root:` (relative to the config file's folder).

### `Could not write ...: the file is locked by another program`

Excel, QGIS, an antivirus scanner or an editor has the file open (typical on Windows/WSL). Close it and
re-run. Short locks are retried automatically; a download whose file stays locked is left for the next run
without using up an attempt.

### The JupyterHub kernel keeps dying during a heavy step

Usually the container ran out of memory (the OOM killer). The pipeline sizes its work from the
container's memory limit, but other notebooks or large variables in the same server share that memory.

- Close other kernels and free large variables.
- Lower `resources.memory_fraction` (e.g. 0.6 → 0.4) or cap `resources.cpu_workers`.
- Re-run: finished work is skipped.

---

## Earth Engine

| Message (contains) | What it means | What the pipeline does | What you do |
|---|---|---|---|
| `User memory limit exceeded` | the computation for one task was too large | task → `SPLIT` into 4 sub-chunks, resubmitted once | nothing; if sub-chunks fail too, use a smaller `grid.chunk_px` in a **new season/aoi key**, or fewer dates per run |
| `Computation timed out` | task took too long | same as memory: split once | same as above |
| `Too many tasks` / `queue` / `429` / `quota` / `rate limit` | project queue full or API rate limit | backoff and retry; does not use up task attempts | wait; check other jobs on the same EE project |
| `Internal error` / `Service unavailable` / `backend error` / 5xx / `deadline` | temporary Earth Engine problem | error class `transient`: retried up to `export.task_max_attempts` | nothing; if rows end FAILED, `export --retry-failed --yes` later |
| Log line `queue full (queued=…, running=…)` | the project queue has no headroom | waits for the next poll | nothing; other jobs share the queue |
| `Earth Engine project ... not registered` / `Permission denied` on EE | service account not registered for EE on that project | stops | ask the project owner to register it; check `auth.project` |
| `was exported in Earth Engine project ...` | your config names a different `auth.project` than the run | stops reading the run | set `auth.project` back to the run's project (the key file may differ) |
| `Asset ... not found` for the DEM or a rain dataset | dataset id changed or no access | stops (audit) or rain = NaN with `source = none` | check the GEE catalog; update `ard.dem` / `rain.*` |
| Task `CANCELLED` | someone cancelled it in the EE Tasks page | row → FAILED (`cancelled`) | decide, then `export --retry-failed --yes` or accept |
| Row FAILED with class `lost` | the task stayed missing from Earth Engine's task list for many polls | looked up by id/description first, then FAILED | `export --retry-failed --yes` |
| `No acquisitions` / empty audit | wrong dates, AOI, or `instrument_mode` | audit writes empty tables | check `season.start/end`, AOI path/CRS |
| Rain columns NaN, `source = none` | rain dataset does not cover that time yet (latency) | recorded, not an error | fine; refresh the audit later |
| `rain_gap_images > 0` | some rain images are missing inside the 24 h window | recorded; the sum under-counts | treat that rain value as a lower bound |

### Why is Sentinel-1D (or a date) missing?

Earth Engine ingests new scenes within a few days. Very recent dates may not exist yet. The audit
reflects what Earth Engine has **on the audit day**; refresh with `audit --force`.

---

## Cloud Storage and download

| Situation | What it means | What to do |
|---|---|---|
| `403` / `Permission denied` on the bucket | service account lacks read/write on the bucket | ask for Storage Object Admin on that bucket (or prefix) |
| Checksum (CRC32C) mismatch during download | download corrupted in transit | automatic retry (up to 3, class `io`); if it persists, check network/disk, then `download --retry-failed --yes` |
| Row `FAILED_DOWNLOAD` with class `verify` | the file is not aligned to the grid, or has a wrong band count, dtype or nodata tag | one fresh download is tried automatically; if the task had been adopted it is re-exported. Otherwise report it (export parameters broken) |
| Row `VERIFIED_EMPTY` | file is correct but holds only nodata (water, outside swath) | usually fine; appears as `EMPTY_CHUNK` in QA |
| Download refuses: data does not fit | not enough free space | free space / bigger volume, re-run |
| `DNS` / `Temporary failure in name resolution` | network down | re-run when back online |

---

## Pipeline errors

| Error | Meaning | What to do |
|---|---|---|
| `ConfirmationRequired` | an export/download was called without `confirmed=True` (notebook) or `--yes` (CLI) | check the printed plan, then confirm |
| `AuditRequired` | `s1.tracks` is empty, no audit exists, or the run's audit folder is missing | run the audit, review it, fill `s1.tracks` |
| `GridMismatch` | a grid already exists and the new one has different geometry, or the run was exported on a different grid | restore the original AOI/settings, or use a new `aoi.key` / `season.key`. Never delete the grid of a season with runs |
| `DecisionRequired` | QA found issues; see `runs/<run>/decisions_required.md` | decide per issue; acknowledge in `qa.acknowledged_issues`, re-run `stack` |
| `FileNotFoundError: No run yet` | no run created yet | `new-run` |
| `export --force` refused | tasks of this run already progressed beyond planning | create a new run instead |

---

## Outputs look wrong

| Symptom | Likely cause | Check |
|---|---|---|
| Stripes/blocks of nodata along one edge | outside the swath of that track (or the optional border-noise mask) | `s1_footprints.gpkg`, `valid_pct_*` in `dates.csv` |
| Whole date much brighter | rain before the pass | `rain_24h_mm` in `dates.csv` |
| First/last dates noisier than the rest | fewer neighbours for the multi-temporal filter | `n_temporal_neighbors` in `dates.csv` |
| A step in all pixels around 2026-06-29 | sensor change S1A → S1D | compare stable targets (buildings/trees) before/after |
| Values exactly 0 inside the AOI | nodata not applied | should not happen; report |
| Very steep slopes facing the satellite look dark | flattening factor tends to zero near the layover limit | expected physics; see [SAR basics §10](01_sar_basics.md#10-terrain-correction-two-different-things) |
| Fields shifted relative to basemap | grid/CRS problem | `grid_def.json` CRS; re-check with QGIS "identify" |
| VRT opens empty after moving the project | VRTs reference chunk files by relative path inside the run | keep `runs/<run>/` structure intact; rebuild with `stack --force` |
