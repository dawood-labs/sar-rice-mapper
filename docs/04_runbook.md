# 04 — Runbook

The exact, ordered procedure to process an AOI. Each step shows the **CLI command** and the matching
**notebook**. Steps marked **⛔ CHECKPOINT** require a human to look at the results before continuing.

> Every command is **safe to re-run**. If something crashes or the kernel dies, run the same command
> again: finished work is skipped. See [Crash-safe re-runs](03_pipeline_overview.md#crash-safe-re-runs).

Throughout this page:

```bash
CFG=config/my_aoi_season.yaml     # must live in config/ (or set project_root)
mkdir -p logs                     # gitignored; keep log files and plots here
```

---

## Step 0 — Prep checks and pre-flight checklist

- [ ] Setup done ([02 Setup](02_setup.md)); `pytest` passes (this includes the public-repo hygiene test).
- [ ] Config filled; `s1.tracks: []` for now.
- [ ] Key file in `secrets/`; `SAR_PIPELINE_CONFIG=$CFG pytest -m gee` passes.
- [ ] Check the machine:

```bash
python -m sar_pipeline --config $CFG resources
```

Confirm CPUs, available RAM and **free disk** look right for this machine.

### 0a — Is the AOI file the AOI you think it is?

```bash
python -m sar_pipeline.prep aoi-qc --config $CFG
python -m sar_pipeline.prep aoi-qc --config $CFG --json          # same numbers, machine-readable
```

Local only; reads `aoi.path` and `grid.crs`, writes nothing. Read:

- **features vs distinct shapes** — a gap between them means the file contains duplicate polygons.
  `aoi-qc` deduplicates before totalling the area, so the acre total is not double-counted.
- **area (acres)** total / median / min / max and the largest AOIs. Tiny polygons (well under an acre,
  i.e. under ~40 pixels) will be mostly measured from their neighbours later.
- **supplied `area` column** — if the file carries one, the report says which unit it appears to be in
  and how far it disagrees with the geometry. Trust the geometry, not the column.

If someone has split the AOI into one file per polygon (dropping duplicates), cross-check that copy
against the original:

```bash
python -m sar_pipeline.prep aoi-qc --config $CFG --split-dir <folder> [--pattern '*/*.gpkg'] [--id-field id]
```

The verdict line must read `OK - faithful deduplication`. The command **exits non-zero** when a
polygon is missing from the split, present in the split but not in the original, orphaned (a dropped
duplicate whose shape no kept file reproduces), or when a file name's number disagrees with the id
stored inside it. Comparing counts alone cannot catch any of these: 160 features becoming 132 files
says nothing about *which* 28 went.

### 0b — Does Sentinel-1 cover the AOI often enough?

```bash
python -m sar_pipeline.prep s1-availability --config $CFG
python -m sar_pipeline.prep s1-availability --config $CFG --start 2026-05-01 --end 2027-02-01
python -m sar_pipeline.prep s1-availability --config $CFG --min-dates 4 --json
```

Earth Engine **metadata only** (no grid, no run folder, no export, no cost). Read:

- **scenes / tracks** — how many tracks cover the area at all, i.e. whether track selection will be a
  real decision in Checkpoint 1.
- **platforms** — which satellites contributed, so a constellation gap does not surprise you later.
- **distinct dates per month**, and the "months with fewer than N distinct dates" line. One track
  revisits every ~12 days, so 2–3 dates per month per track is normal; a month well below the rest of
  the window is a candidate gap that will later show up as an interpolated time bin.

Use `--start` / `--end` to try candidate season windows **before** committing to one in the config.
For rice the window must contain the flooding minimum, the harvest drop and a dry baseline before
flooding (see [01 SAR basics §5](01_sar_basics.md#5-how-crops-look-over-a-season)).

> Counts are over the AOI **bounding box** and a scene that merely touches the box is counted, so
> this step sizes the problem — it does not replace the audit. **Decide `s1.tracks` from the audit
> (Step 2), never from here.**

---

## Step 1 — Grid and pixel index  *(notebook 01)*

```bash
python -m sar_pipeline --config $CFG grid
```

Output: `processed/<aoi>/<season>/grid/` with `grid_def.json`, `chunks.gpkg`, `pixel_index.tif`.

Check:
- the printed grid size and chunk count are plausible (AOI area ÷ chunk area, plus partial edge chunks);
- open `chunks.gpkg` over the AOI in QGIS: the chunks cover the AOI, and `aoi_frac` is small only at the edges.

If the command fails with `GridMismatch`, a grid for this `aoi.key`/`season.key` already exists with
different geometry. **Do not delete it.** Either restore the original settings or use a new key.

---

## Step 2 — Audit  *(notebook 01)*

```bash
python -m sar_pipeline --config $CFG audit
```

Output: `processed/<aoi>/<season>/audit/<YYYYMMDD>/` (UTC date; metadata only; no exports, no cost).

### ⛔ CHECKPOINT 1 — choose tracks

Open and read, in this order:

1. **`track_summary.csv`**: acquisitions per track, first/last date, median and max gap, platforms, mean
   AOI coverage, mean incidence angle.
2. **`gaps_report.md`**: gaps longer than `audit.max_gap_days`, low-coverage acquisitions, rain windows with
   missing images, and which gaps coincide with constellation events (e.g. the late-June 2026 Sentinel-1C manoeuvre).
3. **`track_recommendation.csv`**: suggested `primary` / `secondary` / `unused` roles with the reason.
4. **`chunk_track_coverage.csv`** (large AOIs): whether the primary track really covers every chunk.
5. **`slope_stats.json`**: how steep the terrain is (how much terrain flattening matters).
6. **`rain_flags.csv`**: rain before each acquisition.

Decide, and write into your config:

```yaml
s1:
  tracks:
    - {track_id: RO123_ASC, role: primary}
    - {track_id: RO045_DSC, role: secondary}   # only if it adds value
```

Questions to ask yourself:
- Does the primary track have acquisitions across the **whole** season (sowing → harvest)?
- Is any gap long enough to miss a phenological stage (e.g. > 18 days during early growth)?
- Is a secondary track worth the extra processing, or does it only add near-duplicate dates?

If new acquisitions appear later (e.g. weeks after the first audit), refresh with
`audit --force`, or `audit --audit-date YYYYMMDD --force` to refresh a specific audit folder.

---

## Step 3 — Create a run  *(notebook 02)*

```bash
python -m sar_pipeline --config $CFG new-run
```

Creates `runs/v<NNN>_<YYYYMMDD>/` with a frozen copy of the config (`run_config.yaml`) and a unique
`run_uid.txt`, and updates `LATEST.txt`. All later commands use the newest run unless you pass `--run <run_id>`.

`new-run` refuses (`AuditRequired`) while `s1.tracks` is empty: the order is always
**grid → audit → fill `s1.tracks` → new-run → export/monitor/download/stack/pixel**.

### Why the frozen config wins

From now on, `export`, `monitor`, `download`, `stack` and `pixel` read **the run's frozen
`run_config.yaml`** for everything. Your `--config` file is only used to find the project and season folders.

Why: the files of a run were produced with specific settings (dates, tracks, filters, `export.dtype`,
nodata value, scale). If someone edits the config after exporting, for example switching `float32` →
`int16`, and `stack` then used the edited file, it would decode the rasters wrongly, **silently**.
Freezing the config makes a run always interpreted with the settings that created it.

- If your config differs from the frozen one in a run-defining section (`aoi, season, grid, gcs, s1, ard, export, qa`),
  every run-scoped command prints a **warning listing the differing keys**. The command still runs, with the frozen settings.
  To apply the changes, create a **new run**.
- **`auth.project` is run-defining too.** Tasks, the queue count and duplicate detection live in the Earth Engine
  project the run was exported in. If your config names a different project, the command warns and uses the run's
  project; reading the run's metadata with a mismatching project raises an error.
- **Taken from your config instead:**
  - `qa.acknowledged_issues`, because acknowledging QA issues (Step 7) is a decision you make after the run exists;
  - `auth.key_file` and `resources`, because they describe the machine, not the run. A run exported on a laptop
    can be downloaded and stacked on a cloud notebook server with a different key location and more CPUs/RAM.

---

## Step 4 — Pilot export  *(notebook 02)*

Pick a pilot chunk in QGIS from `grid/chunks.gpkg`: fully inside the AOI (`aoi_frac` ≈ 1), with a mix of
fields, ideally some slopes.

```bash
python -m sar_pipeline --config $CFG export --pilot chunk_r01c01          # dry run: plans (PLANNED), prints counts
python -m sar_pipeline --config $CFG export --pilot chunk_r01c01 --yes    # confirms this scope and starts it
python -m sar_pipeline --config $CFG monitor --yes                        # waits, applies retry policy
python -m sar_pipeline --config $CFG download                             # prints size + free disk
python -m sar_pipeline --config $CFG download --yes
python -m sar_pipeline --config $CFG stack
```

The pilot creates one task **per selected track** (minus tracks that never cover that chunk, shown as
"Not covered"). Planned rows stay `PLANNED` until the same scope is run with `--yes`; `monitor` never submits them.

### ⛔ CHECKPOINT 2 — inspect the pilot

- [ ] Open a date VRT (`stack/track_<id>/VH/VH_<date>.vrt`) in QGIS over a basemap: fields line up
      with field boundaries; no shift.
- [ ] Value ranges (dB): VH roughly −25 … −10, VV roughly −18 … −3 over farmland; buildings brighter,
      water darker. Values like 0 or −9999 inside the AOI mean a nodata problem.
- [ ] Speckle: fields look smooth but field **edges stay sharp**.
- [ ] Slopes: hillsides facing and facing away from the satellite should look similar after flattening.
      Layover/shadow pixels are nodata. Very steep slopes facing the sensor may stay darker (see
      [SAR basics §10](01_sar_basics.md#10-terrain-correction-two-different-things)).
- [ ] `dates.csv`: every expected date present, `valid_pct_*` high (for a pilot, `aoi_planned_pct` is small).
- [ ] Pixel check (Step 8) on a known field: the curve looks like a crop season.

Only when all boxes are ticked, continue.

---

## Step 5 — Full export  *(notebook 02)*

```bash
python -m sar_pipeline --config $CFG export --all          # dry run: planned tasks per track, not-covered chunks
python -m sar_pipeline --config $CFG export --all --yes
nohup python -m sar_pipeline --config $CFG monitor --yes > logs/monitor.log 2>&1 &
```

`monitor` keeps polling until every confirmed task has finished. It logs how many tasks Earth Engine is running,
the project queue, and an ETA. You can stop it and start it again at any time; a second `monitor` on the same run
is refused while the first one is alive.

Afterwards:
- `export_manifest.csv` (+ `export_manifest.journal.jsonl`): state of every task.
- `failed_chunks.csv`: FAILED export and FAILED_DOWNLOAD rows, with state, error class and message.

If `failed_chunks.csv` is not empty: read [06 Troubleshooting](06_troubleshooting.md). Fix the cause, then
put failed exports back into the queue deliberately:

```bash
python -m sar_pipeline --config $CFG export --retry-failed          # shows how many rows would be retried
python -m sar_pipeline --config $CFG export --retry-failed --yes
python -m sar_pipeline --config $CFG monitor --yes
```

---

## Step 6 — Download  *(notebook 03)*

### ⛔ CHECKPOINT 3 — size check

```bash
python -m sar_pipeline --config $CFG download
```

Read `Files`, `Size`, `Disk free`, `Fits`. If it does not fit, free space or move the project to a
bigger volume. The pipeline refuses to start anyway.

```bash
python -m sar_pipeline --config $CFG download --yes
```

Each file is checksum-verified and checked against the grid. Network/checksum errors are retried up to 3 times;
a file failing verification gets one fresh download. Only one `download` may run per run.

Useful options:
- `download --retry-failed --yes`: try `FAILED_DOWNLOAD` rows again (e.g. after a network or disk problem).
- `download --deep --yes`: re-hash every local file instead of trusting size + modification time.

---

## Step 7 — Stack and QA  *(notebook 03)*

```bash
python -m sar_pipeline --config $CFG stack
```

### ⛔ CHECKPOINT 4 — decisions

If QA finds issues, the command stops with `DecisionRequired` and writes
`runs/<run>/decisions_required.md` (+ `.csv`) for all tracks. Issue types:

| Type | Meaning | Typical decision |
|---|---|---|
| `MISSING_ACQUISITION` | in the audit, but not in the exported band layout | re-plan/re-export in a **new run**, or accept |
| `LOW_VALID` | a date has < `qa.min_valid_pct` valid pixels inside the planned scope | accept (partial swath) and handle in analysis, or fix missing chunks first |
| `FAILED_CHUNK` | a planned chunk's export or download failed after all retries | `export --retry-failed --yes` / `download --retry-failed --yes`, or accept a hole |
| `UNVERIFIED_CHUNK` | a planned chunk is not VERIFIED yet (still exporting or downloading) | run `monitor` / `download` until it is |
| `EMPTY_CHUNK` | a chunk downloaded correctly but holds only nodata (e.g. water, outside the swath) | usually accept |

To accept an issue, copy its `issue_id` into **your** config and re-run `stack` (see Step 3 for why this one
setting comes from your config):

```yaml
qa:
  acknowledged_issues:
    - "LOW_VALID:RO123_ASC:20260623"
```

Nothing is dropped automatically: acknowledged dates stay in the stack and are flagged in `dates.csv`.

---

## Step 8 — Pixel explorer  *(notebook 04)*

```bash
python -m sar_pipeline --config $CFG pixel --track RO123_ASC --lon <lon> --lat <lat> --plot logs/pixel.png
python -m sar_pipeline --config $CFG pixel --track RO123_ASC --pid 1234567
```

Prints date, VV, VH, VH − VV, rain and platforms; the plot marks constellation events, long gaps and rainy dates.
Plots may show locations: keep them in gitignored folders such as `logs/`.

---

## Later: new acquisitions during the season

1. `audit --force` (or `audit` on a new UTC day, which creates a new audit folder).
2. Review; keep or update `s1.tracks`.
3. `new-run`, then Steps 4/5–7 again (the pilot can be skipped if nothing else changed).

Old runs stay untouched, so you can compare runs.
