# 07 — Scaling to large AOIs

The pipeline is developed on a small R&D AOI, and the same code is designed to process a
**country-scale AOI**. This page explains what changes with scale, and which settings to adjust.

---

## 1. The numbers

At 10 m, 1 km² = 10,000 pixels. A 512 px chunk covers ≈ 26 km², a 2048 px chunk ≈ 420 km².

| Quantity | Formula | Example: 100,000 km² AOI |
|---|---|---|
| Pixels inside the AOI | area_km² × 10,000 | 10⁹ |
| Grid (bounding box) pixels | bbox area × 10,000 (includes water, gaps) | often several × 10⁹ → `pid_dtype = int64` |
| Chunks at 512 px | ≈ AOI area ÷ 26 km² (+ partial edge chunks) | ~4,000+ |
| Chunks at 2048 px | ≈ AOI area ÷ 420 km² (+ edges) | ~250+ |
| Export tasks | chunks × tracks − not-covered chunks | many tracks cover a large area |

**Storage estimate** for one track (before compression):

```
bytes ≈ AOI pixels × number of acquisitions × 2 (VV, VH) × bytes per value
```

Example: 10⁹ px × 40 dates × 2 × 4 B (float32) ≈ **320 GB**. With `export.dtype: int16`
(2 B, dB × 100, 0.01 dB precision) ≈ **160 GB**. Always check `download` (dry run) before `--yes`.

---

## 2. Settings to change

| Setting | Small AOI | Large AOI | Why |
|---|---|---|---|
| `grid.chunk_px` | 512 | **2048** (suggested; keep a multiple of 512) | fewer tasks, less bookkeeping |
| `export.dtype` | float32 | **int16** | halves storage; precision 0.01 dB is far below speckle noise |
| `rain.level` | aoi | **chunk** | one AOI-mean rainfall is meaningless across a large area |
| `resources.*` | auto | auto | never hard-code; cap only if the server is shared |

**Chunk size trade-off:** bigger chunks mean fewer tasks and less queue waiting, but each task needs
more Earth Engine memory and time. When a task fails with *memory limit* or *timeout*, the pipeline
**automatically splits it into 4 sub-chunks** and retries, so an occasional too-big chunk costs
time but does not stop the run. If many chunks split, choose a smaller `chunk_px` for a **new** grid
(new `aoi.key`/`season.key`): an existing grid never changes. Decide the size from the pilot: note how long one
task runs and how much compute it uses, then extrapolate.

---

## 3. Earth Engine queue limits

- An Earth Engine project accepts at most **~3000 queued tasks** (READY + RUNNING), **shared with every
  other job on the same project** (`export.max_queued_tasks`).
- Before **every** submission round the pipeline counts the project's queued tasks, keeps a safety
  margin (`export.queue_safety_margin`, default 90 % of the limit), and submits only into the remaining
  headroom. The rest waits for the next poll.
- Earth Engine decides how many tasks **run at the same time** (on some tiers roughly a dozen or so).
  The pipeline **measures** this live, logs it each round together with an ETA (`median task duration ×
  pending ÷ running`, shown once a few tasks have completed), and never assumes a number.
- Task states for all active tasks come from **one paged project task listing** per round. The listing stops
  reading as soon as it has passed all active tasks, so a project with a long history (tens of thousands of old
  tasks) does not slow polling down. Individual status lookups are used only for a few tasks missing from the listing.
- Queue-full / rate-limit errors are retried with backoff and do not count as task failures.

Rough time estimate: `tasks × average task minutes ÷ concurrently running tasks`. With 2000 tasks,
10 min each and 13 running: ~26 hours of wall time. `monitor` can be stopped and restarted at any time.

---

## 4. Tracks differ across a large area

Over a small AOI one track covers everything. Over a large area, **different regions are covered by different
tracks**, and a track may cover a chunk only partially.

- The audit writes `chunk_track_coverage.csv`: for every chunk and track, how many acquisitions cover it
  and how completely.
- `track_recommendation.csv` is computed for the whole AOI; for large AOIs **read the per-chunk coverage**
  before deciding. The pipeline does not auto-assign tracks per chunk: that choice stays with you.
- A chunk a selected track **never** covers is planned as `NOT_COVERED` and never exported (no wasted quota).
  A chunk covered only on some dates exports with nodata on the other dates and shows up in QA (`LOW_VALID`),
  never silently.

---

## 5. One CRS for the whole AOI {#single-crs}

The grid uses **one projected CRS** (`grid.crs`) for the whole AOI, typically the UTM zone that covers most of it.

A UTM zone is a 6°-wide strip. Pixels are exactly true-to-scale only near the strip's central meridian and
stretch slowly with distance from it. Using one zone beyond its strip keeps everything simple (one grid, one
pixel-id system, one VRT per date) at the cost of a small scale error. Worst case (at the equator; the error
shrinks towards the poles):

| Distance from the central meridian | Linear scale error (approx.) | Area error (approx.) |
|---|---|---|
| 0° | −0.04 % | −0.08 % |
| 3° (zone edge) | +0.1 % | +0.2 % |
| 5° | +0.3 % | +0.7 % |
| 8° | +0.9 % | +1.9 % |
| 10° | +1.5 % | +3 % |

- For **classification** (is this pixel rice?) a scale difference of ~1 % is irrelevant.
- For **area statistics** (acres from pixel counts — this project never reports hectares or m²), apply a
  per-location scale-factor correction far from the
  central meridian, or compute areas on the polygons in an equal-area CRS.
- If an AOI extends much more than ~8–10° from the chosen central meridian, consider a custom projection (e.g. a
  transverse Mercator or equal-area projection centred on the AOI) for that AOI's grid.

Alternative (not used): one grid per UTM zone. It is exact, but it means several grids, several pixel-id
namespaces and no single VRT, which is more complexity than a ~1 % edge error justifies.

---

## 6. Running on JupyterHub (AWS)

- **Resources are detected at runtime and are container-aware.** Inside a container, `os.cpu_count()` reports the
  host machine; `sar_pipeline.resources` reads the cgroup CPU quota and memory limit of the process's own (possibly
  nested) cgroup and all its parents, and does not count reclaimable page cache as used memory. Check with
  `python -m sar_pipeline --config $CFG resources` on the actual server.
- Download threads scale with CPUs (network-bound). Raster QA workers are limited by both CPUs and the RAM
  budget (`resources.memory_fraction` of *available* memory); each worker's GDAL cache is counted inside that budget.
  Read block sizes follow from the same budget.
- Put the project on a **large attached volume**; the download step refuses to start when the data does not fit.
- Keep `monitor` running in a **terminal** rather than a notebook cell, so closing the browser does not stop it:
  `mkdir -p logs && nohup python -m sar_pipeline --config $CFG monitor --yes > logs/monitor.log 2>&1 &`.
  If it stops anyway, re-run it. A second `monitor` on the same run is refused while the first is alive, and a
  notebook can download finished chunks at the same time safely.
- Continuing a run on a different machine works: only `auth.key_file` and `resources` come from that machine's
  config; `auth.project` must stay the run's project.

---

## 7. What never changes with scale

- Nothing is held in memory for the whole AOI: grid and index are written chunk by chunk, QA reads are windowed,
  stacks are VRTs, and valid-pixel caches are written in parts.
- No Earth Engine request grows with the AOI: geometries are simplified and batched, results are paged, and
  geometry maths runs locally.
- Manifest updates stay fast for tens of thousands of tasks (append-only journal), and downloads list each track
  folder once.
- Crash-safe re-runs, confirmation gates and "never silently skip" rules apply exactly as for a small AOI.
