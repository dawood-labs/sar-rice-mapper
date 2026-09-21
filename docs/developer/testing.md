# Testing guide

## 1. Two kinds of tests

| Kind | Marker | Network | Runs by default | Command |
|---|---|---|---|---|
| **Unit tests** | none | **none** | yes | `pytest` |
| **Live Earth Engine tests** | `@pytest.mark.gee` | Earth Engine (read-only) | no | `pytest -m gee` |

`pyproject.toml` sets `addopts = "-m 'not gee'"`, so plain `pytest` never touches the network.

```bash
pytest                              # all unit tests (includes the public-repo hygiene test)
pytest tests/test_grid.py -q        # one file
pytest -k split -q                  # tests whose name contains "split"
SAR_PIPELINE_CONFIG=config/my_aoi_season.yaml pytest -m gee     # live tests with a real config
```

Live tests read the config named by the `SAR_PIPELINE_CONFIG` environment variable (default
`config/live.yaml`) and skip when that config or its key file is missing.

## 2. Rules

1. **Unit tests: no network.** Earth Engine and Cloud Storage are replaced by fakes:
   - `export.EEBackend`: a fake object with the same methods (`start_export`, `task_statuses`,
     `project_snapshot` and the project queue counters) that records calls and returns scripted states;
     the paged task listing is tested through `export.scan_operations` with a fake page fetcher;
   - `download.GCSBackend`: a fake with `list_blobs` / `download` writing local test files;
   - CLI tests put fake modules into `sys.modules` (the CLI imports stage modules lazily).
2. **Live tests are read-only.** Allowed: `getInfo()` on small results, `ee.data.computePixels` on tiny
   regions (e.g. 64 × 64 px), paged `computeFeatures`. **Never** start export tasks, never write to Cloud Storage.
3. **No private data in tests.** Use the synthetic AOI from `tests/conftest.py` (an arbitrary polygon unrelated
   to any project area). Live tests derive locations at runtime from the local (gitignored) config; never
   hard-code AOI coordinates, bucket or project names, or place names.
4. **Temporary files only** (`tmp_path`). Tests never write into `processed/`.
5. **Test the "why", not only the "what".** E.g. grid tests assert that chunk origins lie on the master grid;
   resume tests crash a stage halfway and check the re-run skips finished work and does not duplicate tasks.
   Where it matters, crashes are real: separate OS processes writing the same manifest, and a subprocess killed
   with `os._exit` while writing the pixel index.

## 3. Fixtures and helpers

| Fixture / helper | Gives you |
|---|---|
| `make_project(overrides=None, aoi_lonlat=None)` (`tests/conftest.py`) | a synthetic project in a temp folder: `config/test.yaml`, a synthetic AOI GeoPackage, and the loaded `cfg` (paths resolve inside the temp folder). `overrides` updates config sections, e.g. `{"grid": {"chunk_px": 128}}`. |
| `live_cfg` (`tests/conftest.py`) | the real config (env `SAR_PIPELINE_CONFIG`, default `config/live.yaml`) with Earth Engine initialised; skips if the config or key is missing. |
| `tests/_stack_fixtures.py` | builder of a complete synthetic run (grid, pixel index, manifest, band layout, audit files, chunk GeoTIFFs) shared by the stack and pixel-query tests. |

## 4. Test files

| File | Covers |
|---|---|
| `tests/test_prep_aoi_qc.py` | duplicate counting, vertex-order-independent shape comparison, cross-check verdicts (lost / added / orphaned / renumbered), acres and the refusal of a geographic CRS, area-column unit detection |
| `tests/test_prep_s1_availability.py` | report building from canned metadata (injected `fetch`, so no network): track grouping by orbit **and** pass, `RO…_ASC` keys, distinct dates per month, thin-month flagging, empty window |
| `tests/test_grid.py`, `tests/test_index.py` | snapping, chunk tiling, `aoi_frac`, geometry-only immutability, bounded lookup cache, pid conversions, sparse index raster, resume after a hard kill |
| `tests/test_audit.py` | slice grouping, gaps, recommendations, coverage maths, request-size bounding, rain window weighting, resume |
| `tests/test_audit_geometry.py` | `audit._ee_geometry`: Z stripped from 3D footprints (with a stubbed `ee.Geometry`), coordinates and `proj`/`geodesic` preserved, clear error for an empty or missing geometry |
| `tests/test_locking.py` | cross-process file locks: exclusion, re-entrance, stale-holder breaking, `run_lock` |
| `tests/test_s1_ard.py` | band naming (incl. same-date suffixes), temporal neighbours, parameter validation, terrain geometry maths mirrors |
| `tests/test_manifest.py` | journal + snapshot storage, compaction, cross-process updates, torn journal lines |
| `tests/test_export.py` | plan/confirm scope, state machine, retry/split policy, queue headroom, adoption after crash, lost tasks, verify re-export, run uid |
| `tests/test_download.py` | single listing per track, verification (grid, dtype, nodata), `VERIFIED_EMPTY`, index resume, locked files, run lock |
| `tests/test_stack.py`, `tests/test_pixel_query.py` | VRT XML from synthetic chunk GeoTIFFs, planned-scope valid %, QA issues, caches, no `osgeo` import, pixel reads |
| `tests/test_cli.py` | argument parsing, `--yes` gating, `--force`/`--retry-failed`/`--deep` pass-through, frozen run config |
| `tests/test_resources.py` | cgroup v1/v2 limits with nesting, page cache, fallbacks, config caps (fake `/proc` and `/sys` trees) |
| `tests/test_analysis.py`, `tests/test_analysis_model.py`, `tests/test_analysis_final_models.py`, `tests/test_analysis_field_labels.py` | ground-truth QC and class codes, feature sets and binning, spatial CV grouping, holdout isolation, tuning/threshold selection, model variants and the field-level model, delineation repair/despike/label/cut/derive/tidy. Class names such as `Maize`/`Rice`/`Sugarcane` in these tests are **arbitrary multi-class labels**, not a statement about the project's crops |
| `tests/test_repo_hygiene.py` | public-repo safety: `.gitignore` rules, no private values in trackable files, no secret files tracked, notebooks without outputs |
| `tests/gee/…` | live, read-only checks of audit queries and preprocessing (incl. terrain-flattening validation) on tiny regions |

## 5. The hygiene test and `secrets/private_terms.txt`

`tests/test_repo_hygiene.py` protects the public repository. It builds a list of private values at test time from:

- every non-example config in `config/` (bucket, base folder, Earth Engine project, key file name, AOI file name);
- the service-account key referenced there (email, key id, client id);
- `secrets/private_terms.txt`: one word per line that must never be published (client name, place names of the
  project area, …).

It then scans every file git would track (`git ls-files --cached --others --exclude-standard`). The private
list itself lives only in gitignored files, so the test contains no private information. On a fresh clone
without those files, the private-value check is skipped and the structural checks still run.

To maintain it: add new sensitive words to `secrets/private_terms.txt` (never to a tracked file), run
`pytest tests/test_repo_hygiene.py -q`, and fix every reported file. Notebooks must be saved without outputs
(clear all outputs before committing).

## 6. Before committing

```bash
pytest
git status        # confirm no secrets/, data/, processed/, logs/, config/*.yaml, *.tif, *.png appear
```
