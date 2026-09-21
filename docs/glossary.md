# Glossary

| Term | Meaning |
|---|---|
| **Acquisition** | One satellite pass over the AOI. It may consist of several Earth Engine images (*slices*) of the same track taken within minutes. Id: `<track_id>_<YYYYMMDD>`. |
| **Adopted task** | An Earth Engine task (or its output files) found after a crash and taken over by the manifest instead of being exported again. |
| **AOI** | Area of interest: the polygon(s) to process. |
| **Ascending / descending** | Satellite moving south→north (evening pass, ≈ 18:00 local solar time, looks east) / north→south (morning pass, ≈ 06:00, looks west). |
| **Atomic write** | Writing a file under a temporary name and renaming it when complete, so a crash never leaves a half file under the real name. |
| **Backscatter (σ⁰, γ⁰)** | Radar power returned to the satellite per unit area. σ⁰ uses ellipsoid ground area; γ⁰ uses the area perpendicular to the beam; terrain-flattened γ⁰ also corrects for DEM slopes. |
| **Band layout** | Which band of a chunk file holds which date and polarisation (`band_layout.csv`). |
| **Border noise** | Low-value artefacts at the far edges of a Sentinel-1 swath. Current GRD data already has it removed; the extra mask is optional. |
| **C-band** | Microwave band around 5.4 GHz (wavelength ≈ 5.55 cm) used by Sentinel-1. |
| **cgroup** | Linux mechanism that limits CPU and memory of a container. The pipeline reads these limits (including parent cgroups) so it does not overuse a JupyterHub container. |
| **Chunk** | A pixel-exact rectangular piece of the master grid, exported and stored separately. |
| **Coherence** | Similarity of the radar phase between two dates (needs SLC data). High = unchanged surface. Not used in this phase. |
| **Confirm (export)** | Moving a planned scope from `PLANNED` to `PENDING` (`export ... --yes`, or `export.confirm_exports`). Only confirmed rows are ever submitted. |
| **CRS** | Coordinate reference system, e.g. a UTM zone such as `EPSG:326zz` (zone zz, northern hemisphere). |
| **dB (decibel)** | `10·log10(linear)`. Used to display backscatter. Never average or filter in dB. |
| **DEM** | Digital elevation model (here Copernicus GLO-30). |
| **Double bounce** | Wave reflected twice (e.g. wall → ground → sensor): very bright. Buildings; flooded vegetation. |
| **ENL** | Equivalent number of looks: how much speckle averaging an image already has. Higher = less noisy. |
| **Export task** | An Earth Engine batch job that computes an image and writes a GeoTIFF to Cloud Storage. |
| **Flooding minimum** | The pronounced dip in VH backscatter when a paddy field is flooded before transplanting: the water surface reflects the beam away from the satellite, so almost nothing returns. The most reliable marker of paddy rice, and in practice the transplanting date. See [01 SAR basics §5](01_sar_basics.md#5-how-crops-look-over-a-season). |
| **GCP / ground truth** | Polygons with known land cover and a confirmed crop label; a model learns from these. Example classes: rice, maize, sugarcane, other vegetation, others. **This project has none** — see [README](../README.md#working-without-ground-truth). |
| **GEE** | Google Earth Engine. |
| **GRD** | Ground Range Detected: Sentinel-1 product with intensity only (no phase), in ground geometry. |
| **Grid (master grid)** | The single pixel grid (CRS, 10 m, snapped origin) every output is aligned to. Immutable. |
| **Incidence angle (θ)** | Angle between the radar beam and the vertical at the ground (≈29°–46° across the IW swath). |
| **IW** | Interferometric Wide swath mode: Sentinel-1's standard land mode (~250 km swath). |
| **Journal (manifest)** | `export_manifest.journal.jsonl`: append-only log of manifest changes since the last CSV snapshot. |
| **Layover / shadow** | Geometric distortions on steep slopes: slope folds over itself (α_r ≥ θ) / is hidden from the beam (α_r ≤ −(90° − θ)). Masked. |
| **Linear power** | Backscatter in its natural (non-log) units. Filtering happens here. |
| **Lock (file lock)** | A small file that lets only one process change the manifest at a time, and only one `monitor`/`download` run per run. |
| **Manifest** | `export_manifest.csv` + journal: one row per export task with its state and history. |
| **Mosaic** | Combining several images (slices) into one. |
| **Nodata** | Special value marking "no valid measurement" (e.g. −9999 in float32 outputs). |
| **NOT_COVERED** | Manifest state for a chunk that a selected track never covers; it is never exported. |
| **Phenology** | The timing of crop growth stages (sowing, emergence, peak, senescence, harvest). |
| **Prep steps** | `python -m sar_pipeline.prep aoi-qc` and `s1-availability`: read-only checks run *before* the grid. They write nothing and start nothing. See [04 Runbook step 0](04_runbook.md#step-0--prep-checks-and-pre-flight-checklist). |
| **Pilot chunk** | One chunk exported and inspected before exporting everything. |
| **Pixel id (pid)** | `row × width + col` on the master grid: a stable address of a 10 m pixel. |
| **PLANNED** | Manifest state written by a dry run; never submitted until confirmed. |
| **Polarisation (VV, VH)** | Orientation of sent and received waves. VV: surface/moisture sensitive. VH: volume/biomass sensitive. |
| **Range slope (α_r)** | The part of the terrain slope along the radar's viewing direction; positive for slopes facing the sensor. |
| **Relative orbit / track** | One of 175 repeating ground paths. Same track = same geometry. Id: `RO123_ASC`. |
| **Run** | One immutable processing attempt: `runs/v<NNN>_<YYYYMMDD>/`. |
| **Run uid** | Globally unique run id (`run_uid.txt`, e.g. `v002_20260915_a1b2c3`) used in Cloud Storage prefixes and task descriptions. |
| **Slice** | One Earth Engine Sentinel-1 image, a ~170 km piece of a pass. |
| **SLC** | Single Look Complex: Sentinel-1 product with phase; enables coherence. Not in Earth Engine. |
| **Specular reflection** | Reflection off a smooth surface (open or standing water) that sends the beam away from the satellite in one direction, so the returned signal is very low. The physical cause of the flooding minimum. Opposite of volume scattering. |
| **Speckle** | Grainy multiplicative noise inherent to coherent radar imaging. |
| **Speckle filter** | Algorithm that reduces speckle (Boxcar, Lee, Refined Lee, Gamma MAP, Lee Sigma, multi-temporal). |
| **Sun-synchronous orbit** | Orbit that passes over each place at the same local solar time. |
| **Terrain flattening** | Radiometric correction removing brightness caused by slopes (σ⁰ → flattened γ⁰). |
| **UTM** | Universal Transverse Mercator: 6°-wide map projection zones. |
| **VERIFIED_EMPTY** | Manifest state for a downloaded file that is correctly aligned but contains only nodata. |
| **Volume scattering** | Wave bouncing many times inside a canopy; the main source of VH. |
| **VRT** | Virtual Raster: a small XML file that presents many files (or bands) as one raster without copying data. |
