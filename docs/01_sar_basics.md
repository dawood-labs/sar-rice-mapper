# 01 — SAR basics for crop mapping

This guide explains, from zero, the radar concepts behind every step of the pipeline. You do not
need prior radar knowledge. Read it once end-to-end; later, use the section headings as a reference.

**Contents**

1. [Why radar?](#1-why-radar)
2. [Backscatter: what the satellite measures](#2-backscatter-what-the-satellite-measures)
3. [Linear power vs decibels (dB)](#3-linear-power-vs-decibels-db)
4. [Polarisation: VV and VH](#4-polarisation-vv-and-vh)
5. [How crops look over a season](#5-how-crops-look-over-a-season)
6. [Orbits: ascending and descending](#6-orbits-ascending-and-descending)
7. [Tracks (relative orbits) and incidence angle](#7-tracks-relative-orbits-and-incidence-angle)
8. [GRD vs SLC](#8-grd-vs-slc)
9. [Speckle and speckle filtering](#9-speckle-and-speckle-filtering)
10. [Terrain correction: two different things](#10-terrain-correction-two-different-things)
11. [Temporal resolution and the 2026 constellation](#11-temporal-resolution-and-the-2026-constellation)
12. [Weather effects: rain, dew, wind](#12-weather-effects-rain-dew-wind)
13. [References](#13-references)

---

## 1. Why radar?

Optical satellites (Sentinel-2, Landsat) record reflected **sunlight**. Clouds block sunlight, so
under persistent cloud cover (common in humid regions during rainy seasons), optical images
may be unusable for weeks or months.

**Sentinel-1** carries a **Synthetic Aperture Radar (SAR)**. It is an *active* sensor: it sends its own
microwave pulses and records the echo.

- Wavelength **≈ 5.55 cm (C-band)**. Clouds and light rain are almost transparent at this wavelength.
- It works **day and night**, since it does not need the sun.
- The price: radar images are harder to interpret, noisy (speckle), and sensitive to geometry and moisture.

---

## 2. Backscatter: what the satellite measures

The radar measures how much of its pulse comes **back** to the antenna. This is **backscatter**, written
**σ⁰ (sigma-nought)**: the returned power per unit ground area.

Three things control backscatter:

| Factor | What it means | Crop example |
|---|---|---|
| **Structure / roughness** | Size, shape and orientation of the things the wave hits, compared to the 5.5 cm wavelength | Tall crop stalks, leaves, a ploughed field vs a smooth one, a flooded paddy vs a dry one |
| **Moisture (dielectric constant)** | Water strongly reflects microwaves. Wetter soil and fresh leaves → more backscatter | Rain on bare soil makes VV jump overnight |
| **Geometry** | Angle between the radar beam and the surface | A slope facing the satellite is brighter |

Two special surfaces to remember:

- **Calm water** acts like a mirror. The pulse bounces away from the satellite, so backscatter is very
  **low** and the image is dark. Flooded rice fields look like this.
- **Buildings** create **double-bounce** (wall → ground → back to the satellite), so backscatter is very
  **high** and the image is bright.

Everything in the pipeline tries to remove the *geometry* effects and the *noise*, so the remaining
signal reflects the crop (structure + water content).

---

## 3. Linear power vs decibels (dB)

Backscatter spans several orders of magnitude, so it is usually shown in **decibels**:

```
dB = 10 · log10(linear)          linear = 10^(dB / 10)
```

| Linear σ⁰ | dB |
|---|---|
| 1.0 | 0 dB |
| 0.1 | −10 dB |
| 0.01 | −20 dB |
| 0.001 | −30 dB |

**Rule: filter and average in linear, display and analyse in dB.**
The mean of dB values is a *geometric* mean of the linear values. It is biased low and distorts
speckle statistics. That is why the pipeline reads `COPERNICUS/S1_GRD_FLOAT` (linear) rather than
`COPERNICUS/S1_GRD` (already in dB), does all filtering in linear power, and converts to dB only at the end.

---

## 4. Polarisation: VV and VH

Radar waves have an orientation (**polarisation**). Sentinel-1 over land usually transmits
**vertically** (V) and receives both **vertically** and **horizontally** (H):

- **VV**: sent vertical, received vertical. Dominated by **surface scattering** (soil) and sensitive to
  **soil moisture** and row structure. Vertical plant stalks interact strongly with vertically polarised
  waves.
- **VH (cross-pol)**: sent vertical, received horizontal. The polarisation only rotates when the wave
  bounces around many times inside a volume of leaves and stems (**volume scattering**). So **VH grows
  with biomass** and is less affected by soil moisture.
- **VH − VV (in dB)**, i.e. the VH/VV ratio: rises as the canopy develops and partly cancels moisture
  effects that change both bands together. A useful derived feature.

---

## 5. How crops look over a season

> **All dB values below are approximate** C-band ranges from the literature and field experience.
> Treat them as intuition and as a way to spot obviously wrong outputs — not as thresholds. The real
> values for your own AOI have to be read off its own time series with the pixel explorer
> ([04 Runbook step 8](04_runbook.md#step-8--pixel-explorer--notebook-04)), because this project has
> **no ground-truth polygons** to calibrate them against
> ([README](../README.md#working-without-ground-truth)).

### Rice (≈120–150 days) — the target crop

Rice is the easiest major crop to recognise in radar, because of something no other crop does: the
field is **deliberately flooded** before planting.

| Stage | What happens physically | VH (approx.) | VV (approx.) |
|---|---|---|---|
| Dry field / land prep | Bare soil, tillage roughness | low, ≈ −20 to −16 dB | ≈ −12 to −7 dB, spikes after rain |
| **Flooding / transplanting** | Shallow water acts as a mirror: the beam reflects *away* from the satellite (specular reflection) | **minimum, often below −20 dB** | **also very low** |
| Early vegetative | Shoots emerge through the water; stem + water surface create **double bounce** | rises quickly | rises quickly, sometimes faster than VH |
| Late vegetative → heading | Dense canopy, volume scattering dominates | high, ≈ −16 to −12 dB | flattens |
| Ripening | Leaves dry, water content falls | slowly declines | variable |
| **Harvest** | Canopy removed, stubble or re-flooded soil | **sharp drop** | drops |

**The two anchors.** The **flooding minimum** and the **harvest drop** are what make rice separable.
The minimum is also, in practice, the transplanting date — published rice calendars key on exactly
this feature.

**Why VH.** Cross-polarised VH responds to the flooding phase more strongly than VV, so the minimum
is deeper and easier to detect. Use VH as the primary band and keep VV for context and for the
VH − VV ratio.

#### Finding rice without ground-truth labels

Over one full season, three statistics of the VH time series separate rice from everything else:

| Feature | Rice | Open water | Other vegetation / built-up |
|---|---|---|---|
| **minimum** VH | **low** (the flood) | low | higher |
| **maximum** VH | **high** (dense canopy) | low | moderate |
| **variance** of VH | **high** (big seasonal swing) | low | low |

Rice is the class that **swings from looking like water to looking like a dense canopy and back**.
Water stays low all year; forest and settlements stay high and flat; a short-season dryland crop
swings less and never reaches the water-like minimum.

> **This is why the season window matters more for rice than for other crops.** The window must
> contain the flooding minimum, the harvest drop, **and** some dry baseline before flooding to
> measure the minimum against. Cut the window short of harvest and both the maximum and the variance
> collapse, and rice stops being separable at all. Pick the window from a crop calendar *before*
> exporting anything.

#### Double cropping

Where two rice crops are grown a year, the whole pattern repeats: **two** flooding minima and **two**
harvest drops within twelve months. A window sized for one crop will slice a double-cropped field in
half and make it look like something else, so decide up front which crop you are mapping and whether
any AOI carries a second cycle.

### Maize (≈100–120 days) — a contrast class

| Stage | What happens physically | VH (approx.) | VV (approx.) |
|---|---|---|---|
| Land prep / sowing | Bare soil, roughness from tillage | low, ≈ −22 to −18 dB | depends on soil moisture, ≈ −14 to −8 dB, spikes after rain |
| Emergence → early vegetative | Small plants, soil still dominates | starts rising | still soil-driven |
| Late vegetative → tasseling (≈60–75 days) | Dense canopy, tall vertical stalks | peak, ≈ −14 to −11 dB | flat or **dips**: vertical stalks attenuate the vertical wave |
| Grain filling → senescence | Leaves dry, water content falls | declines | variable |
| Harvest | Canopy removed, soil or residue | drops back towards soil level | soil/residue level |

Key maize features: a clear VH rise-and-fall over ~4 months, VH − VV rising during vegetative
growth, a sharp drop at harvest. **How it differs from rice:** no flooding minimum. Maize starts from
ordinary dry soil, so its VH minimum is several dB higher than a flooded paddy's, and its seasonal
variance is smaller.

### Sugarcane (≈10–12 months)

- Very dense, tall canopy for **many months**, so **VH stays high** (≈ −14 to −11 dB) for a long plateau.
- The length of the high period separates it from maize (~4 months) and from rice
  (~4–5 months with a flooding minimum at the start).

### Other vegetation (grass, shrubs, trees)

- **Stable**: no clear planting/harvest cycle. Trees keep VH fairly high all year.

### Others (built-up, water, bare land)

- Built-up: very high, stable VV (double bounce).
- Permanent water: very low all year.
- Bare land: low VH, VV driven by rain events.

---

## 6. Orbits: ascending and descending

Sentinel-1 flies a **sun-synchronous, near-polar orbit** at ~693 km, crossing the equator at nearly the
same local solar times every day (a "dawn–dusk" orbit).

| Pass | Direction | Local solar time (anywhere) | Radar looks towards |
|---|---|---|---|
| **Ascending** | South → North | ≈ 18:00 (evening) | **East** |
| **Descending** | North → South | ≈ 06:00 (morning) | **West** |

Sentinel-1 is **right-looking**: it looks to the right of its flight direction. That is why the two
passes see slopes from opposite sides.

Why you should **not blindly mix** ascending and descending images in one time series:

1. **Slopes**: a hillside facing east is bright in ascending images and darker in descending ones.
2. **Time of day**: morning **dew** (descending) raises VV; in climates with afternoon **convective storms**,
   rain shortly before the evening (ascending) pass wets canopy and soil.
3. **Different incidence angles** over the same field (next section).

---

## 7. Tracks (relative orbits) and incidence angle

A single satellite repeats **exactly the same ground path every 12 days**. Within that cycle it flies
**175 distinct paths**, numbered **relative orbit 1–175**. We call such a path a **track**; in this
repository, track id `RO123_ASC` = relative orbit 123, ascending.

- **Same track ⇒ same viewing geometry**, whichever satellite (A, C or D) flew it. Images from the same
  track are directly comparable.
- The Interferometric Wide (IW) swath is ~250 km wide. Across it, the **incidence angle** (angle
  between the radar beam and the vertical) goes from **~29° (near range) to ~46° (far range)**.
- **Backscatter decreases as the incidence angle increases.** The same field can differ by several dB
  between a track that sees it at 32° and one that sees it at 44°. That is as large as a big part of
  the seasonal crop signal.
- Simple normalisations (e.g. dividing by cos θ) assume every surface reacts the same way to the angle.
  In reality bare soil loses backscatter with angle faster than a dense canopy does. So normalising
  and merging tracks always leaves some geometry noise behind.

**Strategy used here:** the audit finds the most consistent track (most acquisitions, smallest gaps)
as the **primary** series. Other tracks are processed as **separate series** and never merged, and
whether they add information is tested later against ground truth.

| Strategy | Pro | Con |
|---|---|---|
| Single best track | Cleanest physics | Fewer dates, gaps cannot be filled |
| **Each track as its own series** (used) | Clean geometry, more dates overall | More features to handle |
| Normalise + merge | One dense series | Normalisation errors hide inside the signal |
| Regular time bins across tracks | Tidy regular grid | Blurs timing, mixes geometry inside a bin |

---

## 8. GRD vs SLC

| | **GRD** (Ground Range Detected) | **SLC** (Single Look Complex) |
|---|---|---|
| Contains | Amplitude/intensity only; **phase discarded** | Amplitude **and phase** |
| Geometry | Projected to ground range, multi-looked | Slant range, bursts |
| Pixel spacing / resolution (IW) | **10 m pixels, but ≈ 20 × 22 m true resolution** | ~2.3 × 14 m |
| Enables | Backscatter time series | + **interferometric coherence** (how much the surface changed between two dates), great for tillage/harvest detection |
| In Earth Engine | **Yes** (`COPERNICUS/S1_GRD`, `_FLOAT`) | No; needs SNAP / ASF processing |

This pipeline uses **GRD**. Because true resolution is ~20 m, pixels at the edges of small fields
(often < 1 ha) mix neighbouring fields. Averaging all pixels inside a ground-truth polygon later is
the most effective way to reduce both speckle and edge mixing.

---

## 9. Speckle and speckle filtering

**What speckle is:** radar is *coherent*. Inside one resolution cell there are many small scatterers
(leaves, clods). Their echoes add up with random phases, sometimes reinforcing and sometimes
cancelling. The result is a grainy, salt-and-pepper texture called **speckle**. It is not sensor
noise you can "calibrate away"; it is part of how coherent imaging works.

- Speckle is **multiplicative**: brighter areas have proportionally larger fluctuations.
- A Sentinel-1 IW GRD image has an **equivalent number of looks (ENL) of about 4–5**, so single-pixel
  values are very noisy.

**Filters** (all applied in **linear** power):

| Filter | Idea | Trade-off |
|---|---|---|
| Boxcar | Plain moving average | Strong smoothing, blurs field edges |
| Lee | Adapts to local mean/variance: smooths flat areas, keeps high-variance areas | Edges partly kept |
| Refined Lee | Lee + chooses an edge-aligned sub-window | Better edges; uses fixed internal windows |
| Gamma MAP | Statistical (Bayesian) estimate assuming gamma-distributed speckle | Good for vegetation, more compute |
| Lee Sigma | Uses only pixels within a statistical range of the centre | Preserves point targets |
| **Multi-temporal (Quegan & Yu 2001)** | Uses **other dates of the same track** to reduce noise | Little spatial blur; needs a time series |

**Window size vs field size:** a 7 × 7 window at 10 m spans 70 m and blends a 1-ha field with its
neighbours. The pipeline therefore uses a **multi-temporal filter** (±3 neighbouring acquisitions of
the same track) with a **small spatial filter** inside.

**Edge dates are noisier.** At the start and end of a series fewer neighbours exist (e.g. 4 instead of 7),
so the first and last dates (including the newest one) keep more speckle. `dates.csv` reports this per date
in `n_temporal_neighbors`; take it into account when a feature depends on the latest acquisition.

Why a **fixed ±N window** instead of all dates: when new acquisitions arrive later in the season, only
the last few dates' filtered values change. The earlier outputs stay stable.

Multi-temporal formula, for date *k* and neighbour dates *i*:

```
J_k = F(I_k) · mean_i( I_i / F(I_i) )
```

`I` = linear intensity, `F` = the spatial filter. Each neighbour contributes its "texture relative to its
own local mean", which averages speckle out while keeping the spatial detail of date *k*.

---

## 10. Terrain correction: two different things

People often say "terrain correction" for two separate operations.

### a) Geometric correction (orthorectification)

Radar measures **distance** to the satellite, not map position. Hills are displaced towards the sensor.
Geometric correction uses a DEM to move each pixel to its true map location.

**Already done by Earth Engine** for every Sentinel-1 GRD image, together with: orbit-file
application, GRD border-noise removal, thermal-noise removal, radiometric calibration (σ⁰), and
Range-Doppler terrain correction with SRTM 30 m (ASTER GDEM above 60° latitude).

### b) Radiometric terrain flattening (σ⁰ → γ⁰)

A slope **facing** the satellite puts more ground area into each radar pixel, so it looks **brighter**
even with no crop difference. A slope facing **away** looks darker. Flattening divides out this
area effect:

- **β⁰**: brightness in radar geometry; **σ⁰**: normalised by ellipsoid ground area; **γ⁰**: normalised by the
  area seen perpendicular to the beam. **Terrain-flattened γ⁰** uses the real DEM slope.
- **Not done by Earth Engine.** This pipeline uses the angular-based method of **Vollrath et al. (2020)**
  with the Copernicus GLO-30 DEM, and the **volume model**, which is appropriate for vegetated land.
- Pixels in **layover** (slope steeper than the beam, so the top of the hill arrives before the bottom)
  and **radar shadow** (slope hidden from the beam) contain no usable information and are **masked**.

**How the code defines the geometry** (important when you read or change `s1_ard/terrain_flattening.py`):

- θ is the incidence angle on flat ground, taken from the image's own `angle` band. That band is a coarse
  tie-point grid, so it is always read with **bilinear** resampling (nearest-neighbour reading would create
  sub-degree steps across the image). The radar look direction is derived from the same band.
- α_r is the **range slope**: the part of the terrain slope in the radar's viewing direction. It is
  **positive for slopes facing the sensor** and negative for slopes facing away.
- **Volume model:** γ⁰_flat = γ⁰ · tan(θ − α_r) / tan θ. On flat ground the factor is 1; facing slopes are
  darkened, away-facing slopes brightened.
- **Layover** is masked where α_r ≥ θ, **shadow** where α_r ≤ −(90° − θ). These are exactly the angles where the
  factor above reaches zero or infinity.
- A DIRECT (surface) model, γ⁰ · cos(α_az) · sin(θ − α_r) / sin θ, is available but less validated.

**Validation on real data.** The original open-source implementation had the slope sign reversed, so true
layover was not masked and away-facing slopes stayed several dB too bright. After the correction, on a hilly
test area (mean slope ≈ 13°, one ascending and one descending acquisition):

| Slope steepness | Facing minus away-facing, raw σ⁰ | After flattening |
|---|---|---|
| 3–10° | ≈ +2 dB | ≈ 0 dB |
| 10–20° | ≈ +5 dB | ≈ +0.3 dB |
| 20–30° | ≈ +9 dB | ≈ −1.2 dB |

≥ 99 % of true layover pixels were masked, and on flat terrain the correction changes values by only ~0.05 dB.
Very steep slopes **facing** the sensor (30–45°), right next to the layover limit, stay too dark because the
factor tends to zero there: interpret such pixels with care.

Even in a mostly flat area, rolling terrain benefits. The audit reports the AOI's slope distribution
(`slope_stats.json`) so you can see how much flattening matters. Flattening also makes ascending and
descending images more comparable on slopes.

---

## 11. Temporal resolution and the 2026 constellation

- One satellite: **12-day** repeat per track. Two satellites 180° apart in the same orbit: **6-day** repeat.
- Rice lasts ~120–150 days, and the key transitions (flooding, peak canopy, harvest) take only
  days to a couple of weeks - the flooding dip can be missed entirely by a coarse revisit.
  12 days ≈ 9–10 observations per season; 6 days ≈ 18–20.

**Constellation changes in 2026** (relevant for any 2026 time series):

| Date | Event | Consequence |
|---|---|---|
| 2026-04-17 | **Sentinel-1D** user data opens (initially flying Sentinel-1C's ground track one day later) | Near-duplicate dates one day apart: useful for denoising, adds little timing information |
| late June 2026 | **Sentinel-1C** phasing manoeuvre, ~2 weeks without S1C acquisitions | Possible **gap** in the time series |
| 2026-06-29 | **Sentinel-1A** end of operations | Sensor change mid-season, so a small **calibration step** is possible |
| afterwards | Sentinel-1C + 1D, 6-day repeat | Stable dense series |

The audit reports gaps and platform changes per track. During QA, the mean backscatter of **stable
targets** (buildings, dense trees) is plotted over time to look for a step at the end of June.

---

## 12. Weather effects: rain, dew, wind

- **Rain** wets soil and leaves, so backscatter rises (strongest in VV, and early in the season when the
  canopy is thin). A one-date spike right after rain is *weather*, not crop growth.
- **Dew** on morning (descending) passes slightly raises VV.
- **Wind** roughens flooded rice paddies, which turns their dark "mirror" signal brighter.

The pipeline stores **rain in the 6 h and 24 h before each acquisition** (GPM IMERG, or GSMaP as a
fallback) next to every date, so spikes can be explained instead of misread. Rain images that only partly
overlap the window are weighted by their overlap, and `rain_gap_images` counts dataset images missing inside
the window (the sum then under-counts).

---

## 13. References

- ESA / Copernicus, *Sentinel-1 SAR User Guide* and technical guides: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar
- Copernicus Data Space Ecosystem, *Sentinel-1D user data opening and future plans* (2026): https://dataspace.copernicus.eu/news/2026-4-2-sentinel-1d-user-data-opening-and-future-plans
- Copernicus Data Space Ecosystem, *Sentinel-1A end of operations* (2026): https://dataspace.copernicus.eu/news/2026-6-30-copernicus-sentinel-1a-satellite-end-operations-after-12-years-service
- Google Earth Engine catalog, *Sentinel-1 SAR GRD*: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
- Google Earth Engine guide, *Sentinel-1 algorithms*: https://developers.google.com/earth-engine/guides/sentinel1
- Vollrath, A., Mullissa, A., Reiche, J. (2020). Angular-based radiometric slope correction for Sentinel-1 on Google Earth Engine. *Remote Sensing* 12(11), 1867. https://doi.org/10.3390/rs12111867
- Mullissa, A. et al. (2021). Sentinel-1 SAR backscatter analysis ready data preparation in Google Earth Engine. *Remote Sensing* 13(10), 1954. https://doi.org/10.3390/rs13101954
- Quegan, S., Yu, J. J. (2001). Filtering of multichannel SAR images. *IEEE TGRS* 39(11), 2373–2379.
- Lee, J.-S. (1980). Digital image enhancement and noise filtering by use of local statistics. *IEEE PAMI* 2(2), 165–168.
- Lee, J.-S., Grunes, M. R., De Grandi, G. (1999). Polarimetric SAR speckle filtering and its implication for classification. *IEEE TGRS* 37(5), 2363–2373.
- Lee, J.-S. et al. (2009). Improved sigma filter for speckle filtering of SAR imagery. *IEEE TGRS* 47(1), 202–213.
- Lopes, A., Touzi, R., Nezry, E. (1990). Adaptive speckle filters and scene heterogeneity. *IEEE TGRS* 28(6), 992–1000.
- Small, D. (2011). Flattening gamma: radiometric terrain correction for SAR imagery. *IEEE TGRS* 49(8), 3081–3093.
