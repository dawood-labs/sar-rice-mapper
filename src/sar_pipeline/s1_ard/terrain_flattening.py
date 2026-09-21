"""Radiometric terrain flattening (sigma0 -> flattened gamma0) with layover/shadow masking.

Originally adapted from gee_s1_ard/python-api/terrain_flattening.py
(https://github.com/adugnag/gee_s1_ard, MIT License, (c) 2021 Adugna Mullissa; see
LICENSE_gee_s1_ard.txt), which implements the angular-based slope correction of
Vollrath, A., Mullissa, A., & Reiche, J. (2020). Angular-Based Radiometric Slope Correction for
Sentinel-1 on Google Earth Engine. Remote Sensing, 12(11), 1867. https://doi.org/10.3390/rs12111867
The geometry and the model equations below were rewritten after a live review (see "What changed").

Why flatten at all
------------------
A slope facing the radar gathers more ground (and more vegetation) into one radar pixel, so it looks
brighter; a slope facing away looks darker, whatever grows on it. Earth Engine's GRD product is only
*geometrically* terrain corrected (pixels are in the right place). This step removes the
*radiometric* slope effect, so the same crop gives similar backscatter on flat and sloping fields.

The angles, in plain words
--------------------------
- theta (θ): the radar incidence angle on a flat ellipsoid, from the `angle` band (about 30-46° for IW).
- look direction: the map azimuth pointing from the ground TOWARD the satellite. The incidence
  angle grows with distance from the satellite, so the look direction is the direction in which
  `angle` DECREASES fastest, i.e. the opposite of its gradient.
- alpha_r (α_r): the terrain slope measured in the range direction. **Positive = the slope faces the
  sensor**, negative = it faces away. α_r = atan(tan(slope) · cos(aspect − look direction)), where
  aspect is the downslope direction of the DEM (a slope whose downhill side points at the sensor faces it).
- alpha_az (α_az): the slope component along the satellite track.

The models
----------
gamma0 = sigma0 / cos(θ) first removes the flat-terrain incidence dependence. Then:

- VOLUME model (vegetated land, the default), Hoekman & Reiche (2015), Vollrath et al. (2020):
  a canopy is a layer of scatterers; tilting it towards the radar packs a longer canopy path into
  one pixel. The flattened value is

      gamma0_flat = gamma0 · tan(θ − α_r) / tan(θ)

  Flat ground (α_r = 0) is unchanged, facing slopes are darkened (factor < 1), slopes facing away
  are brightened (factor > 1).
- DIRECT (surface) model, Ulander (1996) area projection:

      gamma0_flat = gamma0 · cos(α_az) · sin(θ − α_r) / sin(θ)

  Less validated in this project than VOLUME; use VOLUME for crops.

Both factors reach 0 exactly at the layover boundary (α_r = θ) and grow without bound at the shadow
boundary (α_r = −(90° − θ)). Those pixels carry no usable signal and are masked:

- active layover: α_r ≥ θ       (the slope is steeper than the radar beam towards the sensor)
- active shadow:  α_r ≤ −(90° − θ)  (the slope hides behind itself)

An optional buffer (`ard.layover_shadow_buffer_m`) also removes pixels next to those areas.

Why per slice (before mosaicking)
---------------------------------
The look direction comes from each slice's `angle` band and the DEM is resampled into each slice's
native projection; a mosaic has no meaningful native projection. Slope aspect and the look direction
are both measured in that same projection, so they share one "north".

What changed from the original, and why
---------------------------------------
- Sign convention: the original derived the heading from `ee.Terrain.aspect` of the coarse, sheared
  tie-point `angle` grid and negated the DEM aspect. A live review found its α_r was positive for
  slopes facing AWAY (correlation −0.97 with an independent range-slope estimate), so true layover
  was never masked (0 %) and the volume factor over-corrected: +3 dB left on 20-30° slopes and
  +7-8 dB on steep slopes facing away. Now the look direction comes from the gradient of the
  bilinearly resampled `angle` band in the slice projection, α_r is positive towards the sensor,
  and the model equations are written for that convention.
- θ is read with bilinear resampling: the `angle` band is a ~16 km tie-point grid and nearest
  neighbour gives ~0.8° steps.
- The DEM may be an Image or an ImageCollection. `COPERNICUS/DEM/GLO30_2024_1` is a collection: we
  select its first band, mosaic and `setDefaultProjection(<first tile projection>)`; without that
  the slope would be computed on a default 1-degree projection and be wrong.
- The DEM is not clipped to the slice footprint (clipping leaves one-pixel holes where slices meet).
- Only the polarisation bands are flattened; the mask is applied with `updateMask` (the original's
  `Image.mask(mask)` replaced existing masks and re-exposed masked pixels).
"""
from __future__ import annotations

import math

import ee

MODELS = ("VOLUME", "DIRECT")
D2R = math.pi / 180.0
_GRADIENT_SCALE_M = 1000   # angle varies smoothly over tens of km; a coarse grid gives a stable gradient

_DEM_CACHE: dict[str, ee.Image] = {}


def load_dem(dem_id: str) -> ee.Image:
    """Single-band DEM image with a proper default projection (cached per asset id).

    Makes one metadata request per DEM id to find out whether it is an Image or ImageCollection.
    """
    if dem_id not in _DEM_CACHE:
        asset_type = str(ee.data.getAsset(dem_id).get("type", "")).upper()
        if "COLLECTION" in asset_type:
            col = ee.ImageCollection(dem_id).select([0])
            dem = col.mosaic().setDefaultProjection(col.first().projection())
        else:
            dem = ee.Image(dem_id).select([0])
        _DEM_CACHE[dem_id] = dem.rename("elevation")
    return _DEM_CACHE[dem_id]


# ---------------------------------------------------------------- pure-python mirrors (unit-testable)
def toward_sensor_azimuth_deg(grad_x: float, grad_y: float) -> float:
    """Azimuth (degrees clockwise from north) in which the incidence angle decreases fastest.

    grad_x / grad_y: gradient of the angle band towards east / north. Mirrors `look_direction_deg`.
    """
    return math.degrees(math.atan2(-grad_x, -grad_y)) % 360.0


def range_slope_deg(slope_deg: float, aspect_deg: float, toward_deg: float) -> float:
    """α_r in degrees, positive when the slope faces the sensor. Mirrors `slope_geometry`."""
    return math.degrees(math.atan(math.tan(slope_deg * D2R) * math.cos((aspect_deg - toward_deg) * D2R)))


def volume_factor(theta_deg: float, alpha_r_deg: float) -> float:
    """tan(θ − α_r) / tan(θ): multiply gamma0 by this (VOLUME model)."""
    return math.tan((theta_deg - alpha_r_deg) * D2R) / math.tan(theta_deg * D2R)


def direct_factor(theta_deg: float, alpha_r_deg: float, alpha_az_deg: float) -> float:
    """cos(α_az) · sin(θ − α_r) / sin(θ): multiply gamma0 by this (DIRECT model)."""
    return math.cos(alpha_az_deg * D2R) * math.sin((theta_deg - alpha_r_deg) * D2R) / math.sin(theta_deg * D2R)


def is_valid_geometry(theta_deg: float, alpha_r_deg: float) -> bool:
    """False for active layover (α_r ≥ θ) or active shadow (α_r ≤ −(90° − θ))."""
    return -(90.0 - theta_deg) < alpha_r_deg < theta_deg


# ---------------------------------------------------------------- Earth Engine expressions
def look_direction_deg(image: ee.Image) -> ee.Number:
    """Map azimuth (degrees clockwise from north) from the ground towards the sensor, for one slice.

    Uses the gradient of the bilinearly resampled `angle` band in the slice's own map projection,
    averaged over the slice. The angle increases away from the sensor, so the sensor lies in the
    direction opposite to the gradient.
    """
    proj = image.select(0).projection()
    angle = image.select("angle").resample("bilinear").reproject(proj.atScale(_GRADIENT_SCALE_M))
    grad = angle.gradient().reduceRegion(
        reducer=ee.Reducer.mean(), geometry=image.geometry(), scale=2 * _GRADIENT_SCALE_M, bestEffort=True,
    )
    gx, gy = ee.Number(grad.get("x")), ee.Number(grad.get("y"))
    # azimuth = atan2(east, north) of the vector (-gx, -gy). ee.Number(a).atan2(b) is the angle of the
    # vector [x=a, y=b] (i.e. math.atan2(b, a)), so the north component goes first.
    return gy.multiply(-1).atan2(gx.multiply(-1)).multiply(180 / math.pi).add(360).mod(360)


def slope_geometry(image: ee.Image, dem: ee.Image) -> dict:
    """Radar/terrain angles for one slice (radians): theta_i, alpha_r (+ = facing sensor), alpha_az."""
    proj = image.select(0).projection()
    elevation = dem.resample("bilinear").reproject(proj, None, 10)
    theta_i = image.select("angle").resample("bilinear").multiply(D2R).rename("theta_i")
    toward = look_direction_deg(image)
    slope = ee.Terrain.slope(elevation).multiply(D2R)
    aspect = ee.Terrain.aspect(elevation)  # downslope direction, degrees clockwise from north
    relative = aspect.subtract(ee.Image.constant(toward)).multiply(D2R)
    alpha_r = slope.tan().multiply(relative.cos()).atan().rename("alpha_r")
    alpha_az = slope.tan().multiply(relative.sin()).atan().rename("alpha_az")
    return {"theta_i": theta_i, "alpha_r": alpha_r, "alpha_az": alpha_az, "look_direction_deg": toward}


def flattening_factor(model: str, theta_i: ee.Image, alpha_r: ee.Image, alpha_az: ee.Image) -> ee.Image:
    if model == "VOLUME":
        return theta_i.subtract(alpha_r).tan().divide(theta_i.tan())
    if model == "DIRECT":
        return alpha_az.cos().multiply(theta_i.subtract(alpha_r).sin()).divide(theta_i.sin())
    raise ValueError(f"terrain flattening model must be one of {MODELS}, got {model!r}")


def _erode(mask: ee.Image, distance_m: float) -> ee.Image:
    """Shrink valid areas by `distance_m` (buffer around layover/shadow)."""
    d = mask.Not().unmask(1).fastDistanceTransform(30).sqrt().multiply(ee.Image.pixelArea().sqrt())
    return mask.updateMask(d.gt(distance_m))


def layover_shadow_mask(alpha_r: ee.Image, theta_i: ee.Image, buffer_m: float = 0) -> ee.Image:
    """1 where the geometry is usable, 0 for active layover (α_r ≥ θ) or active shadow (α_r ≤ −(90° − θ))."""
    ninety = ee.Image.constant(math.pi / 2)
    not_layover = alpha_r.lt(theta_i)
    not_shadow = alpha_r.gt(ninety.subtract(theta_i).multiply(-1))
    mask = not_layover.And(not_shadow)
    if buffer_m > 0:
        mask = _erode(mask, buffer_m)
    return mask.rename("no_data_mask")


def flatten_slice(image: ee.Image, pols: list[str], model: str, dem: ee.Image, buffer_m: float = 0) -> ee.Image:
    """Return the slice with `pols` converted to flattened gamma0 (linear), plus the `angle` band.

    `image` must be linear power (S1_GRD_FLOAT) with bands `pols` + `angle`, in its native projection.
    """
    if model not in MODELS:
        raise ValueError(f"terrain flattening model must be one of {MODELS}, got {model!r}")
    geo = slope_geometry(image, dem)
    gamma0 = image.select(pols).divide(geo["theta_i"].cos())
    flat = gamma0.multiply(flattening_factor(model, geo["theta_i"], geo["alpha_r"], geo["alpha_az"]))
    mask = layover_shadow_mask(geo["alpha_r"], geo["theta_i"], buffer_m)
    out = flat.updateMask(mask).rename(pols).addBands(image.select("angle"))
    return ee.Image(out.copyProperties(image, image.propertyNames()))
