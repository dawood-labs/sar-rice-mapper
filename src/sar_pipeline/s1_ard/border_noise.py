"""Additional border-noise masking.

Adapted from gee_s1_ard/python-api/border_noise_correction.py
(https://github.com/adugnag/gee_s1_ard, MIT License, (c) 2021 Adugna Mullissa; the approach
follows Hird et al. 2017, Remote Sensing 9(12):1315). See LICENSE_gee_s1_ard.txt.

Why it exists: scenes processed before 2018 showed low-value stripes at the near and far edges of
the IW swath. Those edges correspond to the extreme incidence angles, so masking by angle removed
them cheaply.

Why it is OFF by default (`ard.border_noise_correction: false`): current GRD products already have
ESA's border-noise removal applied. A live check on 2026 scenes showed the valid incidence angles span
about 30.8-45.4 degrees, so the upper limit removed ~5% of good far-range pixels and the lower limit
removed nothing. Enable it only for old data that still shows edge stripes.

Changes from the original: no dB round trip (masking by angle does not need it), the mask is applied
with updateMask so existing masks are kept, and the angle band is resampled bilinearly. The `angle`
band is a coarse tie-point grid (~16 km); read with nearest neighbour it changes in ~0.8 degree steps,
which would give the mask a staircase edge.
"""
from __future__ import annotations

import ee

# Incidence-angle limits from gee_s1_ard (degrees). IW swath spans roughly 29-46 degrees.
ANGLE_MIN_DEG = 30.63993
ANGLE_MAX_DEG = 45.23993


def mask_border_noise(image: ee.Image) -> ee.Image:
    """Mask pixels whose incidence angle is outside (ANGLE_MIN_DEG, ANGLE_MAX_DEG)."""
    angle = image.select("angle").resample("bilinear")
    keep = angle.gt(ANGLE_MIN_DEG).And(angle.lt(ANGLE_MAX_DEG))
    return image.updateMask(keep).copyProperties(image, image.propertyNames())
