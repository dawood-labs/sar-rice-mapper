"""Mono-temporal speckle filters (one image at a time). All filters expect LINEAR power.

Adapted from gee_s1_ard/python-api/speckle_filter.py
(https://github.com/adugnag/gee_s1_ard, MIT License, (c) 2021 Adugna Mullissa; see
LICENSE_gee_s1_ard.txt). Refined Lee is originally by Guido Lemoine.

What speckle is: every radar pixel is the coherent sum of echoes from many small scatterers, so
its brightness fluctuates randomly around the true value (multiplicative noise). Filters estimate
the true value from neighbouring pixels. They must run on linear power, never on dB: averaging dB
values is a geometric mean and biases the result low.

Kernel-size behaviour (`ard.speckle_kernel`, must be an odd integer >= 3):

| Filter      | Uses speckle_kernel? | Notes |
|-------------|----------------------|-------|
| BOXCAR      | yes                  | plain K×K mean |
| LEE         | yes                  | K×K local mean/variance (MMSE) |
| GAMMA MAP   | yes                  | K×K local mean/std-dev |
| REFINED LEE | **no**               | fixed 3×3 statistics sampled inside a fixed 7×7 window with 8 edge-aligned directional sub-windows |
| LEE SIGMA   | partly               | fixed 3×3 target window for point-target detection and the a-priori mean; K×K for the final MMSE step |

Changes from the original (all deliberate):
- Kernels use integer radius K // 2, giving exactly K×K windows (the original passed K/2, a
  float radius, which depends on Earth Engine's rounding).
- Filters loop over explicit polarisation bands in Python instead of server-side band lists.
- GAMMA MAP: the MAP solution uses (alpha - ENL - 1), as in Lopes et al. (1990) eq. 11. The original
  expression used (z*alpha - ENL - 1), which is dimensionally inconsistent.
- LEE SIGMA (Lee et al. 2009):
  - pixels inside the sigma range are those with I1 <= value <= I2 (the original used OR, which keeps
    every pixel);
  - point targets are pixels whose 3×3 window holds at least 7 pixels above the scene's 98th
    percentile, counted with a neighbourhood SUM (the original counted distinct values of a 0/1
    image, which can never reach 7, so point targets were never kept);
  - the 98th percentile is estimated from a fixed-seed random sample of full-resolution (10 m) pixels
    over the acquisition footprint (the `footprint` image property set by `mosaic_acquisition`).
    Full resolution matters: a percentile of averaged coarse pixels is lower and would flag ordinary
    speckle peaks as targets. The fixed seed and the acquisition-wide region give every chunk the same
    threshold (no seams). (The earlier adaptation used `image.geometry()`, which is unbounded for a
    mosaic, so every task failed.)
  - the first-stage MMSE weight is clamped to [0, 1];
  - out-of-range pixels are filtered (the final estimate uses the centre pixel) instead of becoming holes.
- REFINED LEE: the MMSE weight is clamped to [0, 1], and when two directional gradients are equally
  large the first one is used (the original summed their direction codes, giving an invalid direction).
"""
from __future__ import annotations

import math

import ee

FILTERS = ("BOXCAR", "LEE", "GAMMA MAP", "REFINED LEE", "LEE SIGMA")
KERNEL_SIZE_IGNORED = {"REFINED LEE"}

# GRD IW data are multi-looked ~5 times in range (ESA quotes ENL ~4.4); gee_s1_ard uses 5 for
# Lee / Gamma MAP and 4 for Lee Sigma (its look-up table is for 4-look intensity).
ENL_LEE = 5
ENL_SIGMA = 4
SIGMA_PERCENTILE_SAMPLE_PIXELS = 20000  # full-resolution pixels sampled for the 98th percentile


def _square(kernel_size: int) -> ee.Kernel:
    return ee.Kernel.square(kernel_size // 2, "pixels")


def boxcar(image: ee.Image, bands: list[str], kernel_size: int) -> ee.Image:
    """Plain K×K mean (ignores masked neighbours)."""
    return image.select(bands).reduceNeighborhood(ee.Reducer.mean(), _square(kernel_size)).rename(bands)


def lee(image: ee.Image, bands: list[str], kernel_size: int) -> ee.Image:
    """Lee (1980) MMSE filter: x = mean + b·(z − mean), b = var_x / var_z."""
    eta2 = 1.0 / ENL_LEE  # squared speckle coefficient of variation
    out = []
    for b in bands:
        z = image.select(b)
        stats = z.reduceNeighborhood(
            reducer=ee.Reducer.mean().combine(ee.Reducer.variance(), sharedInputs=True),
            kernel=_square(kernel_size), optimization="window",
        )
        z_bar = stats.select(f"{b}_mean")
        var_z = stats.select(f"{b}_variance")
        var_x = var_z.subtract(z_bar.pow(2).multiply(eta2)).divide(1 + eta2)
        weight = var_x.divide(var_z)
        weight = weight.where(weight.lt(0), 0)
        out.append(ee.Image(1).subtract(weight).multiply(z_bar.abs()).add(weight.multiply(z)).rename(b))
    return ee.Image.cat(out)


def gamma_map(image: ee.Image, bands: list[str], kernel_size: int) -> ee.Image:
    """Gamma maximum-a-posteriori filter (Lopes et al. 1990)."""
    enl = ENL_LEE
    cu = 1.0 / math.sqrt(enl)
    cmax = math.sqrt(2.0) * cu
    out = []
    for b in bands:
        img = image.select(b)
        stats = img.reduceNeighborhood(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
            kernel=_square(kernel_size), optimization="window",
        )
        z = stats.select(f"{b}_mean")
        ci = stats.select(f"{b}_stdDev").divide(z)  # observed coefficient of variation
        alpha = ee.Image(1 + cu**2).divide(ci.pow(2).subtract(cu**2))
        a = alpha.subtract(enl + 1)
        q = z.pow(2).multiply(a.pow(2)).add(alpha.multiply(4 * enl).multiply(img).multiply(z))
        r_hat = z.multiply(a).add(q.sqrt()).divide(alpha.multiply(2))
        homogeneous = z.updateMask(ci.lte(cu))                              # -> local mean
        textured = r_hat.updateMask(ci.gt(cu)).updateMask(ci.lt(cmax))      # -> MAP estimate
        point_target = img.updateMask(ci.gte(cmax))                         # -> keep original
        out.append(ee.ImageCollection([homogeneous.rename(b), textured.rename(b), point_target.rename(b)]).sum().rename(b))
    return ee.Image.cat(out)


def _refined_lee_band(img: ee.Image) -> ee.Image:
    weights3 = ee.List.repeat(ee.List.repeat(1, 3), 3)
    kernel3 = ee.Kernel.fixed(3, 3, weights3, 1, 1, False)
    mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3)
    variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3)

    # 3x3 windows sampled inside a 7x7 window -> gradients and directions
    sample_weights = ee.List([[0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0]])
    sample_kernel = ee.Kernel.fixed(7, 7, sample_weights, 3, 3, False)
    sample_mean = mean3.neighborhoodToBands(sample_kernel)
    sample_var = variance3.neighborhoodToBands(sample_kernel)

    gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs()
    gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs())
    gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs())
    gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs())
    # index (0-3) of the largest gradient; ties resolve to the first index (deterministic)
    strongest = gradients.toArray().arrayArgmax().arrayGet([0])

    # for gradient g the edge lies on one side (direction g+1) or the other (direction g+5)
    side = []
    side.append(sample_mean.select(1).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(7))))
    side.append(sample_mean.select(6).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(2))))
    side.append(sample_mean.select(3).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(5))))
    side.append(sample_mean.select(0).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(8))))
    directions = ee.Image(0)
    for g in range(4):
        code = side[g].multiply(g + 1).add(side[g].Not().multiply(g + 5))
        directions = directions.where(strongest.eq(g), code)
    directions = directions.updateMask(gradients.select(0).mask())

    sample_stats = sample_var.divide(sample_mean.multiply(sample_mean))
    sigma_v = sample_stats.toArray().arraySort().arraySlice(0, 0, 5).arrayReduce(ee.Reducer.mean(), [0])

    rect_weights = ee.List.repeat(ee.List.repeat(0, 7), 3).cat(ee.List.repeat(ee.List.repeat(1, 7), 4))
    diag_weights = ee.List([[1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1, 1, 1]])
    rect_kernel = ee.Kernel.fixed(7, 7, rect_weights, 3, 3, False)
    diag_kernel = ee.Kernel.fixed(7, 7, diag_weights, 3, 3, False)

    dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1))
    dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1))
    dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)))
    dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)))
    for i in range(1, 4):
        dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2 * i + 1)))
        dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(directions.eq(2 * i + 1)))
        dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2 * i + 2)))
        dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(directions.eq(2 * i + 2)))
    dir_mean = dir_mean.reduce(ee.Reducer.sum())
    dir_var = dir_var.reduce(ee.Reducer.sum())

    var_x = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigma_v)).divide(sigma_v.add(1.0))
    # sigma_v is a 1-element array image; flatten before clamping (clamp does not accept arrays)
    weight = var_x.divide(dir_var).arrayProject([0]).arrayFlatten([["w"]]).clamp(0, 1)
    result = dir_mean.add(weight.multiply(img.subtract(dir_mean)))
    return result.rename("sum").float()


def refined_lee(image: ee.Image, bands: list[str], kernel_size: int | None = None) -> ee.Image:
    """Refined Lee (Lee et al. 1999; GEE implementation by G. Lemoine). kernel_size is ignored."""
    return ee.Image.cat([_refined_lee_band(image.select(b)).rename(b) for b in bands])


def lee_sigma(image: ee.Image, bands: list[str], kernel_size: int, footprint: ee.Geometry | None = None) -> ee.Image:
    """Improved Lee sigma filter (Lee et al. 2009), sigma = 0.9, 4-look look-up table.

    `footprint` bounds the region used for the scene's 98th percentile. When omitted, the image's
    `footprint` property is used (set by pipeline.mosaic_acquisition).
    """
    tk = 7            # bright pixels needed in a 3x3 window to keep a point target
    target_kernel = 3
    eta = 1.0 / math.sqrt(ENL_SIGMA)
    # Lee et al. 2009 LUT for sigma = 0.9 (4-look intensity)
    i1_factor, i2_factor, new_eta = 0.378, 2.094, 0.3991
    region = footprint if footprint is not None else ee.Geometry(image.get("footprint"))
    out = []
    for b in bands:
        img = image.select(b)
        # scene-wide 98th percentile from full-resolution pixels (fixed seed -> identical in every chunk)
        sample = img.sample(region=region, scale=10, numPixels=SIGMA_PERCENTILE_SAMPLE_PIXELS, seed=0,
                            dropNulls=True, tileScale=4)
        z98 = ee.Number(sample.aggregate_array(b).reduce(ee.Reducer.percentile([98])))
        bright = img.gte(z98)
        n_bright = bright.reduceNeighborhood(ee.Reducer.sum(), _square(target_kernel))
        retain = n_bright.gte(tk)

        stats = img.reduceNeighborhood(
            reducer=ee.Reducer.mean().combine(ee.Reducer.variance(), sharedInputs=True),
            kernel=_square(target_kernel), optimization="window",
        )
        z_bar = stats.select(f"{b}_mean")
        var_z = stats.select(f"{b}_variance")
        var_x = var_z.subtract(z_bar.abs().pow(2).multiply(eta**2)).divide(1 + eta**2)
        weight = var_x.divide(var_z).clamp(0, 1)
        x_tilde = ee.Image(1).subtract(weight).multiply(z_bar.abs()).add(weight.multiply(img))

        lower, upper = x_tilde.multiply(i1_factor), x_tilde.multiply(i2_factor)
        in_range = img.updateMask(img.gte(lower).And(img.lte(upper)))
        stats = in_range.reduceNeighborhood(
            reducer=ee.Reducer.mean().combine(ee.Reducer.variance(), sharedInputs=True),
            kernel=_square(kernel_size), optimization="window",
        )
        z_bar = stats.select(f"{b}_mean")
        var_z = stats.select(f"{b}_variance")
        var_x = var_z.subtract(z_bar.abs().pow(2).multiply(new_eta**2)).divide(1 + new_eta**2)
        weight = var_x.divide(var_z).clamp(0, 1)
        x_hat = ee.Image(1).subtract(weight).multiply(z_bar.abs()).add(weight.multiply(img))
        out.append(img.updateMask(retain).unmask(x_hat).updateMask(img.mask()).rename(b))
    return ee.Image.cat(out)


_DISPATCH = {
    "BOXCAR": boxcar,
    "LEE": lee,
    "GAMMA MAP": gamma_map,
    "REFINED LEE": refined_lee,
    "LEE SIGMA": lee_sigma,
}


def mono_filter(image: ee.Image, name: str, kernel_size: int, bands: list[str]) -> ee.Image:
    """Apply one mono-temporal filter to `bands` of `image`; returns only those bands."""
    if name not in _DISPATCH:
        raise ValueError(f"ard.speckle_filter must be one of {FILTERS}, got {name!r}")
    return _DISPATCH[name](image, bands, kernel_size)
