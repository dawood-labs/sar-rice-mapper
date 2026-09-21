"""Preparation steps that run *before* the pipeline: check the AOI, check data availability.

Why this package exists
-----------------------
Stage 0 of the pipeline (``grid``) assumes two things that nobody has verified yet:

1. the AOI file you point it at is actually the AOI you think it is, and
2. Sentinel-1 really does cover that AOI often enough in the season you care about.

Both assumptions are cheap to check and expensive to get wrong. Building a grid, running an audit
and submitting exports for a mis-deduplicated AOI wastes hours of Earth Engine quota before anyone
notices. So these checks live here as first-class, tested steps instead of throwaway scripts.

Neither step writes anything into a run folder, and neither one starts an Earth Engine export.
``s1-availability`` only reads image *metadata*.

Run them with::

    python -m sar_pipeline.prep aoi-qc          --config config/<name>.yaml
    python -m sar_pipeline.prep s1-availability --config config/<name>.yaml
"""
from __future__ import annotations

from . import aoi_qc, s1_availability

__all__ = ["aoi_qc", "s1_availability"]
