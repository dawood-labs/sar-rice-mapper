"""Shared pytest fixtures.

- Unit tests use a synthetic project in a temp folder (synthetic AOI, synthetic config). No network.
- Live Earth Engine tests are marked `@pytest.mark.gee` and are skipped by default.
  Run them with `pytest -m gee`. They read the real local config given by the
  SAR_PIPELINE_CONFIG env var (default: config/live.yaml) and must be READ-ONLY
  (getInfo / computePixels on tiny regions). They must NEVER start export tasks.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import geopandas as gpd
import pytest
import yaml
from shapely.geometry import Polygon

from sar_pipeline.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

# Synthetic AOI (EPSG:4326): an irregular polygon ~5 x 4 km over the open ocean. The grid uses the
# global equal-area CRS EPSG:6933. Arbitrary, unrelated to any real project area.
SYNTHETIC_AOI_LONLAT = [
    (-30.000, 40.000), (-29.945, 40.002), (-29.942, 40.040),
    (-29.970, 40.046), (-29.998, 40.038), (-30.000, 40.000),
]


def _base_cfg_dict() -> dict:
    example = yaml.safe_load((REPO_ROOT / "config" / "pipeline.example.yaml").read_text())
    cfg = copy.deepcopy(example)
    cfg["aoi"] = {"key": "test_aoi", "path": "data/aoi/test_aoi.gpkg"}
    cfg["season"].update({"key": "test_season", "start": "2026-04-01", "end": "2026-12-01"})
    cfg["auth"] = {"key_file": "secrets/fake.json", "project": "fake-project"}
    cfg["gcs"] = {"bucket": "fake-bucket", "base_folder": "fake_base"}
    cfg["grid"].update({"crs": "EPSG:6933", "res": 10, "buffer_m": 100, "chunk_px": 256})
    return cfg


@pytest.fixture
def make_project(tmp_path):
    """Factory: make_project(overrides: dict | None, aoi_lonlat: list | None) -> cfg dict.

    Creates <tmp>/config/test.yaml and <tmp>/data/aoi/test_aoi.gpkg and returns the loaded config
    (with _root = tmp project folder). `overrides` is a shallow-per-section update, e.g.
    {"grid": {"chunk_px": 128}}.
    """

    def _make(overrides: dict | None = None, aoi_lonlat: list | None = None) -> dict:
        cfg = _base_cfg_dict()
        for section, values in (overrides or {}).items():
            if isinstance(values, dict) and isinstance(cfg.get(section), dict):
                cfg[section].update(values)
            else:
                cfg[section] = values
        (tmp_path / "config").mkdir(exist_ok=True)
        aoi_file = tmp_path / cfg["aoi"]["path"]
        aoi_file.parent.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame(
            {"name": ["aoi"]}, geometry=[Polygon(aoi_lonlat or SYNTHETIC_AOI_LONLAT)], crs="EPSG:4326"
        ).to_file(aoi_file, driver="GPKG")
        cfg_path = tmp_path / "config" / "test.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return load_config(cfg_path)

    return _make


@pytest.fixture(scope="session")
def live_cfg():
    """Real config + initialised Earth Engine, for @pytest.mark.gee tests only."""
    cfg_path = Path(os.environ.get("SAR_PIPELINE_CONFIG", REPO_ROOT / "config" / "live.yaml"))
    if not cfg_path.exists():
        pytest.skip(f"live config not found: {cfg_path}")
    cfg = load_config(cfg_path)
    key = Path(cfg["_root"]) / cfg["auth"]["key_file"]
    if not key.exists():
        pytest.skip(f"service account key not found: {key}")
    from sar_pipeline.auth import init_ee

    init_ee(cfg)
    return cfg
