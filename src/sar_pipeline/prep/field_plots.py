"""Field-surveyed plots: load them, check them, and find which AOIs they fall in.

Why
---
Everything so far has been inferred from imagery, with no ground truth. Field plots — polygons a
survey team drew around fields they visited — are the first independent evidence of what a crop
looks like on the ground. Before any curve is read from them, three things have to be known:

* **what arrived**: how many polygons, from which delivery file, with which attributes, how large
  (in acres), and whether the geometries are valid;
* **where they are relative to the AOIs**: a plot outside every AOI has no exported imagery yet, so
  the work can only start where plots and AOIs overlap;
* **what their attributes say**, before anyone assumes a label.

Deliveries arrive as one zipped shapefile per area. :func:`load` reads every ``*.zip`` in a folder
(and any loose ``*.shp``/``*.gpkg``), adds a ``source`` column with the file it came from and a
``plot_id`` that is unique across files, and reprojects to lon/lat. Nothing is written back.
The folder lives outside the repository: field data is client data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .aoi_qc import SQM_PER_ACRE


def load(folder) -> "gpd.GeoDataFrame":  # noqa: F821
    """Every plot in every delivery file of ``folder``, in EPSG:4326, with ``source`` and ``plot_id``."""
    import zipfile

    import geopandas as gpd

    frames = []
    for path in sorted(Path(folder).resolve().iterdir()):
        if path.suffix == ".zip":
            # the shapefile usually sits in a sub-folder inside the zip; point GDAL at it directly
            inner = [n for n in zipfile.ZipFile(path).namelist() if n.lower().endswith(".shp")]
            uris = [f"/vsizip/{path}/{n}" for n in inner]
        elif path.suffix in (".shp", ".gpkg"):
            uris = [str(path)]
        else:
            continue
        for uri in uris:
            frame = gpd.read_file(uri)
            frame["source"] = path.stem
            frame["crs_in_file"] = str(frame.crs)
            frames.append(frame.to_crs(4326))
    plots = pd.concat(frames, ignore_index=True)
    plots = gpd.GeoDataFrame(plots, geometry="geometry", crs=4326)
    plots.insert(0, "plot_id", range(len(plots)))
    return plots


def with_area(plots) -> "gpd.GeoDataFrame":  # noqa: F821
    """Add ``acres`` and ``valid`` (geometry validity as delivered).

    Area is measured per delivery file in that file's own UTM zone: the deliveries straddle two
    zones, and one zone for all of them would stretch the areas at the far side.
    """
    out = plots.copy()
    out["no_geometry"] = out.geometry.isna() | out.geometry.is_empty
    out["valid"] = out.geometry.is_valid | out["no_geometry"]
    out["acres"] = 0.0
    for _, idx in out.groupby("source").groups.items():
        part = out.loc[idx]
        out.loc[idx, "acres"] = (repair(part).to_crs(part.estimate_utm_crs()).area / SQM_PER_ACRE).round(3)
    return out


def repair(plots):
    """Invalid geometries made valid (self-touching rings, bow-ties), polygon parts only."""
    import shapely

    out = plots.copy()
    fixed = shapely.make_valid(out.geometry.values)
    # make_valid can return a collection with stray lines or points; keep the polygonal part
    out["geometry"] = [g if g is None or g.geom_type in ("Polygon", "MultiPolygon")
                       else shapely.unary_union([p for p in getattr(g, "geoms", []) if p.area > 0])
                       for g in fixed]
    return out


def summary(plots) -> pd.DataFrame:
    """One row per delivery file: plot count, area in acres, invalid geometries, CRS in the file."""
    p = with_area(plots)
    return (p.groupby("source")
            .agg(plots=("plot_id", "size"), total_acres=("acres", "sum"), median_acres=("acres", "median"),
                 min_acres=("acres", "min"), max_acres=("acres", "max"),
                 invalid=("valid", lambda v: int((~v).sum())), no_geometry=("no_geometry", "sum"),
                 crs_in_file=("crs_in_file", "first"))
            .round(2).reset_index())


def attribute_overview(plots, max_values: int = 12) -> pd.DataFrame:
    """For every attribute column: type, how many values are filled, and the most common values."""
    rows = []
    for col in plots.columns:
        if col in ("geometry",):
            continue
        s = plots[col]
        top = s.astype(str).value_counts().head(max_values)
        rows.append({"column": col, "dtype": str(s.dtype), "filled": int(s.notna().sum()),
                     "distinct": int(s.nunique()),
                     "top_values": "; ".join(f"{k} ({v})" for k, v in top.items())})
    return pd.DataFrame(rows)


def in_aois(plots, aoi_polygons: dict) -> pd.DataFrame:
    """For every plot: the AOI it overlaps most (by area), and what share of the plot lies inside it.

    ``aoi_polygons`` maps an AOI key to lon/lat GeoJSON (``season_screen.aoi_polygons``). Plots that
    touch no AOI get ``aoi = None`` and share 0.
    """
    import geopandas as gpd
    from shapely.geometry import shape

    aois = gpd.GeoDataFrame({"aoi": list(aoi_polygons)},
                            geometry=[shape(g) for g in aoi_polygons.values()], crs=4326)
    utm = plots.estimate_utm_crs()
    p = repair(plots[["plot_id", "geometry"]]).to_crs(utm)  # empty records simply match nothing
    a = repair(aois).to_crs(utm)
    pieces = gpd.overlay(p, a, how="intersection", keep_geom_type=True)
    pieces["overlap_sqm"] = pieces.area
    best = pieces.sort_values("overlap_sqm", ascending=False).drop_duplicates("plot_id")
    out = pd.DataFrame({"plot_id": plots["plot_id"], "plot_sqm": p.area.to_numpy()})
    out = out.merge(best[["plot_id", "aoi", "overlap_sqm"]], on="plot_id", how="left")
    out["share_in_aoi"] = (out["overlap_sqm"].fillna(0) / out["plot_sqm"]).round(3)
    return out[["plot_id", "aoi", "share_in_aoi"]]
