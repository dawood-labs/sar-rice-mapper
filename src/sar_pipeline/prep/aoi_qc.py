"""Check an AOI collection, and cross-check a deduplicated copy against the original.

The problem this solves
-----------------------
AOI polygons usually arrive as one file with many features, and someone then splits them into one
file per AOI, dropping duplicates on the way. Two things can silently go wrong:

* a polygon present in the original is **missing** from the split copy (data loss), or
* the split copy contains a polygon that is **not** in the original (something got edited).

Either one is invisible until a map comes out wrong weeks later. Comparing feature *counts* is not
enough: 160 rows collapsing to 132 files tells you nothing about *which* 28 went away.

How it works
------------
Every geometry is reduced to a canonical form with :meth:`shapely.normalize` and then to WKB, so two
polygons that describe the same shape with a different vertex order or ring direction compare equal.
Comparing those byte strings as sets answers all three questions exactly:

* how many distinct shapes the original really holds,
* whether every distinct original shape survived into the split copy,
* which dropped feature each duplicate was a copy of.

Areas
-----
Areas are always reported in **acres** (never hectares or m²), because that is the unit field teams
and clients use. Shapely returns m² in a projected CRS, so the conversion happens here, once, at
:data:`SQM_PER_ACRE`. A 10 m pixel is 0.025 acre, so an acre is about 40 pixels.

If the AOI file carries its own area column, :func:`compare_area_column` checks it against the
geometry instead of trusting it, and reports the unit it appears to be in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd

#: Exact conversion: 1 international acre = 4046.8564224 m².
SQM_PER_ACRE = 4046.8564224

#: Trailing digits in a per-AOI filename, e.g. ``AOI_017.gpkg`` -> 17.
_NUMBER_IN_NAME = re.compile(r"_(\d+)\.[A-Za-z]+$")


def sqm_to_acres(sqm: float | pd.Series) -> float | pd.Series:
    """Convert m² to acres. Kept as a named function so the unit rule is stated in exactly one place."""
    return sqm / SQM_PER_ACRE


@dataclass
class DuplicateReport:
    """How many distinct shapes a single AOI file really contains."""

    n_features: int
    n_unique: int
    #: group size -> how many distinct shapes appear that many times (``{1: 110, 2: 19, 4: 3}``).
    group_sizes: dict[int, int] = field(default_factory=dict)

    @property
    def n_duplicate_rows(self) -> int:
        """Rows that are copies of a shape counted elsewhere."""
        return self.n_features - self.n_unique

    def is_consistent(self) -> bool:
        """Do the group sizes add back up to the feature count? A guard against a counting mistake."""
        return sum(size * count for size, count in self.group_sizes.items()) == self.n_features


@dataclass
class CrosscheckReport:
    """Result of comparing a split/deduplicated AOI set against the original file."""

    duplicates: DuplicateReport
    n_split_files: int
    #: Distinct original shapes that no split file reproduces. Any entry here means data loss.
    missing_from_split: list[str]
    #: Split shapes absent from the original. Any entry here means the split was edited.
    unexpected_in_split: list[str]
    #: ``dropped id -> kept id`` for every duplicate that was removed.
    dropped_to_kept: dict[int, int]
    #: Dropped ids whose shape no kept file reproduces. Must be empty.
    orphaned: list[int]
    #: Split files whose filename number disagrees with the id stored inside the file.
    id_mismatches: list[tuple[int, int]]

    @property
    def ok(self) -> bool:
        """True when the split copy is a faithful deduplication of the original."""
        return not (self.missing_from_split or self.unexpected_in_split
                    or self.orphaned or self.id_mismatches)

    def summary(self) -> str:
        """A short human-readable verdict, suitable for a log or a report."""
        lines = [
            f"original features        : {self.duplicates.n_features}",
            f"distinct shapes          : {self.duplicates.n_unique}",
            f"duplicate rows           : {self.duplicates.n_duplicate_rows}",
            f"group sizes              : {self.duplicates.group_sizes}",
            f"split files              : {self.n_split_files}",
            f"missing from split       : {len(self.missing_from_split)}",
            f"unexpected in split      : {len(self.unexpected_in_split)}",
            f"dropped duplicates       : {len(self.dropped_to_kept)}",
            f"orphaned (no kept twin)  : {len(self.orphaned)}",
            f"filename/id mismatches   : {len(self.id_mismatches)}",
            f"VERDICT                  : {'OK - faithful deduplication' if self.ok else 'PROBLEM - see above'}",
        ]
        return "\n".join(lines)


def _wkb(frame: gpd.GeoDataFrame) -> pd.Series:
    """Canonical bytes per geometry, so equal shapes compare equal regardless of vertex order."""
    return frame.geometry.normalize().to_wkb()


def find_duplicates(frame: gpd.GeoDataFrame) -> DuplicateReport:
    """Count distinct shapes in one AOI file, and how often each repeats."""
    sizes = frame.assign(_wkb=_wkb(frame)).groupby("_wkb").size()
    return DuplicateReport(
        n_features=len(frame),
        n_unique=int(sizes.size),
        group_sizes={int(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
    )


def read_split_aois(paths: Iterable[Path], id_field: str = "id") -> pd.DataFrame:
    """Read one-feature-per-file AOIs into a table of ``file, number, id, wkb``.

    ``number`` comes from the trailing digits of the filename and ``id`` from inside the file; the
    cross-check compares the two, because a split that renumbers its outputs loses traceability
    back to the original.
    """
    rows = []
    for path in sorted(Path(p) for p in paths):
        match = _NUMBER_IN_NAME.search(path.name)
        frame = gpd.read_file(path)
        if len(frame) != 1:
            raise ValueError(f"{path.name}: expected exactly 1 feature, found {len(frame)}")
        rows.append({
            "file": path.name,
            "number": int(match.group(1)) if match else None,
            "id": int(frame[id_field].iloc[0]) if id_field in frame.columns else None,
            "wkb": _wkb(frame).iloc[0],
        })
    return pd.DataFrame(rows)


def crosscheck(original: gpd.GeoDataFrame, split: pd.DataFrame, id_field: str = "id") -> CrosscheckReport:
    """Compare a split AOI set against the original file it came from.

    ``original`` is the multi-feature file; ``split`` is the table from :func:`read_split_aois`.
    """
    frame = original.assign(_wkb=_wkb(original))
    ids = frame[id_field].astype(int) if id_field in frame.columns else pd.Series(range(len(frame)))
    frame = frame.assign(_id=ids)

    original_shapes = set(frame["_wkb"])
    split_shapes = set(split["wkb"].dropna())
    kept_ids = set(split["number"].dropna().astype(int))
    shape_to_kept = dict(zip(split["wkb"], split["number"]))

    dropped = frame[~frame["_id"].isin(kept_ids)]
    dropped_to_kept, orphaned = {}, []
    for _, row in dropped.iterrows():
        twin = shape_to_kept.get(row["_wkb"])
        if twin is None:
            orphaned.append(int(row["_id"]))
        else:
            dropped_to_kept[int(row["_id"])] = int(twin)

    paired = split.dropna(subset=["number", "id"])
    mismatches = [(int(r["number"]), int(r["id"])) for _, r in paired.iterrows()
                  if int(r["number"]) != int(r["id"])]

    return CrosscheckReport(
        duplicates=find_duplicates(original),
        n_split_files=len(split),
        missing_from_split=[b.hex()[:16] for b in sorted(original_shapes - split_shapes)],
        unexpected_in_split=[b.hex()[:16] for b in sorted(split_shapes - original_shapes)],
        dropped_to_kept=dropped_to_kept,
        orphaned=orphaned,
        id_mismatches=mismatches,
    )


def area_table(frame: gpd.GeoDataFrame, crs: str | int, id_field: str = "id") -> pd.DataFrame:
    """Per-AOI area in **acres**, computed from geometry in the given projected CRS.

    Pass the same ``grid.crs`` the pipeline will use, so these numbers match the grid's own idea of
    area. Geographic CRSs (degrees) give meaningless areas, so one is refused outright.
    """
    projected = frame.to_crs(crs)
    if projected.crs is None or projected.crs.is_geographic:
        raise ValueError(f"{crs} is geographic; area needs a projected CRS (e.g. a UTM zone)")
    out = pd.DataFrame({
        "id": frame[id_field].astype(int) if id_field in frame.columns else range(len(frame)),
        "acres": sqm_to_acres(projected.geometry.area),
    })
    return out.sort_values("acres", ascending=False).reset_index(drop=True)


def compare_area_column(frame: gpd.GeoDataFrame, crs: str | int, column: str = "area") -> dict:
    """Check a supplied area column against the geometry, and guess its unit.

    An area column in a delivered file is metadata, not truth: it may be in acres, hectares or m²,
    and it may predate an edit to the geometry. This reports the ratio so the unit is evident and
    the column can be trusted (or not) on evidence.
    """
    computed_acres = sqm_to_acres(frame.to_crs(crs).geometry.area)
    ratio = (frame[column] / computed_acres).median()
    units = {"acres": 1.0, "hectares": 1 / 2.47105, "m2": 4046.8564224}
    unit = min(units, key=lambda u: abs(ratio - units[u]) / units[u])
    disagreement = ((frame[column] - computed_acres).abs() / computed_acres * 100)
    return {
        "column": column,
        "ratio_to_computed_acres": round(float(ratio), 4),
        "likely_unit": unit if abs(ratio - units[unit]) / units[unit] < 0.05 else "unknown",
        "max_disagreement_pct": round(float(disagreement.max()), 2),
        "mean_disagreement_pct": round(float(disagreement.mean()), 2),
    }
