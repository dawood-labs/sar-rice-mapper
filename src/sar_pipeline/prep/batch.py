"""Set up and run many AOIs with one config each, and choose each AOI's tracks by a written rule.

Why one config per AOI instead of one merged AOI
------------------------------------------------
When AOIs are many small polygons scattered over hundreds of kilometres, a single merged AOI means a
single grid over the whole bounding box. For 132 fields spread over roughly 234 x 456 km that is on
the order of a billion grid pixels, almost all of them empty, against about nine million for 132
tight per-AOI grids. Per-AOI configs also let each AOI use the Sentinel-1 tracks that actually cover
it well, which differ across a region that size.

Why track choice is a rule and not a judgement call
---------------------------------------------------
With one AOI a person reads the audit and picks tracks. With a hundred, that stops being reviewable,
and quietly different choices per AOI make the results hard to compare. :func:`choose_tracks`
applies one rule everywhere and returns its **reasons**, so every choice can be audited afterwards.

The rule, in plain words:

1. A track must cover the AOI well (``min_coverage_pct``) and have a reasonable number of
   acquisitions (at least ``min_share`` of the best track's count).
2. A track is **excluded** if any of its gaps longer than ``max_gap_days`` (default **20 days**)
   overlaps the **flooding window**. For paddy rice the flooding minimum is the most diagnostic
   event of the season and lasts roughly three weeks; a gap shorter than that still leaves an
   observation inside the flooded period, a longer one can miss it entirely.

   The 20-day default was set so the rule reproduces the choice made by hand on the first AOIs:
   one track there was blind for 22 + 24 consecutive days across May-June (about 46 days, the whole
   early flooding period) and was rejected, while another had a single 18-day gap in July and was
   kept. The physical argument above supports the number; the data did not tune it.
3. The eligible track with the most acquisitions is **primary** (ties: shorter longest gap). The
   next one is **secondary**. At most two tracks: a second geometry lets the two tracks confirm each
   other, and a third adds cost without adding a new kind of evidence.
4. **Fallback secondary** (``fallback_secondary=True``): if no second track is eligible, a track that
   was excluded *only* for a flooding-window gap may still be used as secondary. The features are
   computed from the primary; the secondary is used only to confirm that the primary's seasonal
   shape is real (between-track correlation), and a blind spot in May-June merely removes a few bins
   from that comparison. Such a track is never promoted to primary.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_NUMBER_IN_NAME = re.compile(r"_(\d+)$")


@dataclass
class TrackChoice:
    """The outcome of :func:`choose_tracks` for one AOI."""

    primary: str | None
    secondary: str | None
    #: ``track_id -> reason`` for every track that was considered, chosen or not.
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def tracks(self) -> list[dict]:
        """The chosen tracks in the shape the config's ``s1.tracks`` expects."""
        out = []
        if self.primary:
            out.append({"track_id": self.primary, "role": "primary"})
        if self.secondary:
            out.append({"track_id": self.secondary, "role": "secondary"})
        return out


def _as_date(value) -> dt.date:
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def choose_tracks(track_summary: pd.DataFrame, gaps: pd.DataFrame, flood_window,
                  max_gap_days: int = 20, min_coverage_pct: float = 90.0,
                  min_share: float = 0.5, fallback_secondary: bool = False) -> TrackChoice:
    """Pick a primary and an optional secondary track from one AOI's audit tables.

    ``track_summary`` and ``gaps`` are the audit's ``track_summary.csv`` and ``gaps.csv``.
    ``flood_window`` is ``(start, end)``; a gap longer than ``max_gap_days`` that overlaps it
    disqualifies the track.
    """
    flood_start, flood_end = (_as_date(x) for x in flood_window)
    reasons: dict[str, str] = {}
    if track_summary.empty:
        return TrackChoice(None, None, {"-": "audit found no tracks"})

    best_count = int(track_summary["n_acq_ok_coverage"].max())
    eligible = []
    gap_only = []   # otherwise fine, excluded only for a flooding-window gap
    for row in track_summary.itertuples():
        tid = row.track_id
        if row.mean_aoi_coverage_pct < min_coverage_pct:
            reasons[tid] = f"excluded: mean AOI coverage {row.mean_aoi_coverage_pct:.0f}% < {min_coverage_pct:.0f}%"
            continue
        if row.n_acq_ok_coverage < min_share * best_count:
            reasons[tid] = (f"excluded: {row.n_acq_ok_coverage} well-covering acquisitions, "
                            f"under {min_share:.0%} of the best track's {best_count}")
            continue
        blocking = []
        for g in gaps[gaps["track_id"] == tid].itertuples():
            start, end = _as_date(g.gap_start), _as_date(g.gap_end)
            if g.gap_days > max_gap_days and start <= flood_end and end >= flood_start:
                blocking.append(f"{g.gap_days} d gap {start}..{end}")
        if blocking:
            reasons[tid] = "excluded: " + "; ".join(blocking) + " overlaps the flooding window"
            gap_only.append(row)
            continue
        eligible.append(row)

    eligible.sort(key=lambda r: (-r.n_acq_ok_coverage, r.max_gap_days, r.track_id))
    primary = eligible[0].track_id if eligible else None
    secondary = eligible[1].track_id if len(eligible) > 1 else None
    for rank, row in enumerate(eligible):
        role = "primary" if rank == 0 else "secondary" if rank == 1 else "not used (third eligible track)"
        reasons[row.track_id] = (f"{role}: {row.n_acq_ok_coverage} acquisitions, longest gap "
                                 f"{row.max_gap_days:.0f} d, coverage {row.mean_aoi_coverage_pct:.0f}%")
    if fallback_secondary and primary and secondary is None and gap_only:
        gap_only.sort(key=lambda r: (-r.n_acq_ok_coverage, r.max_gap_days, r.track_id))
        row = gap_only[0]
        secondary = row.track_id
        reasons[secondary] = ("secondary (fallback, confirmation only): " + reasons[secondary]
                              .replace("excluded: ", "") + "; tolerated because features come from the primary")
    return TrackChoice(primary, secondary, reasons)


def choose_tracks_from_audit(audit_dir, flood_window, **kwargs) -> TrackChoice:
    """:func:`choose_tracks` reading the two CSVs from an audit folder."""
    audit_dir = Path(audit_dir)
    summary = pd.read_csv(audit_dir / "track_summary.csv")
    gaps_path = audit_dir / "gaps.csv"
    gaps = pd.read_csv(gaps_path) if gaps_path.exists() else pd.DataFrame(
        columns=["track_id", "gap_start", "gap_end", "gap_days"])
    return choose_tracks(summary, gaps, flood_window, **kwargs)


def aoi_ids_in(split_dir, pattern: str = "*") -> list[int]:
    """AOI ids from a folder of one-folder-per-AOI splits, read from the trailing number."""
    ids = []
    for path in sorted(Path(split_dir).glob(pattern)):
        match = _NUMBER_IN_NAME.search(path.name)
        if path.is_dir() and match:
            ids.append(int(match.group(1)))
    return ids


def render_config(template: dict, aoi_id: int, aoi_path: str, season_key: str,
                  start: str, end: str, tracks: list[dict] | None = None) -> str:
    """Build one AOI's config from a template config (a parsed YAML dict) and return it as YAML text.

    Only ``aoi.key``, ``aoi.path``, ``season.key``, ``season.start``, ``season.end``,
    ``analysis.bins.start`` and ``s1.tracks`` change; every other setting is copied from the
    template, so all AOIs share identical processing settings by construction.

    The YAML is regenerated rather than edited as text, which drops the template's comments. That is
    deliberate: a text substitution that silently misses a line in one of a hundred files is far
    worse than a file without comments. A header points back to the template for the explanations.
    """
    import copy
    import yaml

    cfg = copy.deepcopy(template)
    cfg["aoi"]["key"] = f"aoi{aoi_id}"
    cfg["aoi"]["path"] = aoi_path
    cfg["season"]["key"] = season_key
    cfg["season"]["start"] = start
    cfg["season"]["end"] = end
    cfg.setdefault("analysis", {}).setdefault("bins", {})["start"] = start
    cfg["s1"]["tracks"] = list(tracks or [])
    header = (f"# AOI {aoi_id}, season {season_key}: generated by sar_pipeline.prep.batch from a template.\n"
              f"# Setting explanations live in config/pipeline.example.yaml. Gitignored: names a bucket.\n")
    return header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None)


def aoi_index(folder="data/aoi") -> pd.DataFrame:
    """``aoi``, ``lon``, ``lat``, ``acres`` for every per-AOI file ``aoi_NNN.gpkg`` in ``folder``.

    The area comes from the file's own ``area`` column (already in acres) when there is one,
    otherwise from the geometry in UTM zone 46N. Used to order and spread a batch.
    """
    import geopandas as gpd

    rows = []
    for path in sorted(Path(folder).glob("aoi_[0-9]*.gpkg")):
        frame = gpd.read_file(path)
        centroid = frame.to_crs(4326).geometry.union_all().centroid
        acres = float(frame["area"].sum()) if "area" in frame else float(frame.to_crs(32646).area.sum() / 4046.8564224)
        rows.append({"aoi": int(path.stem.split("_")[1]), "lon": round(centroid.x, 4),
                     "lat": round(centroid.y, 4), "acres": round(acres, 1)})
    return pd.DataFrame(rows)


def pick_spread(index: pd.DataFrame, n: int, min_km: float = 15.0, exclude=()) -> list[int]:
    """Choose ``n`` AOI ids, largest first, skipping any AOI within ``min_km`` of one already chosen.

    Why: a batch that is only the largest AOIs would cluster in one district, and the rule was
    validated on three regions with different calendars. Taking the biggest AOI first and then the
    biggest one at least ``min_km`` away gives large AOIs from many places; when the spread rule
    leaves fewer than ``n``, the largest remaining AOIs fill the batch regardless of distance.
    Distances are on the centroids, in kilometres (111 km per degree, longitude scaled by latitude).
    """
    import math

    ordered = index[~index["aoi"].isin(set(exclude))].sort_values("acres", ascending=False)
    chosen: list[int] = []
    taken: list[tuple[float, float]] = []

    def far_enough(lon, lat):
        for lon2, lat2 in taken:
            dx = (lon - lon2) * 111.0 * math.cos(math.radians(lat))
            dy = (lat - lat2) * 111.0
            if math.hypot(dx, dy) < min_km:
                return False
        return True

    for _, row in ordered.iterrows():
        if len(chosen) >= n:
            break
        if far_enough(row["lon"], row["lat"]):
            chosen.append(int(row["aoi"]))
            taken.append((row["lon"], row["lat"]))
    for _, row in ordered.iterrows():
        if len(chosen) >= n:
            break
        if int(row["aoi"]) not in chosen:
            chosen.append(int(row["aoi"]))
    return chosen
