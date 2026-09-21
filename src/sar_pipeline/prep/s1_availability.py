"""Ask Earth Engine which Sentinel-1 tracks and dates exist over an AOI. Metadata only.

Why run this before anything else
---------------------------------
The pipeline's ``audit`` step (Stage 1) answers this question thoroughly — per chunk, with rain and
slope and coverage percentages — but it needs a grid to exist first, and on a large AOI it is not
cheap. This step is the five-second version, and it is meant to be run while you are still *choosing*
the season window:

* Does the AOI have enough acquisitions in the window to see a crop cycle at all?
* How many tracks cover it, so you know whether track selection will be a real decision?
* Which satellites contributed, so a constellation gap does not surprise you later?

It reads image **metadata** only. It never exports, never writes to Cloud Storage, and never builds
a grid, so it is safe to run repeatedly while experimenting with dates.

This is not a substitute for ``audit``
--------------------------------------
Counts here are over the AOI's **bounding box**, not the polygons themselves, and a scene that
merely touches the box is counted. On an AOI made of scattered polygons spread over hundreds of
kilometres, the box can intersect tracks that cover no actual field. Use this to size the problem;
use ``audit`` to decide ``s1.tracks``.
"""
from __future__ import annotations

import collections
import datetime as dt
from dataclasses import dataclass, field
from typing import Callable, Sequence

#: One row of returned metadata: (epoch ms, relative orbit, pass direction, platform letter).
SceneRow = tuple[int, int, str, str]


@dataclass
class Track:
    """One Sentinel-1 track: a relative orbit flown in one direction."""

    relative_orbit: int
    orbit_pass: str
    n_scenes: int

    @property
    def key(self) -> str:
        """Short label, e.g. ``RO143_ASC`` — the same shape the config's ``s1.tracks`` uses."""
        return f"RO{self.relative_orbit}_{self.orbit_pass[:3].upper()}"


@dataclass
class AvailabilityReport:
    """What Sentinel-1 offers over an AOI in a window."""

    start: str
    end: str
    n_scenes: int
    tracks: list[Track]
    platforms: dict[str, int]
    #: ``"YYYY-MM" -> number of distinct acquisition dates`` across all tracks.
    dates_per_month: dict[str, int] = field(default_factory=dict)

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)

    def months_below(self, minimum: int) -> list[str]:
        """Months with fewer distinct dates than ``minimum`` — candidate gaps worth investigating.

        A single track revisits every ~12 days, so roughly 2–3 dates per month per track is normal.
        A month well below the rest of the window usually means a constellation outage or a
        seasonal acquisition-plan change, and it will show up later as an interpolated time bin.
        """
        return [month for month, n in sorted(self.dates_per_month.items()) if n < minimum]

    def summary(self) -> str:
        """Human-readable report, ordered the way you actually read it."""
        lines = [
            f"window                : {self.start} .. {self.end}",
            f"scenes (IW, bbox)     : {self.n_scenes}",
            f"tracks                : {self.n_tracks}",
        ]
        for track in self.tracks:
            lines.append(f"  {track.key:14s} orbit {track.relative_orbit:3d} "
                         f"{track.orbit_pass:11s} {track.n_scenes:5d} scenes")
        lines.append(f"platforms             : {self.platforms}")
        lines.append("distinct dates / month:")
        for month, n in sorted(self.dates_per_month.items()):
            lines.append(f"  {month}: {n:3d}")
        return "\n".join(lines)


def build_report(rows: Sequence[SceneRow], start: str, end: str) -> AvailabilityReport:
    """Turn raw metadata rows into a report. Pure function, so it is unit-testable without Earth Engine."""
    track_counts: collections.Counter = collections.Counter()
    platforms: collections.Counter = collections.Counter()
    months: dict[str, set] = collections.defaultdict(set)

    for epoch_ms, relative_orbit, orbit_pass, platform in rows:
        stamp = dt.datetime.fromtimestamp(epoch_ms / 1000, tz=dt.timezone.utc)
        track_counts[(int(relative_orbit), orbit_pass)] += 1
        platforms[platform] += 1
        months[stamp.strftime("%Y-%m")].add(stamp.date())

    tracks = [Track(relative_orbit=orbit, orbit_pass=direction, n_scenes=n)
              for (orbit, direction), n in track_counts.most_common()]
    return AvailabilityReport(
        start=start, end=end, n_scenes=len(rows), tracks=tracks,
        platforms=dict(platforms.most_common()),
        dates_per_month={month: len(dates) for month, dates in months.items()},
    )


def fetch_rows(bounds: tuple[float, float, float, float], start: str, end: str,
               collection: str = "COPERNICUS/S1_GRD", instrument_mode: str = "IW") -> list[SceneRow]:
    """Read Sentinel-1 scene metadata over ``bounds`` from Earth Engine. No exports.

    ``bounds`` is ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326. Earth Engine must already
    be initialised — use :func:`sar_pipeline.auth.init_ee`.

    Only four properties are pulled back, in one ``reduceColumns`` call, so the payload stays small
    even for thousands of scenes.
    """
    import ee

    region = ee.Geometry.Rectangle(list(bounds))
    images = (ee.ImageCollection(collection)
              .filterBounds(region)
              .filterDate(start, end)
              .filter(ee.Filter.eq("instrumentMode", instrument_mode)))
    reducer = ee.Reducer.toList(4)
    columns = ["system:time_start", "relativeOrbitNumber_start",
               "orbitProperties_pass", "platform_number"]
    raw = images.reduceColumns(reducer, columns).getInfo()["list"]
    return [(int(t), int(orbit), str(direction), str(platform))
            for t, orbit, direction, platform in raw]


def availability(bounds: tuple[float, float, float, float], start: str, end: str,
                 fetch: Callable[..., list[SceneRow]] = fetch_rows, **kwargs) -> AvailabilityReport:
    """Fetch and summarise Sentinel-1 availability over ``bounds``.

    ``fetch`` is injectable so tests can supply canned metadata and run with no network.
    """
    return build_report(fetch(bounds, start, end, **kwargs), start, end)
