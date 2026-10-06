# Field review brief (for reviewers judging fields by eye)

Why this file: the same brief for every reviewer (person or assistant), so verdicts are comparable across AOIs and
rounds. It encodes the user's class definitions and the mistakes found when reviewers were checked against the
user's own pixel labels (5 Oct 2026: reviewers called flooded fields "young" and optical-only drops "harvested").
Module: `sar_pipeline.analysis.field_review` (pick, render, score).

## Blind test

Use only the three sheet files of each field: `<id>_curve.png`, `<id>_chips.png`, `<id>.txt`. Never open
`fields.csv`, `user_checks.csv`, `score.csv`, other verdicts, class rasters, field layers, locked outputs, rule trials or
label files. Do not run analysis code. Open all three files for every field, small ones included.

## What the sheets show

- `_curve.png` top: field-median NDVI (filled dots clear views, hollow hazy, line fitted), dashed LSWI.
  Bottom: Sentinel-1 VV (solid) and VH (dotted) per track, dB.
- `_chips.png`: Sentinel-2 5-3-2 chips around the field, field outline in magenta.
- `.txt`: the numbers: clear-view median NDVI / LSWI per date, radar VV / VH per pass.

The map date is the newest data in the sheets. Where optical views are missing (clouds), the radar decides.

## Classes (exactly one per field)

- **rice standing transplanted**: rice standing now, planted more than 50 days before the map date. Any sign of
  water before or while the crop was small makes it transplanted (radar VV or VH falling toward its season bottom,
  or a clear view with LSWI above NDVI / NDVI near 0).
- **rice standing direct seeded**: rice standing now, more than 50 days old, with no sign of water at all.
- **young rice**: rice planted 50 days or less before the map date. Decided mainly by the radar: the field sat at
  water level and then VH **and** VV have been climbing out of it **over several passes (weeks)**; NDVI supports.
  Seedlings are green, so NDVI 0.5-0.7 three to four weeks after transplanting is possible.
  **Not young**: a field whose radar is still at water level, or rose on the last one or two passes only, with
  NDVI still low (about 0.3 or less) and LSWI near or above NDVI. That is **flooded / bare**.
- **rice harvested**: a grown canopy that was cut: high NDVI before, then **at least two** newest clear views clearly
  lower **and** the radar dropping. A drop seen only in the optical (one or two views) with the radar still at its
  canopy level is **not** a harvest: keep the standing class (ripening, haze, shadow) and say so in the reason.
- **flooded / bare**: open water or bare soil now, no crop coming up yet.
- **other vegetation**: green cover that is not this season's rice: never emptied (NDVI never near bare), grass,
  weeds, a non-rice crop, a strip along a road, canal or bund.
- **tree/orchard**: never bare across the whole season including dry March-May, radar bright and flat all season,
  often by villages. A cover whose NDVI low comes in the monsoon (July-August) is not a tree.

A field that is not field-shaped (a road or a narrow strip) is "other vegetation" with a note "not a field".

## Write

One JSON per field: `<review folder>/verdicts/<id>.json` with exactly the keys `field_id`, `class`, `sowing`
(YYYY-MM-DD: transplanted = near the end of the water spell when the radar starts rising; direct seeded = when NDVI
leaves bare; "" for non-rice), `confidence` (high / medium / low) and `reason` (one or two sentences citing dates and
values). Be honest: low confidence where the evidence is thin.
