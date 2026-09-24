# 17. Review of the first map and the fixes (September 2026)

*Status: draft, numbers to be filled from the stage-4 re-run.*

## Why this document

The first delivered map (docs/15) was reviewed field by field in a GIS on two AOIs by the
project lead, then systematically on 20 AOIs with `sar_pipeline.review` (one panel per AOI: the
latest clear Sentinel-2 image, the radar rice composite, the class map; per-field evidence tables;
"suspect" categories; curve and chip sheets for the largest suspects). About 430 fields received a
verdict by eye. The review found 23 kinds of problem, from a code bug to data artefacts to rule
weaknesses. This document lists them by cause, says what was changed, and how each change was
measured before it was adopted. Nothing was changed without a measurement: every fix is scored on
the surveyed plots (recall), on the negatives (evergreen, water, bare, cut fields: false alarms) and
on the reviewed fields (wrong fields must flip, right fields must stay), against a frozen copy of
the first map (`analysis/compare_runs`).

## The problems, by root cause

| root cause | what the review saw | size in the reviewed AOIs |
|---|---|---|
| A. Optical cloud mask (QA60 only) | the cirrus bit removed whole clear scenes (34 AOI-dates in 22 AOIs, including the only clear view of a transplanting flood); hazy scenes passed and turned standing rice into "harvested" at the series end; cloud-contaminated low values inside gaps served as fake troughs | one AOI: 370-570 ac of standing rice in "harvested"; another: ~190 ac |
| B. Radar reader and pass screening | (1) a code bug: the 5x5 box mean spread one no-data pixel along its whole row and column, so a 2 % no-data strip blanked a pass over up to 96 % of an AOI (92 passes, 14 AOIs); (2) one pass broken in VH only (5-20 dB low on fields) stayed under the 3 dB artefact cut and faked transplanting water; (3) the pass audit could not re-judge a pass once marked bad; (4) four AOIs had radar only from 1 May | class 1 resting on the broken pass alone: 625 ac over 14 AOIs |
| C. Standing / harvested decided on the optical end only | hazy or missing end values made standing rice "harvested" or "young" | see A |
| D. Trough and water logic | a trough hidden by cloud (the fit interpolates across the gap) lost textbook paddies; a fallow shorter than 40 days after a summer crop failed; v1 took a previous crop's harvest for water; shallow water just under the thresholds; support only from the same polarisation | ~1,000 ac counted in docs/15 as "trough above 0.40 with radar water and canopy" |
| E. "Never bare" on class 3 only | tree lines, gardens and houses in classes 1, 2, 4 and 6 | tens of acres per AOI |
| F. Field delineation | outlines drawn around groups of fields ("monsters"), roads / canals / ponds traced as fields, tails along roads, slivers, areas computed in Web Mercator (~9 % high) | 10 % of the polygon acres |

## What was changed

### Inputs
* **Radar reader** (`analysis/sar_curve.box_mean`): a NaN-aware box mean (sum of valid power over
  count of valid pixels; a pixel keeps a value while half its window has data).
* **Pass screening v2** (`analysis/final_audit.screen_passes`): a pass is bad when the stable
  ground jumps by 3 dB, or when one polarisation drops 2 dB while the other is flat; a pass bad in
  three AOIs of a track and 1 dB low in 20 % of the track's AOIs is dropped on the whole track. The
  audit reads raw stacks, so passes are re-judged every time. Result: one pass (VH, one descending
  track, 11 June) dropped track-wide; the 15 "artefacts" of docs/15 were products of the reader bug.
* **VV/VH consistency** in both water tests: on the flood pass the other polarisation may not have
  risen by more than 1 dB (water lowers both).
* **Four AOIs** re-exported from 15 March (`scripts/s1_reexport.sh`).
* **Cloud mask** (`analysis/ndvi_5day.clear_mask`): QA60 opaque cloud; Cloud Score+ on bright
  pixels only (a plain Cloud Score+ mask removed the flooded-field observations: it treats dark
  water as cloud shadow, and on one plot AOI 77 ac of surveyed rice became "not rice"); dark
  observations (NIR < 1,500, NDVI < 0.3) always kept; a vegetated observation with blue reflectance
  above 900 removed as haze (clear canopies measured at 150-770, hazy ones at 1,100-2,700; the one
  hazy scene Cloud Score+ called clear sat at 1,115-1,266). The Cloud Score+ threshold was chosen
  by score (`analysis/mask_experiment`): TODO final choice and table.

### Rule
* **Radar-defined trough** (`monsoon_rule.radar_trough_events`): a confirmed radar flood can stand
  in for an optical trough hidden by cloud or cut short by a short fallow; the pixel then needs a
  canopy after the flood, its own optical history must show bare ground within 60 days before /
  30 days after the flood (the radar box is 5 x 5 pixels: a tree line beside flooded paddies
  floods in the radar), and no canopy may have been seen in the 25 days before the drop (that is a
  harvest).
* **Canopy on the flood date** judged on clear observations, not on the fitted curve.
* **Flood support** from either polarisation of any track; a very dark, very deep pass (VH <= -24,
  drop >= 8 dB) needs no second pass.
* **Never bare** applied to every rice-like class unless a flood was confirmed.
* **Fit floor**: the fitted NDVI may not fall below the pixel's lowest observation by more than 0.05.
* **Report classes**: 7 "flooded, not yet green" (a confirmed flood within 45 days of the map date,
  no canopy yet: the next map's rice); 8 "cut crop, water not confirmed" (class 4 is now harvested
  paddy with water).
* **No radar harvest test**: the radar end-of-season drop does not separate harvested from standing
  fields here (reviewed harvested fields: median drop 0.9 dB; standing: 0.5 dB), so "harvested"
  stays an optical decision on clear observations; a field with no clear view in the last 45 days
  is delivered with `label_confidence = low`.
* **Confidence notes** at field level: "late flood after an earlier crop" (delivered rice flooded
  after 25 July on a field that was green in the 60 days before: the data cannot tell a second
  transplanting from rain under a standing crop; listed for a field check) and "no clear view in
  the last 45 days".

### Field delineation (`analysis/field_refine`)
Overlaps resolved (the smaller polygon wins); an outline that loses half its area to smaller
polygons is cut to its remainder or dropped (remainder under 4 pixels, a strip, or scattered in
more than 3 pieces); tails narrower than 6 m removed by a morphological opening; long thin polygons
(no part wider than 15 m, longer than 150 m) flagged as strips; all-season water flagged as ponds;
slivers under 4 pixels merged into a same-label neighbour; areas recomputed in the local UTM zone.
Flagged polygons stay in the file with `is_field = False` and a `refine_flag`. Every polygon keeps
its `field_id`.

## Measurements

TODO after the stage-4 re-run: class transition table against the first map, plot and negative
scores, reviewed-field agreement, refined delineation totals.
