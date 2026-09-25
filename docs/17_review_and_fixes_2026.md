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
  by score (`analysis/mask_experiment`) on the 12 surveyed-plot AOIs and the 17 reviewed AOIs.
  Plot-interior recall (%) in the three surveyed regions, raw rule maps, same rule for every mask:

  | mask | delta | region B | region N | reviewed fields: right kept / wrong fixed |
  |---|---|---|---|---|
  | QA60 both bits (first map's mask, new rule) | 97.5 | 91.3 | 95.0 | 14/17, 6/14 (plot AOIs) |
  | hybrid, Cloud Score+ >= 20 | 97.7 | 91.3 | 93.5 | 12/17, 11/14 |
  | hybrid, Cloud Score+ >= 30 | 97.4 | 93.0 | 93.9 | 13/17, 11/14 |
  | **hybrid, Cloud Score+ >= 40 (chosen)** | 97.0 | 94.7 | 93.9 | 14/17, 11/14 |
  | hybrid, Cloud Score+ >= 50 | 95.7 | 92.1 | 94.0 | 13/17, 10/14 |
  | hybrid, Cloud Score+ >= 60 | 90.1 | 89.2 | 92.6 | 11/17, 10/14 |
  | haze test only, no Cloud Score+ | 40.1 | 91.0 | 89.9 | 13/17, 10/14 |

  Without Cloud Score+ real clouds get in (the delta collapses); above 40 the stricter thresholds
  remove too many observations in the two drier regions. The blue-band haze rule has a second part:
  a blue-bright observation whose blue is at least 0.9 x its red is haze or thin cloud whatever
  its NDVI (haze had pushed a young canopy to NDVI 0.26 and slipped past the canopy test).

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

## Measurements (stage-4 re-run of all 132 AOIs, 25 September 2026)

**Class acres inside the AOIs, pixel map, first map -> new map** (`report/compare/stage4_matrix.csv`):

| class | first map | new map | where the difference went |
|---|---|---|---|
| rice (1) | 42,741 | 45,308 | +2,967 from "harvested", +2,699 from "not rice", +1,845 from class 3; -3,086 to class 3, -2,292 to "not rice" |
| young rice (6) | 6,072 | 8,405 | +3,411 from "not rice", +531 from "young" |
| **delivered (1 + 6)** | **48,812** | **53,714** | |
| rice-like, water not confirmed (3) | 11,454 | 12,535 | |
| harvested (4) | 7,021 | 588 | 2,967 to rice, 1,468 to class 3, 1,419 to "not rice", 494 to class 8 |
| young (2) | 3,629 | 2,155 | |
| never bare (5) | 3,998 | 2,478 | 2,862 to "not rice" (haze troughs gone with the mask) |
| flooded, not yet green (7, new) | - | 5,081 | 4,862 from "not rice" |
| cut crop, water not confirmed (8, new) | - | 716 | |
| not rice (0) | 38,491 | 36,141 | |

Field labels inside the AOIs (refined delineation): rice 47,056 ac, young rice 7,857 ac
(delivered 54,913; first map 49,586), class 3 11,584, flooded not green 5,028, harvested 481,
never bare 1,892, not rice 37,394.

**Surveyed plots** (share of plot-interior pixels delivered as rice, three established regions):

| | delta | region B | region N |
|---|---|---|---|
| first map, field labels | 96.9 | 94.8 | 98.3 |
| new map, pixel map | 97.9 | 95.2 | 93.6 |
| new map, field labels | 98.9 | 97.7 | 96.7 |

Negatives: evergreen, water, bare / built and "cut before the map date" pixels are delivered as
rice in 0 % of cases in the rule map; the 4-pixel minimum mapping unit then absorbs isolated tree
pixels inside rice fields (31 of 221 evergreen pixels in the delta plot AOIs, all of them created
by the sieve, as in the first map). Region N is 1.6 points under the first map at field level: its
remaining misses are fields whose water is shallower than the thresholds (VH -17.5 to -18.5 dB,
3-4 dB drop); accepting them would also accept dark dry soils in the dry-zone AOIs, so they stay
in class 3.

**Reviewed fields** (432 fields judged by eye in 20 AOIs, final field labels): of the fields the
reviewers called right, 198 of 216 decidable keep a label that agrees with them (first map 213); of
the fields called wrong, 127 of 148 now agree (first map 38); of the uncertain ones 20 of 34 (14).
Agreement over all decidable fields: 89 % against 69 % for the first map.

**Refined delineation**: 428,349 polygons -> 422,763 fields; 123,279 attribute acres -> 112,618
UTM acres -> 101,204 refined field acres (overlaps and tails removed, 597 ac of monster outlines
dropped and 193 ac cut, 232 ac of strips flagged).
