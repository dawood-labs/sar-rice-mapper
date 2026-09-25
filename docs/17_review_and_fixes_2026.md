# 17. Review of the first map and the fixes (September 2026)

*Status: final numbers from the stage-6 re-run of 25 September 2026 (section "Measurements").*

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
  canopy after the flood, and its own optical history must show ground without a canopy (NDVI
  <= 0.45, the canopy level less a margin) at some clear date from 90 days before the flood to the
  optical climb (the radar box is 5 x 5 pixels: a tree line beside flooded paddies floods in the
  radar). A canopy seen in the 10 days before a drop makes that drop a harvest, not a flood (25 days
  at first; that refused the transplanting flood of a second crop cut 2-3 weeks earlier).
* **Canopy on the flood date** judged on clear observations, not on the fitted curve.
* **The flood is the largest drop that passes the whole water test**, not the season's largest drop
  (round 2, aoi110). Every monsoon pass is tested on its own (drop >= 4 dB below the pass's own
  earlier level, dark enough for water, no canopy in the 10 days before it, a second pass within
  14 days or a very deep single pass). On a double-crop plain the season's largest drop is the
  summer crop's ripening and harvest (a bright canopy falling to -14..-16 dB VH, which is not
  water); judging only that pass hid the real August transplanting flood, and ~150 ac of monsoon
  rice in one AOI lost their water. The largest drop of all is still reported where no pass
  qualifies, so the reason for a failed test stays visible.
* **No "other polarisation" test on the v2 flood.** It was added against a broken pass (issue 15:
  VH 5-20 dB low while VV was bright) but also refused real transplanting floods, where VV rises
  2-4 dB by double bounce off the seedlings while VH falls into the water. Broken passes are now
  removed by the pass screening before any rule runs, and the flood keeps its support test; the v1
  dip, which has no support test, keeps the check.
* **Radar canopy**: a young rice seen only by the radar is class 1 when the canopy is full (VH rose
  >= 8 dB from the flood to >= -15 dB at the season end) and class 7 while its optical value is
  still open water (< 0.20). The radar canopy may only stand in where the optical end is stale: a
  clear view in the last 20 days that reads below 0.30 contradicts it (open water and bare mud
  beside bright bunds showed a 12-16 dB "rise" in the 5 x 5 radar box and were delivered as rice).
* **Fit ceiling**: the fitted NDVI may not exceed the pixel's highest observation by more than 0.05
  (a peak invented across a gap made "harvested" fields).
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
areas recomputed in the local UTM zone. Flagged polygons stay in the file with `is_field = False`
and a `refine_flag`. Every polygon keeps its `field_id`.

**Geometry hygiene (after the user's QGIS review of the delivered aoi116 file, 25 September):** the
first refined files still carried invalid outlines, line-shaped holes where an overlapping
neighbour had been cut out, crumbs left by the cut (one field in several pieces), mitred corners of
the tail-removing opening that spiked into neighbours, duplicate outlines, tiny polygons that had
not merged, and small same-class pieces sitting inside a bigger field. Now, always starting from
the raw delineation: only valid polygons are kept (never lines or geometry collections); holes that
are thin (under 6 m wide) or under 4 pixels are filled, wider holes (a pond, a house) stay; the
opening is clipped to the original outline; duplicates are cut to nothing; a polygon left in
pieces keeps one outline per field (pieces of 4 pixels or more become fields of their own,
`field_id` + a letter, flag `split_part`; smaller pieces are dropped, their share recorded in
`crumbs_dropped_share`); after labelling, crumbs under 4 pixels merge into the neighbour they share
most outline with (whatever its label: 1-3 pixels carry no label), small pieces under 0.5 ac that
share at least half their outline with one same-label field at least twice their size merge into
it, and merged outlines are labelled again from their final shape. `field_refine audit` measures
all of this on a delivered file and `field_refine.review_crops` draws before/after crops.
Delivered fields of aoi116, before -> after: invalid 327 -> 0, line-shaped holes 2,652 -> 1,
polygons in pieces 862 -> 8 (all strips), polygons under 4 pixels 1,495 -> 1, overlapping pairs
1,562 -> 0, small same-class pieces inside a field 578 -> 0; polygons 7,917 -> 5,982, delivered
acres unchanged (rice 4,904 + young rice 60). The delivered GeoPackage layer now carries the file
name, so two AOIs or two versions opened together in QGIS are told apart.

## Measurements (final re-run of all 132 AOIs, 25 September 2026, "stage 7")

**Class acres inside the AOIs, pixel map, first map -> final map** (`report/compare/stage7_matrix.csv`):

| class | first map | final map | where the difference went |
|---|---|---|---|
| rice (1) | 42,740 | 47,206 | +3,064 from "harvested", +3,219 from "not rice", +2,032 from class 3; -2,649 to class 3, -1,843 to "not rice" |
| young rice (6) | 6,072 | 7,419 | +2,512 from "not rice", +483 from "young" |
| **delivered (1 + 6)** | **48,812** | **54,626** | |
| rice-like, water not confirmed (3) | 11,454 | 11,870 | |
| harvested (4) | 7,020 | 628 | 3,064 to rice, 1,419 to class 3, 1,359 to "not rice", 444 to class 8 |
| young (2) | 3,630 | 2,878 | |
| never bare (5) | 3,998 | 2,476 | 2,861 to "not rice" (haze troughs gone with the mask) |
| flooded, not yet green (7, new) | - | 5,901 | 5,632 from "not rice" |
| cut crop, water not confirmed (8, new) | - | 642 | |
| not rice (0) | 38,491 | 34,386 | |

Field labels inside the AOIs (refined delineation): rice 48,824 ac, young rice 6,939 ac
(delivered 55,763; first map 49,586), class 3 11,042, flooded not green 5,859, young 2,126,
never bare 1,899, harvested 520, cut crop 576, not rice 35,639.

**Surveyed plots** (share of plot-interior pixels delivered as rice, three established regions):

| | delta | region B | region N |
|---|---|---|---|
| first map, field labels | 96.9 | 94.8 | 98.3 |
| final map, pixel map | 98.0 | 97.0 | 94.4 |
| final map, field labels | 99.0 | 98.0 | 96.8 |

Negatives (rule map, before the 4-pixel sieve): water and bare / built pixels 0 %; "cut before the
map date" 0 % in the delta and 1.2 % in the fourth region (3 of 249 pixels; the 12-pixel set of
region B is 7 fields last seen clear in July whose radar canopy grew afterwards); evergreen 1.4 %
in the delta (3 of 221) and 0.7 % in the fourth region (6 of 853): tree lines beside paddies whose
5 x 5 radar box floods and whose clear views once read 0.36-0.45 (first map and stage 5: 0 %).
The 4-pixel minimum mapping unit then absorbs isolated tree pixels inside rice fields (17 % of
the delta evergreen pixels, as in the first map). Region N is 1.5 points under the first map at
field level: its remaining misses are fields whose water is shallower than the thresholds
(VH -17.5 to -18.5 dB, 3-4 dB drop); accepting them would also accept dark dry soils in the
dry-zone AOIs, so they stay in class 3.

**Reviewed fields** (432 fields judged by eye in 20 AOIs, final field labels): of the fields the
reviewers called right, 197 of 216 decidable keep a label that agrees with them (first map 213); of
the fields called wrong, 128 of 148 now agree (first map 38); of the uncertain ones 21 of 34 (14).
Agreement over the right and wrong fields: 89 % against 69 % for the first map. The last two
rule changes (flood pick among all passes, no other-polarisation test) moved 12 reviewed fields:
4 towards the reviewers, 5 against them (a 5-acre polygon over trees and paddies labelled by
plurality, a 0.3-acre tree strip, and three "non-paddy / dry-land / stream bed" fields that carry
a -19.6 to -22 dB flood on 3-4 passes and a full optical cycle, which the data cannot refuse).

**Double-crop check** (aoi110, the AOI the round-2 review found worse): 6 of the 7 fields named
by the reviewer are delivered (the seventh has a 3.3 dB water drop, too weak); the AOI gained
65 ac against stage 5 and no AOI lost delivered area.

**Refined delineation**: 428,349 polygons -> 422,763 fields; 123,279 attribute acres -> 112,618
UTM acres -> 101,204 refined field acres (overlaps and tails removed, 597 ac of monster outlines
dropped and 193 ac cut, 232 ac of strips flagged).
