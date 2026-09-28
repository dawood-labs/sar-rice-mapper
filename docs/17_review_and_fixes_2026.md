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
* **A 140-day lookback as a fallback** (28 September, user review of aoi160): the 110-day window
  missed crops sown in the first half of May (their bare period lies before 3 June); the wider
  window is tried only where the 110-day one found no cycle at all, because applied everywhere it
  shifts the water anchor and ~3 % of the confirmed rice loses its flood.
* **Class 3 -> rice where it is the same crop** (`analysis/class3_phenology`): per AOI, the class-3
  pixels' green-up date, canopy peak, leaf water (LSWI at the peak) and fall by the map date are
  compared with those of the AOI's own confirmed rice; where they match (canopy within 0.06 NDVI /
  0.06 LSWI / 0.08 fall, green-up not more than 45 days earlier or 15 later, both looking like the
  surveyed rice) class 3 is relabelled rice through a generated override file, with
  `label_confidence = low` and a note. Why: in many AOIs rice is sown dry or into water too shallow
  for the radar; its class 3 followed exactly the calendar of the confirmed rice (aoi160: green-up
  8 vs 18 July, peak 0.80 vs 0.80, LSWI 0.35 vs 0.37) while true dry-land crops green up 20-80 days
  earlier with a lower canopy. On the surveyed plots the relabel raised region B by 3 points and
  region N by 2-3 points with no change in the negatives.
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

## Measurements (final re-run of all 132 AOIs, 28 September 2026, "stage 10")

The map of 25 September ("stage 7", pixel rule) plus the delineation hygiene of 26 September
("stage 8") and the two rules of 28 September (140-day fallback; class 3 -> rice where it is the
AOI's own crop). `report/compare/stage10_*.csv`; the earlier stages' numbers are in the log of
the plan file.

**Class acres inside the AOIs, pixel map, first map -> final map:**

| class | first map | stage 7 | final (stage 10) |
|---|---|---|---|
| rice (1) | 42,741 | 47,206 | 57,311 |
| young rice (6) | 6,072 | 7,419 | 7,477 |
| **delivered (1 + 6)** | **48,812** | **54,625** | **64,788** |
| rice-like, water not confirmed (3) | 11,454 | 11,870 | 8,095 |
| harvested (4) | 7,020 | 628 | 953 |
| young (2) | 3,630 | 2,878 | 3,404 |
| never bare (5) | 3,998 | 2,476 | 3,369 |
| flooded, not yet green (7) | - | 5,901 | 5,642 |
| cut crop, water not confirmed (8) | - | 642 | 879 |
| not rice (0) | 38,491 | 34,386 | 25,172 |

Field labels inside the AOIs (refined delineation): rice 58,975 ac, young rice 6,969 ac
(delivered 65,944; first map 49,586; stage 8 55,869), class 3 7,743, flooded not green 5,918,
young 2,606, never bare 2,493, harvested 811, cut crop 1,731, not rice 26,176. Of the delivered
rice, 11,767 ac come from the class-3 relabel in 59 AOIs (`label_confidence = low`, note "water
not seen; same crop cycle as the area's confirmed rice"); the 140-day fallback alone added 424 ac
in the other AOIs. The largest changes against stage 8: one AOI without plots 350 -> 2,123 ac
(its class 3 greens up 25 days before its confirmed rice with the same canopy), aoi160 714 -> 1,352,
then 500-550 ac in three AOIs. The dry-zone AOIs (class 3 greening 55-80 days before the rice, or
with a lower canopy) were not relabelled and keep their class 3.

**Surveyed plots** (share of plot-interior pixels delivered as rice, three established regions):

| | delta | region B | region N |
|---|---|---|---|
| first map, field labels | 96.9 | 94.8 | 98.3 |
| stage 7, field labels | 99.0 | 98.0 | 96.8 |
| final, pixel map | 98.6 | 99.5 | 96.4 |
| final, field labels | 99.3 | 99.4 | 98.4 |

Negatives (final, field labels): "cut before the map date" 6.7 % (15 px) / 75 % (12 px) / 2.8 %;
evergreen 24.0 / 6.4 / 12.1 / 2.4 % (pixel map 18.6 / 6.4 / 3.0 / 0.8): the relabel added a
handful of tree-line pixels in two regions (1 and 10 pixels), the rest is the 4-pixel sieve
filling isolated tree pixels inside rice fields, as before. Water and bare / built: 0 %.

**Reviewed fields** (432 fields judged by eye in 20 AOIs, delivered polygons only): right
158 of 180 (first map 177), wrong 100 of 127 (32), uncertain 19 of 27 (10). The relabel costs
five "right" and five "wrong" agreements against stage 8: reviewers had judged those class-3
fields "dry-land" by eye; the phenology test puts them with the AOI's rice, and the user's
decision was to count them. Geometry audit of the 132 delivered fields files unchanged
(0 invalid, 4 overlapping pairs over one pixel).

**Refined delineation**: 428,349 polygons -> 422,763 fields; 123,279 attribute acres -> 112,618
UTM acres -> 101,204 refined field acres (overlaps and tails removed, 597 ac of monster outlines
dropped and 193 ac cut, 232 ac of strips flagged).
