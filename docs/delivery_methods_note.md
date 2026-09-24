# Standing monsoon rice map — methods note

**What the map shows.** For every AOI, one GeoTIFF (10 m pixels) with the state of each pixel on the
map date (the last five-day window of the series, written in the file name and in the file's
`map_date` tag). Class codes, also stored as `class_*` tags and as a colour table in each file:

| code | class | meaning |
|---|---|---|
| 1 | rice, standing, water confirmed | a rice-like crop cycle standing on the map date, whose transplanting water was confirmed by radar |
| 6 | rice, standing, young | a young rice crop: transplanting water confirmed by radar and canopy already visible (NDVI at least 0.30); delivered as rice |
| 3 | rice-like, water not confirmed | a rice-like cycle standing on the map date, but the radar saw no transplanting water |
| 2 | young | a crop has started but is not yet visible as a canopy, or its water is not confirmed |
| 7 | flooded, not yet green | under water on the map date: transplanting water confirmed by radar within the last 60 days, no crop visible yet (the next map's rice) |
| 4 | harvested | a rice-like cycle with confirmed water, already cut before the map date |
| 8 | cut crop, water not confirmed | a rice-like cycle cut before the map date with no transplanting water in the radar: mostly rain-fed dry-land crops |
| 5 | rice-like curve, never bare | the optical curve looks like rice but the radar shows trees or buildings all season: not rice |
| 0 | not rice | no rice-like cycle this monsoon (water, trees, settlements, bare ground, other crops) |
| 255 | no data | outside the AOI or never observed |

The delivered rice is classes **1** and **6**. Class 3 is reported separately: it has the optical shape of rice
but no radar evidence of the water rice is transplanted into; a field check is needed to name it.
Classes 2, 4 and 7 are reported so that a later map, with newer images, can pick up the young crop.

**Data.** Sentinel-2 surface reflectance, every date from September of the previous year to the map
date, with the product's own cloud flags and the Cloud Score+ clear score; Sentinel-1 radar (VV and
VH, two viewing geometries per AOI), every pass from mid-March to the map date.

**Cloud mask.** An observation is kept when the product's opaque-cloud flag is clear and either
Cloud Score+ rates it at least 40 % clear or it is dark (near-infrared below 0.15 reflectance and
NDVI below 0.3: flooded fields and wet soil, which the cloud score mistakes for shadow), and it is
not haze: a canopy with blue reflectance above 0.09, or any blue-bright observation whose blue is at
least 0.9 times its red, is removed (haze and thin cloud are grey; soils are brown and water is
dark). The mask was chosen by scoring seven alternatives on the surveyed plots.

**How a pixel is decided.**

1. *Optical crop cycle.* NDVI is composited to five-day windows (the clearest value in each), and a
   smooth curve is fitted that follows the upper envelope of the observations (never below the
   lowest observation), so haze and thin cloud cannot pull it down. A pixel is rice-like when,
   within the last 110 days, the field was low (NDVI at most 0.40) for at least 40 days, then
   climbed by at least 0.30 to a canopy of at least 0.50, and is still standing on the map date
   (last value at least 0.40 and not more than 0.35 below the peak: a ripening crop is standing).
2. *Water.* Rice is transplanted into standing water, which shows in radar as a sharp fall in
   backscatter in both polarisations. The water is confirmed when the radar falls at least 3 dB
   below the field's own dry level around the optical trough and ends dark, or when, anywhere in
   the bare period before the canopy climb, a monsoon pass falls at least 4 dB below the weeks
   before it, is dark enough to be water (VH at or below -19 dB, or -18 dB with a larger drop seen
   on more passes), falls on a field where no canopy was observed in the weeks before, and is
   supported by a second pass in either polarisation (a very dark, very deep pass needs none).
3. *Radar-defined start.* Where cloud hid the weeks around transplanting, the fitted curve runs
   straight across the gap and shows no trough. A confirmed radar flood then stands in for it: the
   pixel is rice when a canopy of at least 0.50 follows the flood and is still standing, provided
   the field's own clear observations showed bare ground within 90 days before or 30 days after
   the flood (the radar is read over a 50 m box, so a tree line beside flooded paddies floods in
   the radar too). A young crop transplanted late is recognised by the radar canopy rising at
   least 4 dB from the water (5 dB to deliver it as young rice) when no clear view shows it yet.
4. *Never bare.* Where the radar shows a canopy or buildings throughout, the low optical values were
   haze over a long cloud gap, and the pixel is not rice (class 5). This applies to every rice-like
   class unless a flood was confirmed.

**Finishing.** Where the user reviewed the evidence and decided that a whole class in an AOI is
the same crop as its confirmed rice, that class is relabelled (recorded with its reason). Finally,
patches smaller than 4 pixels (about 0.1 acre, smaller than 90 % of the surveyed fields) take the
class that surrounds them.

**Radar quality.** Passes that moved ground which cannot change within a week (trees, buildings) by
3 dB or more, or by 2 dB in one polarisation while the other stayed flat, are treated as artefacts;
a pass broken in three AOIs of a track and low in a fifth of them is dropped on the whole track.
One such pass (11 June, one descending track, VH) was dropped everywhere.

**Fields.** The delineated field polygons (traced on high-resolution imagery) were refined before
labelling: overlaps resolved (the smaller polygon wins), outlines drawn around groups of fields cut
to their remainder or dropped, tails narrower than 6 m removed, long thin polygons (roads, canals,
tree lines) and all-season water flagged (`is_field = false`, `refine_flag` says why; they stay in
the file but carry no field acres), slivers under 4 pixels merged into a same-class neighbour, and
`area_acres` recomputed in the local UTM zone (the original attribute, kept as
`area_acres_delivered`, was computed in Web Mercator and ran about 9 % high). Every polygon keeps
its `field_id`. Each field carries one label: the class holding most of its pixels (`label`,
`class_name`), with `rice_share` and `unconfirmed_share` so another threshold can be applied,
`pixels` (0 for fields smaller than one pixel, labelled from the pixel under them), and
`label_confidence`: `high` where the same rule re-run on the field's mean curves gives the same
class, `mixed` where it does not, `low` with a `confidence_note` where the label deserves a field
check ("late flood after an earlier crop": rice flooded after 25 July on a field that was green
before, which may be a second transplanting or rain under a standing crop; "no clear view in the
last 45 days"). One GeoPackage per AOI in `fields/`; `field_acres_by_class.csv` gives, per AOI, the
acres of each class on the pixel map and on the field-labelled map inside the AOI (field polygons
can extend past the AOI edge; their full area is in `area_acres`).

**Accuracy.** TODO (stage 4): plot recall per region at pixel and field level, negatives, the
reviewed-field agreement (about 430 fields judged by eye: 90 % agreement against 71 % for the first
map). Dates of the crop events are accurate to about one to two weeks. Where plots were transplanted
late (late August) the crop is young on the map date: class 6 when the canopy is visible (optical or
radar), class 2 or 7 otherwise.

**Areas.** All areas are in acres, from the 10 m pixels (one pixel = 0.0247 acre);
`acres_by_class.csv` lists every class per AOI and the totals.
