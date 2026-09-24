# Standing monsoon rice map — methods note

**What the map shows.** For every AOI, one GeoTIFF (10 m pixels) with the state of each pixel on the
map date (the last five-day window of the series, written in the file name and in the file's
`map_date` tag). Class codes, also stored as `class_*` tags and as a colour table in each file:

| code | class | meaning |
|---|---|---|
| 1 | rice, standing, water confirmed | a rice-like crop cycle standing on the map date, whose transplanting water was confirmed by radar |
| 3 | rice-like, water not confirmed | a rice-like cycle standing on the map date, but the radar saw no transplanting water |
| 2 | young | a crop has started but has no full canopy yet on the map date |
| 4 | harvested | a rice-like cycle already cut before the map date |
| 5 | rice-like curve, never bare | the optical curve looks like rice but the radar shows trees or buildings all season: not rice |
| 0 | not rice | no rice-like cycle this monsoon (water, trees, settlements, bare ground, other crops) |
| 255 | no data | outside the AOI or never observed |

The delivered rice is class **1**. Class 3 is reported separately: it has the optical shape of rice
but no radar evidence of the water rice is transplanted into; a field check is needed to name it.
Classes 2 and 4 are reported so that a later map, with newer images, can pick up the young crop.

**Data.** Sentinel-2 surface reflectance, every date from September of the previous year to the map
date, with the product's own cloud flags; Sentinel-1 radar (VV and VH, two viewing geometries per
AOI), every pass of the monsoon season.

**How a pixel is decided.**

1. *Optical crop cycle.* NDVI is composited to five-day windows (the clearest value in each), and a
   smooth curve is fitted that follows the upper envelope of the observations, so haze and thin
   cloud cannot pull it down. A pixel is rice-like when, within the last 110 days, the field was low
   (NDVI at most 0.40) for at least 40 days, then climbed by at least 0.30 to a canopy of at least
   0.50, and is still standing on the map date.
2. *Water.* Rice is transplanted into standing water, which shows in radar as a sharp fall in
   backscatter. The water is confirmed when the radar falls at least 3 dB below the field's own dry
   level around the optical trough, or when, anywhere in the bare period before the canopy climb,
   a monsoon pass falls at least 4 dB below the weeks before it, is dark enough to be water, falls
   on a field without canopy, and is supported by a second pass.
3. *Never bare.* Where the radar shows a canopy or buildings throughout, the low optical values were
   haze over a long cloud gap, and the pixel is not rice (class 5).

**Finishing.** Where the user reviewed the evidence and decided that a whole class in an AOI is
the same crop as its confirmed rice, that class is relabelled (recorded with its reason). Finally,
patches smaller than 4 pixels (about 0.1 acre, smaller than 90 % of the surveyed fields) take the
class that surrounds them.

**Fields.** Every delineated field polygon (traced on high-resolution imagery) carries one label:
the class holding most of its pixels (`label`, `class_name`), with `rice_share` and
`unconfirmed_share` so another threshold can be applied, and `pixels` (0 for fields smaller than one
pixel, labelled from the pixel under them). One GeoPackage per AOI in `fields/`;
`field_acres_by_class.csv` gives, per AOI, the acres of each class on the pixel map and on the
field-labelled map inside the AOI (field polygons can extend past the AOI edge; their full area is
in `area_acres`).

**Accuracy.** Checked against the field plots reported as standing rice: 89-95 % of plot pixels in
the three established regions are mapped as rice with water confirmed. Areas known not to be rice
(evergreen cover, permanent water, bare and built ground, fields cut before the map date) are mapped
as rice in 0 % of pixels, and an AOI without a monsoon crop has 0 acres of rice. Dates of the crop
events are accurate to about one to two weeks. Where plots were transplanted late (late August) the
crop is still young on the map date and is class 2, not rice.

**Areas.** All areas are in acres, from the 10 m pixels (one pixel = 0.0247 acre);
`acres_by_class.csv` lists every class per AOI and the totals.
