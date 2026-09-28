# Standing monsoon rice map: what each class means (one-page guide)

*Map date 21 September 2026. Delivered as rice: classes 1 and 6. Every other class is reported so
the reader knows what the ground was doing, but it is not counted as rice.*

## Three questions, asked of every 10 m pixel

1. **Did a crop cycle happen?** (Sentinel-2, a cloud-free value every 5 days). In the last 110 days
   the field was low or bare for at least 40 days (NDVI at or below 0.40), then greened up by at
   least 0.30 to a canopy of at least 0.50, and is still standing on the map date.
2. **Was the field flooded when the crop was planted?** (Sentinel-1 radar, which sees through
   cloud). Standing water makes the radar signal fall at least 4 dB below the field's own level of
   the previous weeks, to a level only water reaches (VH at or below -19 dB), on more than one pass.
   Rice is transplanted into standing water; other crops are not. Water is what separates rice from
   everything else with a similar green-up.
3. **What is the field doing on the map date?** Full canopy, still small, or already cut.

Fields (the delineated polygons) take the class of the majority of their pixels.

## The classes

| code | name | what it means | counted as rice? |
|---|---|---|---|
| 1 | rice, standing | crop cycle + flood at planting + full canopy standing on the map date | **yes** |
| 6 | young rice | flood at planting + a crop growing but not yet a full canopy (planted late) | **yes** |
| 2 | young | a small crop growing, no flood found: may be rice, may be another crop | no, reported |
| 7 | flooded, not yet green | flooded in the last 60 days, no green-up yet: next month's rice, not standing rice today | no, reported |
| 3 | rice-like, water not confirmed | a rice-like crop cycle but the radar never saw the field flooded: mostly rain-fed dry-land crops; some shallow-water rice | no, reported |
| 4 | harvested | crop cycle + flood, but cut before the map date | no, reported |
| 8 | cut crop, water not confirmed | a crop cycle that was cut, and no flood | no, reported |
| 5 | never bare | the optical series hinted at a cycle, but the radar shows the ground was never bare or smooth all season: trees, gardens, houses | no |
| 0 | not rice | no rice-like cycle: water, built-up, forest, other land uses | no |
| 255 | no data | outside the area or no usable observations | - |

In one sentence: **rice = a bare field, then water, then a crop, standing today; young rice = the
same with the crop still small. Every other class names which of those four things is missing.**

## What a reader should know

* Accuracy: on 2,950 surveyed rice plots in three regions, 97-99 % of plot-interior pixels are
  delivered as rice; areas known not to be rice (evergreen cover, water, bare and built ground,
  fields cut before the map date) are called rice in 0-1 % of pixels by the rule. Dates of the crop
  events are accurate to one to two weeks.
* Class 3 is the largest grey area (about 11,000 acres over the 132 areas). The decision was: no
  radar evidence of water, no rice. A field check can move fields from class 3 to rice.
* Plots transplanted in late August are young on the map date (class 6 where the canopy already
  shows, class 7 while still under water); they become rice on a later map.
* Each field carries `label_confidence` (high / mixed / low) and, where relevant, a note (a late
  flood after an earlier crop; no clear optical view in the last 45 days; rice by an area-level
  relabel), so a reviewer knows which fields to look at first.
