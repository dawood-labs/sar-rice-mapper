# Standing monsoon rice map: what each class means

*A guide for anyone who opens the map, no remote-sensing background needed. It describes the rules of the next
delivery (fix round 3). Numbers marked **[to refresh]** are filled in after the full re-run.*

**Delivered as rice: classes 1 (rice) and 6 (young rice).** Every other class is reported so the reader knows
what the ground was doing, but it is not counted as rice.

---

## 1. What the map shows

* **One question per 10 m pixel:** *is there monsoon rice standing on this spot on the map date?*
  "Monsoon rice" means rice sown in May 2026 or later. The dry-season crop before it is not counted.
* **The map date** is the last date the satellites could see (written in each file name).
* **Fields:** the pixels are also summarised per field outline. A field takes the class of most of its
  pixels, and carries a confidence (high / mixed / low) and, where useful, a short note.

## 2. The two satellites and why both are needed

| | Radar (Sentinel-1) | Optical (Sentinel-2) |
|---|---|---|
| what it measures | how much of its own microwave signal comes back ("backscatter", in dB) | the colour of the ground; from it the greenness index NDVI (0 = bare or water, near 1 = dense green crop) |
| clouds | sees through clouds, day and night | blind under clouds and haze |
| how often | every 12 days per orbit, 2-3 orbits per area | every 5 days, but in the monsoon fewer than 1 in 10 views is clear |
| what it tells us about rice | **water** (a flooded field looks dark) and **the crop growing** (the signal climbs back) | **bare field, green-up, full canopy, harvest**, whenever the sky is clear |

Cloudy and hazy views are removed before anything is decided (cloud flag, a per-pixel cloud score, a
blue-haze test). A view whose cloud score was not produced yet is treated as "unknown", not as cloud.

## 3. How a pixel is judged: four questions

No question uses a fixed cut-off such as "4 dB" or "NDVI 0.4". Each pixel is compared with **its own history**
(its own normal level and its own day-to-day noise) and with **the other fields of the same area** (for example,
how dark this area's clearly flooded fields are on the same day).

**Q1. When was it sown?** The last time in the season (from 1 May) that the field was at its own lowest, bare
point, for at least two views in a row (one low view alone is usually haze).

**Q2. Was there water at sowing?** Rice is transplanted into standing water, and calm water makes the radar
signal drop sharply. Water is accepted when the signal falls clearly below the field's own recent level (more
than twice its normal noise) **and** is as dark as this area's clearly flooded fields on that day, confirmed by
the next pass, by another orbit, or by a clear optical view that shows open water. One orbit is enough: an orbit
that saw nothing is "unknown", not "no". Ponds (open water all year) never count as flooded fields.

**Q3. Did a crop grow after it?** Optically: greenness rose clearly above the field's bare point. With the
radar: after the water, the signal climbed back by more than twice the field's own noise, all orbits together.
Tree lines are ruled out because their "bare point" is never as low as the area's fields at sowing.

**Q4. What is it doing on the map date?** Full canopy (standing), still small (young), already cut
(harvested), or still under water with no crop seen yet.

## 4. The classes

| code | name | in plain words | how it is recognised | rice? |
|---|---|---|---|---|
| **1** | **rice** | rice standing on the map date | water at sowing (Q2) + a crop that grew (Q3) + still standing (Q4); also a crop sown before 20 July that is still small, and a crop that follows exactly the area's rice calendar where the water was not seen (see "low confidence" below) | **yes** |
| **6** | **young rice** | rice sown late, still small | same as rice, but sown **after 20 July 2026** and the canopy is not full yet | **yes** |
| 7 | flooded, not yet green | water in the field, no crop visible yet | water at sowing, but neither the radar nor the optical shows a crop since; also when the latest clear view (taken after the flood) still shows open water and the radar has not risen after it | no, reported |
| 3 | rice-like, water not confirmed | looks like a rice crop, but no water was seen, and it does not follow the area's rice calendar | Q1 and Q3 yes, Q2 no | no, reported |
| 9 | rice-like, no sign of water | follows the area's rice calendar, but the radar brightened right after sowing instead of going dark | a crop sown into dry soil (a dry-land crop, or dry-seeded rice) | no, reported |
| 4 | harvested | rice that was cut before the map date | water at sowing + a crop that grew + fallen back to bare before the map date | no, reported |
| 8 | cut crop, water not confirmed | a crop that was cut, no water seen at sowing | like 4, without the water | no, reported |
| 5 | never bare | trees, gardens, houses | the radar shows the ground was never bare or wet all season | no |
| 0 | not rice | everything else | no rice-like cycle: water bodies, built-up, forest, bare land, other crops | no |
| 2 | young (old maps only) | not used any more | since round 3 a young crop is young rice or not rice | - |
| 255 | no data | outside the area or never seen | - | - |

**In one sentence:** rice = a bare field, water at sowing, then a crop standing today; young rice = the same, sown
after 20 July; every other class says which of these is missing.

## 5. Examples (what the curves look like)

* **Rice (1):** bare in May (NDVI about 0.1), radar drops by about 10 dB in June (water), climbs back in July;
  greenness reaches 0.7-0.8 and stays high to the map date.
* **Double crop, rice (1):** a dry-season crop until June, harvest, then the field floods (radar drops), and a new
  crop grows. The monsoon crop is what the map reports.
* **Young rice (6):** water in August, radar starts to climb in September, greenness still 0.3-0.4 on the map date.
* **Flooded, not yet green (7):** radar dark since August, the last clear view in September still shows water.
* **Rice-like, no sign of water (9):** bare in May, then the radar brightens at once (no dark water phase), greenness
  climbs like the area's rice.
* **Harvested (4):** a full canopy until mid-September, then the clear view after it reads bare soil.

## 6. Reading the field file

* `label` / `class_name`: the class of most of the field's pixels.
* `label_confidence`:
  * **high**: the pixels agree and a second, field-level check agrees;
  * **mixed**: the pixels or the two checks disagree (look at the field);
  * **low**: the label rests on weaker evidence, with a note: *water not seen in the radar; same crop cycle as the
    area's confirmed rice*, *no clear optical view in the last weeks*, *a late flood after an earlier crop*.
* `is_field = false`: strips, canals and ponds kept for context, not counted as fields.
* A reader who wants only radar-confirmed rice can drop the **low**-confidence rice fields.

## 7. What a reader should know (limits)

* **Accuracy [to refresh]:** share of ground-surveyed rice plots (three regions) mapped as rice; share of known
  non-rice areas (tree cover, water, bare and built ground, cut fields) mapped as rice.
* **Dates** of sowing, water and harvest are accurate to about one to two weeks (a clear view every 5 days at best,
  a radar pass every 12 days per orbit).
* **Shallow or short water** can drain between two radar passes; such rice may land in class 3 or 9 instead of 1.
* **Class 9 and class 3** are the grey zone: dry-seeded rice and other rain-fed crops look alike from space. A field
  visit settles them.
* **Late crops:** fields sown in late August or September are young rice or flooded-not-green on this map; they
  become rice on a later map.
* **The map date matters:** a field cut a few days after the map date is still rice on this map.
