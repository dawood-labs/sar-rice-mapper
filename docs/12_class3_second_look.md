# 12 — Class 3, second look: every check possible without field data

**Question.** After the v2 water test (docs/11) about 12,000 acres stayed in class 3: a standing crop
whose optical curve looks like rice, but with no transplanting water in the radar. The AOIs were
chosen as rice areas and rice is always transplanted into water, so the water detection was
suspected first once more. Code: `analysis/class3_audit.py`, `analysis/external_checks.py`,
`analysis/water_investigation.py` (`gallery`), `analysis/rice_similarity.py`.

## Checks and results

Acre-weighted over all 132 AOIs; the null is the same test on pixels known not to be flooded
paddies (evergreen, bare/built, fields cut before the map date) with random dates.

| Check | Class 1 (rice) | Class 3 | Null: cut fields | Null: evergreen |
|---|---|---|---|---|
| flood test on 3 x 3 radar means | 79 % | 8 % | 12 % | 0.1 % |
| flood test on single pixels | 82 % | 20 % | 15 % | 0.5 % |
| two dark monsoon passes (VH <= -20) while bare, no drop needed | 74 % | 11 % | 7 % | 0 % |
| optical water (NDWI > 0 or LSWI > NDVI, clear obs) | 34 % | 9 % | 9 % | 0.1 % |

* The optical water test also fired on 64 % of bare/built pixels; it is not used.
* On class 3 the extra tests fire at about their null rate; on class 1 they fire at 74-82 %. No hidden
  water was found in class 3 by a smaller radar window, by darkness without a drop, or optically.
* **Rainfall** (GPM IMERG, May-August): 1,050-1,290 mm in the AOIs with the most class 3 against
  1,800-2,350 mm in the delta. Enough for rainfed paddies to hold water; not a drought year there.
* **ALOS-2 PALSAR-2 L-band** (ScanSAR, read-only sample, 5 dates): where class 1 is flooded the L-band
  is dark too (e.g. HH -20.7 / HV -25.4 dB in June in a delta AOI); class 3 of the same AOI never is
  (HH -9 to -14). An independent sensor agrees with the C-band.
* **Galleries** (the user's method: many fields, not one or two): 20 field-interior blocks per class
  in ten AOIs, NDVI and VV/VH of every track side by side.
  - In every AOI but one, the class-1 blocks show the paddy signature (VH falling to -20 to -26 dB for
    weeks, then the canopy) and the class-3 blocks never do. Instead the class-3 radar jumps at the
    first rains (late May; VV to -3 to -6 dB, VH to -12 to -15, the signature of ploughed wet soil
    and then a crop on it) and stays bright all monsoon.
  - Common class-3 optical patterns: a crop sown with the first rains (May-July) and a second crop
    from August, the gap between them read as a "trough"; troughs invented by the curve fit across a
    three-month cloud gap (fitted NDVI as low as -0.25 with no observation there); a summer-irrigated
    rice (flooded in April) followed by green regrowth in September without new water.
  - **One AOI is different:** its class-3 blocks have the same curves as its class-1 blocks, including
    the June flood, only a little shallower (VH -18 to -19 instead of -20.5). There class 3 is the same
    crop as the confirmed rice and missed the threshold.
* **Field context:** 1,707 acres of class 3 have at least half of their 9 x 9 neighbourhood in class 1
  (edges and weak-signal parts of confirmed rice fields).
* **Same-crop similarity** (`rice_similarity`): not discriminative (22-41 % of other crops also came
  out "like rice"); not used.

## Conclusion

Most of class 3 is not paddy rice missed by the radar: it is ground on which the radar sees the whole
season clearly, including the floods of the neighbouring rice, and sees none. Its optical rule-pass
comes from rainfed dry-land cropping (often two crops) and from cloud-gap artefacts. What is
recoverable, and needs the user's verdict before any relabel:

1. the AOI whose class 3 matches its class 1 (about 470 acres);
2. class 3 inside confirmed rice fields (about 1,700 acres), best settled per field with the
   delineated field polygons (phase 7);
3. summer-rice fields with September regrowth: rice or not depends on whether ratoon or regrowth
   counts as monsoon rice.

Also found: one Sentinel-1 date on one descending track is anomalously dark over several AOIs in
mid-June (VH -26 to -28 dB on otherwise bright ground); the v2 support condition already stops it
from counting as a flood.
