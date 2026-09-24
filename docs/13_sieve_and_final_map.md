# 13 — The delivered map: user relabels and the minimum mapping unit

Code: `analysis/finalize.py` (order: rule -> relabels -> sieve), `analysis/sieve.py`
(`sieve_classes`, `evaluate`), relabels in the local `config/class_overrides_monsoon2026.yaml`.

## Relabels (decided by the user after docs/12)

* In one AOI the class-3 fields had the same NDVI and radar curves as its confirmed rice, including
  the June flood, only shallower: class 3 -> rice there (about 470 acres).
* Green regrowth after summer rice is weeds, not rice (user verdict): it stays out of the rice class.

## Minimum mapping unit from the field plots

Plot sizes (3,533 surveyed plots): median 0.31 acre (about 12 pixels); 10 % are under 0.09 acre
(about 4 pixels), 5 % under 0.07 acre (about 3 pixels). A sieve larger than the smallest real fields
could delete real fields, so the unit was bounded by the plots and then scored:

| sieve (pixels) | 1 | 3 | 4 | 6 | 10 | 21 | 40 |
|---|---|---|---|---|---|---|---|
| plot interiors called rice, delta | 94.8 | 95.2 | 95.3 | 95.7 | 96.4 | 97.2 | 98.0 |
| region B | 89.6 | 89.7 | 89.9 | 90.3 | 90.6 | 93.1 | 94.7 |
| capital area | 92.6 | 93.3 | 93.5 | 93.9 | 94.7 | 95.8 | 96.8 |
| plot edges called rice, region B | 82.5 | 83.4 | 83.9 | 84.7 | 85.9 | 87.3 | 90.0 |
| evergreen negatives called rice | 0.0 | 0.1 | 0.3 | 0.4 | 0.8 | 1.2 | 2.5 |

Larger sieves raise plot recall (plots sit inside larger rice areas) but also turn tree specks into
rice, and anything above the plots' small end can erase real small fields. **4 pixels (~0.1 acre,
the plots' 10th percentile)** was chosen: plot recall +0.3 to +1.4 points, class 3 on plots down
0.1-0.7 points, evergreen false positives 0.3 %.

## Effect (all 132 AOIs, acres)

| class | rule (v2) | final |
|---|---|---|
| rice, water confirmed | 41,798 | 42,690 |
| rice-like, water not confirmed | 12,082 | 11,332 |
| young | 10,966 | 10,283 |
| harvested | 7,221 | 6,996 |
| never bare | 4,036 | 3,952 |
| not rice | 37,324 | 38,174 |

Of the +892 acres of rice, 478 come from the relabel and 415 from the sieve. The sieve alone does
not resolve class 3 along the edges of rice fields (about 1,700 acres, docs/12): those patches are
larger than any sieve that is safe for small fields; they are left to the field polygons (phase 7).

```bash
python -m sar_pipeline.analysis.finalize          # writes <aoi>_monsoon2026_final.tif + acres CSV
python -m sar_pipeline.delivery                   # packages the final maps
```
