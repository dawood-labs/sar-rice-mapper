# 14 — One label per delineated field (phase 7)

Code: `analysis/field_rice.py`; input: the SAMGeo field delineation (428,349 polygons over all 132
AOIs, median 0.10 acre, a quarter smaller than one 10 m pixel); map: `<aoi>_monsoon2026_final.tif`
(docs/13).

## Method

Fields are burnt onto each AOI's 10 m grid (largest first, so where tracings overlap the finer one
keeps the pixel); the class pixels inside each field are counted; the field takes the class with the
most pixels. A field that covers no pixel centre (13 %, all tiny) takes the class of the pixel under
its representative point and is flagged `pixels = 0`. `rice_share` and `unconfirmed_share` are kept.
87-92 % of each AOI's pixels fall inside a field; the rest keep their own class.

## Validation (same plots and negatives as docs/10-13)

| Share called rice | pixel map | field labels |
|---|---|---|
| plot interiors, delta | 95.3 % | 96.6 % |
| plot interiors, region B | 89.9 % | 91.9 % |
| plot interiors, capital area | 93.5 % | 96.5 % |
| plot edges, region B | 83.9 % | 86.7 % |
| plot edges, capital area | 92.9 % | 96.1 % |
| evergreen negatives | 0.3 % | 0.6 % |
| fields cut before the map date | 0.0 % | 0.1 % |

Class 3 on the plots falls to 0.0-2.1 %. Region Y is still young and unchanged.

## Effect inside the AOIs (acres)

| class | pixel map | field labels |
|---|---|---|
| rice, water confirmed | 42,690 | 44,146 |
| rice-like, water not confirmed | 11,332 | 10,263 |
| young | 10,283 | 9,328 |
| harvested | 6,996 | 6,711 |
| never bare | 3,952 | 3,625 |
| not rice | 38,174 | 39,355 |

Most of the class 3 that sat on the edges of confirmed rice fields (docs/12) is resolved by the
field label; what remains in class 3 are whole fields without water.

```bash
python -m sar_pipeline.analysis.field_rice      # fields/<aoi>_fields_monsoon2026.gpkg + field_acres_by_class.csv
python -m sar_pipeline.delivery                 # adds fields/ and the field acres to the package
```
