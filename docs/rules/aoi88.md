# aoi88: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[88]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_LOW_SHARE` = `0.05`
- `ANY_WATER_TRANSPLANTED` = `True`
- `SIEVE_ACRES` = `0.15`
- `YOUNG_NEEDS_AGE` = `True`

## History





### 2026-10-06 06:57:00 - rule set chosen

`aoi28` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi28 | 36.5 | 42.5 |
| aoi28+long_flood | 36.5 | 42.5 |
| aoi28+young_while_radar_low | 36.5 | 42.5 |
| aoi28+any_water_transplanted | 36.5 | 42.5 |
| aoi125 | 36.0 | 42.0 |
| aoi160 | 36.0 | 42.0 |
| aoi28+sowing_from_radar | 36.0 | 42.0 |
| aoi39 | 35.0 | 60.0 |
| aoi83 | 28.5 | 41.0 |
| aoi13 | 28.0 | 50.5 |
| aoi20 | 27.0 | 41.0 |
| aoi118 | 25.5 | 42.5 |
| aoi72 | 25.0 | 52.0 |
| aoi116 | 23.0 | 50.0 |
| aoi33 | 23.0 | 47.5 |

### 2026-10-06 06:58:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi88/ (24 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-26) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 544 rice fields, 0.0 of 366.91 ac (none).

### 2026-10-08 11:11:00 - v2 rules (weak AOI)

`aoi39` -> `aoi88_fields_v2.gpkg` (delivered files unchanged)

Why: rice vs not right on reviewed fields 57.0 -> 71.5 % (held-out half 61.0 -> 69.5 %); user, 7 Oct 2026

### 2026-10-08 16:31:00 - v3 rules (weak AOI)

`aoi39+harvest_not_if_radar_top+rice_needs_empty_field` -> `aoi88_fields_v3.gpkg` (delivered files unchanged)

Why: rice vs not right on reviewed fields 71.5 -> 81.5 % (held-out half 69.5 -> 79.0 %, vs v2)
