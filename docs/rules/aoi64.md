# aoi64: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[64]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_LOW_VIEW` = `True`
- `AGE_LOW_SHARE` = `0.05`
- `LAST_CROP_ONLY` = `True`
- `SIEVE_ACRES` = `0.15`
- `SOWING_FROM_RADAR` = `True`
- `YOUNG_NEEDS_AGE` = `True`

## History





### 2026-10-06 05:41:00 - rule set chosen

`aoi20` (switches below)

Why: best of 15 sets / switches on 21 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi20 | 42.9 | 42.9 |
| aoi20+sowing_from_radar | 42.9 | 42.9 |
| aoi20+long_flood | 42.9 | 42.9 |
| aoi20+any_water_transplanted | 42.9 | 42.9 |
| aoi20+young_while_radar_low | 33.3 | 33.3 |
| aoi160 | 28.6 | 28.6 |
| aoi28 | 28.6 | 28.6 |
| aoi125 | 28.6 | 28.6 |
| aoi39 | 14.3 | 14.3 |
| aoi116 | 14.3 | 14.3 |
| aoi72 | 14.3 | 14.3 |
| aoi118 | 14.3 | 14.3 |
| aoi13 | 14.3 | 14.3 |
| aoi83 | 14.3 | 14.3 |
| aoi33 | 0.0 | 0.0 |

### 2026-10-06 05:41:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi64/ (18 S2 dates >= 80 % clear)

### 2026-10-06 06:19:00 - rule set chosen

`aoi20+sowing_from_radar` (switches below)

Why: best of 15 sets / switches on 82 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi20 | 36.6 | 41.5 |
| aoi20+sowing_from_radar | 36.6 | 42.7 |
| aoi20+long_flood | 36.6 | 41.5 |
| aoi20+any_water_transplanted | 35.4 | 41.5 |
| aoi125 | 35.4 | 36.6 |
| aoi160 | 35.4 | 36.6 |
| aoi28 | 35.4 | 36.6 |
| aoi20+young_while_radar_low | 32.9 | 37.8 |
| aoi83 | 30.5 | 31.7 |
| aoi13 | 30.5 | 31.7 |
| aoi39 | 26.8 | 32.9 |
| aoi72 | 25.6 | 31.7 |
| aoi116 | 25.6 | 31.7 |
| aoi118 | 25.6 | 31.7 |
| aoi33 | 4.9 | 11.0 |

### 2026-10-06 06:19:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi64/ (18 S2 dates >= 80 % clear)
