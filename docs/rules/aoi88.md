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
