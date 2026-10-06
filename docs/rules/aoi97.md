# aoi97: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[97]` (code comments name the pixel or field behind each).

## Switches now

- `ANY_WATER_TRANSPLANTED` = `True`
- `SIEVE_ACRES` = `0.15`
- `YOUNG_WHILE_RADAR_LOW` = `True`

## History



### 2026-10-06 06:59:00 - rule set chosen

`aoi125` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi125+young_while_radar_low | 55.0 | 66.5 |
| aoi125+any_water_transplanted | 55.0 | 66.5 |
| aoi125 | 55.0 | 66.5 |
| aoi28 | 54.5 | 67.5 |
| aoi160 | 54.5 | 66.5 |
| aoi13 | 54.0 | 68.5 |
| aoi125+long_flood | 53.0 | 66.0 |
| aoi83 | 52.0 | 69.5 |
| aoi39 | 50.0 | 64.0 |
| aoi118 | 47.5 | 69.0 |
| aoi125+sowing_from_radar | 47.0 | 65.0 |
| aoi116 | 46.5 | 68.5 |
| aoi72 | 44.0 | 65.5 |
| aoi33 | 21.0 | 70.0 |
| aoi20 | 21.0 | 70.0 |

### 2026-10-06 06:59:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi97/ (13 S2 dates >= 80 % clear)
