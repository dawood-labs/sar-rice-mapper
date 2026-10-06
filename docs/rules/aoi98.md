# aoi98: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[98]` (code comments name the pixel or field behind each).

## Switches now

- `ANY_WATER_TRANSPLANTED` = `True`
- `SIEVE_ACRES` = `0.15`
- `YOUNG_WHILE_RADAR_LOW` = `True`

## History



### 2026-10-06 07:02:00 - rule set chosen

`aoi125` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi125+young_while_radar_low | 62.5 | 77.0 |
| aoi125+any_water_transplanted | 62.5 | 77.0 |
| aoi125 | 62.5 | 77.0 |
| aoi160 | 61.0 | 75.5 |
| aoi39 | 60.5 | 76.5 |
| aoi125+long_flood | 60.0 | 77.0 |
| aoi28 | 60.0 | 77.0 |
| aoi83 | 57.0 | 76.5 |
| aoi13 | 55.0 | 77.5 |
| aoi125+sowing_from_radar | 54.5 | 76.0 |
| aoi118 | 51.5 | 78.0 |
| aoi72 | 49.5 | 76.5 |
| aoi116 | 46.5 | 78.0 |
| aoi20 | 14.0 | 75.5 |
| aoi33 | 13.5 | 75.0 |

### 2026-10-06 07:02:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi98/ (15 S2 dates >= 80 % clear)
