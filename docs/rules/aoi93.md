# aoi93: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[93]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_LOW_SHARE` = `0.05`
- `ANY_WATER_TRANSPLANTED` = `True`
- `SIEVE_ACRES` = `0.15`
- `SOWING_FROM_RADAR` = `True`
- `YOUNG_NEEDS_AGE` = `True`

## History



### 2026-10-06 07:31:00 - rule set chosen

`aoi28+sowing_from_radar` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi28 | 70.0 | 77.0 |
| aoi28+sowing_from_radar | 70.0 | 77.5 |
| aoi28+young_while_radar_low | 70.0 | 77.0 |
| aoi28+any_water_transplanted | 70.0 | 77.0 |
| aoi13 | 69.5 | 78.5 |
| aoi28+long_flood | 59.5 | 76.0 |
| aoi118 | 58.0 | 79.0 |
| aoi116 | 56.5 | 78.5 |
| aoi72 | 54.5 | 76.0 |
| aoi39 | 46.5 | 53.5 |
| aoi83 | 45.5 | 79.0 |
| aoi160 | 40.5 | 54.5 |
| aoi125 | 40.5 | 54.5 |
| aoi33 | 17.5 | 79.0 |
| aoi20 | 17.5 | 75.5 |

### 2026-10-06 07:31:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi93/ (18 S2 dates >= 80 % clear)
