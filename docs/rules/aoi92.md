# aoi92: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[92]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_LOW_SHARE` = `0.05`
- `ANY_WATER_TRANSPLANTED` = `True`
- `SIEVE_ACRES` = `0.15`
- `YOUNG_NEEDS_AGE` = `True`

## History



### 2026-10-06 07:27:00 - rule set chosen

`aoi28` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi28 | 61.5 | 72.5 |
| aoi28+any_water_transplanted | 61.5 | 72.5 |
| aoi28+sowing_from_radar | 59.5 | 71.5 |
| aoi28+long_flood | 59.5 | 71.0 |
| aoi28+young_while_radar_low | 59.5 | 73.0 |
| aoi13 | 56.5 | 68.0 |
| aoi72 | 48.5 | 73.5 |
| aoi39 | 46.5 | 60.0 |
| aoi116 | 45.5 | 69.0 |
| aoi118 | 45.0 | 70.0 |
| aoi83 | 44.0 | 69.5 |
| aoi160 | 36.5 | 55.0 |
| aoi125 | 36.5 | 56.5 |
| aoi20 | 20.5 | 76.5 |
| aoi33 | 20.0 | 76.0 |

### 2026-10-06 07:28:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi92/ (15 S2 dates >= 80 % clear)
