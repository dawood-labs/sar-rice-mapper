# aoi2: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[2]` (code comments name the pixel or field behind each).

## Switches now

- `ANY_WATER_TRANSPLANTED` = `True`
- `SIEVE_ACRES` = `0.15`

## History



### 2026-10-06 06:18:00 - rule set chosen

`aoi160` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi160 | 57.0 | 62.0 |
| aoi160+any_water_transplanted | 57.0 | 62.0 |
| aoi125 | 57.0 | 62.0 |
| aoi160+young_while_radar_low | 57.0 | 62.0 |
| aoi160+long_flood | 54.5 | 62.0 |
| aoi160+sowing_from_radar | 52.0 | 62.0 |
| aoi13 | 46.0 | 50.5 |
| aoi116 | 45.0 | 50.0 |
| aoi39 | 41.0 | 67.0 |
| aoi20 | 38.5 | 66.0 |
| aoi28 | 35.5 | 54.0 |
| aoi33 | 32.0 | 61.0 |
| aoi118 | 31.5 | 51.5 |
| aoi72 | 30.5 | 52.5 |
| aoi83 | 30.0 | 48.5 |

### 2026-10-06 06:19:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi2/ (15 S2 dates >= 80 % clear)
