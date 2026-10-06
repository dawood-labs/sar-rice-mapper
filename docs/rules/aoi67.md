# aoi67: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[67]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_LOW_VIEW` = `True`
- `AGE_LOW_SHARE` = `0.05`
- `LAST_CROP_ONLY` = `True`
- `SIEVE_ACRES` = `0.15`
- `YOUNG_NEEDS_AGE` = `True`

## History



### 2026-10-06 05:41:00 - rule set chosen

`aoi20` (switches below)

Why: best of 15 sets / switches on 23 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi20 | 26.1 | 39.1 |
| aoi72 | 26.1 | 39.1 |
| aoi33 | 26.1 | 39.1 |
| aoi116 | 26.1 | 39.1 |
| aoi39 | 26.1 | 39.1 |
| aoi20+sowing_from_radar | 26.1 | 39.1 |
| aoi118 | 26.1 | 39.1 |
| aoi20+long_flood | 26.1 | 39.1 |
| aoi20+young_while_radar_low | 26.1 | 39.1 |
| aoi83 | 17.4 | 43.5 |
| aoi28 | 13.0 | 39.1 |
| aoi160 | 13.0 | 39.1 |
| aoi13 | 13.0 | 39.1 |
| aoi125 | 13.0 | 39.1 |
| aoi20+any_water_transplanted | 13.0 | 39.1 |

### 2026-10-06 05:41:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi67/ (17 S2 dates >= 80 % clear)
