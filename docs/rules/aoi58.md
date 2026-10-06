# aoi58: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[58]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_WATER` = `True`
- `AGE_LOW_SHARE` = `0.05`
- `ANY_WATER_TRANSPLANTED` = `True`
- `SIEVE_ACRES` = `0.15`
- `SOWING_FROM_RADAR` = `True`
- `TREE_NEEDS_NO_WATER` = `True`
- `WATER_BOTTOM` = `0.3`
- `WATER_END_TRACKS` = `earliest_unless_wet_view`
- `WATER_FALL_LOOKBACK_DAYS` = `45`
- `WATER_FALL_POLS` = `VV`
- `WATER_SPELL_DAYS` = `30`
- `YOUNG_AFTER_LAST_WATER` = `True`
- `YOUNG_NEEDS_AGE` = `True`
- `YOUNG_WHILE_RADAR_LOW` = `True`

## History



### 2026-10-06 07:17:00 - rule set chosen

`aoi13` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi13+young_while_radar_low | 69.0 | 89.5 |
| aoi13+long_flood | 69.0 | 89.5 |
| aoi13+sowing_from_radar | 69.0 | 89.5 |
| aoi13+any_water_transplanted | 69.0 | 89.5 |
| aoi13 | 69.0 | 89.5 |
| aoi83 | 67.0 | 87.5 |
| aoi39 | 66.5 | 91.5 |
| aoi160 | 58.5 | 90.0 |
| aoi28 | 58.5 | 90.0 |
| aoi125 | 58.5 | 90.0 |
| aoi72 | 38.5 | 90.5 |
| aoi116 | 38.0 | 90.0 |
| aoi118 | 38.0 | 90.0 |
| aoi33 | 14.5 | 94.5 |
| aoi20 | 14.5 | 90.0 |

### 2026-10-06 07:18:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi58/ (33 S2 dates >= 80 % clear)
