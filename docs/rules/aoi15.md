# aoi15: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[15]` (code comments name the pixel or field behind each).

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



### 2026-10-06 13:12:00 - rule set chosen

`aoi13` (switches below)

Why: best of 18 sets / switches on 120 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi13+long_flood | 64.2 | 65.8 |
| aoi13 | 64.2 | 65.8 |
| aoi13+young_while_radar_low | 64.2 | 65.8 |
| aoi13+any_water_transplanted | 64.2 | 65.8 |
| aoi13+sowing_from_radar | 64.2 | 65.8 |
| aoi83 | 63.3 | 67.5 |
| aoi116 | 62.5 | 66.7 |
| aoi72 | 60.0 | 66.7 |
| aoi118 | 60.0 | 67.5 |
| aoi39 | 45.0 | 58.3 |
| aoi63 | 36.7 | 44.2 |
| aoi160 | 34.2 | 39.2 |
| aoi28 | 34.2 | 40.0 |
| aoi125 | 34.2 | 39.2 |
| universal | 34.2 | 35.8 |
| aoi13+universal | 33.3 | 35.0 |
| aoi20 | 24.2 | 41.7 |
| aoi33 | 15.0 | 33.3 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-08-17) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 11 of 58 rice fields, 7.69 of 29.03 ac (young rice 7.69 ac).

### 2026-10-06 13:12:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi15/ (15 S2 dates >= 80 % clear)
