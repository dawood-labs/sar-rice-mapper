# aoi5: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[5]` (code comments name the pixel or field behind each).

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



### 2026-10-06 13:36:00 - rule set chosen

`aoi13` (switches below)

Why: best of 18 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi13+long_flood | 64.0 | 69.5 |
| aoi13 | 64.0 | 69.5 |
| aoi13+young_while_radar_low | 64.0 | 69.5 |
| aoi13+any_water_transplanted | 64.0 | 69.5 |
| aoi13+sowing_from_radar | 64.0 | 69.5 |
| aoi116 | 62.5 | 69.0 |
| aoi83 | 54.0 | 69.0 |
| aoi13+universal | 49.5 | 58.5 |
| aoi125 | 40.5 | 57.5 |
| aoi160 | 40.0 | 57.0 |
| aoi72 | 39.5 | 70.5 |
| universal | 35.0 | 57.0 |
| aoi118 | 34.0 | 70.0 |
| aoi28 | 30.5 | 59.0 |
| aoi39 | 29.5 | 77.0 |
| aoi20 | 22.5 | 64.0 |
| aoi33 | 22.0 | 63.5 |
| aoi63 | 15.0 | 59.5 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-10-01) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 473 rice fields, 0.0 of 202.5 ac (none).

### 2026-10-06 13:36:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi5/ (11 S2 dates >= 80 % clear)
