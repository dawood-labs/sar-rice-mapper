# aoi8: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[8]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_WATER` = `True`
- `AGE_LOW_SHARE` = `0.05`
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



### 2026-10-06 14:24:00 - rule set chosen

`aoi116` (switches below)

Why: best of 18 sets / switches on 68 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi116+young_while_radar_low | 63.2 | 85.3 |
| aoi116 | 63.2 | 85.3 |
| aoi116+sowing_from_radar | 63.2 | 85.3 |
| aoi116+long_flood | 63.2 | 85.3 |
| aoi116+universal | 57.4 | 79.4 |
| aoi13 | 55.9 | 85.3 |
| aoi116+any_water_transplanted | 55.9 | 85.3 |
| aoi83 | 54.4 | 80.9 |
| universal | 51.5 | 77.9 |
| aoi72 | 50.0 | 86.8 |
| aoi118 | 48.5 | 85.3 |
| aoi125 | 47.1 | 73.5 |
| aoi160 | 44.1 | 72.1 |
| aoi33 | 35.3 | 76.5 |
| aoi20 | 35.3 | 77.9 |
| aoi28 | 32.4 | 75.0 |
| aoi39 | 26.5 | 76.5 |
| aoi63 | 16.2 | 67.6 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-08-17) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 46 rice fields, 0.0 of 15.99 ac (none).

### 2026-10-06 14:24:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi8/ (14 S2 dates >= 80 % clear)
