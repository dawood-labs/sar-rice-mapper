# aoi11: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[11]` (code comments name the pixel or field behind each).

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



### 2026-10-06 13:06:00 - rule set chosen

`aoi116` (switches below)

Why: best of 18 sets / switches on 77 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi13 | 85.7 | 89.6 |
| aoi116 | 85.7 | 89.6 |
| aoi116+young_while_radar_low | 85.7 | 89.6 |
| aoi116+long_flood | 85.7 | 89.6 |
| aoi116+sowing_from_radar | 85.7 | 89.6 |
| aoi116+any_water_transplanted | 85.7 | 89.6 |
| aoi116+universal | 83.1 | 87.0 |
| aoi83 | 72.7 | 89.6 |
| universal | 55.8 | 81.8 |
| aoi160 | 46.8 | 84.4 |
| aoi125 | 46.8 | 84.4 |
| aoi72 | 33.8 | 90.9 |
| aoi118 | 33.8 | 92.2 |
| aoi39 | 27.3 | 81.8 |
| aoi28 | 26.0 | 84.4 |
| aoi63 | 26.0 | 87.0 |
| aoi20 | 19.5 | 92.2 |
| aoi33 | 18.2 | 92.2 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-10-01) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 71 rice fields, 0.0 of 36.77 ac (none).

### 2026-10-06 13:06:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi11/ (14 S2 dates >= 80 % clear)
