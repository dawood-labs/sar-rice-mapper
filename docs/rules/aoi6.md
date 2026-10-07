# aoi6: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[6]` (code comments name the pixel or field behind each).

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



### 2026-10-06 13:02:00 - rule set chosen

`aoi13` (switches below)

Why: best of 18 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi13+long_flood | 57.0 | 75.0 |
| aoi13 | 57.0 | 75.0 |
| aoi13+young_while_radar_low | 57.0 | 75.0 |
| aoi13+any_water_transplanted | 57.0 | 75.0 |
| aoi13+sowing_from_radar | 57.0 | 75.0 |
| aoi116 | 56.0 | 75.0 |
| aoi83 | 50.0 | 74.5 |
| aoi13+universal | 49.0 | 67.0 |
| aoi118 | 48.0 | 75.0 |
| aoi72 | 46.0 | 70.5 |
| aoi160 | 37.0 | 56.5 |
| aoi125 | 36.5 | 57.0 |
| universal | 36.0 | 63.0 |
| aoi39 | 34.0 | 67.5 |
| aoi63 | 30.5 | 62.0 |
| aoi28 | 29.0 | 56.5 |
| aoi20 | 17.5 | 65.5 |
| aoi33 | 17.0 | 65.5 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-08-17) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 5 of 651 rice fields, 1.97 of 247.31 ac (young rice 1.97 ac).

### 2026-10-06 13:02:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi6/ (11 S2 dates >= 80 % clear)
