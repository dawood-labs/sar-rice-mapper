# aoi18: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[18]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_WATER` = `True`
- `AGE_LOW_SHARE` = `0.05`
- `ANY_WATER_TRANSPLANTED` = `True`
- `HARVEST_NEEDS_BOTH_POLS` = `True`
- `RADAR_DECIDES_WITHOUT_OPTICAL` = `True`
- `RADAR_STORY_IN_GAP` = `True`
- `SIEVE_ACRES` = `0.15`
- `SOWING_FROM_RADAR` = `True`
- `TONE_AND_CURVE` = `True`
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



### 2026-10-06 18:48:00 - rule set chosen

`aoi13+universal` (switches below)

Why: best of 18 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi13+universal | 41.0 | 62.5 |
| aoi13 | 40.0 | 59.0 |
| aoi13+long_flood | 40.0 | 59.0 |
| aoi13+sowing_from_radar | 40.0 | 59.0 |
| aoi13+any_water_transplanted | 40.0 | 59.0 |
| aoi13+young_while_radar_low | 40.0 | 59.0 |
| aoi63 | 37.0 | 59.0 |
| aoi39 | 36.0 | 58.5 |
| aoi72 | 36.0 | 63.5 |
| aoi116 | 36.0 | 59.0 |
| aoi118 | 33.0 | 60.5 |
| aoi83 | 32.5 | 61.5 |
| aoi28 | 32.0 | 63.5 |
| aoi125 | 30.5 | 59.5 |
| aoi160 | 30.0 | 59.5 |
| universal | 27.0 | 65.0 |
| aoi33 | 19.0 | 66.0 |
| aoi20 | 19.0 | 65.0 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-10-01) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 318 rice fields, 0.0 of 100.61 ac (none).

### 2026-10-06 18:48:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi18/ (12 S2 dates >= 80 % clear)
