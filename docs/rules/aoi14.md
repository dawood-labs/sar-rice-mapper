# aoi14: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[14]` (code comments name the pixel or field behind each).

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



### 2026-10-06 13:09:00 - rule set chosen

`aoi13+universal` (switches below)

Why: best of 18 sets / switches on 121 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi13+universal | 62.8 | 70.2 |
| aoi116 | 61.2 | 70.2 |
| aoi13+long_flood | 61.2 | 71.1 |
| aoi13+young_while_radar_low | 61.2 | 71.1 |
| aoi13+any_water_transplanted | 61.2 | 71.1 |
| aoi13 | 61.2 | 71.1 |
| aoi13+sowing_from_radar | 61.2 | 71.1 |
| aoi83 | 56.2 | 72.7 |
| universal | 41.3 | 69.4 |
| aoi125 | 26.4 | 62.0 |
| aoi160 | 25.6 | 61.2 |
| aoi72 | 20.7 | 67.8 |
| aoi118 | 20.7 | 68.6 |
| aoi39 | 18.2 | 74.4 |
| aoi28 | 16.5 | 64.5 |
| aoi63 | 14.9 | 74.4 |
| aoi20 | 13.2 | 71.9 |
| aoi33 | 9.1 | 68.6 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-08-17) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 9 of 76 rice fields, 4.8 of 36.48 ac (rice standing transplanted 0.26 ac, young rice 4.54 ac).

### 2026-10-06 13:09:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi14/ (14 S2 dates >= 80 % clear)
