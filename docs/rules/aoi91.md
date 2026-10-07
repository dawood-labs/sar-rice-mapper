# aoi91: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[91]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_WATER` = `True`
- `CANOPY_NOT_FLOODED` = `True`
- `CANOPY_OVER_RADAR_WATER` = `True`
- `GREEN_VIEW_NOT_FLOODED` = `True`
- `GROWN_NOT_IF_WATER_VIEW` = `True`
- `HARVEST_NEEDS_BOTH_POLS` = `True`
- `HARVEST_NOT_IF_RADAR_RISING` = `False`
- `POND_IF_DRY_WATER` = `True`
- `POND_NEEDS_NO_CROP` = `True`
- `RADAR_DECIDES_WITHOUT_OPTICAL` = `True`
- `RADAR_STORY_IN_GAP` = `True`
- `SECOND_CROP_BY_RADAR` = `True`
- `SIEVE_ACRES` = `0.15`
- `TONE_AND_CURVE` = `True`
- `WATER_BOTTOM` = `0.3`
- `WATER_FALL_LOOKBACK_DAYS` = `45`
- `WATER_SMALL_UNSEEN` = `True`
- `WATER_SPELL_MIN_PASSES` = `1`
- `YOUNG_AFTER_LAST_WATER` = `True`
- `YOUNG_MIN_RISE_DAYS` = `40`
- `YOUNG_NEEDS_WATER` = `True`
- `YOUNG_RADAR_LOW_NEEDS_RISE` = `True`
- `YOUNG_WHILE_RADAR_LOW` = `True`

## History



### 2026-10-06 14:07:00 - rule set chosen

`aoi63+universal` (switches below)

Why: best of 18 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi63+universal | 71.5 | 77.5 |
| aoi63+any_water_transplanted | 70.5 | 76.0 |
| aoi63+sowing_from_radar | 70.0 | 75.5 |
| aoi63+long_flood | 69.5 | 75.5 |
| aoi63+young_while_radar_low | 69.5 | 75.5 |
| aoi63 | 69.5 | 75.5 |
| aoi39 | 67.5 | 76.0 |
| aoi83 | 65.0 | 73.5 |
| aoi13 | 64.5 | 72.5 |
| aoi160 | 59.0 | 74.5 |
| aoi125 | 59.0 | 74.5 |
| aoi28 | 57.5 | 74.0 |
| aoi118 | 57.0 | 74.0 |
| aoi116 | 56.5 | 73.0 |
| aoi72 | 53.0 | 73.5 |
| universal | 30.5 | 75.5 |
| aoi20 | 27.5 | 74.0 |
| aoi33 | 24.5 | 73.0 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-10-01) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 2125 rice fields, 0.0 of 1083.76 ac (none).

### 2026-10-06 14:08:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi91/ (23 S2 dates >= 80 % clear)
