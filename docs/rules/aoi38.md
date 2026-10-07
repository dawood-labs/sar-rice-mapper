# aoi38: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[38]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_WATER` = `True`
- `CANOPY_NOT_FLOODED` = `True`
- `CANOPY_OVER_RADAR_WATER` = `True`
- `GREEN_VIEW_NOT_FLOODED` = `True`
- `GROWN_NOT_IF_WATER_VIEW` = `True`
- `HARVEST_NOT_IF_RADAR_RISING` = `False`
- `POND_IF_DRY_WATER` = `True`
- `POND_NEEDS_NO_CROP` = `True`
- `RADAR_DECIDES_WITHOUT_OPTICAL` = `True`
- `RADAR_STORY_IN_GAP` = `True`
- `SECOND_CROP_BY_RADAR` = `True`
- `SIEVE_ACRES` = `0.15`
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



### 2026-10-06 13:55:00 - rule set chosen

`aoi63` (switches below)

Why: best of 18 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi63+universal | 78.0 | 79.5 |
| aoi63+young_while_radar_low | 78.0 | 79.5 |
| aoi63+any_water_transplanted | 78.0 | 79.5 |
| aoi63 | 78.0 | 79.5 |
| aoi13 | 76.0 | 81.5 |
| aoi63+sowing_from_radar | 75.5 | 79.5 |
| aoi39 | 75.0 | 82.0 |
| aoi72 | 70.5 | 82.5 |
| aoi118 | 70.0 | 81.0 |
| aoi116 | 69.0 | 81.5 |
| aoi28 | 66.5 | 81.0 |
| aoi160 | 65.0 | 80.0 |
| aoi125 | 64.5 | 81.0 |
| aoi63+long_flood | 45.0 | 79.0 |
| aoi83 | 42.0 | 82.5 |
| universal | 41.5 | 81.5 |
| aoi33 | 39.0 | 81.5 |
| aoi20 | 39.0 | 81.5 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-23) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 530 rice fields, 0.0 of 560.99 ac (none).

### 2026-10-06 13:55:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi38/ (13 S2 dates >= 80 % clear)
