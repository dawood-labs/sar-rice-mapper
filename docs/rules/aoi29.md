# aoi29: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[29]` (code comments name the pixel or field behind each).

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



### 2026-10-06 13:49:00 - rule set chosen

`aoi63` (switches below)

Why: best of 18 sets / switches on 143 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi63+universal | 77.6 | 89.5 |
| aoi63+young_while_radar_low | 77.6 | 89.5 |
| aoi63+any_water_transplanted | 77.6 | 89.5 |
| aoi63 | 77.6 | 89.5 |
| aoi63+sowing_from_radar | 72.7 | 83.9 |
| aoi28 | 65.0 | 72.7 |
| aoi63+long_flood | 59.4 | 80.4 |
| aoi118 | 57.3 | 72.7 |
| aoi13 | 56.6 | 70.6 |
| aoi72 | 56.6 | 71.3 |
| aoi116 | 50.3 | 67.1 |
| aoi125 | 49.0 | 56.6 |
| aoi160 | 46.2 | 49.7 |
| aoi39 | 45.5 | 50.3 |
| universal | 42.0 | 88.8 |
| aoi83 | 40.6 | 64.3 |
| aoi33 | 38.5 | 81.8 |
| aoi20 | 38.5 | 79.7 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-28) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 137 rice fields, 0.0 of 95.99 ac (none).

### 2026-10-06 13:50:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi29/ (34 S2 dates >= 80 % clear)
