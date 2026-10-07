# aoi90: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[90]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_WATER` = `True`
- `ANY_WATER_TRANSPLANTED` = `True`
- `GROWN_NOT_IF_WATER_VIEW` = `True`
- `HARVEST_NOT_IF_RADAR_RISING` = `True`
- `POND_IF_DRY_WATER` = `True`
- `RADAR_DECIDES_WITHOUT_OPTICAL` = `True`
- `SECOND_CROP_BY_RADAR` = `True`
- `SIEVE_ACRES` = `0.15`
- `WATER_BOTTOM` = `0.3`
- `WATER_FALL_LOOKBACK_DAYS` = `45`
- `WATER_SPELL_MIN_PASSES` = `1`
- `YOUNG_AFTER_LAST_WATER` = `True`
- `YOUNG_MIN_RISE_DAYS` = `40`
- `YOUNG_NEEDS_WATER` = `True`
- `YOUNG_RADAR_LOW_NEEDS_RISE` = `True`
- `YOUNG_WHILE_RADAR_LOW` = `True`

## History



### 2026-10-06 14:03:00 - rule set chosen

`aoi39+any_water_transplanted` (switches below)

Why: best of 18 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi39+any_water_transplanted | 61.5 | 72.5 |
| aoi39 | 56.0 | 72.5 |
| aoi39+young_while_radar_low | 56.0 | 72.5 |
| aoi39+sowing_from_radar | 55.0 | 72.5 |
| aoi39+long_flood | 53.0 | 72.5 |
| aoi39+universal | 50.0 | 67.5 |
| aoi63 | 49.0 | 67.0 |
| aoi28 | 49.0 | 62.5 |
| aoi160 | 48.5 | 63.0 |
| aoi125 | 48.5 | 63.0 |
| aoi83 | 47.0 | 63.0 |
| aoi13 | 46.0 | 62.5 |
| aoi72 | 44.0 | 65.0 |
| aoi118 | 44.0 | 64.0 |
| aoi116 | 38.5 | 63.5 |
| universal | 25.0 | 66.5 |
| aoi20 | 20.5 | 66.5 |
| aoi33 | 19.5 | 74.5 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-10-01) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 1765 rice fields, 0.0 of 681.45 ac (none).

### 2026-10-06 14:03:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi90/ (21 S2 dates >= 80 % clear)
