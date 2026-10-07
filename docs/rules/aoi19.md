# aoi19: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[19]` (code comments name the pixel or field behind each).

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



### 2026-10-06 06:04:00 - rule set chosen

`aoi39+any_water_transplanted` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi39+any_water_transplanted | 73.0 | 78.5 |
| aoi39+sowing_from_radar | 71.5 | 79.0 |
| aoi39+long_flood | 69.5 | 80.5 |
| aoi39+young_while_radar_low | 68.5 | 78.0 |
| aoi39 | 68.5 | 78.0 |
| aoi160 | 66.5 | 74.5 |
| aoi125 | 66.5 | 74.5 |
| aoi83 | 66.5 | 73.0 |
| aoi28 | 66.0 | 74.5 |
| aoi13 | 65.0 | 73.5 |
| aoi72 | 57.0 | 74.0 |
| aoi118 | 57.0 | 74.0 |
| aoi116 | 56.0 | 73.5 |
| aoi20 | 45.0 | 73.5 |
| aoi33 | 22.5 | 51.0 |

### 2026-10-06 06:05:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi19/ (13 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-13) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 32 of 1600 rice fields, 10.39 of 538.91 ac (rice standing transplanted 10.13 ac, young rice 0.26 ac).
