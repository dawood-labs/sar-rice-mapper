# aoi70: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[70]` (code comments name the pixel or field behind each).

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



### 2026-10-06 06:54:00 - rule set chosen

`aoi39+any_water_transplanted` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi39+any_water_transplanted | 64.0 | 77.5 |
| aoi39+sowing_from_radar | 54.0 | 75.5 |
| aoi39 | 54.0 | 76.0 |
| aoi39+long_flood | 54.0 | 76.0 |
| aoi39+young_while_radar_low | 54.0 | 76.0 |
| aoi160 | 50.5 | 69.0 |
| aoi83 | 50.5 | 67.0 |
| aoi125 | 50.0 | 69.5 |
| aoi28 | 48.5 | 67.5 |
| aoi13 | 48.0 | 66.5 |
| aoi116 | 33.5 | 65.0 |
| aoi72 | 31.0 | 66.0 |
| aoi118 | 31.0 | 66.0 |
| aoi20 | 22.5 | 69.0 |
| aoi33 | 17.0 | 69.0 |

### 2026-10-06 06:56:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi70/ (27 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-26) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 2 of 1838 rice fields, 0.59 of 784.57 ac (young rice 0.59 ac).
