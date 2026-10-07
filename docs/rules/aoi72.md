# aoi72: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[72]` (code comments name the pixel or field behind each).

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




### 2026-10-05 (back-filled from the handover table) - rules and reasons until 5 Oct

as 28 + `YOUNG_AFTER_LAST_WATER True`, `AGE_FROM_WATER True`, `WATER_BOTTOM 0.3`, `WATER_FALL_LOOKBACK_DAYS 45`, `YOUNG_WHILE_RADAR_LOW True`, `WATER_SPELL_DAYS 30`; sieve 0.15 ac, final fields sliver 0.15

Why: 42109 / 44280 (last view water, radar up after: young rice), 27550 (transplanted mid July: age from the water spell), 26433 / 40765 (two-month flood: transplanted); 2 Oct (new session): 19247 (radar still low and rising, NDVI 0.47 still climbing: young; the fit held flat after the last view looked like 'held at top'), 42979 (water 18 May DSC + 14 Jun ASC, the DSC mid-June VH is an artefact pass: 30-day spell -> transplanted); 20586 young and 17310 transplanted accepted as they were. 9/9 labels agree. **Locked 2 Oct** on the OLD inputs (series to 28 Sep, radar v001).

### 2026-10-06 05:34:00 - rule set chosen

`aoi39+any_water_transplanted` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi39+any_water_transplanted | 80.5 | 88.0 |
| aoi39+young_while_radar_low | 80.0 | 88.0 |
| aoi39 | 80.0 | 88.0 |
| aoi39+sowing_from_radar | 80.0 | 89.0 |
| aoi118 | 79.0 | 86.0 |
| aoi72 | 79.0 | 86.0 |
| aoi28 | 78.0 | 84.5 |
| aoi160 | 77.0 | 84.5 |
| aoi13 | 76.5 | 86.0 |
| aoi125 | 76.5 | 84.5 |
| aoi116 | 75.0 | 86.0 |
| aoi83 | 75.0 | 86.0 |
| aoi39+long_flood | 74.5 | 88.0 |
| aoi33 | 26.0 | 84.5 |
| aoi20 | 26.0 | 84.5 |

### 2026-10-06 05:35:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi72/ (14 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-13) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 9 of 729 rice fields, 3.82 of 446.63 ac (rice standing transplanted 0.54 ac, young rice 3.27 ac).
