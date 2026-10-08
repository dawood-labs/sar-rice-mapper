# aoi66: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[66]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_WATER` = `True`
- `AGE_LOW_SHARE` = `0.05`
- `ANY_WATER_TRANSPLANTED` = `True`
- `LONG_WATER_DAYS` = `30`
- `LONG_WATER_GREEN_LEAD` = `0`
- `LONG_WATER_LEVEL` = `0.4`
- `LONG_WATER_WET_VIEW` = `True`
- `NEVER_EMPTY_OTHER_VEG` = `0.35`
- `SIEVE_ACRES` = `0.15`
- `SOWING_FROM_RADAR` = `False`
- `TREE_LOW_BEFORE` = `2026-07-01`
- `TREE_NEEDS_NO_WATER` = `True`
- `VV_JUMP` = `inf`
- `WATER_BOTTOM` = `0.3`
- `WATER_END_TRACKS` = `earliest_unless_wet_view`
- `WATER_FALL_LOOKBACK_DAYS` = `45`
- `WATER_FALL_POLS` = `VV`
- `WATER_SPELL_DAYS` = `30`
- `YOUNG_AFTER_LAST_WATER` = `True`
- `YOUNG_NEEDS_AGE` = `True`
- `YOUNG_WHILE_RADAR_LOW` = `True`

## History






### 2026-10-06 05:41:00 - rule set chosen

`aoi83` (switches below)

Why: best of 15 sets / switches on 5 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi83+young_while_radar_low | 80.0 | 80.0 |
| aoi83+long_flood | 80.0 | 80.0 |
| aoi83+sowing_from_radar | 80.0 | 80.0 |
| aoi83+any_water_transplanted | 80.0 | 80.0 |
| aoi83 | 80.0 | 80.0 |
| aoi118 | 40.0 | 40.0 |
| aoi116 | 40.0 | 40.0 |
| aoi13 | 40.0 | 40.0 |
| aoi28 | 20.0 | 20.0 |
| aoi160 | 20.0 | 20.0 |
| aoi72 | 20.0 | 20.0 |
| aoi125 | 20.0 | 20.0 |
| aoi39 | 0.0 | 0.0 |
| aoi20 | 0.0 | 0.0 |
| aoi33 | 0.0 | 0.0 |

### 2026-10-06 05:41:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi66/ (10 S2 dates >= 80 % clear)

### 2026-10-06 06:01:00 - rule set chosen

`aoi83` (switches below)

Why: best of 15 sets / switches on 25 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi83+young_while_radar_low | 36.0 | 36.0 |
| aoi83+long_flood | 36.0 | 36.0 |
| aoi83+sowing_from_radar | 36.0 | 36.0 |
| aoi83+any_water_transplanted | 36.0 | 36.0 |
| aoi83 | 36.0 | 36.0 |
| aoi118 | 24.0 | 24.0 |
| aoi116 | 24.0 | 24.0 |
| aoi13 | 24.0 | 24.0 |
| aoi28 | 12.0 | 12.0 |
| aoi160 | 12.0 | 12.0 |
| aoi72 | 12.0 | 12.0 |
| aoi125 | 12.0 | 12.0 |
| aoi39 | 4.0 | 4.0 |
| aoi20 | 4.0 | 4.0 |
| aoi33 | 4.0 | 4.0 |

### 2026-10-06 06:01:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi66/ (10 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-18) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 4 rice fields, 0.0 of 0.81 ac (none).

### 2026-10-08 11:10:00 - v2 rules (weak AOI)

`aoi160` -> `aoi66_fields_v2.gpkg` (delivered files unchanged)

Why: rice vs not right on reviewed fields 88.0 -> 100.0 % (held-out half 91.7 -> 100.0 %); user, 7 Oct 2026
