# aoi83: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[83]` (code comments name the pixel or field behind each).

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



### 2026-10-05 - 5 Oct

the aoi118 set + LONG_WATER_DAYS 30, NEVER_EMPTY_OTHER_VEG 0.35, TREE_LOW_BEFORE 2026-07-01, LONG_WATER_LEVEL 0.4, LONG_WATER_WET_VIEW, LONG_WATER_GREEN_LEAD 0, ANY_WATER_TRANSPLANTED

Why: pixels 86401 (flooded 18 May, planted late July: sowing at the end of a long water spell), 69008 / 67677 (never emptied, no water: other vegetation), 66338 (low in the monsoon: not a tree); field review of 79 fields: fields flooded into late Aug / Sep were young (user confirmed 7/7); any water sign -> transplanted (user rule)

### 2026-10-05 - locked and delivered

sliver 0.15 field layer; delivery rice_map_2026-10-05 (rice = standing direct seeded + standing transplanted + young; S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-13) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 14 of 1793 rice fields, 4.71 of 939.8 ac (rice standing transplanted 0.22 ac, young rice 4.5 ac).
