# aoi13: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[13]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_WATER` = `True`
- `AGE_LOW_SHARE` = `0.05`
- `ANY_WATER_TRANSPLANTED` = `True`
- `SIEVE_ACRES` = `0.15`
- `SOWING_FROM_RADAR` = `True`
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



### 2026-10-05 - 5 Oct

the aoi116 set + ANY_WATER_TRANSPLANTED

Why: 160 fields judged by agents: aoi116 set 74 % vs aoi83 set 64 %, aoi118 42 %; fields flooded June-August and planted at the END of the water spell (SOWING_FROM_RADAR); no clear S2 view 17 Aug - 1 Oct, radar decides; young vs transplanted mix-ups accepted by the user

### 2026-10-05 - locked and delivered

sliver 0.15 field layer; delivery rice_map_2026-10-05 (rice = standing direct seeded + standing transplanted + young; S2 dates >= 80 % clear)
