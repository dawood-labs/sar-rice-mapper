# aoi28: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[28]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_LOW_SHARE` = `0.05`
- `ANY_WATER_TRANSPLANTED` = `True`
- `SIEVE_ACRES` = `0.15`
- `YOUNG_NEEDS_AGE` = `True`

## History




### 2026-10-05 (back-filled from the handover table) - rules and reasons until 5 Oct

`AGE_LOW_SHARE 0.05`, `YOUNG_NEEDS_AGE True`, `SIEVE_ACRES 0.15`; final fields sliver 0.15

Why: 3345 sown end July shown as 45 days / young; sieve 0.5 erased all young rice. A tree rule (`TREE_LOW_BEFORE`) was tried for 7826 and **reverted** by the user. Locked.

### 2026-10-05 - 5 Oct

ANY_WATER_TRANSPLANTED added; re-locked

Why: user rule 5 Oct: direct seeded 58.6 -> 0 ac, transplanted 25.0 -> 83.6 ac; first lock kept in locked/aoi28_v1

### 2026-10-05 - locked and delivered

sliver 0.15 field layer; delivery rice_map_2026-10-05 (rice = standing direct seeded + standing transplanted + young; S2 dates >= 80 % clear)
