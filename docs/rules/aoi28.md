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

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-28) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 123 rice fields, 0.0 of 82.39 ac (none).
