# aoi72: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[72]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_WATER` = `True`
- `AGE_LOW_SHARE` = `0.05`
- `SIEVE_ACRES` = `0.15`
- `WATER_BOTTOM` = `0.3`
- `WATER_FALL_LOOKBACK_DAYS` = `45`
- `WATER_SPELL_DAYS` = `30`
- `YOUNG_AFTER_LAST_WATER` = `True`
- `YOUNG_NEEDS_AGE` = `True`
- `YOUNG_WHILE_RADAR_LOW` = `True`

## History


### 2026-10-05 (back-filled from the handover table) - rules and reasons until 5 Oct

as 28 + `YOUNG_AFTER_LAST_WATER True`, `AGE_FROM_WATER True`, `WATER_BOTTOM 0.3`, `WATER_FALL_LOOKBACK_DAYS 45`, `YOUNG_WHILE_RADAR_LOW True`, `WATER_SPELL_DAYS 30`; sieve 0.15 ac, final fields sliver 0.15

Why: 42109 / 44280 (last view water, radar up after: young rice), 27550 (transplanted mid July: age from the water spell), 26433 / 40765 (two-month flood: transplanted); 2 Oct (new session): 19247 (radar still low and rising, NDVI 0.47 still climbing: young; the fit held flat after the last view looked like 'held at top'), 42979 (water 18 May DSC + 14 Jun ASC, the DSC mid-June VH is an artefact pass: 30-day spell -> transplanted); 20586 young and 17310 transplanted accepted as they were. 9/9 labels agree. **Locked 2 Oct** on the OLD inputs (series to 28 Sep, radar v001).
