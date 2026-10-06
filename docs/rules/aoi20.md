# aoi20: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[20]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_LOW_VIEW` = `True`
- `AGE_LOW_SHARE` = `0.05`
- `LAST_CROP_ONLY` = `True`
- `SIEVE_ACRES` = `0.15`
- `YOUNG_NEEDS_AGE` = `True`

## History



### 2026-10-05 (back-filled from the handover table) - rules and reasons until 5 Oct

the aoi28 set + `LAST_CROP_ONLY True` (judge only the crop standing now; tree test still on the whole season; `LAST_CROP_LOW_VIEW_DAYS 15`) + `AGE_FROM_LOW_VIEW True`; NEW inputs: radar v002_20261005 (DSC to 4 Oct, ASC to 25 Sep), series hyb40m1late_20261001 (S2 to 1 Oct)

Why: 5 Oct: 13014 (summer crop Apr-Jun, monsoon crop sown 7 Aug judged on the summer peak -> other veg: judge the last crop only); 12187 (the low view 3 Jul was masked, a cloud-shadowed 1 Oct view became the "low" -> harvested: keep views from 15 days before the low); 9454 / 10887 (empty field seen 2 Aug, fitted low a week later -> young at 50 days: age from the lowest clear view, now 60 days). Ground plots 99.4 % rice. Raw: direct seeded 153.5, transplanted 132.6, tree 11.7, harvested 5.5, flooded 1.0, young 0.6 ac. Sieve 0.15 + fields sliver 0.15 done (875 polygons, overlap 0). **Not locked yet** (user to accept: `cr.lock(20, 0.15)` + add 20 to `LOCKED_AOIS`).

### 2026-10-05 - locked and delivered

sliver 0.15 field layer; delivery rice_map_2026-10-05 (rice = standing direct seeded + standing transplanted + young; S2 dates >= 80 % clear)
