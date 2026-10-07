# aoi33: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[33]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_LOW_VIEW` = `True`
- `AGE_LOW_SHARE` = `0.05`
- `FAST_RISE_OTHER_VEG` = `False`
- `LAST_CROP_ONLY` = `True`
- `LONE_LOW_VIEW_DAYS` = `5`
- `RADAR_TOP` = `0.6`
- `SIEVE_ACRES` = `0.15`
- `YOUNG_NEEDS_AGE` = `True`

## History



### 2026-10-05 (back-filled from the handover table) - rules and reasons until 5 Oct

the aoi20 set + `LONE_LOW_VIEW_DAYS 5`, `RADAR_TOP LOW_NOW (0.6)`, `FAST_RISE_OTHER_VEG False`; NEW inputs: radar v002_20261005, series hyb40m1late_20261001

Why: 5 Oct: 51135 / 9412 (standing and clear 26 Sep, cloudy 28 Sep view NDVI 0.17 / 0.24 -> harvested: a lone low newest view within 5 days of a held view, with VH not in the low part of its range, is a cloud, not a cut; harvested 61.1 -> 23.6 ac); 41218 / 41453 (user: rice; rain-sown mid May, NDVI up in 25-30 days to a late-June top, flat ~0.5-0.6 after, no water: a fast green-up is not "other vegetation" here; all 829 other-veg pixels had this pattern, other veg 20.7 -> 0 ac); 43566 (same pattern, radar dipped on the last passes -> flooded: with the switch off a fast green-up held at its top is a grown crop). 68529 transplanted explained (radar + 23 Jun clear view both show water), no change. The old map called 79 % of aoi33 rain-fed dry-land crops; the user's labels say rice. Raw: direct seeded 1,076.5, harvested 23.6, transplanted 12.7, flooded 1.1, tree 0.7, young 0.2 ac. Sieve 0.15 + fields done. **Not locked yet.**

### 2026-10-05 - locked and delivered

sliver 0.15 field layer; delivery rice_map_2026-10-05 (rice = standing direct seeded + standing transplanted + young; S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-26) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 2751 rice fields, 0.0 of 952.0 ac (none).
