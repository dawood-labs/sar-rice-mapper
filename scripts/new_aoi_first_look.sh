#!/usr/bin/env bash
# After scripts/update_aoi_imagery.sh: everything from the new radar run to the first rule-set comparison of a NEW AOI.
# Why a script: the same steps per AOI (README "Bringing in newer imagery" + "Second fresh start"), logged per step,
# so each AOI can go on as soon as its imagery has arrived. Never use it on a locked AOI (it re-pins the inputs).
# Steps: screen the new radar passes (final_audit new-passes) -> pin the run -> build + pin the 5-day series ->
# first clear views, vegetation (step 1), sowing (step 2) -> try_rules with the given AOIs' rule sets.
# Usage: scripts/new_aoi_first_look.sh <aoi> <run> <series variant> <log folder> [rule-set AOIs ...]
set -euo pipefail
cd "$(dirname "$0")/.."
aoi=$1; run=$2; variant=$3; logs=$4; shift 4
sources=${*:-"116 72 33 160"}
cfg=config/aoi${aoi}_monsoon2026.yaml
sr=processed/_batch/s2_2026_${variant}
py=".venv/bin/python -W ignore"
export GDAL_NUM_THREADS=ALL_CPUS PYTHONPATH=src
mkdir -p "$logs"
$py -c "import sys; from sar_pipeline.analysis import curve_rules as cr; sys.exit(f'aoi$aoi is LOCKED' if $aoi in cr.LOCKED_AOIS else 0)"
echo "== aoi$aoi first look $(date -u +%H:%M:%S) run $run series $variant"
$py -m sar_pipeline.analysis.final_audit new-passes --ids "$aoi" --run "$run" > "$logs/aoi${aoi}_newpasses.log" 2>&1
echo "   flagged new bad passes: $(grep -c ' True' "$logs/aoi${aoi}_newpasses.log" || true)"
$py -m sar_pipeline --config "$cfg" pin-run --run "$run" > "$logs/aoi${aoi}_pin.log" 2>&1
$py -m sar_pipeline.analysis.mask_experiment build --ids "$aoi" --variants "$variant" --jobs 1 > "$logs/aoi${aoi}_series.log" 2>&1
$py -c "from sar_pipeline.analysis import ndvi_5day as nd; print(nd.pin_analysis_series($aoi, '$sr'))" >> "$logs/aoi${aoi}_pin.log" 2>&1
$py -m sar_pipeline.analysis.first_clear --aoi "$aoi" --start 2026-09-01 --end 2026-09-30 > "$logs/aoi${aoi}_firstclear.log" 2>&1
$py -m sar_pipeline.analysis.first_clear --aoi "$aoi" --vegetation >> "$logs/aoi${aoi}_firstclear.log" 2>&1
$py -m sar_pipeline.analysis.vegetation_types --aoi "$aoi" --series-root "$sr" > "$logs/aoi${aoi}_step1.log" 2>&1
$py -m sar_pipeline.analysis.sowing_fresh --aoi "$aoi" --series-root "$sr" > "$logs/aoi${aoi}_step2.log" 2>&1
$py -m sar_pipeline.analysis.curve_rules try-rules --aoi "$aoi" --sources $sources > "$logs/aoi${aoi}_trials.log" 2>&1
echo "== aoi$aoi first look done $(date -u +%H:%M:%S)"
