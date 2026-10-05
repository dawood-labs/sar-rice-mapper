#!/usr/bin/env bash
# Bring one AOI's radar run and Sentinel-2 dates up to the newest imagery (README "Bringing in newer imagery").
# Why a script: the same seven steps per AOI, logged per step, stopping at the first failure. season.end in the
# AOI's config must already be moved to the new end date. Starts Earth Engine exports: run only on the user's word.
# Usage: scripts/update_aoi_imagery.sh <aoi> <first new S2 date> <day after the last S2 date> <log folder>
set -euo pipefail
cd "$(dirname "$0")/.."
aoi=$1; s2_start=$2; s2_end=$3; logs=$4
cfg=config/aoi${aoi}_monsoon2026.yaml
py=.venv/bin/python
export GDAL_NUM_THREADS=ALL_CPUS
mkdir -p "$logs"
echo "== aoi$aoi $(date -u +%H:%M:%S) S2 dates $s2_start..$s2_end"
$py -m sar_pipeline.optical_export dates --configs "$cfg" --start "$s2_start" --end "$s2_end" > "$logs/aoi${aoi}_s2.log" 2>&1
$py -m sar_pipeline --config "$cfg" audit > "$logs/aoi${aoi}_audit.log" 2>&1
$py -m sar_pipeline --config "$cfg" new-run > "$logs/aoi${aoi}_newrun.log" 2>&1
run=$(sed -n 's#.*/runs/\([^ ]*\).*#\1#p' "$logs/aoi${aoi}_newrun.log" | head -1)
echo "   new radar run: $run"
for stage in export monitor download; do
  echo "   $stage $(date -u +%H:%M:%S)"
  extra=""; [ "$stage" = export ] && extra="--all"
  $py -m sar_pipeline --config "$cfg" $stage --run "$run" $extra --yes > "$logs/aoi${aoi}_${stage}.log" 2>&1
done
$py -m sar_pipeline --config "$cfg" stack --run "$run" > "$logs/aoi${aoi}_stack.log" 2>&1
echo "== done $(date -u +%H:%M:%S) run $run"
