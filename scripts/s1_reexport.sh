#!/usr/bin/env bash
# Re-export the Sentinel-1 season for AOIs whose run started too late (fix plan, issue 12: four
# AOIs were exported from 1 May, so they have no dry-season radar before the transplanting floods).
#
# Why a script: the same six stages in a fixed order per AOI, each skipping what is already done,
# so the job can be re-run after an interruption. It STARTS EARTH ENGINE EXPORTS (approved in the
# fix plan of 24 Sep 2026). The season start/end are rewritten in the AOI's config first; new-run
# then freezes that config, and every later stage (and the rule, through the newest complete run)
# picks the new run up automatically.
#
# Usage: scripts/s1_reexport.sh <id> [<id> ...]        (logs in logs/aoi<N>_monsoon2026_reexport.log)
set -u
PY=${PY:-.venv/bin/python}
SEASON=monsoon2026; START=2026-03-15; END=2026-09-24
mkdir -p logs
for id in "$@"; do
  C=config/aoi${id}_${SEASON}.yaml
  sed -i "s/start: '[0-9-]*', end: '[0-9-]*'/start: '$START', end: '$END'/" "$C"
  grep -q "start: '$START', end: '$END'" "$C" || { echo "aoi$id: season line not rewritten in $C"; continue; }
done
export PY SEASON
echo "$@" | tr ' ' '\n' | xargs -P 4 -I{} sh -c '
  C=config/aoi{}_$SEASON.yaml; L=logs/aoi{}_${SEASON}_reexport.log
  ( $PY -m sar_pipeline --config $C grid && $PY -m sar_pipeline --config $C audit \
    && $PY -m sar_pipeline --config $C new-run && $PY -m sar_pipeline --config $C export --all --yes \
    && $PY -m sar_pipeline --config $C monitor --yes && $PY -m sar_pipeline --config $C download --yes \
    && $PY -m sar_pipeline --config $C stack ) > $L 2>&1 && echo "aoi{} re-export DONE" || echo "aoi{} re-export FAILED (see $L)"'
