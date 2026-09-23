#!/usr/bin/env bash
# Run the standing-rice map end to end for one batch of AOIs (docs/09, README "Running many AOIs").
#
# Why a script: a batch is ten commands in a fixed order, some run in parallel, and running them by
# hand once left an AOI half-downloaded. Every step skips what is already done, so the script can be
# re-run after an interruption. It STARTS EARTH ENGINE EXPORTS: confirm the batch before running it.
#
# Usage: scripts/monsoon_batch_chain.sh <template_config> <split_dir> <batch_name> <id> [<id> ...]
#   e.g. scripts/monsoon_batch_chain.sh config/<working>.yaml <split_folder> batch3 12 34 56
# Output: per-AOI logs in logs/, one acres CSV per AOI in processed/_batch/s2_2026/<batch_name>_parts/,
# the evidence report per AOI in processed/_batch/s2_2026/report/.
set -u
TEMPLATE=$1; SPLIT=$2; NAME=$3; shift 3; IDS="$*"
PY=${PY:-.venv/bin/python}
SEASON=monsoon2026; START=2026-03-15; END=2026-09-24
mkdir -p logs "processed/_batch/s2_2026/${NAME}_parts"
export PY SEASON NAME
t0=$(date +%s); stamp() { echo "[$(( ($(date +%s) - t0) / 60 )) min] $*"; }

stamp "configs"
$PY -m sar_pipeline.prep batch-configs --template "$TEMPLATE" --split-dir "$SPLIT" --season-key $SEASON \
    --start $START --end $END --ids $IDS | tail -1
stamp "grid + audit"
echo $IDS | tr ' ' '\n' | xargs -P 6 -I{} sh -c 'C=config/aoi{}_$SEASON.yaml; ($PY -m sar_pipeline --config $C grid && $PY -m sar_pipeline --config $C audit) > logs/aoi{}_${SEASON}_prep.log 2>&1 || echo "aoi{} prep FAILED"'
stamp "tracks"
$PY -m sar_pipeline.prep batch-tracks --season-key $SEASON --flood-start 2026-05-01 --flood-end 2026-08-31 --ids $IDS | tail -3
cp processed/_batch/${SEASON}_track_choice.csv processed/_batch/${SEASON}_track_choice_${NAME}.csv
stamp "Sentinel-1 new-run + export"
echo $IDS | tr ' ' '\n' | xargs -P 4 -I{} sh -c 'C=config/aoi{}_$SEASON.yaml; ($PY -m sar_pipeline --config $C new-run && $PY -m sar_pipeline --config $C export --all --yes) > logs/aoi{}_${SEASON}_export.log 2>&1 || echo "aoi{} s1 export FAILED"'
stamp "Sentinel-1 monitor/download/stack (background) + Sentinel-2 export"
( echo $IDS | tr ' ' '\n' | xargs -P 8 -I{} sh -c 'C=config/aoi{}_$SEASON.yaml; ($PY -m sar_pipeline --config $C monitor --yes && $PY -m sar_pipeline --config $C download --yes && $PY -m sar_pipeline --config $C stack) > logs/aoi{}_${SEASON}_mds.log 2>&1 || echo "aoi{} s1 stack FAILED"' ) &
$PY -u -m sar_pipeline.monsoon_batch s2-export --ids $IDS --yes > logs/${NAME}_s2_export.log 2>&1; tail -1 logs/${NAME}_s2_export.log
wait
stamp "rule (6 at a time)"
echo $IDS | tr ' ' '\n' | xargs -P 6 -I{} sh -c '$PY -u -m sar_pipeline.monsoon_batch rule --ids {} --out processed/_batch/s2_2026/${NAME}_parts/aoi{}.csv > logs/${NAME}_rule_aoi{}.log 2>&1 || echo "aoi{} rule FAILED"'
stamp "evidence report (4 at a time)"
echo $IDS | tr ' ' '\n' | xargs -P 4 -I{} sh -c '$PY -m sar_pipeline.analysis.batch_report --ids {} > logs/report_aoi{}.log 2>&1 || echo "aoi{} report FAILED"'
stamp "$NAME DONE"
