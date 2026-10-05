#!/usr/bin/env bash
# Runs scripts/new_aoi_first_look.sh for each AOI as soon as its imagery update (a chain of
# scripts/update_aoi_imagery.sh calls teeing into <chain log>) prints "== done ... run <run>".
# Why: the imagery of several AOIs arrives one after another; the user asked (5 Oct) to go on with each AOI as soon
# as its data is there instead of waiting for all of them.
# Usage: scripts/follow_imagery_chain.sh <chain log> <series variant> <log folder> <aoi> [<aoi> ...]
set -uo pipefail
cd "$(dirname "$0")/.."
chain=$1; variant=$2; logs=$3; shift 3
for aoi in "$@"; do
  until run=$(awk -v a="== aoi$aoi " 'index($0, a) == 1 {f = 1; next} f && /^== aoi/ {exit} f && /^== done/ {print $NF; exit}' "$chain") && [ -n "$run" ]; do
    grep -q "FAILED aoi$aoi" "$chain" && { echo "aoi$aoi: imagery update FAILED, skipped"; continue 2; }
    sleep 60
  done
  scripts/new_aoi_first_look.sh "$aoi" "$run" "$variant" "$logs" || echo "aoi$aoi: first look FAILED"
done
