#!/usr/bin/env bash
# Freeze the current results as a named baseline before a new round of fixes.
#
# Why: every change of a fix round is judged against the map as it was delivered (plots, negatives,
# noted fields, acres). The copy must never be touched by a later run, so it is a real copy (not a
# link: rasterio rewrites files in place) into new folders, and the script refuses to overwrite.
#
# Use:   scripts/freeze_baseline.sh v4
# Makes: processed/_batch/s2_2026/delivery_<name>_baseline/   (the delivery folder as it is)
#        processed/_batch/s2_2026/baseline_<name>/             (raw and final rule maps of every AOI + fields/)
set -euo pipefail
name="${1:?usage: scripts/freeze_baseline.sh <name, e.g. v4>}"
src=processed/_batch/s2_2026
out_d="$src/delivery_${name}_baseline"
out_m="$src/baseline_${name}"
for d in "$out_d" "$out_m"; do
  [ -e "$d" ] && { echo "refusing: $d exists" >&2; exit 1; }
done
cp -a "$src/delivery" "$out_d"
mkdir -p "$out_m"
cp -a "$src"/aoi*/aoi*_monsoon2026.tif "$src"/aoi*/aoi*_monsoon2026_final.tif "$out_m"/
cp -a "$src/fields" "$out_m/fields"
echo "delivery -> $out_d ($(ls "$out_d" | wc -l) entries)"
echo "rule maps -> $out_m ($(ls "$out_m"/*.tif | wc -l) rasters), fields -> $out_m/fields"
