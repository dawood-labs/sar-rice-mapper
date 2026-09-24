#!/usr/bin/env bash
# Draw curve + chip sheets for a few fields of ONE AOI, safely next to other jobs.
#
# Why: several reviewers may ask for sheets at the same time while the review batch runs; each call
# loads a whole AOI (2-7 GB). A lock lets only one render run at a time, and it waits until
# MIN_FREE_GB of RAM are available, so the machine is never pushed into an out-of-memory kill.
#
# Usage: scripts/review_render.sh <field_id> [<field_id> ...]     (all from the same AOI)
# Output: processed/_batch/s2_2026/review/<aoi>/fields_extra/<field_id>_curve.png / _chips.png
set -u
MIN_FREE_GB=${MIN_FREE_GB:-9}
LOCK=${LOCK:-/tmp/review_render.lock}
exec 9>"$LOCK"
flock 9
while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt "$MIN_FREE_GB" ]; do sleep 15; done
MPLBACKEND=Agg .venv/bin/python -m sar_pipeline.review --render "$@"
