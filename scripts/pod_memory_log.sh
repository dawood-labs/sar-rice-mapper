#!/usr/bin/env bash
# Append the container's memory use to logs/pod_memory.log once a minute.
#
# Why: the pod was replaced eight times in 30 hours (docs outside the repo: instance_restarts note);
# `free` and `uptime` show the NODE, not the container, so they never showed the pressure. The
# cgroup files below are the numbers the kernel enforces (memory.max) and records (memory.peak,
# oom_kill in memory.events). A log line per minute means the next restart carries evidence.
#
# Usage: scripts/pod_memory_log.sh &      (stops when the container does; safe to leave running)
set -u
mkdir -p logs
G=/sys/fs/cgroup
echo "# time  current_GiB  peak_GiB  max_GiB  oom_kill  load1" >> logs/pod_memory.log
while true; do
  cur=$(awk '{printf "%.1f", $1/2^30}' $G/memory.current 2>/dev/null || echo na)
  peak=$(awk '{printf "%.1f", $1/2^30}' $G/memory.peak 2>/dev/null || echo na)
  max=$(awk '{if ($1=="max") print "max"; else printf "%.1f", $1/2^30}' $G/memory.max 2>/dev/null || echo na)
  oom=$(awk '$1=="oom_kill"{print $2}' $G/memory.events 2>/dev/null || echo na)
  load=$(cut -d' ' -f1 /proc/loadavg)
  echo "$(date -u +%FT%TZ)  $cur  $peak  $max  $oom  $load" >> logs/pod_memory.log
  sleep 60
done
