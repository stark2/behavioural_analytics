#!/usr/bin/env bash
# Loop behavio_client eval and print "<YYYY-mm-dd HH:MM:SS> <similarity_score>"
# Interval seconds can be set via INTERVAL_S env var. Default 300.
set -euo pipefail

PY="/Users/david/miniforge3/bin/python"
CLIENT="/Users/david/Documents/Data/python-workspace/behavioural_analytics/behavio_client.py"
CSV="session_mickey_1.csv"
USER="Mickey Mouse"
CONTRACT="795131459"
TOPN="10"
INTERVAL="${INTERVAL_S:-300}"

run_once() {
  "$PY" "$CLIENT" eval \
    --csv "$CSV" \
    --username "$USER" \
    --contract "$CONTRACT" \
    --explain \
    --topN "$TOPN"
}

extract_score() {
  # Prefer the compact block to avoid duplicates
  awk '
    /^=== EVAL response \(compact\) ===/ {inblock=1; next}
    /^=== EVAL response/ && inblock {exit}
    inblock {print}
  ' | sed -n 's/.*"similarity_score"[[:space:]]*:[[:space:]]*\([-0-9.+eE]*\).*/\1/p' | head -n1
}

while :; do
  start=$(date +%s)
  out="$(run_once 2>/dev/null || true)"
  score="$(printf '%s\n' "$out" | extract_score)"
  ts="$(date '+%F %T')"
  if [ -n "${score:-}" ]; then
    printf '%s %s\n' "$ts" "$score"
  else
    printf '%s NaN\n' "$ts"
  fi
  now=$(date +%s)
  elapsed=$(( now - start ))
  sleep_for=$(( INTERVAL - elapsed ))
  if [ $sleep_for -gt 0 ]; then sleep "$sleep_for"; fi
done
