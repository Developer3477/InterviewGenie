#!/usr/bin/env bash
# Run InterviewGenie's server and restart it automatically if it dies.
#
# The sandbox that hosts this repo resets processes at every turn boundary, so
# a server started by hand is gone the moment a turn ends. This wrapper keeps
# retrying, which means the preview comes back on its own instead of waiting
# for someone to notice and restart it.
#
#   ./scripts/run.sh            # port 8422
#   PORT=9000 ./scripts/run.sh

set -u

PORT="${PORT:-8422}"
HOST="${HOST:-0.0.0.0}"
LOG="${IG_LOG:-/tmp/interviewgenie-server.log}"

cd "$(dirname "$0")/.." || exit 1

echo "interviewgenie → http://${HOST}:${PORT}  (log: ${LOG})"

while true; do
  INTERVIEWGENIE_LOG_LEVEL="${INTERVIEWGENIE_LOG_LEVEL:-INFO}" \
    python3 -m interviewgenie serve --port "${PORT}" --host "${HOST}" \
    >>"${LOG}" 2>&1
  code=$?
  echo "$(date '+%H:%M:%S') server exited (${code}); restarting in 2s" | tee -a "${LOG}"
  sleep 2
done
