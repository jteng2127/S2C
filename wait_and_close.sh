#!/bin/bash

PID=$1
CCS_ID=$2

# Check if PID and CCS_ID is provided
if [ -z "$PID" ] || [ -z "$CCS_ID" ]; then
  echo "Usage: $0 <pid> <ccs_id>"
  exit 1
fi

# Check if the process is running
if ! kill -0 "$PID" 2>/dev/null; then
  echo "Process $PID is not running."
  exit 1
fi

echo "Precess $PID is running, waiting it to end..."

# Wait for the process to exit
while kill -0 "$PID" 2>/dev/null; do
  sleep 1
done

echo "Process $PID has exited."
twccli rm ccs -fs $CCS_ID
