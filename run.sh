#!/usr/bin/env bash
set -euo pipefail
if [ -z "${1-}" ]; then
  exec java -cp out com.example.App
else
  exec java -cp out com.example.App "$@"
fi
