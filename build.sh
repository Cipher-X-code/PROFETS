#!/usr/bin/env bash
set -euo pipefail
mkdir -p out
echo "Finding .java sources..."
find src -name '*.java' > .javac_sources
if [ ! -s .javac_sources ]; then
  echo "No Java sources found under src/"
  rm -f .javac_sources
  exit 1
fi
javac -d out @.javac_sources
rm -f .javac_sources
echo "Compiled classes placed in out/"
