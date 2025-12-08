#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "Step 1: download data"
python3 download_data.py

echo "Step 2: prepare features (quick check)"
python3 prepare_data.py

echo "Step 3: train model"
python3 train_model.py

echo "Step 4: backtest / pick generation"
python3 backtest.py

echo "Pipeline complete. See predictor/picks.csv"
