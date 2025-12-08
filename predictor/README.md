# Predictor prototype

This folder contains a minimal prototype to build a soccer (association football) prediction
pipeline that can be used to identify potential value bets (real-money use requires
responsible gambling practices — see top-level README).

Structure
- `requirements.txt` — Python deps
- `download_data.py` — downloads historical CSVs from football-data.co.uk
- `prepare_data.py` — loads CSVs and creates simple features + labels
- `train_model.py` — trains an XGBoost multiclass model and saves it
- `backtest.py` — computes EV from model vs bookmaker odds and simulates simple staking
- `run_pipeline.sh` — convenience wrapper to run the steps

Calibration
- The training script now calibrates model probabilities using Platt scaling (`CalibratedClassifierCV`)
  so `predictor/model.joblib` contains a calibrated classifier that produces better probability
  estimates for expected-value calculations.

Notes
- This prototype uses public CSVs from football-data.co.uk (no API key). Downloading
  these files may be rate-limited; respect the source terms.
- The model is a starting point — tune features, hyperparameters, and validation carefully
  before risking real money.
