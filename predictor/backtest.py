"""Backtest simple expected-value-based betting using model predictions and bookmaker odds.

This script expects `predictor/model.joblib` to exist. It loads features and model, computes
predicted probabilities, compares to normalized implied probabilities from bookmakers, and
outputs matches with positive EV.
"""
import joblib
from pathlib import Path
import pandas as pd
from prepare_data import build_features

MODEL = Path(__file__).parent / "model.joblib"


def backtest(ev_threshold=0.05):
    if not MODEL.exists():
        raise FileNotFoundError("Model not found. Run train_model.py first.")
    model = joblib.load(MODEL)
    df = build_features()

    # Drop non-feature columns including FTR (actual result)
    X = df.drop(columns=["label", "FTR", "date", "HomeTeam", "AwayTeam", "odds_home", "odds_draw", "odds_away"])
    probs = model.predict_proba(X)
    df[["p_home", "p_draw", "p_away"]] = probs

    # implied probabilities already computed as imp_*_n in prepare_data, reuse them
    # compute EV for each outcome: EV = prob * (odds) - 1
    df["ev_home"] = df["p_home"] * df["odds_home"] - 1
    df["ev_draw"] = df["p_draw"] * df["odds_draw"] - 1
    df["ev_away"] = df["p_away"] * df["odds_away"] - 1

    picks = []
    for _, r in df.iterrows():
        for side, ev, odd in [("home", r["ev_home"], r["odds_home"]), ("draw", r["ev_draw"], r["odds_draw"]), ("away", r["ev_away"], r["odds_away"])]:
            if pd.isna(ev) or pd.isna(odd):
                continue
            if ev > ev_threshold:
                picks.append({
                    "date": r["date"],
                    "home": r["HomeTeam"],
                    "away": r["AwayTeam"],
                    "side": side,
                    "odds": odd,
                    "model_prob": r[f"p_{side}"],
                    "ev": ev,
                })

    out = pd.DataFrame(picks).sort_values("ev", ascending=False)
    out_path = Path(__file__).parent / "picks.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} picks to {out_path}")
    return out


if __name__ == "__main__":
    backtest()
