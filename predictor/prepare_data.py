"""Load football-data CSVs, create basic features and labels.

This file implements a simple feature generation pipeline:
- result label: 0=home win, 1=draw, 2=away win
- features: home team historical form (last 5 matches), away form, goal difference averages,
  home advantage flag, and bookmaker implied probabilities (if available).
"""
import pandas as pd
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"


def load_all_csvs(data_dir=DATA_DIR):
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. Run download_data.py first.")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = f.name
            dfs.append(df)
        except Exception as e:
            print(f"Failed reading {f}: {e}")
    return pd.concat(dfs, ignore_index=True)


def normalize_odds(df):
    # Try commonly present bookmaker columns (Bet365) and compute implied probabilities
    for prefix in ["B365", "PS"]:
        h = f"{prefix}H"
        d = f"{prefix}D"
        a = f"{prefix}A"
        if {h, d, a}.issubset(df.columns):
            df["odds_home"] = df[h]
            df["odds_draw"] = df[d]
            df["odds_away"] = df[a]
            break
    # implied probabilities (na-safe)
    df["imp_home"] = 1 / df["odds_home"]
    df["imp_draw"] = 1 / df["odds_draw"]
    df["imp_away"] = 1 / df["odds_away"]
    # normalize so they sum to 1 (remove vigorish)
    s = df[["imp_home", "imp_draw", "imp_away"]].sum(axis=1)
    df["imp_home_n"] = df["imp_home"] / s
    df["imp_draw_n"] = df["imp_draw"] / s
    df["imp_away_n"] = df["imp_away"] / s
    return df


def label_result(df):
    # FTR: 'H','D','A'
    mapping = {"H": 0, "D": 1, "A": 2}
    if "FTR" not in df.columns:
        raise KeyError("Input CSVs do not contain FTR (full-time result) column")
    df["label"] = df["FTR"].map(mapping)
    return df


def compute_form_features(df, n=5):
    # basic rolling features per team: last n matches goal diff mean and win rate
    df = df.copy()
    df["date"] = pd.to_datetime(df.get("Date", df.get("date", pd.NaT)), dayfirst=True, errors="coerce")
    df = df.sort_values("date")

    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    team_stats = {t: {"gd": [], "res": []} for t in teams}

    rows = []
    for _, r in df.iterrows():
        home = r["HomeTeam"]
        away = r["AwayTeam"]
        fhome = team_stats[home]
        faway = team_stats[away]

        # compute aggregates
        def agg(lst):
            if not lst:
                return {"gd_mean": 0.0, "win_rate": 0.0}
            arr = np.array(lst[-n:])
            return {"gd_mean": float(arr.mean()), "win_rate": float((arr > 0).mean())}

        home_agg = agg(fhome["gd"]) if fhome else {"gd_mean": 0.0, "win_rate": 0.0}
        away_agg = agg(faway["gd"]) if faway else {"gd_mean": 0.0, "win_rate": 0.0}

        row = r.to_dict()
        row.update({
            "home_gd_mean": home_agg["gd_mean"],
            "home_win_rate": home_agg["win_rate"],
            "away_gd_mean": away_agg["gd_mean"],
            "away_win_rate": away_agg["win_rate"],
            "home_adv": 1,
        })
        rows.append(row)

        # update stats with this match result
        hg = r.get("FTHG", 0)
        ag = r.get("FTAG", 0)
        gd_home = hg - ag
        gd_away = ag - hg
        # store goal diffs
        team_stats[home]["gd"].append(gd_home)
        team_stats[away]["gd"].append(gd_away)
        # store res as numeric for win rate
        # home result
        if r.get("FTR") == "H":
            team_stats[home]["res"].append(1)
            team_stats[away]["res"].append(0)
        elif r.get("FTR") == "A":
            team_stats[home]["res"].append(0)
            team_stats[away]["res"].append(1)
        else:
            team_stats[home]["res"].append(0)
            team_stats[away]["res"].append(0)

    out = pd.DataFrame(rows)
    return out


def build_features(data_dir=DATA_DIR):
    df = load_all_csvs(data_dir)
    df = label_result(df)
    df = normalize_odds(df)
    df = compute_form_features(df)

    # select useful columns
    features = [
        "home_gd_mean",
        "home_win_rate",
        "away_gd_mean",
        "away_win_rate",
        "imp_home_n",
        "imp_draw_n",
        "imp_away_n",
    ]
    # Keep `FTR` (full-time result) so backtest/evaluation can access actual outcomes
    df_features = df[[*features, "label", "FTR", "date", "HomeTeam", "AwayTeam", "odds_home", "odds_draw", "odds_away"]].copy()
    return df_features.dropna()


if __name__ == "__main__":
    out = build_features()
    print("Built features, sample:")
    print(out.head())
