"""Evaluate calibrated model calibration and backtest historical bets.

Outputs:
- `predictor/calibration.csv` : calibration bins for each class (home/draw/away)
- `predictor/backtest_report.md` : summary metrics (Brier score, log loss, accuracy, ROI, max drawdown)
- prints a short summary to stdout
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

from prepare_data import build_features

MODEL = Path(__file__).parent / "model.joblib"
OUT_DIR = Path(__file__).parent


def brier_score_multi(y_true, probas, n_classes=3):
    # y_true: (n,), probas: (n, n_classes)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    return np.mean(np.sum((probas - y_bin) ** 2, axis=1))


def compute_calibration(probas, y_true, n_bins=10):
    # returns dataframe with columns: class, bin_lower, bin_upper, prob_pred, prob_true, count
    rows = []
    n_classes = probas.shape[1]
    for c in range(n_classes):
        prob = probas[:, c]
        true = (y_true == c).astype(int)
        frac_pos, mean_prob = calibration_curve(true, prob, n_bins=n_bins, strategy='uniform')
        # calibration_curve returns arrays length <= n_bins; allocate bins
        for i in range(len(frac_pos)):
            rows.append({
                'class': c,
                'bin_idx': i,
                'mean_pred_prob': float(mean_prob[i]),
                'frac_pos': float(frac_pos[i]),
            })
    return pd.DataFrame(rows)


def backtest_and_metrics(df, probas, ev_threshold=0.05, stake=1.0):
    df = df.copy()
    df[['p_home', 'p_draw', 'p_away']] = probas
    df['ev_home'] = df['p_home'] * df['odds_home'] - 1
    df['ev_draw'] = df['p_draw'] * df['odds_draw'] - 1
    df['ev_away'] = df['p_away'] * df['odds_away'] - 1

    picks = []
    for _, r in df.iterrows():
        for side, ev_col, odd_col in [('home','ev_home','odds_home'),('draw','ev_draw','odds_draw'),('away','ev_away','odds_away')]:
            ev = r[ev_col]
            odd = r[odd_col]
            if pd.isna(ev) or pd.isna(odd):
                continue
            if ev > ev_threshold:
                picks.append({
                    'date': r['date'],
                    'home': r['HomeTeam'],
                    'away': r['AwayTeam'],
                    'side': side,
                    'odds': odd,
                    'model_prob': r['p_'+side],
                    'ev': ev,
                    'actual': r.get('FTR'),
                })
    picks_df = pd.DataFrame(picks).sort_values('date')

    # simulate staking 1 unit per bet
    def pick_won(row):
        if row['side'] == 'home' and row['actual'] == 'H':
            return True
        if row['side'] == 'draw' and row['actual'] == 'D':
            return True
        if row['side'] == 'away' and row['actual'] == 'A':
            return True
        return False

    if picks_df.empty:
        return picks_df, {'n_bets':0}

    picks_df['won'] = picks_df.apply(pick_won, axis=1)
    picks_df['pnl'] = picks_df.apply(lambda r: (r['odds'] - 1) * stake if r['won'] else -stake, axis=1)
    picks_df['cum_pnl'] = picks_df['pnl'].cumsum()

    total_staked = (picks_df.shape[0]) * stake
    total_pnl = picks_df['pnl'].sum()
    roi = total_pnl / total_staked if total_staked > 0 else 0.0

    # max drawdown
    cum = picks_df['cum_pnl']
    peak = cum.cummax()
    drawdown = (peak - cum)
    max_drawdown = drawdown.max() if not drawdown.empty else 0.0

    stats = {
        'n_bets': int(picks_df.shape[0]),
        'wins': int(picks_df['won'].sum()),
        'losses': int((~picks_df['won']).sum()),
        'total_staked': float(total_staked),
        'total_pnl': float(total_pnl),
        'roi': float(roi),
        'avg_odds': float(picks_df['odds'].mean()),
        'max_drawdown': float(max_drawdown),
    }
    return picks_df, stats


def main():
    if not MODEL.exists():
        raise FileNotFoundError('Model not found. Run train_model.py first to create predictor/model.joblib')
    model = joblib.load(MODEL)

    df = build_features()
    # Drop non-numeric columns (including `FTR`) before prediction
    X = df.drop(columns=['label','FTR','date','HomeTeam','AwayTeam','odds_home','odds_draw','odds_away'])
    y = df['label']

    probas = model.predict_proba(X)

    # metrics
    brier = brier_score_multi(y.values, probas, n_classes=3)
    ll = log_loss(y.values, probas)
    preds = probas.argmax(axis=1)
    acc = accuracy_score(y.values, preds)

    calib_df = compute_calibration(probas, y.values, n_bins=10)
    calib_out = OUT_DIR / 'calibration.csv'
    calib_df.to_csv(calib_out, index=False)

    picks_df, stats = backtest_and_metrics(df, probas, ev_threshold=0.05, stake=1.0)
    picks_out = OUT_DIR / 'picks_eval.csv'
    picks_df.to_csv(picks_out, index=False)

    # write report
    rpt = OUT_DIR / 'backtest_report.md'
    with rpt.open('w') as f:
        f.write('# Backtest & Calibration Report\n\n')
        f.write('## Model metrics\n')
        f.write(f'- Brier score (multiclass): {brier:.6f}\n')
        f.write(f'- Log loss: {ll:.6f}\n')
        f.write(f'- Accuracy: {acc:.4f}\n')
        f.write('\n')
        f.write('## Backtest stats (EV threshold = 0.05, stake=1 unit)\n')
        for k,v in stats.items():
            f.write(f'- {k}: {v}\n')
        f.write('\n')
        f.write('Calibration bins saved to `predictor/calibration.csv`.\n')
        f.write('Detailed picks saved to `predictor/picks_eval.csv`.\n')

    # print summary
    print('Evaluation complete')
    print(f'Brier score: {brier:.6f}, LogLoss: {ll:.6f}, Acc: {acc:.4f}')
    print('Backtest stats:')
    for k,v in stats.items():
        print(f' - {k}: {v}')
    print(f'Wrote {calib_out} and {rpt} and {picks_out}')


if __name__ == '__main__':
    main()
